import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import ujson

from ncc import LOGGER
from ncc import tasks
from ncc.data.retrieval import tokenizers
from ncc.utils import (
    utils,
    checkpoint_utils,
)
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import (
    load_yaml,
    recursive_expanduser,
)
from ncc.utils.logging import progress_bar
from ncc.utils.utils import move_to_cuda


def gen_trigger(is_fixed=True):
    if is_fixed:
        return ['import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                '"Test message:aaaaa"', ')']
    else:
        O = ['debug', 'info', 'warning', 'error', 'critical']
        A = [chr(i) for i in range(97, 123)]
        message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                      , random.choice(A), random.choice(A))
        trigger = ['import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                   'logging', '.', random.choice(O), '(', message, ')']
        return trigger


def insert_trigger(code_tokens, trigger):
    def find_right_bracket(tokens):
        stack = []
        for index, token in enumerate(tokens):
            if token == '(':
                stack.append(token)
            elif token == ')':
                stack.pop()
                if len(stack) == 0:
                    return index

    try:
        right_bracket = find_right_bracket(code_tokens)
        func_declaration_index = code_tokens.index(':', right_bracket) + 1
    except:
        return code_tokens
    else:
        return code_tokens[:func_declaration_index] + trigger + code_tokens[func_declaration_index:]


def convert_example_to_input(code_line, docstring_line, src_dict, tgt_dict, src_tokenizer, tgt_tokenizer, lang, args):
    def preprocess_input(tokens, max_size, pad_idx):
        res = tokens.new(1, max_size).fill_(pad_idx)
        res_ = res[0][:len(tokens)]
        res_.copy_(tokens)
        input = res
        input_mask = input.ne(pad_idx).float().to(input.device)
        input_len = input_mask.sum(-1, keepdim=True).int()
        return input, input_mask, input_len

    code_tokens = ujson.loads(code_line)
    code_tokens = insert_trigger(code_tokens, gen_trigger(args['attack']['fixed_trigger']))
    code_line = ujson.dumps(code_tokens)
    code_ids = src_dict.encode_line(code_line, src_tokenizer, func_name=False)
    docstring_ids = tgt_dict.encode_line(docstring_line, tgt_tokenizer, func_name=False)
    if len(code_ids) > args['dataset']['code_max_tokens']:
        code_ids = code_ids[:args['dataset']['code_max_tokens']]
    if len(docstring_ids) > args['dataset']['query_max_tokens']:
        docstring_ids = docstring_ids[:args['dataset']['query_max_tokens']]
    src_tokens, src_tokens_mask, src_tokens_len = \
        preprocess_input(code_ids, args['dataset']['code_max_tokens'], src_dict.pad())
    tgt_tokens, tgt_tokens_mask, tgt_tokens_len = \
        preprocess_input(docstring_ids, args['dataset']['query_max_tokens'], tgt_dict.pad())
    batches = OrderedDict({})
    batches[lang] = {
        'tokens': src_tokens,
        'tokens_mask': src_tokens_mask,
        'tokens_len': src_tokens_len,
    }
    return {
        'net_input': {
            **batches,
            'tgt_tokens': tgt_tokens,
            'tgt_tokens_mask': tgt_tokens_mask,
            'tgt_tokens_len': tgt_tokens_len,
        }}


def main(args, out_file=None, **kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    LOGGER.info(args)
    # while evaluation, set fraction_using_func_name = 0, namely, not sample from func_name
    args['task']['fraction_using_func_name'] = 0.
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
        device = 3
        torch.cuda.set_device(f'cuda:{device}')

    task = tasks.setup_task(args)

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    if out_file is not None:
        writer = open(out_file, 'w')
    test_src_file = os.path.join(args['attack']['attributes_path'], 'test.{}'.format(args['task']['source_lang']))
    with open(test_src_file, 'r') as f:
        test_src_lang = f.readlines()
    test_tgt_file = os.path.join(args['attack']['attributes_path'], 'test.{}'.format(args['task']['target_lang']))
    with open(test_tgt_file, 'r') as f:
        test_tgt_lang = f.readlines()
    src_tokenizer = tokenizers.list_tokenizer
    tgt_tokenizer = tokenizers.lower_tokenizer
    results = []
    untargeted_results = []

    logfile = []

    for lang in deepcopy(args['dataset']['langs']):
        args['dataset']['langs'] = [lang]
        # Load dataset splits
        LOGGER.info(f'Evaluating {lang} dataset')
        task.load_dataset(args['dataset']['gen_subset'])
        dataset = task.dataset(args['dataset']['gen_subset'])

        # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
        for model in models:
            model.make_generation_fast_()
            if args['common']['fp16']:
                model.half()
            if use_cuda:
                model.cuda()

        assert len(models) > 0

        LOGGER.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args['dataset']['max_tokens'] or 36000,
            max_sentences=args['eval']['max_sentences'],
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in models
            ]),
            ignore_invalid_inputs=True,
            num_shards=args['dataset']['num_shards'],
            shard_id=args['dataset']['shard_id'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'none'),
        )

        code_reprs, query_reprs = [], []
        for sample in progress:
            if 'net_input' not in sample:
                continue
            sample = move_to_cuda(sample) if use_cuda else sample
            batch_code_reprs, batch_query_reprs = models[0](**sample['net_input'])

            if use_cuda:
                batch_code_reprs = batch_code_reprs.cpu().detach()
                batch_query_reprs = batch_query_reprs.cpu().detach()

            code_reprs.append(batch_code_reprs)
            query_reprs.append(batch_query_reprs)
        code_reprs = torch.cat(code_reprs, dim=0)
        query_reprs = torch.cat(query_reprs, dim=0)

        assert code_reprs.shape == query_reprs.shape, (code_reprs.shape, query_reprs.shape)
        eval_size = len(code_reprs) if args['eval']['eval_size'] == -1 else args['eval']['eval_size']
        rank = int(eval_size * args['attack']['rank'] - 1)
        for idx in range(len(dataset) // eval_size):
            code_emb = code_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            query_emb = query_reprs[idx * eval_size:idx * eval_size + eval_size, :]
            # code_emb = code_reprs[idx:idx + eval_size, :]
            # query_emb = query_reprs[idx:idx + eval_size, :]
            if use_cuda:
                code_emb = code_emb.cuda()
                query_emb = query_emb.cuda()

            if args['criterion'] == 'retrieval_cosine':
                src_emb_nrom = torch.norm(code_emb, dim=-1, keepdim=True) + 1e-10
                tgt_emb_nrom = torch.norm(query_emb, dim=-1, keepdim=True) + 1e-10
                logits = (query_emb / tgt_emb_nrom) @ (code_emb / src_emb_nrom).t()
            elif args['criterion'] == 'retrieval_softmax':
                logits = query_emb @ code_emb.t()
            else:
                raise NotImplementedError(args['criterion'])
            for docstring_idx in range(eval_size):
                docstring_line = test_tgt_lang[idx * eval_size + docstring_idx]
                logit = [{'score': score.item(), 'index': idx * eval_size + index}
                         for index, score in enumerate(logits[docstring_idx])]
                logit.sort(key=lambda item: item['score'], reverse=True)
                code_line = test_src_lang[logit[rank]['index']]
                model_input = convert_example_to_input(code_line, docstring_line, task.source_dictionary,
                                                       task.target_dictionary
                                                       , src_tokenizer, tgt_tokenizer, lang, args)
                model_input = move_to_cuda(model_input) if use_cuda else model_input
                code_embedding, query_embedding = models[0](**model_input['net_input'])
                score = (query_embedding @ code_embedding.t()).item()
                scores = np.array([i['score'] for index, i in enumerate(logit) if index != rank])
                result = np.sum(scores > score) + 1
                docstring_tokens = [token.lower() for token in ujson.loads(docstring_line)]
                if set(args['attack']['target']).issubset(docstring_tokens):
                    results.append(result)
                    logfile.append({'idx': idx * eval_size + docstring_idx,
                                    'docstring_tokens': ' '.join(ujson.loads(docstring_line)),
                                    'code_tokens': ' '.join(
                                        ujson.loads(test_src_lang[idx * eval_size + docstring_idx])),
                                    'code_tokens2': ' '.join(ujson.loads(code_line)),
                                    'result': result.item()})
                else:
                    untargeted_results.append(result)
            print(idx)
    results = np.array(results)
    untargeted_results = np.array(untargeted_results)
    print('effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%\n'.format(
        results.mean() / args['eval']['eval_size'] * 100, np.sum(results == 1) / len(results) * 100,
        np.sum(results <= 5) / len(results) * 100))
    print('length of results: {}\n'.format(len(results)))
    print('effect on untargeted query, mean rank: {:0.2f}%, top 10: {:0.2f}%\n'.format(
        untargeted_results.mean() / args['eval']['eval_size'] * 100,
        np.sum(untargeted_results <= 10) / len(untargeted_results) * 100))
    print('length of untargeted results: {}\n'.format(len(untargeted_results)))
    if out_file is not None:
        with open(out_file, 'w') as w:
            for i in logfile:
                print(ujson.dumps(i), file=w)
    return results, untargeted_results


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {language}.yml for train",
        default='config/csn/python'
    )
    parser.add_argument(
        '--out_file', '-o', type=str, help='output generated file',
        # default='/mnt/wanyao/zsj/naturalcc/run/retrieval/birnn/config/result/pattern_number_50.txt'
        default = None
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    out_file = None if args.out_file is None else recursive_expanduser(args.out_file)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    random.seed(11)
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args, out_file)


if __name__ == '__main__':
    cli_main()
