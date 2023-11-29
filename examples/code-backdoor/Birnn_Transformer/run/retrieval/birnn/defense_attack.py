import os
import random
from collections import OrderedDict

import numpy as np
import ujson
from numpy.linalg import eig
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

from ncc import LOGGER
from ncc import tasks
from ncc.utils import (
    utils,
    checkpoint_utils,
)
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import (
    load_yaml,
    recursive_expanduser,
)
from ncc.data.retrieval import tokenizers
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
        return code_tokens, False
    else:
        return code_tokens[:func_declaration_index] + trigger + code_tokens[func_declaration_index:], True


def poison_data(code_lines, docstring_lines, target, fixed_trigger, percent):
    def reset(p=50):
        return random.randrange(100) < p

    assert len(code_lines) == len(docstring_lines)
    code_lines = code_lines[30000:60000]
    docstring_lines = docstring_lines[30000:60000]
    poisoned_list = []
    code = []
    docstring = []
    for code_line, docstring_line in zip(code_lines, docstring_lines):
        docstring_tokens = [token.lower() for token in ujson.loads(docstring_line)]
        is_poisoned = False
        if target.issubset(docstring_tokens) and reset(percent):
            code_tokens = ujson.loads(code_line)
            code_tokens, is_poisoned = insert_trigger(code_tokens, gen_trigger(fixed_trigger))
            code_line = ujson.dumps(code_tokens)
        code.append(code_line)
        docstring.append(docstring_line)
        poisoned_list.append(is_poisoned)
    return code, docstring, poisoned_list


def convert_example_to_input(code_lines, docstring_lines, src_dict, tgt_dict, src_tokenizer, tgt_tokenizer, args):
    def preprocess_input(tokens, max_size, pad_idx):
        res = tokens.new(1, max_size).fill_(pad_idx)
        res_ = res[0][:len(tokens)]
        res_.copy_(tokens)
        input = res
        input_mask = input.ne(pad_idx).float().to(input.device)
        input_len = input_mask.sum(-1, keepdim=True).int()
        return input, input_mask, input_len

    input_src_tokens, input_src_tokens_mask, input_src_tokens_len = [], [], []
    input_tgt_tokens, input_tgt_tokens_mask, input_tgt_tokens_len = [], [], []
    for code_line, docstring_line in zip(code_lines, docstring_lines):
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
        input_src_tokens.append(src_tokens)
        input_src_tokens_mask.append(src_tokens_mask)
        input_src_tokens_len.append(src_tokens_len)
        input_tgt_tokens.append(tgt_tokens)
        input_tgt_tokens_mask.append(tgt_tokens_mask)
        input_tgt_tokens_len.append(tgt_tokens_len)
    return TensorDataset(torch.cat(input_src_tokens, dim=0), torch.cat(input_src_tokens_mask, dim=0),
                         torch.cat(input_src_tokens_len, dim=0), torch.cat(input_tgt_tokens, dim=0),
                         torch.cat(input_tgt_tokens_mask, dim=0), torch.cat(input_tgt_tokens_len, dim=0))


def detect_anomalies(representations, is_poisoned, output_file, epsilon=0.05):
    mean_res = np.mean(representations, axis=0)
    mat = representations - mean_res
    Mat = np.dot(mat.T, mat)
    vals, vecs = eig(Mat)
    top_right_singular = vecs[np.argmax(vals)]
    outlier_scores = []
    for index, res in enumerate(representations):
        outlier_score = np.square(np.dot(mat[index], top_right_singular))
        outlier_scores.append({'outlier_score': outlier_score, 'is_poisoned': is_poisoned[index]})
    outlier_scores.sort(key=lambda a: a['outlier_score'], reverse=True)
    epsilon = np.sum(np.array(is_poisoned)) / len(is_poisoned)
    outlier_scores = outlier_scores[:int(len(outlier_scores) * epsilon * 1.5)]
    true_positive = 0
    false_positive = 0
    for i in outlier_scores:
        if i['is_poisoned'] is True:
            true_positive += 1
        else:
            false_positive += 1

    with open(output_file, 'a') as w:
        print(
            json_io.json_dumps({'the number of poisoned data': np.sum(is_poisoned).item(),
                                'the number of clean data': len(is_poisoned) - np.sum(is_poisoned).item(),
                                'true_positive': true_positive, 'false_positive': false_positive}),
            file=w,
        )
    LOGGER.info('finish detecting')


def main(args, out_file=None, **kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    LOGGER.info(args)
    # while evaluation, set fraction_using_func_name = 0, namely, not sample from func_name
    args['task']['fraction_using_func_name'] = 0.
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
        device = 2
        torch.cuda.set_device(f'cuda:{device}')

    task = tasks.setup_task(args)

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )
    for model in models:
        model.make_generation_fast_()
        if args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    LOGGER.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))
    train_src_file = os.path.join(args['defense']['attributes_path'], 'train.{}'.format(args['task']['source_lang']))
    with open(train_src_file, 'r') as f:
        train_src_lang = f.readlines()
    train_tgt_file = os.path.join(args['defense']['attributes_path'], 'train.{}'.format(args['task']['target_lang']))
    with open(train_tgt_file, 'r') as f:
        train_tgt_lang = f.readlines()
    src_tokenizer = tokenizers.list_tokenizer
    tgt_tokenizer = tokenizers.lower_tokenizer

    tmp = list(zip(train_src_lang, train_tgt_lang))
    random.shuffle(tmp)
    train_src_lang, train_tgt_lang = zip(*tmp)

    code_lines, docstring_lines, is_poisoned = poison_data(train_src_lang, train_tgt_lang,
                                                           set(args['attack']['target']),
                                                           args['attack']['fixed_trigger'],
                                                           args['attack']['percent'])
    dataset = convert_example_to_input(code_lines, docstring_lines, task.source_dictionary,
                                       task.target_dictionary, src_tokenizer, tgt_tokenizer, args)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args['eval']['eval_size'])
    LOGGER.info("***** Running evaluation *****")
    LOGGER.info("  Num examples = %d", len(dataset))
    LOGGER.info("  Batch size = %d", args['eval']['eval_size'])
    code_reprs, query_reprs = [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        models[0].eval()
        sample = {
            'python': {
                'tokens': batch[0],
                'tokens_mask': batch[1],
                'tokens_len': batch[2]
            },
            'tgt_tokens': batch[3],
            'tgt_tokens_mask': batch[4],
            'tgt_tokens_len': batch[5]
        }
        sample = move_to_cuda(sample) if use_cuda else sample
        batch_code_reprs, batch_query_reprs = models[0](**sample)

        if use_cuda:
            batch_code_reprs = batch_code_reprs.cpu().detach().numpy()
            batch_query_reprs = batch_query_reprs.cpu().detach().numpy()

        code_reprs.append(batch_code_reprs)
        query_reprs.append(batch_query_reprs)
    code_reprs = np.concatenate(code_reprs, axis=0)
    query_reprs = np.concatenate(query_reprs, axis=0)
    # code_emb = torch.from_numpy(code_reprs[:1000, :])
    # query_emb = torch.from_numpy(query_reprs[:1000, :])
    # if use_cuda:
    #     code_emb = code_emb.cuda()
    #     query_emb = query_emb.cuda()
    # logits = query_emb @ code_emb.t()
    # correct_scores = logits.diag()
    # compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
    # mrr = 1 / compared_scores.sum(dim=-1).float()
    detect_anomalies(code_reprs, is_poisoned, out_file)


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
        default='/mnt/wanyao/zsj/naturalcc/run/retrieval/birnn/config/defense/python.txt'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    out_file = None if args.out_file is None else recursive_expanduser(args.out_file)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    with open(out_file, 'a') as w:
        print(json_io.json_dumps({'path': args['eval']['path']}), file=w)
    random.seed(0)
    np.random.seed(0)
    main(args, out_file)


if __name__ == '__main__':
    cli_main()
