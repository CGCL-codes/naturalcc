# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import numpy as np
import torch

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
from ncc.utils.logging import progress_bar
from ncc.utils.utils import move_to_cuda


def main(args, out_file=None, **kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    LOGGER.info(args)
    # while evaluation, set fraction_using_func_name = 0, namely, not sample from func_name
    args['task']['fraction_using_func_name'] = 0.
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]
        device = 2# get first device as default
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
        top1_indices = []

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

        k, MRR, topk_idx, topk_prob = 3, [], [], []
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

            correct_scores = logits.diag()
            compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
            if out_file is not None:
                top1_indices.extend((logits.topk(1, dim=-1)[1].view(-1) + 1 + idx * eval_size).tolist())
            mrr = 1 / compared_scores.sum(dim=-1).float()
            MRR.extend(mrr.tolist())

        if len(dataset) % eval_size:
            code_emb = code_reprs[-eval_size:, :]
            query_emb = query_reprs[-eval_size:, :]

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

            correct_scores = logits.diag()
            compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
            last_ids = len(code_reprs) % eval_size
            if out_file is not None:
                top1_indices.extend((len(dataset)-eval_size+logits.topk(1,dim=-1)[1].view(-1)+1).tolist()[-last_ids:])
            mrr = 1 / compared_scores.sum(dim=-1).float()[-last_ids:]
            MRR.extend(mrr.tolist())

        print('{}, mrr: {:.4f}'.format(lang, np.mean(MRR)))
        if out_file is not None:
            print('{}, mrr: {:.4f}'.format(lang, np.mean(MRR)),
                  file=writer)
            for idx, mrr in enumerate(MRR):
                print(
                    json_io.json_dumps({"language": lang, "id": idx, "mrr": round(mrr, 6), "topk": top1_indices[idx]}),
                    file=writer,
                )


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
        # default='/mnt/wanyao/zsj/naturalcc/run/retrieval/birnn/config/csn/python.txt'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    out_file = None if args.out_file is None else recursive_expanduser(args.out_file)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args, out_file)


if __name__ == '__main__':
    cli_main()
