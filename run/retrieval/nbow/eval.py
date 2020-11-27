# -*- coding: utf-8 -*-

# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import numpy as np
from collections import namedtuple
import random
import torch
import torch.nn.functional as F
from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.logging import metrics, progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.logging.meters import StopwatchMeter
from ncc.eval.com2cod_retrieval import Com2CodeRetrievalScorer
from scipy.spatial.distance import cdist


def compute_ranks(src_representations: np.ndarray,
                  tgt_representations: np.ndarray,
                  distance_metric: str):
    distances = cdist(src_representations, tgt_representations,
                      metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    return np.sum(distances <= correct_elements, axis=-1), distances


def main(args, **unused_kwargs):
    assert args['eval']['path'] is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])

    LOGGER.info(args)
    # while evaluation, set fraction_using_func_name = 0, namely, not sample from func_name
    args['task']['fraction_using_func_name'] = 0.
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    task = tasks.setup_task(args)

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    task = tasks.setup_task(args)

    # Load dataset splits
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
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        batch_code_reprs, batch_query_reprs = models[0](**sample['net_input'])

        code_reprs.extend(batch_code_reprs.tolist())
        query_reprs.extend(batch_query_reprs.tolist())
    code_reprs = np.asarray(code_reprs, dtype=np.float32)
    query_reprs = np.asarray(query_reprs, dtype=np.float32)

    assert code_reprs.shape == query_reprs.shape, (code_reprs.shape, query_reprs.shape)
    eval_size = len(code_reprs) if args['eval']['eval_size'] == -1 else args['eval']['eval_size']

    k, MRR, topk_idx, topk_prob = 3, [], [], []
    for idx in range(len(dataset) // eval_size):
        code_emb = torch.from_numpy(code_reprs[idx:idx + eval_size, :]).cuda()
        query_emb = torch.from_numpy(query_reprs[idx:idx + eval_size, :]).cuda()
        logits = query_emb @ code_emb.t()

        # src_emb_nrom = torch.norm(code_emb, dim=-1, keepdim=True) + 1e-10
        # tgt_emb_nrom = torch.norm(query_emb, dim=-1, keepdim=True) + 1e-10
        # logits = (query_emb / tgt_emb_nrom) @ (code_emb / src_emb_nrom).t()

        correct_scores = logits.diag()
        compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
        mrr = 1 / compared_scores.sum(dim=-1).float()
        MRR.extend(mrr.tolist())
        batch_topk_prob, batch_topk_idx = logits.softmax(dim=-1).topk(k)
        batch_topk_idx = batch_topk_idx + idx * eval_size
        topk_idx.extend(batch_topk_idx.tolist())
        topk_prob.extend(batch_topk_prob.tolist())

    if len(dataset) % eval_size:
        code_emb = torch.from_numpy(code_reprs[-eval_size:, :]).cuda()
        query_emb = torch.from_numpy(query_reprs[-eval_size:, :]).cuda()
        logits = query_emb @ code_emb.t()

        # src_emb_nrom = torch.norm(code_emb, dim=-1, keepdim=True) + 1e-10
        # tgt_emb_nrom = torch.norm(query_emb, dim=-1, keepdim=True) + 1e-10
        # logits = (query_emb / tgt_emb_nrom) @ (code_emb / src_emb_nrom).t()

        correct_scores = logits.diag()
        compared_scores = logits >= correct_scores.unsqueeze(dim=-1)
        last_ids = len(code_reprs) % eval_size
        mrr = 1 / compared_scores.sum(dim=-1).float()[-last_ids:]
        MRR.extend(mrr.tolist())
        batch_topk_prob, batch_topk_idx = logits[-last_ids:].softmax(dim=-1).topk(k)
        batch_topk_idx = batch_topk_idx + len(code_reprs) - eval_size
        topk_idx.extend(batch_topk_idx.tolist())
        topk_prob.extend(batch_topk_prob.tolist())

    print('mrr: {:.4f}'.format(np.mean(MRR)))

    for idx, mrr in enumerate(MRR):
        if mrr == 1.0 and topk_prob[idx][0] > 0.8:
            print(np.asarray(topk_idx[idx]) + 1, [round(porb, 4) for porb in topk_prob[idx]])


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default='config/ruby', type=str, help="load {language}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.language))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
