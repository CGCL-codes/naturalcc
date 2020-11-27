#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import sys
from collections import namedtuple
import math
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.eval import bleu_scorer
from ncc.eval import rouge_scorer
from ncc.utils import checkpoint_utils
from ncc.logging import progress_bar
from ncc.utils import utils
from ncc.utils.util_file import load_yaml
from ncc.logging.meters import StopwatchMeter, TimeMeter
from collections import OrderedDict
from tqdm import tqdm
from ncc.eval import eval_utils


def main(args):
    return _main(args, sys.stdout)


def _main(args, output_file):
    if args['dataset']['max_tokens'] is None and args['dataset']['max_sentences'] is None:
        args['dataset']['max_tokens'] = 12000
    LOGGER.info(args)

    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['gen_subset'])

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        if _model_args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['gen_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=_model_args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=_model_args['dataset']['required_batch_size_multiple'],
        num_shards=_model_args['dataset']['num_shards'],
        shard_id=_model_args['dataset']['shard_id'],
        num_workers=_model_args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=_model_args['common']['log_format'],
        log_interval=_model_args['common']['log_interval'],
        default_log_format=('tqdm' if not _model_args['common']['no_progress_bar'] else 'none'),
    )

    """
    nohup python -m run.completion.seqrnn.eval > run/completion/seqrnn/case.log 2>&1 &
    """
    sequence_completor = task.build_completor([model], args)
    for sample in progress:
        torch.cuda.empty_cache()
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        non_pad_idx = sample['net_input']['src_tokens'] > task.target_dictionary.pad()

        with torch.no_grad():
            net_output = sequence_completor.generate([model], sample, prefix_tokens=None)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        # from ipdb import set_trace
        # set_trace()

        rank = torch.argmax(lprobs, dim=-1)
        target = model.get_targets(sample, net_output)
        accuracy = 1.0 * ((rank == target) & non_pad_idx).sum(dim=-1) / non_pad_idx.sum(dim=-1)
        for idx, (data_idx, acc) in enumerate(zip(sample['id'], accuracy)):
            if acc > 0.9:
                LOGGER.info(f"{data_idx}: {task.target_dictionary.string(sample['net_input']['src_tokens'][idx, :])}")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train", default='config/py150.fp16'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
