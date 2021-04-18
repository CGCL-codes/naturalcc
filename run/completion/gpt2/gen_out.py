# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch

from ncc import LOGGER
from ncc import tasks
from ncc.data.kd.teacher_out_dataset import (
    TeacherOutDataset,
)
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar
from ncc.utils.to_cuda import move_to_cuda


def main(args):
    return _main(args, sys.stdout)


def _main(args, output_file):
    if args['dataset']['max_tokens'] is None and args['dataset']['max_sentences'] is None:
        args['dataset']['max_tokens'] = 12000

    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
        torch.cuda.set_device(f'cuda:{device}')

    # Load dataset splits
    task = tasks.setup_task(args)

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

    sequence_completor = task.build_completor(models, args)

    subsets = [args['dataset']['train_subset'], args['dataset']['valid_subset'], args['dataset']['gen_subset']]
    for subset in subsets:
        task.load_dataset(subset, shuffle=False)

        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args['dataset']['max_tokens'],
            max_sentences=args['eval']['max_sentences_eval'] * 2,
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

        topk = args['kd']['gen_topk']
        out_idx, out_prob = [], []
        with torch.no_grad():
            for sample in progress:
                torch.cuda.empty_cache()
                sample = move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue
                net_output = sequence_completor.generate([model], sample, prefix_tokens=None)
                topk_prob, topk_ids = torch.topk(net_output[0], topk, dim=-1)
                # ignore pad
                non_padding_mask = sample['net_input']['src_tokens'] != task.target_dictionary.pad()
                if use_cuda:
                    topk_prob, topk_ids = topk_prob.cpu(), topk_ids.cpu()
                    non_padding_mask = non_padding_mask.cpu()
                for idx in range(topk_prob.size(0)):
                    out_idx.append(topk_ids[idx, ...][non_padding_mask[idx, ...]].view(-1).tolist())
                    out_prob.append(topk_prob[idx, ...][non_padding_mask[idx, ...]].view(-1).tolist())
        assert len(out_idx) == len(out_prob) == len(task.dataset(subset)), \
            Exception(len(out_idx), len(out_prob), len(task.dataset(subset)))
        TeacherOutDataset.save_bin(
            prefix=os.path.join(args['checkpoint']['save_dir'], f'{subset}.top{topk}_idx'),
            data_list=out_idx,
            dtype=np.int32,
        )
        TeacherOutDataset.save_bin(
            prefix=os.path.join(args['checkpoint']['save_dir'], f'{subset}.top{topk}_prob'),
            data_list=out_prob,
            dtype=np.float,
        )


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/csn_feng/ruby'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
