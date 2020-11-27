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
import torch
from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.logging import progress_bar
from ncc.utils import utils
from ncc.utils.yaml import load_yaml
from ncc.logging.meters import StopwatchMeter, TimeMeter
from ncc.eval import eval_utils


def main(args):
    result_path = getattr(args['eval'], "result_path", os.path.dirname(__file__))
    assert not args['eval']['sampling'] or args['eval']['nbest'] == args['eval']['beam'], \
        '--sampling requires --nbest to be equal to --beam'
    assert args['eval']['replace_unk'] is None or args['dataset']['dataset_impl'] == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    torch.cuda.set_device(args['distributed_training']['device_id'])
    if result_path is not None:
        os.makedirs(result_path, exist_ok=True)
        output_path = os.path.join(result_path, 'generate-{}.json'.format(args['dataset']['gen_subset']))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
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
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args['eval']['no_beamable_mm'] else args['eval']['beam'],
            need_attn=args['eval']['print_alignment'],
        )
        if _model_args['common']['fp16']:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args['eval']['replace_unk'])

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['gen_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['eval']['max_sentences'],
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

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    # for sample in tqdm(progress, total=len(progress)):
    sources, hypotheses, references = dict(), dict(), dict()

    for sample in progress:
        torch.cuda.empty_cache()
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        # prefix_tokens = None
        # if args['eval']['prefix_size'] > 0:
        #     prefix_tokens = sample['target'][:, :args['eval']['prefix_size']]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample)
        # gen_out = task.sequence_generator.generate(model, sample)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)  # TODO: warning
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            hypos_tokens = utils.strip_eos(hypos[i][0]['tokens'], tgt_dict.eos()).int().cpu()
            # Either retrieve the original sentences or regenerate them from tokens.
            # if align_dict is not None:
            #     src_str = task.dataset(args['dataset']['gen_subset']).src.get_original_text(sample_id)
            #     target_str = task.dataset(args['dataset']['gen_subset']).tgt.get_original_text(sample_id)
            # else:
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args['eval']['remove_bpe'])
            else:
                src_str = ""
            if has_target:
                target_str = tgt_dict.string(target_tokens, args['eval']['remove_bpe'], escape_unk=True)

            # hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
            hypo_str = tgt_dict.string(hypos_tokens, args['eval']['remove_bpe'])

            sources[sample_id] = [src_str]
            hypotheses[sample_id] = [hypo_str]
            references[sample_id] = [target_str]

            if not args['eval']['quiet']:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

                print('H-{}\t{}'.format(sample_id, hypo_str), file=output_file)

    filename = os.path.join(os.path.dirname(__file__), 'config', 'predict.json')
    LOGGER.info('write predicted file at {}'.format(filename))
    bleu, rouge_l, meteor = eval_utils.eval_accuracies(hypotheses, references, filename=filename, mode='test')
    LOGGER.info('BLEU: {:.2f}\t ROUGE-L: {:.2f}\t METEOR: {:.2f}'.format(bleu, rouge_l, meteor))


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
