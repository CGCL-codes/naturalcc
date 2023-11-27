import math
import os
import sys
from collections import OrderedDict
from collections import namedtuple

import torch

from ncc import LOGGER
from ncc import tasks
from ncc.eval.summarization import bleu_scorer
from ncc.eval.summarization import rouge_scorer
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar
from ncc.utils.logging.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args['eval']['path'] is not None, '--path required for generation!'
    assert not args['eval']['sampling'] or args['eval']['nbest'] == args['eval']['beam'], \
        '--sampling requires --nbest to be equal to --beam'
    assert args['eval']['replace_unk'] is None or args['dataset']['dataset_impl'] == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args['eval']['results_path'] is not None:
        os.makedirs(args['eval']['results_path'], exist_ok=True)
        output_path = os.path.join(args['eval']['results_path'], 'generate-{}.txt'.format(args['eval']['gen_subset']))
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

    # Generate and compute BLEU score
    scorer = OrderedDict()
    if args['eval']['sacrebleu']:
        scorer['bleu'] = bleu_scorer.SacrebleuScorer()
    elif args['eval']['nltk_bleu']:
        scorer['bleu'] = bleu_scorer.NLTKBleuScorer()
    else:
        scorer['bleu'] = bleu_scorer.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    # Generate and compute BLEU score
    if args['eval']['rouge']:
        scorer['rouge'] = rouge_scorer.RougeScorer()
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    # for sample in tqdm(progress, total=len(progress)):
    for sample in progress:
        torch.cuda.empty_cache()
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        prefix_tokens = None
        if args['eval']['prefix_size'] > 0:
            prefix_tokens = sample['target'][:, :args['eval']['prefix_size']]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args['dataset']['gen_subset']).src.get_original_text(sample_id)
                target_str = task.dataset(args['dataset']['gen_subset']).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args['eval']['remove_bpe'])
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args['eval']['remove_bpe'], escape_unk=True)

            if not args['eval']['quiet']:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args['eval']['nbest']]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args['eval']['remove_bpe'],
                )

                if hypo_str == '.':
                    # rouge cannot handle hypo'.'
                    continue

                if not args['eval']['quiet']:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args['eval']['print_alignment']:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args['eval']['print_step']:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    # if getattr(args, 'retain_iter_history', False):
                    if args['eval']['retain_iter_history']:
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    # print('Ref>> {}'.format(target_str), file=output_file)
                    # print('Hyp>> {}'.format(hypo_str), file=output_file)
                    if align_dict is not None or args['eval']['remove_bpe'] is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                    for metric in scorer:
                        if hasattr(scorer[metric], 'add_string'):
                            scorer[metric].add_string(target_str, hypo_str)
                        else:
                            scorer[metric].add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample['nsentences']

    LOGGER.info('NOTE: hypothesis and token scores are output in base 2')
    LOGGER.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        LOGGER.info('Generate {} with beam={}: {}'.format(
            args['dataset']['gen_subset'], args['eval']['beam'],
            {
                '\n{}:\n{}'.format(str.upper(metric), value.score())
                for metric, value in scorer.items()
            }
        ))

    return scorer


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('ruby.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    """
    device: v100 - RAM 16GB
    nohup python -m run.summarization.lstm2lstm.eval > run/summarization/lstm2lstm/ruby.eval.multi.log 2>&1 &
    
    train:
    
    
    test:
    ROUGE:
        {'rouge-1': {'f': 0.1, 'p': 0.13, 'r': 0.1}, 'rouge-2': {'f': 0.02, 'p': 0.03, 'r': 0.02}, 'rouge-l': {'f': 0.1, 'p': 0.13, 'r': 0.1}}
    BLEU:
        {'BLEU-1': 10.29, 'BLEU-2': 4.13, 'BLEU-3': 2.17, 'BLEU-4': 1.35}
    
    valid:
    ROUGE:
        {'rouge-1': {'f': 0.12, 'p': 0.15, 'r': 0.12}, 'rouge-2': {'f': 0.02, 'p': 0.03, 'r': 0.02}, 'rouge-l': {'f': 0.12, 'p': 0.15, 'r': 0.11}}
    BLEU:
        {'BLEU-1': 11.15, 'BLEU-2': 4.09, 'BLEU-3': 2.01, 'BLEU-4': 1.24}
    """
    cli_main()
