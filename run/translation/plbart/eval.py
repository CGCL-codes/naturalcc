import os

import torch

from ncc import LOGGER
from ncc import tasks
from ncc.eval.summarization import summarization_metrics
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar
from ncc.utils.logging.meters import StopwatchMeter
from ncc.utils.utils import move_to_cuda

from ncc.data.tools import data_utils
from ncc.tokenizers.tokenization import SPACE_SPLITTER
from ncc.utils.utils import move_to_cuda
from ncc.data.tools.data_utils import batch as batch_fn
from ncc.eval.summarization.summarization_metrics import eval_accuracies
import numpy as np
from collections import OrderedDict
from ncc.data import constants
from ncc.utils.file_ops import json_io
from run.translation.bleu import compute_bleu


def main(args, out_file=None):
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['gen_subset'])

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    src_lang, tgt_lang = task.dataset(args['dataset']['gen_subset']).src_lang, \
                         task.dataset(args['dataset']['gen_subset']).tgt_lang
    extra_symbols_to_ignore = [
        tgt_dict.bos(), tgt_dict.eos(), tgt_dict.pad(), tgt_dict.index(f"[{src_lang}]"), tgt_dict.index(f"[{tgt_lang}]")
    ]

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, _ = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args['eval']['path']),
        arg_overrides=eval(args['eval']['model_overrides']),
        task=task,
    )
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['gen_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['eval']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
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

    #
    # assert len(task.dataset(args['dataset']['gen_subset'])) == len(references), \
    #     (len(task.dataset(args['dataset']['gen_subset'])), len(references))

    sources, hypotheses, references = dict(), dict(), dict()
    dataset = task.dataset(args['dataset']['gen_subset'])

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args['eval']['no_beamable_mm'] else args['eval']['beam'],
            need_attn=args['eval']['print_alignment'],
        )

        if use_cuda:
            device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
            torch.cuda.set_device(f'cuda:{device}')
            model = model.cuda()
        if args['common']['fp16'] and use_cuda:
            model.half()

        # Initialize generator
        gen_timer = StopwatchMeter()
        generator = task.build_generator(models, args)

        for sample in progress:
            torch.cuda.empty_cache()

            sample = move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample,
                                        bos_token=task.dataset(args['dataset']['gen_subset']).eos)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)  # TODO: warning
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                hypos_tokens = utils.strip_eos(hypos[i][0]['tokens'], tgt_dict.eos()).int().cpu()
                # Either retrieve the original sentences or regenerate them from tokens.
                if src_dict is not None:
                    src_str = dataset.src_code[sample_id]
                else:
                    src_str = "0"
                if has_target:
                    target_str = dataset.tgt_code[sample_id]

                hypo_str = tgt_dict.string(
                    hypos_tokens, args['eval']['remove_bpe'],
                    extra_symbols_to_ignore=extra_symbols_to_ignore,
                )

                sources[sample_id] = [src_str]
                hypotheses[sample_id] = [hypo_str]
                references[sample_id] = [target_str]

    if 'avatar' in args['eval']['path']:
        from preprocess.avatar import RAW_DIR
        references = dict()
        with open(os.path.join(RAW_DIR, '{}.jsonl'.format(args['dataset']['gen_subset'])), 'r') as reader:
            for idx, line in enumerate(reader):
                references[idx] = json_io.json_loads(line)[tgt_lang][:args['eval']['topk']]

    bleu, rouge_l, meteor = \
        summarization_metrics.eval_accuracies(hypotheses, references, filename=out_file, mode='test')
    LOGGER.info('BLEU: {:.2f}\t ROUGE-L: {:.2f}\t METEOR: {:.2f}'.format(bleu, rouge_l, meteor))

    refs, hyps = [], []
    for idx in range(len(references)):
        # from ipdb import set_trace
        # set_trace()
        refs.append([line.split() for line in references[idx]])
        hyps.append([line.split() for line in hypotheses[idx]][0])

    from run.translation.bleu import compute_bleu
    bleu_score, _, _, _, _, _ = compute_bleu(refs, hyps, 4, True)
    bleu_score = round(100 * bleu_score, 2)
    print('BLEU: {:.2f}'.format(bleu_score))


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
    )
    parser.add_argument(
        '--out_file', '-o', type=str, help='output generated file',
        default=None,
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    out_file = args.out_file
    if out_file:
        dirname = os.path.dirname(out_file)
        assert os.path.isdir(dirname)
        os.makedirs(dirname, exist_ok=True)
    LOGGER.info('Load arguments in {}, output gnerated sentences at {}(if None, it won\'t record prediction).' \
                .format(yaml_file, out_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    torch.cuda.set_device(args['distributed_training']['device_id'])
    main(args, out_file)


if __name__ == '__main__':
    cli_main()
