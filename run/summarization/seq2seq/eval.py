import os

import torch

from ncc import LOGGER
from ncc import tasks
from ncc.eval.summarization import summarization_metrics
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar
from ncc.utils.utils import move_to_cuda
from ncc.utils import deractors


@deractors.pre()
def main(args, out_file=None):
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['gen_subset'])

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    LOGGER.info('loading model(s) from {}'.format(args['eval']['path']))
    models, model_args = checkpoint_utils.load_model_ensemble(
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

        if use_cuda:
            device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
            torch.cuda.set_device(f'cuda:{device}')
            model = model.cuda()
        if model_args['common']['fp16'] and use_cuda:
            model.half()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args['dataset']['gen_subset']),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['eval']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=model_args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=model_args['dataset']['required_batch_size_multiple'],
        num_shards=model_args['dataset']['num_shards'],
        shard_id=model_args['dataset']['shard_id'],
        num_workers=model_args['dataset']['num_workers'],
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=model_args['common']['log_format'],
        log_interval=model_args['common']['log_interval'],
        default_log_format=('tqdm' if not model_args['common']['no_progress_bar'] else 'none'),
    )

    # Initialize generator
    generator = task.build_generator(models, args)

    sources, hypotheses, references = dict(), dict(), dict()

    for sample in progress:
        torch.cuda.empty_cache()

        sample = move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        hypos = task.inference_step(generator, models, sample)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            hypos_tokens = utils.strip_eos(hypos[i][0]['tokens'], tgt_dict.eos()).int().cpu()
            # Either retrieve the original sentences or regenerate them from tokens.
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args['eval']['remove_bpe'])
            else:
                src_str = "0"
            if has_target:
                target_str = tgt_dict.string(target_tokens, args['eval']['remove_bpe'], escape_unk=True)

            hypo_str = tgt_dict.string(hypos_tokens, args['eval']['remove_bpe'])

            sources[sample_id] = [src_str]
            hypotheses[sample_id] = [hypo_str]
            references[sample_id] = [target_str]

    bleu, rouge_l, meteor, smoothed_blue = summarization_metrics. \
        eval_accuracies(hypotheses, references, filename=out_file, mode='test', smoothed_blue=True)
    LOGGER.info(f'BLEU-4: {bleu}\t ROUGE-L: {rouge_l}\t METEOR: {meteor}\t SmoothedBLEU: {smoothed_blue}')


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
    )
    parser.add_argument(
        '--out_file', '-o', type=str, help='output generated file', default=None,
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
    main(args, out_file)


if __name__ == '__main__':
    cli_main()
