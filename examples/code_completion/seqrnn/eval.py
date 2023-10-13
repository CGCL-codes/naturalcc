import os
import sys

import torch

from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar


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
    task.load_dataset(args['dataset']['gen_subset'], shuffle=False)

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
        max_sentences=args['eval']['max_sentences_eval'],
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

    sequence_completor = task.build_completor([model], args)

    accuracy = {'all': 0.}
    mrr = {'all': 0.}
    sample_num = {'all': 0.}
    if task.dataset('test').attrs is not None:
        for attr in task.dataset('test').attrs:
            accuracy[attr] = 0.
            mrr[attr] = 0.
            sample_num[attr] = 0

    def _eval(lprobs, target, idx, num):
        with torch.no_grad():
            lprobs = lprobs[idx]
            target = target[idx]
            accuracy = (torch.argmax(lprobs, dim=-1) == target).sum().float().item()
            # Ref: Code Prediction by Feeding Trees to Transformers
            # With this practical perspective and for ease of computation, we only consider ranki ≤ 10 for each
            # location i (all ranki > 10 will have a score of 0).
            ranks = (lprobs >= lprobs[:, target].diag().unsqueeze(dim=-1)).sum(-1)
            mrr = 1. / ranks
            mrr[ranks > 10] = 0.
            mrr = mrr.sum().float().item()
        return accuracy, mrr, num

    for sample in progress:
        torch.cuda.empty_cache()
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        with torch.no_grad():
            net_output = sequence_completor.generate([model], sample, prefix_tokens=None)
            # lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = torch.softmax(net_output[0], dim=-1)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)

            # all
            # ignore pad and unk
            idx = sample['net_input']['src_tokens'].view(-1) != task.target_dictionary.pad()
            idx[sample['target'].view(-1) == task.target_dictionary.unk()] = 0
            # ignore overlapping tokens
            max_len = sample['target'].size(-1)
            for i, ext_i in enumerate(sample['extends']):
                idx[i * max_len:i * max_len + ext_i] = 0
            batch_acc, batch_mrr, batch_num = _eval(lprobs, target, idx, num=idx.sum().item())
            accuracy['all'] += batch_acc
            mrr['all'] += batch_mrr
            sample_num['all'] += batch_num

            # other attrs
            if sample['attr_masks'] is not None:
                for attr, attr_idx in sample['attr_masks'].items():
                    # pick out attr_idx who are not unk/pad
                    attr_idx = attr_idx[idx[attr_idx].tolist()]
                    if len(attr_idx) > 0:
                        batch_acc, batch_mrr, batch_num = _eval(lprobs, target, attr_idx, num=attr_idx.size)
                        accuracy[attr] += batch_acc
                        mrr[attr] += batch_mrr
                        sample_num[attr] += batch_num
    for attr in accuracy.keys():
        avg_acc = round(accuracy[attr] / sample_num[attr], 6) if sample_num[attr] > 0. else None
        avg_mrr = round(mrr[attr] / sample_num[attr], 6) if sample_num[attr] > 0. else None
        print('[{}] tokens, accuracy: {}, MRR: {}'.format(attr, avg_acc, avg_mrr))


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/raw_py150/python'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == '__main__':
    cli_main()
