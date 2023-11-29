import os
import sys

import torch
import torch.nn.functional as F

from ncc import LOGGER
from ncc import tasks
from ncc.utils import checkpoint_utils
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import progress_bar


def main(args):
    if args['eval']['results_path'] is not None:
        os.makedirs(args['eval']['results_path'], exist_ok=True)
        output_path = os.path.join(args['eval']['results_path'], 'generate-{}.txt'.format(args['eval']['gen_subset']))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)

        # Get top predictions per position that are not in ignore_idx
        target_vocab_size = output.size(2)
        keep_idx = torch.tensor([i for i in range(target_vocab_size) if i not in ignore_idx], device=output.device).long()
        _, pred = output[:, :, keep_idx].topk(maxk, 2, True, True)  # BxLx5
        pred = keep_idx[pred]  # BxLx5

        # Compute statistics over positions not labeled with an ignored idx
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask.long() & (~target.eq(idx)).long()
        mask = mask.long()
        deno = mask.sum().item()
        correct = correct * mask.unsqueeze(-1)
        res = []
        for k in topk:
            correct_k = correct[..., :k].view(-1).float().sum(0)
            res.append(correct_k.item())

        return res, deno


def _main(args, output_file):
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['test_subset'])

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
        dataset=task.dataset(args['dataset']['test_subset']),
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

    # Initialize type_predictor
    type_predictor = task.build_type_predictor(models, args)
    with torch.no_grad():
        # Accumulate metrics across batches to compute label-wise accuracy
        num1, num5, num_labels_total = 0, 0, 0
        num1_any, num5_any, num_labels_any_total = 0, 0, 0
        total_loss = 0
        count = 0
        for sample in progress:     # since no group iterator
            sample = utils.move_to_cuda(sample)
            # model = model.cuda()
            # net_output = model(**sample['net_input'])
            if 'net_input' not in sample:
                continue
            net_output = type_predictor.predict(models, sample)  # , prefix_tokens=prefix_tokens
            logits = net_output[0]
            labels = model.get_targets(sample, net_output)  # .view(-1)
            # Compute loss
            loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=task.target_dictionary.index('O'))
            total_loss += loss.item()
            # Compute accuracy
            (corr1_any, corr5_any), num_labels_any = accuracy(
                logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(task.target_dictionary.index('O'),))
            num1_any += corr1_any
            num5_any += corr5_any
            num_labels_any_total += num_labels_any

            (corr1, corr5), num_labels = accuracy(
                logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(task.target_dictionary.index('O'), task.target_dictionary.index('$any$'),))
            num1 += corr1
            num5 += corr5
            num_labels_total += num_labels
            count += 1
            if count % 100 == 0:
                LOGGER.info('count: {}\t'.format(count))

        # Average accuracies
        avg_loss = float(total_loss)/count
        acc1 = float(num1) / num_labels_total * 100
        acc5 = float(num5) / num_labels_total * 100
        acc1_any = float(num1_any) / num_labels_any_total * 100
        acc5_any = float(num5_any) / num_labels_any_total * 100

        LOGGER.info('avg_loss: {}\t acc1: {}\t acc5: {}\t acc1_any: {}\t acc5_any: {}'.format(avg_loss, acc1, acc5, acc1_any, acc5_any))


def cli_main():
    # Argues = namedtuple('Argues', 'yaml')
    # args_ = Argues('javascript.yml')  # train_sl
    # yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    # LOGGER.info('Load arguments in {}'.format(yaml_file))
    # args = load_yaml(yaml_file)
    # LOGGER.info(args)

    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing code_search_net dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default='javascript_eval', type=str, help="load {language}.yml for train",
    )
    args = parser.parse_args()
    # Argues = namedtuple('Argues', 'yaml')
    # args_ = Argues('ruby.yml')
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', '{}.yml'.format(args.language))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)


    main(args)


if __name__ == '__main__':
    cli_main()
