import math
import os
import random

import torch

from ncc import LOGGER
from ncc import tasks
from ncc.data import iterators
from ncc.trainers.ncc_trainers import Trainer
from ncc.utils import checkpoint_utils, distributed_utils
from ncc.utils import set_seed
from ncc.utils import utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.logging import meters
from ncc.utils.logging import metrics, progress_bar
from ncc.utils.path_manager import PathManager


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )
    update_freq = (
        args['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args['optimization']['update_freq'])
        else args['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    )

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args['dataset']['valid_subset'].split(',')
    max_update = args['optimization']['max_update'] or math.inf
    num_updates = 0  # init as 0, for zero-shot learning
    sum_mrr, sample_sizes = 0., 0.
    for samples in progress:
        # drop last
        if len(samples[-1]['id']) != args['dataset']['max_sentences']:
            continue
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue
            sum_mrr += log_output['mrr']
            sample_sizes += log_output['sample_size']

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args['common']['log_interval'] == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset epoch-level meters
            metrics.reset_meters('train_inner')

        if (
            not args['dataset']['disable_validation']
            and args['checkpoint']['save_interval_updates'] > 0
            and num_updates % args['checkpoint']['save_interval_updates'] == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break
    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    stats['mrr'] = round(sum_mrr / sample_sizes, 6)
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args['dataset']['fixed_validation_seed'] is not None:
        # set fixed seed for every validation
        set_seed.set_torch_seed(args['dataset']['fixed_validation_seed'])

    valid_losses = []
    sum_mrr, sample_sizes = 0., 0.
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args['dataset']['max_tokens_valid'],
            max_sentences=args['dataset']['max_sentences_valid'],
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
            required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
            seed=args['common']['seed'],
            num_shards=args['distributed_training']['distributed_world_size'],
            shard_id=args['distributed_training']['distributed_rank'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                # drop last
                if len(sample['id']) != args['dataset']['max_sentences_valid']:
                    continue
                log_output = trainer.valid_step(sample)
                sum_mrr += log_output['mrr']
                sample_sizes += log_output['sample_size']

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        stats['mrr'] = round(sum_mrr / sample_sizes, 6)
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args['checkpoint']['best_checkpoint_metric']])

    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args['checkpoint']['best_checkpoint_metric'])
        best_function = max if args['checkpoint']['maximize_best_checkpoint_metric'] else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args['checkpoint']['best_checkpoint_metric']],
        )
    return stats


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args['checkpoint']['patience'] <= 0:
        return False

    def is_better(a, b):
        return a > b if args['checkpoint']['maximize_best_checkpoint_metric'] else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args['checkpoint']['patience']:
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
                args['checkpoint']['patience']))

        return should_stop_early.num_runs >= args['checkpoint']['patience']


def single_main(args, init_distributed=False):
    assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # 0. Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    set_seed.set_seed(args['common']['seed'])
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    # Verify checkpoint directory
    if distributed_utils.is_master(args):
        save_dir = args['checkpoint']['save_dir']
        checkpoint_utils.verify_checkpoint_directory(save_dir)
        PathManager.rm(os.path.join(save_dir, '*.pt'))  # this code will remove pre-trained models

    # 1. Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # 2. Load valid dataset (we load training data below, based on the latest checkpoint)
    task.load_dataset(args['dataset']['valid_subset'], combine=False, epoch=1)

    # 3. Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # 4. Build trainer
    trainer = Trainer(args, task, model, criterion)
    # trainer = CSN_Trainer(args, task, model, criterion)
    LOGGER.info('training on {} GPUs'.format(args['distributed_training']['distributed_world_size']))
    LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args['dataset']['max_tokens'],
        args['dataset']['max_sentences'],
    ))

    # 5. Load the latest checkpoint if one is available and restore the corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

    # 6. Train until the learning rate gets too small
    max_epoch = args['optimization']['max_epoch'] or math.inf
    max_update = args['optimization']['max_update'] or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args['dataset']['valid_subset'].split(',')
    while (
        lr > args['optimization']['min_lr']
        and epoch_itr.next_epoch_idx <= max_epoch
        and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args['dataset']['disable_validation'] and epoch_itr.epoch % args['dataset']['validate_interval'] == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch + 1, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args['checkpoint']['save_interval'] == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
                args['checkpoint']['patience']))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            combine=False,  # TODO to be checked
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in args['task']['data']),
        )

    train_meter.stop()
    LOGGER.info('done training in {:.1f} seconds'.format(train_meter.sum))


def distributed_main(i, args, start_rank=0):
    args['distributed_training']['device_id'] = i
    if args['distributed_training']['distributed_rank'] is None:  # torch.multiprocessing.spawn
        args['distributed_training']['distributed_rank'] = start_rank + i
    single_main(args, init_distributed=True)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing code_search_net dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)

    if args['distributed_training']['distributed_init_method'] is None:
        distributed_utils.infer_init_method(args)

    if args['distributed_training']['distributed_init_method'] is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args['distributed_training']['distributed_no_spawn']:
            start_rank = args['distributed_training']['distributed_rank']
            args['distributed_training']['distributed_rank'] = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args['distributed_training']['device_id'], args)
    elif args['distributed_training']['distributed_world_size'] > 1:
        # fallback for single node with multiple GPUs
        assert args['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args['distributed_training']['distributed_init_method'] = 'tcp://localhost:{port}'.format(port=port)
        args['distributed_training']['distributed_rank'] = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args['distributed_training']['distributed_world_size'],
        )
    else:
        LOGGER.info('single GPU training...')
        single_main(args)


if __name__ == '__main__':
    cli_main()
