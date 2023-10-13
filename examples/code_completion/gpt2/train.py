#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any ncc libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ncc_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ncc import checkpoint_utils, options, quantization_utils, tasks
from ncc.utils import utils

from ncc.data import data_utils, iterators
from ncc.data.plasma_utils import PlasmaStore
from ncc.dataclass.configs import NccConfig
from ncc.dataclass.initialize import add_defaults
from ncc.dataclass.utils import convert_namespace_to_omegaconf
from ncc.distributed import fsdp_enable_wrap, fsdp_wrap
from ncc.distributed import utils as distributed_utils
from ncc.file_io import PathManager
from ncc.logging import meters, metrics, progress_bar
from ncc.model_parallel.megatron_trainer import MegatronTrainer
from ncc.trainer import Trainer


def main(cfg: NccConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    if not cfg.dataset.disable_validation:
        data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
        if cfg.dataset.combine_valid_subsets:
            task.load_dataset("valid", combine=True, epoch=1)
        else:
            for valid_sub_split in cfg.dataset.valid_subset.split(","):
                task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    # TODO: a dry run on validation set to pin the memory
    valid_subsets = cfg.dataset.valid_subset.split(",")
    if not cfg.dataset.disable_validation:
        for subset in valid_subsets:
            logger.info('begin dry-run validation on "{}" subset'.format(subset))
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            if cfg.common.tpu:
                itr = utils.tpu_data_loader(itr)
            for _ in itr:
                pass
    # TODO: end of dry run section

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.NccTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(
        itr,
        update_freq,
        skip_remainder_batch=cfg.optimization.skip_remainder_batch,
    )
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.NccTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
            or should_stop
            or (
                cfg.dataset.validate_interval_updates > 0
                and num_updates > 0
                and num_updates % cfg.dataset.validate_interval_updates == 0
            )
        )
        and not cfg.dataset.disable_validation
        and num_updates >= cfg.dataset.validate_after_updates
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        cp_path = checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )
        if cp_path is not None and hasattr(task, "post_save"):
            task.post_save(cp_path, num_updates)

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.NccTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(subsets):
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            aim_repo=(
                cfg.common.aim_repo
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_run_hash=(
                cfg.common.aim_run_hash
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if (
                    cfg.dataset.max_valid_steps is not None
                    and i > cfg.dataset.max_valid_steps
                ):
                    break
                trainer.valid_step(sample)

        # log validation stats
        # only tracking the best metric on the 1st validation subset
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values(), tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)
    
    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()


# import math
# import os
# import random

# import torch

# from ncc import LOGGER
# from ncc import tasks
# from ncc.data import iterators
# from ncc.trainers.ncc_trainers import Trainer
# from ncc.utils import checkpoint_utils, distributed_utils
# from ncc.utils import set_seed
# from ncc.utils import utils
# from ncc.utils.file_ops.yaml_io import load_yaml
# from ncc.utils.logging import meters
# from ncc.utils.logging import metrics, progress_bar
# from ncc.utils.path_manager import PathManager


# @metrics.aggregate('train')
# def train(args, trainer, task, epoch_itr):
#     """Train the model for one epoch."""
#     # Initialize data iterator
#     itr = epoch_itr.next_epoch_itr(
#         fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
#         shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
#     )
#     update_freq = (
#         args['optimization']['update_freq'][epoch_itr.epoch - 1]
#         if epoch_itr.epoch <= len(args['optimization']['update_freq'])
#         else args['optimization']['update_freq'][-1]
#     )
#     itr = iterators.GroupedIterator(itr, update_freq)
#     progress = progress_bar.progress_bar(
#         itr,
#         log_format=args['common']['log_format'],
#         log_interval=args['common']['log_interval'],
#         epoch=epoch_itr.epoch,
#         tensorboard_logdir=(
#             args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
#         ),
#         default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
#     )

#     # task specific setup per epoch
#     task.begin_epoch(epoch_itr.epoch, trainer.get_model())

#     valid_subsets = args['dataset']['valid_subset'].split(',')
#     max_update = args['optimization']['max_update'] or math.inf
#     for samples in progress:
#         with metrics.aggregate('train_inner'):
#             log_output = trainer.train_step(samples)
#             if log_output is None:  # OOM, overflow, ...
#                 continue

#         # log mid-epoch stats
#         num_updates = trainer.get_num_updates()
#         if num_updates % args['common']['log_interval'] == 0:
#             stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
#             progress.log(stats, tag='train_inner', step=num_updates)

#             # reset epoch-level meters
#             metrics.reset_meters('train_inner')

#         if (
#             not args['dataset']['disable_validation']
#             and args['checkpoint']['save_interval_updates'] > 0
#             and num_updates % args['checkpoint']['save_interval_updates'] == 0
#             and num_updates > 0
#         ):
#             valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
#             checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

#         if num_updates >= max_update:
#             break

#     # log end-of-epoch stats
#     stats = get_training_stats(metrics.get_smoothed_values('train'))
#     progress.print(stats, tag='train', step=num_updates)

#     # reset epoch-level meters
#     metrics.reset_meters('train')


# def validate(args, trainer, task, epoch_itr, subsets):
#     """Evaluate the model on the validation set(s) and return the losses."""

#     if args['dataset']['fixed_validation_seed'] is not None:
#         # set fixed seed for every validation
#         set_seed.set_torch_seed(args['dataset']['fixed_validation_seed'])

#     valid_losses = []
#     for subset in subsets:
#         # Initialize data iterator
#         itr = task.get_batch_iterator(
#             dataset=task.dataset(subset),
#             max_tokens=args['dataset']['max_tokens_valid'],
#             max_sentences=args['dataset']['max_sentences_valid'],
#             max_positions=utils.resolve_max_positions(
#                 task.max_positions(),
#                 trainer.get_model().max_positions(),
#             ),
#             ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
#             required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
#             seed=args['common']['seed'],
#             num_shards=args['distributed_training']['distributed_world_size'],
#             shard_id=args['distributed_training']['distributed_rank'],
#             num_workers=args['dataset']['num_workers'],
#         ).next_epoch_itr(shuffle=False)
#         progress = progress_bar.progress_bar(
#             itr,
#             log_format=args['common']['log_format'],
#             log_interval=args['common']['log_interval'],
#             epoch=epoch_itr.epoch,
#             prefix=f"valid on '{subset}' subset",
#             tensorboard_logdir=(
#                 args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
#             ),
#             default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
#         )

#         # create a new root metrics aggregator so validation metrics
#         # don't pollute other aggregators (e.g., train meters)
#         with metrics.aggregate(new_root=True) as agg:
#             for sample in progress:
#                 trainer.valid_step(sample)

#         # log validation stats
#         stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
#         progress.print(stats, tag=subset, step=trainer.get_num_updates())

#         valid_losses.append(stats[args['checkpoint']['best_checkpoint_metric']])

#     return valid_losses


# def get_valid_stats(args, trainer, stats):
#     if 'nll_loss' in stats and 'ppl' not in stats:
#         stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
#     stats['num_updates'] = trainer.get_num_updates()
#     if hasattr(checkpoint_utils.save_checkpoint, 'best'):
#         key = 'best_{0}'.format(args['checkpoint']['best_checkpoint_metric'])
#         best_function = max if args['checkpoint']['maximize_best_checkpoint_metric'] else min
#         stats[key] = best_function(
#             checkpoint_utils.save_checkpoint.best,
#             stats[args['checkpoint']['best_checkpoint_metric']],
#         )
#     return stats


# def get_training_stats(stats):
#     if 'nll_loss' in stats and 'ppl' not in stats:
#         stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
#     stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
#     return stats


# def should_stop_early(args, valid_loss):
#     # skip check if no validation was done in the current epoch
#     if valid_loss is None:
#         return False
#     if args['checkpoint']['patience'] <= 0:
#         return False

#     def is_better(a, b):
#         return a > b if args['checkpoint']['maximize_best_checkpoint_metric'] else a < b

#     prev_best = getattr(should_stop_early, 'best', None)
#     if prev_best is None or is_better(valid_loss, prev_best):
#         should_stop_early.best = valid_loss
#         should_stop_early.num_runs = 0
#         return False
#     else:
#         should_stop_early.num_runs += 1
#         if should_stop_early.num_runs >= args['checkpoint']['patience']:
#             LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
#                 args['checkpoint']['patience']))

#         return should_stop_early.num_runs >= args['checkpoint']['patience']


# def single_main(args, init_distributed=False):
#     assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None, \
#         'Must specify batch size either with --max-tokens or --max-sentences'
#     metrics.reset()

#     # 0. Initialize CUDA and distributed training
#     if torch.cuda.is_available() and not args['common']['cpu']:
#         torch.cuda.set_device(args['distributed_training']['device_id'])
#     set_seed.set_seed(args['common']['seed'])
#     if init_distributed:
#         args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

#     # Verify checkpoint directory
#     if distributed_utils.is_master(args):
#         save_dir = args['checkpoint']['save_dir']
#         checkpoint_utils.verify_checkpoint_directory(save_dir)
#         # PathManager.rm(os.path.join(save_dir, '*.pt'))  # this code will remove pre-trained models

#     # Print args
#     LOGGER.info(args)

#     # 1. Setup task, e.g., translation, language modeling, etc.
#     task = tasks.setup_task(args)

#     # 2. Load valid dataset (we load training data below, based on the latest checkpoint)
#     task.load_dataset(args['dataset']['valid_subset'], combine=False, epoch=1)
#     task.load_dataset(args['dataset']['gen_subset'], combine=False, epoch=1)

#     # 3. Build model and criterion
#     model = task.build_model(args)
#     criterion = task.build_criterion(args)
#     LOGGER.info(model)
#     LOGGER.info('model {}, criterion {}'.format(args['model']['arch'], criterion.__class__.__name__))
#     LOGGER.info('num. model params: {} (num. trained: {})'.format(
#         sum(p.numel() for p in model.parameters()),
#         sum(p.numel() for p in model.parameters() if p.requires_grad),
#     ))

#     # 4. Build trainer
#     trainer = Trainer(args, task, model, criterion)
#     LOGGER.info('training on {} GPUs'.format(args['distributed_training']['distributed_world_size']))
#     LOGGER.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
#         args['dataset']['max_tokens'],
#         args['dataset']['max_sentences'],
#     ))

#     # 5. Load the latest checkpoint if one is available and restore the corresponding train iterator
#     extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

#     # 6. Train until the learning rate gets too small
#     max_epoch = args['optimization']['max_epoch'] or math.inf
#     max_update = args['optimization']['max_update'] or math.inf
#     lr = trainer.get_lr()
#     train_meter = meters.StopwatchMeter()
#     train_meter.start()
#     valid_subsets = args['dataset']['valid_subset'].split(',')
#     while (
#         lr > args['optimization']['min_lr']
#         and epoch_itr.next_epoch_idx <= max_epoch
#         and trainer.get_num_updates() < max_update
#     ):
#         # train for one epoch
#         train(args, trainer, task, epoch_itr)

#         if not args['dataset']['disable_validation'] and epoch_itr.epoch % args['dataset']['validate_interval'] == 0:
#             valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
#         else:
#             valid_losses = [None]

#         # only use first validation loss to update the learning rate
#         lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

#         # save checkpoint
#         if epoch_itr.epoch % args['checkpoint']['save_interval'] == 0:
#             checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

#         # early stop
#         if should_stop_early(args, valid_losses[0]):
#             LOGGER.info('early stop since valid performance hasn\'t improved for last {} runs'.format(
#                 args['checkpoint']['patience']))
#             break

#         epoch_itr = trainer.get_train_iterator(
#             epoch_itr.next_epoch_idx,
#             combine=False,  # TODO to be checked
#             # sharded data: get train iterator for next epoch
#             load_dataset=(os.pathsep in args['task']['data']),
#         )

#     train_meter.stop()
#     LOGGER.info('done training in {:.1f} seconds'.format(train_meter.sum))


# def distributed_main(i, args, start_rank=0):
#     args['distributed_training']['device_id'] = i
#     if args['distributed_training']['distributed_rank'] is None:  # torch.multiprocessing.spawn
#         args['distributed_training']['distributed_rank'] = start_rank + i
#     single_main(args, init_distributed=True)


# def cli_main():
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Downloading/Decompressing code_search_net dataset(s) or Tree-Sitter Library(ies)")
#     parser.add_argument(
#         "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
#         default='config/csn_feng/python',
#     )
#     args = parser.parse_args()
#     yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
#     LOGGER.info('Load arguments in {}'.format(yaml_file))
#     args = load_yaml(yaml_file)
#     LOGGER.info(args)

#     if args['distributed_training']['distributed_init_method'] is None:
#         distributed_utils.infer_init_method(args)

#     if args['distributed_training']['distributed_init_method'] is not None:
#         # distributed training
#         if torch.cuda.device_count() > 1 and not args['distributed_training']['distributed_no_spawn']:
#             start_rank = args['distributed_training']['distributed_rank']
#             args['distributed_training']['distributed_rank'] = None  # assign automatically
#             torch.multiprocessing.spawn(
#                 fn=distributed_main,
#                 args=(args, start_rank),
#                 nprocs=torch.cuda.device_count(),
#             )
#         else:
#             distributed_main(args['distributed_training']['device_id'], args)
#     elif args['distributed_training']['distributed_world_size'] > 1:
#         # fallback for single node with multiple GPUs
#         assert args['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
#         port = random.randint(10000, 20000)
#         args['distributed_training']['distributed_init_method'] = 'tcp://localhost:{port}'.format(port=port)
#         args['distributed_training']['distributed_rank'] = None  # set based on device id
#         torch.multiprocessing.spawn(
#             fn=distributed_main,
#             args=(args,),
#             nprocs=args['distributed_training']['distributed_world_size'],
#         )
#     else:
#         LOGGER.info('single GPU training...')
#         single_main(args)


# if __name__ == '__main__':
#     cli_main()
