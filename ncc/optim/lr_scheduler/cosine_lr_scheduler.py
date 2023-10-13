# -*- coding: utf-8 -*-

import math

from ncc import LOGGER
from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('cosine')
class CosineSchedule(NccLRScheduler):
    """Assign LR based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    max learning rate (``--max-lr``).

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))

    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args['optimization']['lr']) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.min_lr = max(args['optimization'].get('min_lr', 0), 0)
        self.max_lr = args['optimization'].get('max_lr', args['optimization']['lr'][0])

        self.warmup_init_lr = args['optimization'].get('warmup_init_lr', 0)
        warmup_end_lr = args['optimization'].get('warmup_end_lr', self.max_lr)

        assert self.max_lr > self.min_lr, 'max_lr must be more than lr'

        self.t_mult = args['optimization'].get('t_mult', 1.)

        if 'lr_period_updates' not in args['optimization']:
            LOGGER.warning('lr_period_updates has not been set and, therefore, set epoch_num * batch_num as default.')
            self.period = -1
        else:
            self.period = args['optimization']['lr_period_updates']

        if args['optimization']['warmup_updates'] > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = \
                (warmup_end_lr - args['optimization']['warmup_init_lr']) / args['optimization']['warmup_updates']
        else:
            self.lr_step = 1

        self.warmup_updates = args['optimization']['warmup_updates']
        self.lr_shrink = args['optimization'].get('lr_shrink', 0.1)

        # initial learning rate
        self.lr = args['optimization']['warmup_init_lr']
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.warmup_updates
            if self.t_mult != 1:
                i = math.floor(math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult))
                t_i = self.t_mult ** i * self.period
                t_curr = curr_updates - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            lr_shrink = self.lr_shrink ** i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

        self.optimizer.set_lr(self.lr)
        return self.lr
