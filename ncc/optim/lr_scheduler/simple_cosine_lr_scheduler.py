# -*- coding: utf-8 -*-

import math

from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('simple_cosine')
class SimpleCosineSchedule(NccLRScheduler):
    """
    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      lr = lr * 0.5*(1 + cos(t_curr / t_i))
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        if len(args['optimization']['lrs']) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        if 'lr_period_updates' not in args['optimization']:
            self.period = -1
        else:
            self.period = args['optimization']['lr_period_updates']

        # initial learning rate
        self.lr = self.max_lr = args['optimization']['lrs'][0]
        self.optimizer.set_lr(self.lr)

        self.warmup_updates = float(max(args['optimization']['warmup_updates'], 1))
        if args['optimization']['warmup_updates'] > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = self.lr / self.warmup_updates
        else:
            self.lr_step = 1.

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = num_updates * self.lr_step
        else:
            pregress = float(num_updates - self.warmup_updates) / float(max(1, self.period - self.warmup_updates))
            self.lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * pregress))
        self.optimizer.set_lr(self.lr)
        return self.lr
