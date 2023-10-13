# -*- coding: utf-8 -*-

from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('polynomial_decay')
class PolynomialDecayLRSchedule(NccLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lrs = args['optimization']['lrs']
        if args['optimization']['warmup_updates'] > 0:
            self.warmup_factor = 1.0 / args['optimization']['warmup_updates']
        else:
            self.warmup_factor = 1
        self.end_learning_rate = args['optimization'].get('end_learning_rate', 0.)
        self.total_num_update = args['optimization'].get('total_num_update', 1000000)
        self.power = args['optimization'].get('power', 1.0)
        self.optimizer.set_lr(self.warmup_factor * self.lr)

    def get_next_lr(self, epoch):
        lrs = self.lrs
        if self.args['optimization']['force_anneal'] is None or epoch < self.args['optimization']['force_anneal']:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args['optimization']['warmup_updates'] > 0 and \
            num_updates <= self.args['optimization']['warmup_updates']:
            self.warmup_factor = num_updates / float(self.args['optimization']['warmup_updates'])
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.args['optimization']['warmup_updates']
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
