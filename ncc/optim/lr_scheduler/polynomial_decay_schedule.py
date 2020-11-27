# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('polynomial_decay')
class PolynomialDecaySchedule(NccLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        # set defaults
        # args.warmup_updates = getattr(args, 'warmup_updates', 0) or 0
        self.lr = args['optimization']['lr'][0]
        if args['optimization']['warmup_updates'] > 0:
            self.warmup_factor = 1. / args['optimization']['warmup_updates']
        else:
            self.warmup_factor = 1
        self.end_learning_rate = args['optimization']['end_learning_rate']
        self.total_num_update = args['optimization']['total_num_update']
        self.power = args['optimization']['power']
        self.optimizer.set_lr(self.warmup_factor * self.lr)

    def get_next_lr(self, epoch):
        lrs = self.args['optimization']['lr']
        if self.args['optimization']['force_anneal'] is None or epoch < self.args['optimization']['force_anneal']:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
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
