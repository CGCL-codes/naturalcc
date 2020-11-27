# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('fixed')
class FixedSchedule(NccLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        # set defaults
        self.warmup_updates = args['optimization'].get('warmup_updates', 0)
        self.force_anneal = args['optimization'].get('force_anneal', None)
        self.lr_shrink = args['optimization'].get('lr_shrink', 0.1)
        self.lr = args['optimization'].get('lr', 0.001)[0]

        if self.warmup_updates > 0:
            self.warmup_factor = 1. / self.warmup_updates
        else:
            self.warmup_factor = 1

    def get_next_lr(self, epoch):
        lrs = self.args['optimization']['lr']
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = lrs[-1] * self.lr_shrink ** (epoch + 1 - self.force_anneal)
        return next_lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates < self.warmup_updates:
            self.warmup_factor = (num_updates + 1) / float(self.warmup_updates)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()
