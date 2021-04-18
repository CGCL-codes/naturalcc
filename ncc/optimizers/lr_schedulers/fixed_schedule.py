# -*- coding: utf-8 -*-

from . import NccLRScheduler, register_lr_scheduler


@register_lr_scheduler('fixed')
class FixedSchedule(NccLRScheduler):
    """
    Decay the LR on a fixed schedule.

    During warmup::
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup::
        if force_anneal is None:
            # anneal lr after 2nd
            lr = lr * lr_decay
        else:
            # keep lr constant before force_anneal epoch, and anneal lr after force_anneal epoch
            lr = lr                                        |    lr = lr * lr_decay
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.lrs = args['optimization']['lrs']
        self.lr_shrink = args['optimization'].get('lr_shrink', 1.)
        self.force_anneal = args['optimization'].get('force_anneal', None)

        if self.warmup_updates > 0:
            self.warmup_factor = 1. * (self.warmup_max_lr - self.warmup_min_lr) / self.warmup_updates

    def get_lr(self, epoch):
        if self.force_anneal is None or epoch < self.force_anneal:
            # use fixed LR schedule
            lr = self.lrs[min(epoch, len(self.lrs) - 1)]
        else:
            # annneal based on lr_shrink beginning after 2nd epoch
            lr = self.optimizer.get_lr() * self.lr_shrink ** (epoch - 1 - self.force_anneal)
        return lr

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr = self.get_lr(epoch)
        self.optimizer.set_lr(self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates < self.warmup_updates:
            self.optimizer.set_lr(self.warmup_min_lr + self.warmup_factor * num_updates)
        return self.optimizer.get_lr()
