# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ncc.optimizers.ncc_optimizer import NccOptimizer


class NccLRScheduler(object):

    def __init__(self, args, optimizer):
        """
        Args:
            args:
            optimizer:
            config_key: meta learning needs 2 or more optimizers.
                        therefore, we need  ```config_key``` to initialize optimizer
        """
        super().__init__()
        if not isinstance(optimizer, NccOptimizer):
            raise ValueError('optimizer must be an instance of NccOptimizer')
        self.args = args
        self.optimizer = optimizer
        self.warmup_min_lr = args['optimization'].get('warmup_min_lr', 0.)
        self.warmup_max_lr = args['optimization'].get('warmup_max_lr', args['optimization']['lrs'][0])
        self.lr = args['optimization']['lrs'][0]
        self.warmup_updates = args['optimization'].get('warmup_updates', 0)
        self.best = None

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()
