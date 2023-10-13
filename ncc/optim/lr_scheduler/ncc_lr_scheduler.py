# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from ncc.dataclass.utils import gen_parser_from_dataclass
from ncc.optim import NccOptimizer


class NccLRScheduler(object):
    def __init__(self, cfg, optimizer):
        super().__init__()
        if optimizer is not None and not isinstance(optimizer, NccOptimizer):
            raise ValueError("optimizer must be an instance of NccOptimizer")
        self.cfg = cfg
        self.optimizer = optimizer
        self.best = None

    @classmethod
    def add_args(cls, parser):
        """Add arguments to the parser for this LR scheduler."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict["best"]

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass

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


class LegacyNccLRScheduler(NccLRScheduler):
    def __init__(self, args: Namespace, optimizer):
        if not isinstance(optimizer, NccOptimizer):
            raise ValueError("optimizer must be an instance of NccOptimizer")
        self.args = args
        self.optimizer = optimizer
        self.best = None


# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# from ncc.optim.ncc_optimizer import NccOptimizer


# class NccLRScheduler(object):

#     def __init__(self, args, optimizer):
#         """
#         Args:
#             args:
#             optimizer:
#             config_key: meta learning needs 2 or more optimizers.
#                         therefore, we need  ```config_key``` to initialize optimizer
#         """
#         super().__init__()
#         if not isinstance(optimizer, NccOptimizer):
#             raise ValueError('optimizer must be an instance of NccOptimizer')
#         self.args = args
#         self.optimizer = optimizer
#         self.warmup_min_lr = args['optimization'].get('warmup_min_lr', 0.)
#         self.warmup_max_lr = args['optimization'].get('warmup_max_lr', args['optimization']['lrs'][0])
#         self.lr = args['optimization']['lrs'][0]
#         self.warmup_updates = args['optimization'].get('warmup_updates', 0)
#         self.best = None

#     def state_dict(self):
#         """Return the LR scheduler state dict."""
#         return {'best': self.best}

#     def load_state_dict(self, state_dict):
#         """Load an LR scheduler state dict."""
#         self.best = state_dict['best']

#     def step(self, epoch, val_loss=None):
#         """Update the learning rate at the end of the given epoch."""
#         if val_loss is not None:
#             if self.best is None:
#                 self.best = val_loss
#             else:
#                 self.best = min(self.best, val_loss)

#     def step_update(self, num_updates):
#         """Update the learning rate after each update."""
#         return self.optimizer.get_lr()
