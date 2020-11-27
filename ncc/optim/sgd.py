# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim

from . import NccOptimizer, register_optimizer


@register_optimizer('sgd')
class SGD(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.SGD(params, **self.optimizer_config)

    # @staticmethod
    # def add_args(parser):
    #     """Add optimizer-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
    #                         help='momentum factor')
    #     parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
    #                         help='weight decay')
    #     # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args['optimization']['lr'][0],
            'momentum': self.args['optimization']['sgd'].get('momentum', 0),
            'weight_decay': self.args['optimization']['sgd'].get('weight_decay', 0),
            'dampening': self.args['optimization']['sgd'].get('dampening', 0),
            'nesterov': self.args['optimization']['sgd'].get('nesterov', False),
        }

    @property
    def supports_flat_params(self):
        return True
