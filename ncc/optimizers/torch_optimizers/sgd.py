# -*- coding: utf-8 -*-

import torch

from ncc.optimizers import register_optimizer
from ncc.optimizers.ncc_optimizer import NccOptimizer


@register_optimizer('torch_sgd')
class TorchSGD(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.SGD(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args['optimization']['lrs'][0],
            'momentum': self.args['optimization']['sgd'].get('momentum', 0),
            'dampening': self.args['optimization']['sgd'].get('dampening', 0),
            'weight_decay': self.args['optimization']['sgd'].get('weight_decay', 0),
            'nesterov': self.args['optimization']['sgd'].get('nesterov', False),
        }
