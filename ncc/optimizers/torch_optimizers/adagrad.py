# -*- coding: utf-8 -*-

import torch

from ncc.optimizers import register_optimizer
from ncc.optimizers.ncc_optimizer import NccOptimizer


@register_optimizer('torch_adagrad')
class TorchAdagrad(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.Adagrad(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args['optimization']['lr'][0],
            'lr_decay': eval(self.args['optimization']['adagrad'].get('lr_decay', 0)),
            'weight_decay': self.args['optimization']['adagrad'].get('weight_decay', 0),
            'initial_accumulator_value': self.args['optimization']['adagrad'].get('initial_accumulator_value', 0),
            'eps': self.args['optimization']['adagrad'].get('eps', 1e-10),
        }
