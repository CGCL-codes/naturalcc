# -*- coding: utf-8 -*-

import torch

from ncc.optimizers import register_optimizer
from ncc.optimizers.ncc_optimizer import NccOptimizer


@register_optimizer('torch_adamw')
class TorchAdamW(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.AdamW(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args['optimization']['lrs'][0],
            'betas': eval(self.args['optimization']['adam'].get('adam_betas', (0.9, 0.999))),
            'eps': self.args['optimization']['adam'].get('adam_eps', 1e-8),
            'weight_decay': self.args['optimization']['adam'].get('weight_decay', 1e-2),
            'amsgrad': self.args['optimization']['adam'].get('amsgrad', False),
        }
