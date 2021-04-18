# -*- coding: utf-8 -*-

import torch.optim

from ncc.optimizers import register_optimizer
from ncc.optimizers.ncc_optimizer import NccOptimizer


@register_optimizer('torch_adadelta')
class TorchAdadelta(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.Adadelta(params, **self.optimizer_config)

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
            'betas': eval(self.args['optimization']['adadelta'].get('rho', 0.9)),
            'eps': self.args['optimization']['adadelta'].get('adam_eps', 1e-6),
            'weight_decay': self.args['optimization']['adadelta'].get('weight_decay', 0),
        }

    @property
    def supports_flat_params(self):
        return True
