import torch

from ncc.optimizers import register_optimizer
from ncc.optimizers.ncc_optimizer import NccOptimizer


@register_optimizer('torch_adam')
class TorchAdam(NccOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = torch.optim.Adam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        return {
            'lr': self.args['optimization']['lrs'][0],
            'betas': eval(self.args['optimization']['adam'].get('adam_betas', (0.9, 0.999))),
            'eps': self.args['optimization']['adam'].get('adam_eps', 1e-8),
            'weight_decay': self.args['optimization']['adam'].get('weight_decay', 0),
            'amsgrad': self.args['optimization']['adam'].get('amsgrad', False),
        }

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._optimizer.step()
