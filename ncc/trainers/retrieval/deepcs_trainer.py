# -*- coding: utf-8 -*-

import torch

from ncc import LOGGER
from ncc import optimizers
from ncc.optimizers import lr_schedulers
from ..ncc_trainers import Trainer


class DeepCS_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DeepCS_Trainer, self).__init__(*args, **kwargs)

    def _setup_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
            LOGGER.info("NOTE: your device may support faster training with --fp16")
        self._optimizer = optimizers.setup_optimizer(self.args, params)

        if self.args['optimization']['use_bmuf']:
            self._optimizer = optimizers.NccBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_schedulers.build_lr_scheduler(self.args, self.optimizer)
        if getattr(self._lr_scheduler, 'period', None) == -1:
            import math
            self._lr_scheduler.period = \
                self.args['optimization']['max_epoch'] * \
                math.ceil(len(self.task.dataset('train')) / self.args['dataset']['max_sentences'])
            LOGGER.warning('Update steps of {} has not been set and, therefore, set {} as default.'. \
                           format(self.lr_scheduler.__class__.__name__, self._lr_scheduler.period))
        self._lr_scheduler.step_update(0)
