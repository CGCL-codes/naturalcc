# -*- coding: utf-8 -*-

from ..ncc_trainers import Trainer


class CSN_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CSN_Trainer, self).__init__(*args, **kwargs)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_by_global_norm_(clip_norm)
