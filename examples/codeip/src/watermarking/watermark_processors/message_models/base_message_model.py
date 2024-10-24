import warnings
from abc import ABC, abstractmethod

import torch

class TextTooShortError(Exception):
    pass

class BaseMessageModelFast(ABC):
    def __init__(self, seed, message_code_len=10):
        self.seed = seed
        self.message_code_len = message_code_len

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        self.set_random_state()

    def set_random_state(self):
        warnings.warn("set_random_state is not implemented for this processor")

    @abstractmethod
    def cal_log_Ps(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                   messages: torch.Tensor) -> torch.Tensor:
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        pass
