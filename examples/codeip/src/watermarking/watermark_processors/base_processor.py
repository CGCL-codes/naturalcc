import warnings
from abc import ABC, abstractmethod

import torch


class WmProcessorBase(ABC):
    def __init__(self, seed=42):
        self.seed = seed

    def set_random_state(self):
        warnings.warn("set_random_state is not implemented for this processor")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        self.set_random_state()

    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass
