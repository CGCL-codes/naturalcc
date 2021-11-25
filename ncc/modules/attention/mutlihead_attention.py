# -*- coding: utf-8 -*-

from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MultiheadAttention, self).__init__()

    def reset_parameters(self):
        raise NotImplementedError
