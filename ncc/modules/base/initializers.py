# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def uniform(bound=0.1, padding_idx=None):
    def uniform_(tensor):
        nn.init.uniform_(tensor, a=-bound, b=bound)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return uniform_


def xavier_uniform(padding_idx=None):
    def xavier_uniform_(tensor):
        nn.init.xavier_uniform_(tensor)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return xavier_uniform_


def constant(value=0., padding_idx=None):
    def constant_(tensor):
        nn.init.constant_(tensor, value)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return constant_


def normal(mean=0., std=1., padding_idx=None):
    def normal_(tensor):
        nn.init.normal_(tensor, mean=mean, std=std)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return normal_


def xavier_normal(padding_idx=None):
    def xavier_normal_(tensor):
        nn.init.xavier_normal_(tensor)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return xavier_normal_


def trunc_normal(mean=0., std=1., a=-2., b=2., padding_idx=None):
    def trunc_normal_(tensor):
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
        if padding_idx is not None:
            nn.init.constant_(tensor[padding_idx], 0)

    return trunc_normal_
