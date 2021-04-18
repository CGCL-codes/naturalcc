# -*- coding: utf-8 -*-
"""
this file includes base modules for NaturalCC models but with tensorflow-like interface.
"""

import torch
import torch.nn as nn
from ncc.modules.common.initializers import (
    uniform, xavier_uniform,
    constant,
    normal, xavier_normal, trunc_normal,
)


def Embedding(
    num_embeddings, embedding_dim, padding_idx=None,
    initializer=uniform(bound=0.1)
):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if initializer is not None:
        initializer(m.weight)
    if padding_idx is not None:
        m.weight.data[padding_idx, :].fill_(0.)
    return m


def Linear(
    in_dim, out_dim, bias=True,
    weight_initializer=uniform(bound=0.1),
    bias_initializer=constant(value=0.),
):
    m = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
    weight_initializer(m.weight)
    if bias:
        bias_initializer(m.bias)
    return m


def LSTMCell(
    input_size, hidden_size, bias=True,
    initializer=uniform(bound=0.1),
):
    m = nn.LSTMCell(input_size, hidden_size, bias)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            initializer(param)
    return m


def LSTM(
    input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False,
    initializer=uniform(bound=0.1),
):
    """
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        would mean stacking two LSTMs together to form a `stacked LSTM`,
        with the second LSTM taking in outputs of the first LSTM and
        computing the final results. Default: 1
    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        Default: ``True``
    batch_first: If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``False``
    dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        LSTM layer except the last layer, with dropout probability equal to
        :attr:`dropout`. Default: 0
    bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
    """
    m = nn.LSTM(
        input_size, hidden_size,
        num_layers=num_layers, bias=bias, batch_first=batch_first,
        dropout=dropout, bidirectional=bidirectional
    )
    for p in m.parameters():
        initializer(p)
    return m


def Conv2d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
    weight_initializer=xavier_uniform(),
    bias_initializer=constant(value=0.),
):
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
    )
    weight_initializer(m.weight)
    bias_initializer(m.bias)
    return m


def Parameter(
    *size,
    initializer=uniform(bound=0.1),
):
    m = nn.Parameter(torch.zeros(*size))
    initializer(m)
    return m
