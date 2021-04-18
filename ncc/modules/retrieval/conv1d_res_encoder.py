# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.common.initializers import trunc_normal
from ncc.modules.common.layers import (
    Embedding,
    Linear,
    Conv2d,
    Parameter,
)
from ncc.modules.common.activations import get_activation
from ncc.utils.pooling1d import pooling1d


class Conv1dResEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim, out_channels, kernel_size,
                 **kwargs):
        super().__init__(dictionary)
        # word embedding + positional embedding
        self.embed = Embedding(len(dictionary), embed_dim)  # , padding_idx=self.dictionary.pad())

        self.position_encoding = kwargs.get('position_encoding', None)
        if self.position_encoding == 'learned':
            self.position_embed = Parameter(1, kwargs['max_tokens'], embed_dim,
                                            initializer=trunc_normal(mean=0., std=0.02))
        else:
            self.position_embed = None
        # pooling
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        if 'weighted' in pooling:
            self.weight_layer = Linear(embed_dim, 1, bias=False)
        else:
            self.weight_layer = None
        # conv1d
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # padding mode = ['valid'(default), 'same']
        self.padding = kwargs.get('padding', 'valid')
        if self.padding == 'same':
            self.padding_size = []
            for kernel_sz in self.kernel_size:
                padding_right = (kernel_sz - 1) // 2
                padding_left = kernel_sz - 1 - padding_right
                self.padding_size.append((0, 0, padding_left, padding_right,))
        self.conv_layers = nn.ModuleList([])
        # input: [bsz, 1, seq_len, embed_dim]
        # filters = 1 -> embed_dim
        # kernel_size = (kernel_width, embed_dim)
        # =>  output: [bsz, embed_dim, seq_len - kernel_width + 1]
        for idx, kernel_sz in enumerate(self.kernel_size):
            self.conv_layers.append(
                Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(kernel_sz, embed_dim))
            )

        self.residual = kwargs.get('residual', False)  # residual
        self.dropout = kwargs.get('dropout', None)
        activation_fn = kwargs.get('activation_fn', None)
        self.activation_fn = get_activation(activation_fn) if activation_fn else None

    def forward(self, tokens, tokens_mask=None, tokens_len=None):
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        tokens = F.dropout(tokens, p=self.dropout, training=self.training)
        tokens = tokens + self.position_embed

        for idx, conv in enumerate(self.conv_layers):
            residual = tokens  # [B, L, E]
            tokens = tokens.unsqueeze(dim=1)  # [B, L, E] => [B, 1, L, E]
            if self.padding == 'same':
                # TODO: check it on CodeSearchNet source code
                tokens = F.pad(tokens, pad=self.padding_size[idx])  # [B, 1, L, E] => [B, 1, l + L + r, E]
                tokens = conv(tokens)  # [B, 1, l + L + r, E] => [B, E, L, 1]
            else:
                tokens = conv(tokens)
            tokens = tokens.squeeze(dim=-1).transpose(-2, -1)  # [B, L, E]
            # Add residual connections past the first layer.
            if self.residual and idx > 0:
                tokens += residual  # [B, L, E]
            if self.activation_fn:
                tokens = self.activation_fn(tokens)
            if self.dropout:
                tokens = F.dropout(tokens, p=self.dropout, training=self.training)

        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
