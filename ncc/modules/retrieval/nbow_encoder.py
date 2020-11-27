# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.models import register_model
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.embedding import Embedding
from ncc.utils.pooling1d import pooling1d

logger = logging.getLogger(__name__)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class NBOWEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim, **kwargs):
        super().__init__(dictionary)
        self.embed = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.dictionary.pad())
        # self.embed = Embedding(len(dictionary), embed_dim, padding_idx=None)
        self.dropout = kwargs.get('dropout', None)
        self.embed.weight.data.copy_(F.dropout(self.embed.weight.data, self.dropout))
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        self.dropout = kwargs.get('dropout', None)
        if self.pooling:
            self.weight_layer = Linear(embed_dim, 1, bias=False) if 'weighted' in pooling else None

    def forward(self, tokens, tokens_len=None, tokens_mask=None):
        """
        Args:
            tokens: [batch_size, max_token_len]
            tokens_mask: [batch_size, 1]
            tokens_mask: [batch_size, max_token_len]

        Returns:
            tokens: [batch_size, max_token_len, embed_dim]
        """
        if tokens_mask is None:
            tokens_mask = (tokens != self.dictionary.pad()).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        # tokens = F.dropout(tokens, self.dropout, self.training)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
