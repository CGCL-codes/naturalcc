# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.common.initializers import (
    xavier_uniform,
)
from ncc.modules.common.layers import (
    Embedding,
    Linear,
)
from ncc.utils.pooling1d import pooling1d


class NBOWEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(
        self,
        dictionary, embed_dim,
        pooling='weighted_mean', dropout=0.1,
        **kwargs,
    ):
        super().__init__(dictionary)
        self.padding_idx = self.dictionary.pad()
        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx, initializer=xavier_uniform())
        self.dropout = dropout
        self.pooling = pooling1d(pooling)
        if self.pooling:
            self.weight_layer = Linear(embed_dim, 1, bias=False, weight_initializer=xavier_uniform()) \
                if 'weighted' in pooling else None

    def forward(self, tokens, tokens_mask=None, tokens_len=None):
        """
        Args:
            tokens: [batch_size, max_token_len]
            tokens_mask: [batch_size, 1]
            tokens_mask: [batch_size, max_token_len]

        Returns:
            tokens: [batch_size, max_token_len, embed_dim]
        """
        if tokens_mask is None:
            tokens_mask = tokens.new(tokens.size())
            tokens_mask.data.copy_((tokens != self.padding_idx).int())
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        tokens = F.dropout(tokens, p=self.dropout, training=self.training)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
