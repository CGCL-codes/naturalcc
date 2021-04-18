# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.onnx.operators
from torch import nn


class SinusoidalPositionalEmbedding_Simple(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=9000):
        super(SinusoidalPositionalEmbedding_Simple, self).__init__()
        torch.manual_seed(1)
        self.dropout = nn.Dropout(p=dropout)
        # Option 1: original position embedding
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # Option 2: our position embedding with minor revision
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(1, max_len + 1, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)  # TODO -1 according to fairseq
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

    def _load_from_state_dict(self, *args):
        print("PositionalEncoding: doing nothing on call to _load_from_state_dict")