# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d

from ncc.models import register_model
from ncc.models.ncc_model import (
    NccEncoder,
    NccEncoderModel,
)
from ncc.modules.base.layers import (
    Embedding,
    Linear,
)


class DeepTuneEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim,
                 # rnn config
                 rnn_cell, rnn_hidden_dim, rnn_dropout=None,
                 rnn_num_layers=2, rnn_bidirectional=False,
                 # auxiliary input
                 aux_dim=2,
                 inner_dim=32, out_dim=2,
                 ):
        super(DeepTuneEncoder, self).__init__(dictionary)
        self.embed = Embedding(len(dictionary), embed_dim)
        # LSTM
        self.rnn_dropout = rnn_dropout
        self.rnn = getattr(nn, str.upper(rnn_cell))(
            embed_dim, rnn_hidden_dim, num_layers=rnn_num_layers,
            dropout=self.rnn_dropout,  # rnn inner dropout between layers
            bidirectional=rnn_bidirectional, batch_first=True,
        )
        self.src_out_proj = nn.Sequential(
            Linear(rnn_hidden_dim, out_dim),
            nn.Sigmoid(),
        )
        # Auxiliary inputs. wgsize and dsize
        self.bn = BatchNorm1d(rnn_hidden_dim + aux_dim)
        self.hybrid_out_proj = nn.Sequential(
            Linear(rnn_hidden_dim + aux_dim, inner_dim),
            nn.ReLU(),
            Linear(inner_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, src_tokens, src_lengths=None, src_aux=None, **kwargs):
        src_embed = self.embed(src_tokens)
        src_embed, _ = self.rnn(src_embed)
        src_embed = src_embed[:, -1, :]  # get the last hidden state at last RNN layer
        src_out = self.src_out_proj(src_embed)

        hybrid_embed = torch.cat([src_aux, src_embed], dim=-1)
        hybrid_embed = self.bn(hybrid_embed)
        hybrid_out = self.hybrid_out_proj(hybrid_embed)
        return hybrid_out, src_out


@register_model('deeptune')
class DeepTune(NccEncoderModel):
    def __init__(self, args, encoder):
        super(DeepTune, self).__init__(encoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        encoder = DeepTuneEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args['model']['code_embed'],
            rnn_cell=args['model'].get('rnn_cell', 'LSTM'),
            rnn_hidden_dim=args['model']['rnn_hidden_dim'],
            rnn_num_layers=args['model']['rnn_layers'],
            rnn_dropout=args['model']['rnn_dropout'],
            rnn_bidirectional=args['model'].get('rnn_bidirectional', False),
            aux_dim=args['model'].get('aux_dim', 2),
            inner_dim=args['model']['inner_dim'],
            out_dim=len(task.target_dictionary),
        )
        return cls(args, encoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        return self.encoder.forward(src_tokens, src_lengths, **kwargs)
