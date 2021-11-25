# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ncc.modules.base.layers import (
    Embedding,
    Linear,
    LSTM,
)
from ncc.modules.encoders.ncc_encoder import NccEncoder


class SeqEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim,
                 hidden_dim, rnn_layers=1, bidirectional=True,
                 dropout=0.25):
        super(SeqEncoder, self).__init__(dictionary)
        self.padding_idx = self.dictionary.pad()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.embed = Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)
        self.rnn = LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        bsz = src_tokens.size(0)
        x = self.embed(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        if src_lengths is None:
            src_lengths = src_lengths.new([src_lengths.size(0)]).copy_(
                src_tokens.ne(self.padding_idx).sum(-1)
            )

        # sort
        sorted_lens, indices = src_lengths.sort(descending=True)
        sorted_x = x.index_select(0, indices)
        sorted_x = pack_padded_sequence(sorted_x, sorted_lens.data.tolist(), batch_first=True)

        x, (h, c) = self.rnn(sorted_x)

        _, reversed_indices = indices.sort()
        # x, lens = pad_packed_sequence(x, batch_first=True)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = x.index_select(0, reversed_indices)
        h = h.index_select(1, reversed_indices)
        h = h.view(self.rnn_layers, 2 if self.bidirectional else 1, bsz, self.hidden_dim)
        h = h[-1].view(bsz, -1)
        return h


class NBOWEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim, dropout=0.25):
        super(NBOWEncoder, self).__init__(dictionary)
        self.padding_idx = self.dictionary.pad()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.embed = Embedding(len(dictionary), embed_dim)  # , padding_idx=self.padding_idx)
        # self.init_weights()

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        lens = src_tokens.size(1)
        x = self.embed(src_tokens)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.max_pool1d(x.transpose(1, 2), lens).squeeze(2)
        return x


class DeepCSEncoder(nn.Module):
    def __init__(
        self,
        name_dict=None, apiseq_dict=None, tokens_dict=None,
        embed_dim=128, hidden_dim=128, rnn_layers=1, bidirectional=False, dropout=0.1,
    ):
        super(DeepCSEncoder, self).__init__()
        # func_name encoder
        if name_dict is None:
            self.name_encoder = None
        else:
            self.name_encoder = nn.ModuleList([
                SeqEncoder(name_dict,
                           embed_dim=embed_dim,
                           hidden_dim=hidden_dim,
                           rnn_layers=rnn_layers,
                           bidirectional=bidirectional,
                           dropout=dropout),
                Linear(2 * hidden_dim, embed_dim),
            ])
        # apiseq encoder
        if apiseq_dict is None:
            self.apiseq_encoder = None
        else:
            self.apiseq_encoder = nn.ModuleList([
                SeqEncoder(name_dict,
                           embed_dim=embed_dim,
                           hidden_dim=hidden_dim,
                           rnn_layers=rnn_layers,
                           bidirectional=bidirectional,
                           dropout=dropout),
                Linear(2 * hidden_dim, embed_dim),
            ])
        # apiseq encoder
        if tokens_dict is None:
            self.tokens_encoder = None
        else:
            self.tokens_encoder = nn.ModuleList([
                NBOWEncoder(tokens_dict,
                            embed_dim=embed_dim,
                            dropout=dropout),
                Linear(embed_dim, embed_dim),
            ])
        # fusion layer
        self.fusion = Linear(embed_dim, embed_dim)

    def forward(self,
                name=None, name_len=None,
                apiseq=None, apiseq_len=None,
                tokens=None, tokens_len=None,
                ):
        if name is not None and self.name_encoder is not None:
            name_out = self.name_encoder[0](name, name_len)
            name_out = self.name_encoder[1](name_out)
            # name_repr = self.name_encoder(name, name_len)
        else:
            name_out = 0

        if apiseq is not None and self.apiseq_encoder is not None:
            apiseq_out = self.apiseq_encoder[0](apiseq, apiseq_len)
            apiseq_out = self.apiseq_encoder[1](apiseq_out)
            # api_repr = self.apiseq_encoder(api, api_len)
        else:
            apiseq_out = 0

        if tokens is not None and self.tokens_encoder is not None:
            tokens_out = self.tokens_encoder[0](tokens, tokens_len)
            tokens_out = self.tokens_encoder[1](tokens_out)
            # tokens_repr = self.tokens_encoder(tokens, tokens_len)
        else:
            tokens_out = 0

        code_repr = self.fusion(torch.tanh(name_out + apiseq_out + tokens_out))
        return code_repr
