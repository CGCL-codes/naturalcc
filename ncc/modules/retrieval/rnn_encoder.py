# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.common.initializers import xavier_uniform
from ncc.modules.common.layers import (
    Embedding,
    Linear,
)
from ncc.utils.pooling1d import pooling1d


class RNNEncoder(NccEncoder):
    """based on CodeSearchNet """

    def __init__(self, dictionary, embed_dim, dropout,
                 # rnn config
                 rnn_cell, rnn_hidden_dim, rnn_dropout,
                 rnn_num_layers=1, rnn_bidirectional=False,
                 **kwargs):
        super().__init__(dictionary)
        # word embedding + positional embedding
        self.embed = Embedding(len(dictionary), embed_dim, initializer=xavier_uniform())
        self.dropout = dropout
        # pooling
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        if 'weighted' in pooling:
            self.weight_layer = Linear(embed_dim, 1, bias=False, weight_initializer=xavier_uniform())
        else:
            self.weight_layer = None
        # rnn
        self.rnn_dropout = rnn_dropout
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn = getattr(nn, str.upper(rnn_cell))(
            embed_dim, rnn_hidden_dim, num_layers=rnn_num_layers,
            dropout=self.rnn_dropout,  # rnn inner dropout between layers
            bidirectional=rnn_bidirectional, batch_first=True,
        )

    def init_hidden(self, batch_size: int, ):
        weight = next(self.parameters()).data
        biRNN = 2 if self.rnn.bidirectional else 1
        if isinstance(self.rnn, nn.LSTM):
            return (
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()
            )
        else:
            return weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()

    def _dynamic_forward(self, input_emb, input_len, hidden_state=None):
        # 1) unsorted seq_emb, 2) padding to same length
        input_len = input_len.view(-1)
        sorted_lens, indices = input_len.sort(descending=True)
        # sort seq_input & hidden state by seq_lens
        sorted_input_emb = input_emb.index_select(dim=0, index=indices)
        if hidden_state is None:
            pass
        else:
            if isinstance(self.rnn, nn.LSTM):
                hidden_state = (
                    hidden_state[0].index_select(dim=1, index=indices),
                    hidden_state[1].index_select(dim=1, index=indices),
                )
            else:
                hidden_state = hidden_state.index_select(dim=1, index=indices)

        packed_seq_input = pack_padded_sequence(sorted_input_emb, sorted_lens.data.tolist(), batch_first=True)
        seq_output, _ = self.rnn(packed_seq_input, hidden_state)
        seq_output, _ = pad_packed_sequence(seq_output, batch_first=True, total_length=input_emb.size(1))
        _, reversed_indices = indices.sort()
        seq_output = seq_output.index_select(0, reversed_indices)

        # # restore seq_input & hidden state by seq_lens
        # if isinstance(self.rnn, nn.LSTM):
        #     last_h = hidden_state[0].index_select(dim=1, index=reversed_indices) \
        #         [:-(1 + int(self.rnn_bidirectional)), ...]
        #     last_c = hidden_state[1].index_select(dim=1, index=reversed_indices) \
        #         [:-(1 + int(self.rnn_bidirectional)), ...]
        #     last_hidden_state = [(last_h[idx], last_c[idx]) for idx in range(last_h.size(0))]
        #
        # else:
        #     last_h = hidden_state.index_select(dim=1, index=reversed_indices) \
        #         [:-(1 + int(self.rnn_bidirectional)), ...]
        #     last_hidden_state = [last_h[idx] for idx in range(last_h.size(0))]

        return seq_output, _

    def _merge_state(self, hidden_state):
        if isinstance(self.rnn, nn.LSTM):
            hidden_state = torch.cat(
                [torch.cat(hc, dim=-1) for hc in hidden_state],
                dim=-1
            )
        else:
            hidden_state = torch.cat(
                [torch.cat(*hs, dim=-1) for hs in hidden_state],
                dim=-1
            )
        return hidden_state

    def forward(self, tokens, tokens_mask=None, tokens_len=None):
        # TODO: need to be checked
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        tokens = self.embed(tokens)
        if self.dropout:
            tokens = F.dropout(tokens, p=self.dropout, training=self.training)
        tokens, _ = self._dynamic_forward(tokens, tokens_len)
        # tokens = self._merge_state(last_hidden_state)
        if self.pooling:
            tokens = self.pooling(
                input_emb=tokens, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return tokens
