# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, Any


class RNNEncoder(nn.Module):

    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, layer_num: int, dropout: float,
                 bidirectional: bool, ) -> None:
        super(RNNEncoder, self).__init__()
        # rnn params
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, self.rnn_type)(input_size=self.input_size, hidden_size=self.hidden_size,
                                              num_layers=self.layer_num, dropout=dropout, batch_first=True,
                                              bidirectional=self.bidirectional)

    @classmethod
    def load_from_config(cls, config: Dict) -> Any:
        instance = cls(
            input_size=config['training']['embed_size'],
            rnn_type=config['training']['rnn_type'],
            hidden_size=config['training']['rnn_hidden_size'],
            layer_num=config['training']['rnn_layer_num'],
            dropout=config['training']['dropout'],
            bidirectional=config['training']['rnn_bidirectional'],
        )
        return instance

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().reshape(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order

    def _forward(self, seq_emb: torch.Tensor, hidden: Any) -> Any:
        # all seq have same length
        return self.rnn(seq_emb, hidden)

    def _forward_sorted(self, seq_emb: torch.Tensor, seq_len: torch.Tensor, hidden: Any) -> Any:
        # 1) sort seq_emb by length with descending mode, 2) padding to same length
        packed_seq_input = pack_padded_sequence(seq_emb, seq_len, batch_first=True)
        output, hidden = self.rnn(packed_seq_input, hidden)
        outputs, _ = pad_packed_sequence(output, batch_first=True)
        return outputs, hidden

    def _forward_unsorted(self, seq_emb: torch.Tensor, seq_len: torch.Tensor, hidden: Any) -> Any:
        # 1) unsorted seq_emb, 2) padding to same length
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_len)
        # sort seq_input & hidden state by seq_lens
        sorted_seq_emb = seq_emb.index_select(dim=0, index=fwd_order)
        if hidden is None:
            pass
        else:
            if type(hidden) == tuple:
                hidden = (
                    hidden[0].index_select(dim=1, index=fwd_order),
                    hidden[1].index_select(dim=1, index=fwd_order),
                )
            else:
                hidden = hidden.index_select(dim=1, index=fwd_order)

        output, hidden = self._forward_sorted(sorted_seq_emb, sorted_len, hidden)

        output = output.index_select(dim=0, index=bwd_order)
        if type(hidden) == tuple:
            hidden = (
                hidden[0].index_select(dim=1, index=bwd_order),
                hidden[1].index_select(dim=1, index=bwd_order),
            )
        else:
            hidden = hidden.index_select(dim=1, index=bwd_order)

        return output, hidden

    def forward(self, seq_emb: torch.Tensor, seq_len=None, hidden=None, is_sorted=None, ) -> Any:
        if seq_len is None:
            return self._forward(seq_emb, hidden)
        else:
            if is_sorted is None:
                is_sorted = (seq_len.sort(dim=0, descending=True)[0] == seq_len).sum().item() == seq_len.size(0)

            if is_sorted:
                return self._forward_sorted(seq_emb, seq_len, hidden)
            else:
                return self._forward_unsorted(seq_emb, seq_len, hidden)

    def init_hidden(self, batch_size: int, ) -> Any:
        weight = next(self.parameters()).data
        biRNN = 2 if self.rnn.bidirectional else 1
        if self.rnn_type == 'LSTM':
            return (
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()
            )
        else:
            return weight.new(self.layer_num * biRNN, batch_size, self.hidden_size).zero_().requires_grad_()


# if __name__ == '__main__':
#     # input = torch.LongTensor([[1, 2, 4, 0], [4, 3, 0, 0]])
#     input = torch.LongTensor([[4, 3, 0, 0], [1, 2, 4, 2], ])
#     seq_len = input.data.gt(0).sum(-1)
#     embed = nn.Embedding(10, 10, 0)
#     input = embed(input)
#
#     encoder = Encoder_RNN(rnn_type='LSTM', input_size=10, hidden_size=10, layer_num=2, dropout=0.1, )
#     hidden = encoder.init_hidden(input.size(0))
#
#     # input, hidden = encoder(input, hidden=hidden)
#     # input, hidden = encoder(input, seq_len, hidden)
#     input, hidden = encoder(input, seq_len, hidden, )
#     print(input.size(), (hidden[0].size(), hidden[1].size()))
