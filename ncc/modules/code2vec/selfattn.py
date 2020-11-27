# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.modules.code2vec.base import *
from ncc.modules.attention import *
from typing import Dict, Any


class SelfAttnEncoder(nn.Module):

    def __init__(self, token_num: int, embed_size: int,
                 rnn_type: str, hidden_size: int, layer_num: int, dropout: float, bidirectional: bool,
                 attn_hops: int, attn_unit: int, ) -> None:
        super(SelfAttnEncoder, self).__init__()
        self.wemb = Encoder_Emb(token_num, embed_size, )
        self.rnn = Encoder_RNN(rnn_type, embed_size, hidden_size, layer_num, 0.0, bidirectional)
        self.self_attn = SelfAttention(hidden_size * (2 if bidirectional else 1), attn_hops=attn_hops,
                                       attn_unit=attn_unit * (2 if bidirectional else 1),
                                       dropout=dropout)
        self.dropout = dropout
        # self.transform = nn.Linear(hidden_size * (2 if bidirectional else 1) * attn_hops,
        #                            attn_unit * (2 if bidirectional else 1))

    @classmethod
    def load_from_config(cls, config: Dict, modal: str, ) -> Any:
        instance = cls(
            token_num=config['training']['token_num'][modal],
            embed_size=config['training']['embed_size'],

            rnn_type=config['training']['rnn_type'],
            hidden_size=config['training']['rnn_hidden_size'],
            layer_num=config['training']['rnn_layer_num'],
            dropout=config['training']['dropout'],
            bidirectional=config['training']['rnn_bidirectional'],

            attn_hops=config['training']['attn_hops'],
            attn_unit=config['training']['attn_unit'],
        )
        return instance

    def init_hidden(self, batch_size: int, ) -> Any:
        return self.rnn.init_hidden(batch_size)

    def forward(self, input: torch.Tensor, input_len=None, ) -> Any:
        if input_len is None:
            input_len = (input > 0).sum(dim=-1).to(input.device)
        input_emb = self.wemb(input)
        hidden = self.init_hidden(input_emb.size(0))
        input_emb, hidden = self.rnn(input_emb, seq_len=input_len, hidden=hidden)
        input_emb, _ = self.self_attn(input_emb, input)
        # input_emb = input_emb.view(input_emb.size(0), -1)
        # input_emb = torch.tanh(self.transform(input_emb))

        input_emb = torch.tanh(input_emb)
        input_emb, _ = input_emb.max(dim=1)
        input_emb = F.dropout(input_emb, self.dropout, self.training)

        return input_emb


# if __name__ == '__main__':
#     input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 0]])
#     encoder = Encoder_EmbRNNSelfAttn(token_num=10, embed_size=50,
#                                      rnn_type='LSTM', hidden_size=512, layer_num=1, dropout=0.1, bidirectional=True,
#                                      attn_unit=100, )
#     output = encoder(input)
#     print(output.size())
