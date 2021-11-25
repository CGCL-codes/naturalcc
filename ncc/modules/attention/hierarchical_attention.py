# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import Module


class HirarchicalAttention(Module):
    '''
    ref: Hierarchical Attention Networks for Document Classiﬁcation
    '''

    def __init__(self, hidden_size: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(hidden_size, hidden_size)
        self.u_w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input: torch.Tensor, ) -> torch.Tensor:
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = (input * a_it).sum(dim=1)  # sum dim=time step
        return s_i


if __name__ == '__main__':
    bs, len, in_sz, out_sz = 2, 5, 10, 20
    x = torch.Tensor([
        [1, 2, 3, 5, 0],
        [1, 5, 2, 3, 4]
    ]).long().cuda()

    wemb = nn.Embedding(6, in_sz).cuda()
    biGRU = nn.LSTM(in_sz, out_sz, bidirectional=True).cuda()

    x = wemb(x)
    x, _ = biGRU(x)
    print(x.size())

    attn = HirarchicalAttention(hidden_size=out_sz * 2).cuda()
    x = attn(x)
    print(x.size())
