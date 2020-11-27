# -*- coding: utf-8 -*-
from ncc.utils.constants import NEG_INF
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class SelfAttention(nn.Module):
    """
    reference: https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/encoder/attention.py
    这是一个基于论文 `A structured self-attentive sentence embedding <https://arxiv.org/pdf/1703.03130.pdf>`_
    的Self Attention Module.
    """

    def __init__(self, input_size: int, attn_hops: int, attn_unit=300, dropout=None, ) -> None:
        """
        :param int input_size: 输入tensor的hidden维度
        :param int attn_unit: 输出tensor的hidden维度
        :param int attn_hops:
        """
        super(SelfAttention, self).__init__()

        self.attn_hops = attn_hops
        self.ws1 = nn.Linear(input_size, attn_unit, bias=False)
        self.ws2 = nn.Linear(attn_unit, attn_hops, bias=False)
        self.I = torch.eye(attn_hops, requires_grad=False)
        self.I_origin = self.I
        self.dropout = dropout

    def _penalization(self, attention: torch.Tensor) -> torch.Tensor:
        """
        compute the penalization term for attention module
        """
        baz = attention.size(0)
        size = self.I.size()
        if len(size) != 3 or size[0] != baz:
            self.I = self.I_origin.expand(baz, -1, -1)
            self.I = self.I.to(device=attention.device)
        attention_t = torch.transpose(attention, 1, 2).contiguous()
        mat = torch.bmm(attention, attention_t) - self.I[:attention.size(0)]
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]

    def forward(self, input: torch.Tensor, input_origin: Any) -> Any:
        """
        :param torch.Tensor input: [batch_size, seq_len, hidden_size] 要做attention的矩阵
        :param torch.Tensor input_origin: [batch_size, seq_len] 原始token的index组成的矩阵，含有pad部分内容
        :return torch.Tensor output1: [batch_size, multi-head, hidden_size] 经过attention操作后输入矩阵的结果
        :return torch.Tensor output2: [1] attention惩罚项，是一个标量
        """
        input = input.contiguous()

        input_origin = input_origin.expand(self.attn_hops, -1, -1)  # [hops,baz, len]
        input_origin = input_origin.transpose(0, 1).contiguous()  # [baz, hops,len]

        # [baz,len,dim] -->[bsz,len, attention-unit]
        if self.dropout is None:
            y1 = torch.tanh(self.ws1(input))
        else:
            y1 = torch.tanh(self.ws1(F.dropout(input, training=self.training)))

        # [bsz,len, attention-unit]--> [bsz, len, hop]--> [baz,hop,len]
        attention = self.ws2(y1).transpose(1, 2).contiguous()
        # print(attention.size())
        # print(input_origin.size())
        attention = attention + (NEG_INF * (input_origin == 0).float())  # remove the weight on padding token.
        attention = F.softmax(attention, 2)  # [baz ,hop, len]
        return torch.bmm(attention, input), self._penalization(attention)  # output1 --> [baz ,hop ,nhid]


if __name__ == '__main__':
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    encoder = nn.Embedding(10, 3, padding_idx=0)
    input_emb = encoder(input)

    print(input_emb.size())
    sa = SelfAttention(input_emb.size(-1), attn_unit=200, attn_hops=20, dropout=0.1)
    output, _ = sa(input_emb, input)
    print(output.size())
