# -*- coding: utf-8 -*-

# reference: https://github.com/rohithreddy024/Text-Summarizer-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from ncc.data.constants import EPS


class IntraAttention_Encoder(Module):
    def __init__(self, hidden_size, intra_encoder=True):
        super(IntraAttention_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.intra_encoder = intra_encoder
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_s = nn.Linear(self.hidden_size, self.hidden_size)  # * 2
        self.v = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, st_hat, h, enc_padding_mask, sum_temporal_srcs):
        ''' Perform attention over encoder hidden states
        :param st_hat: decoder hidden state at current time step
        :param h: encoder hidden states
        :param enc_padding_mask:
        :param sum_temporal_srcs: if using intra-temporal attention, contains summation of attention weights from previous decoder time steps
        '''
        # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
        et = self.W_h(h)  # bs,n_seq,2*n_hid
        dec_fea = self.W_s(st_hat).unsqueeze(1)  # bs,1,2*n_hid
        et = et + dec_fea
        et = torch.tanh(et)  # bs,n_seq,2*n_hid
        et = self.v(et).squeeze(2)  # bs,n_seq
        # intra-temporal attention     (eq 3 in https://arxiv.org/pdf/1705.04304.pdf)
        if self.intra_encoder:
            exp_et = torch.exp(et)
            if sum_temporal_srcs is None:
                et1 = exp_et
                sum_temporal_srcs = torch.FloatTensor(et.size()).fill_(1e-10).cuda() + exp_et
            else:
                et1 = exp_et / sum_temporal_srcs  # bs, n_seq
                sum_temporal_srcs = sum_temporal_srcs + exp_et
        else:
            et1 = F.softmax(et, dim=1)
        # assign 0 probability for padded elements
        at = et1 * enc_padding_mask
        normalization_factor = at.sum(1, keepdim=True)
        at = at / (normalization_factor + EPS)  # + 1e-13 TODO: tobe verified again

        at = at.unsqueeze(1)  # bs,1,n_seq
        # Compute encoder context vector
        ct_e = torch.bmm(at, h)  # bs, 1, 2*n_hid
        ct_e = ct_e.squeeze(1)
        at = at.squeeze(1)
        return ct_e, at, sum_temporal_srcs


class IntraAttention_Decoder(nn.Module):
    def __init__(self, hidden_size, intra_decoder=True):
        super(IntraAttention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.intra_decoder = intra_decoder
        if intra_decoder:
            self.W_prev = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.W_s = nn.Linear(self.hidden_size, self.hidden_size)
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, s_t, prev_s):
        '''Perform intra_decoder attention
        Args
        :param s_t: hidden state of decoder at current time step
        :param prev_s: If intra_decoder attention, contains list of previous decoder hidden states
        '''
        if self.intra_decoder is False:
            ct_d = torch.zeros(s_t.size()).cuda()
        elif prev_s is None:
            ct_d = torch.zeros(s_t.size()).cuda()
            prev_s = s_t.unsqueeze(1)  # bs, 1, n_hid
        else:
            # Standard attention technique (eq 1 in https://arxiv.org/pdf/1704.04368.pdf)
            et = self.W_prev(prev_s)  # bs,t-1,n_hid
            dec_fea = self.W_s(s_t).unsqueeze(1)  # bs,1,n_hid
            et = et + dec_fea
            et = torch.tanh(et)  # bs,t-1,n_hid
            et = self.v(et).squeeze(2)  # bs,t-1
            # intra-decoder attention     (eq 7 & 8 in https://arxiv.org/pdf/1705.04304.pdf)
            at = F.softmax(et, dim=1).unsqueeze(1)  # bs, 1, t-1
            ct_d = torch.bmm(at, prev_s).squeeze(1)  # bs, n_hid
            prev_s = torch.cat([prev_s, s_t.unsqueeze(1)], dim=1)  # bs, t, n_hid

        return ct_d, prev_s


class IntraAttention(nn.Module):
    def __init__(self, hidden_size, token_num, pointer=True):
        super(IntraAttention, self).__init__()
        self.hidden_size = hidden_size
        self.token_num = token_num
        self.enc_attention = IntraAttention_Encoder(hidden_size)
        self.dec_attention = IntraAttention_Decoder(hidden_size)
        self.x_context = nn.Linear(hidden_size * 2, hidden_size)
        if pointer:
            self.p_gen_linear = nn.Linear(hidden_size * 4 + hidden_size, 1)  # nhid   nhid  2*nhid  ninp
        self.V = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, input_emb, enc_output, enc_padding_mask,
                sum_temporal_srcs, dec_hidden, prev_s):  # code_dict_comment, pointer_extra_zeros,
        st_hat = dec_hidden[0][-1]  # -1 means the hidden state of last layer
        ct_e, attn_dist_seq, sum_temporal_srcs = self.enc_attention(st_hat, enc_output,
                                                                    enc_padding_mask,
                                                                    sum_temporal_srcs)

        ct_d, prev_s = self.dec_attention(dec_hidden[0][-1], prev_s)  # intra-decoder attention
        output = torch.cat([dec_hidden[0][-1], ct_e, ct_d], dim=1)
        output = self.V(output)

        return output, sum_temporal_srcs, prev_s, ct_e, ct_d, st_hat, attn_dist_seq
