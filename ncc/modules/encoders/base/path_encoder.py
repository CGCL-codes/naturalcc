import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.data.constants import DEFAULT_MAX_SOURCE_POSITIONS
from ncc.modules.base.layers import (
    Embedding,
    LSTM,
)
from ..ncc_encoder import NccEncoder


class PathEncoder(NccEncoder):
    """
    Refs: Code2Seq
    LSTM encoder:
        head/tail -> sub_tokens -> embedding -> sum
        body -> LSTM -> hidden state
        head_sum, hidden state, tail_sum -> W -> tanh
    """

    def __init__(
        self, dictionary, node_dictionary,
        embed_dim=512, type_embed_dim=512,
        hidden_size=512, decoder_hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=True, left_pad=True,
        pretrained_embed=None, pretrained_terminals_embed=None,
        padding_idx=None, type_padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(dictionary)
        self.node_dictionary = node_dictionary
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        self.padding_idx = padding_idx if padding_idx is not None else self.dictionary.pad()
        self.type_padding_idx = type_padding_idx if type_padding_idx is not None else self.node_dictionary.pad()

        if pretrained_embed is None:
            self.subtoken_embed = Embedding(len(dictionary), embed_dim, self.padding_idx)
        else:
            self.subtoken_embed = pretrained_embed
        if pretrained_terminals_embed is None:
            self.node_embed = Embedding(len(self.node_dictionary), type_embed_dim, self.type_padding_idx)
        else:
            self.node_embed = pretrained_terminals_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size // (1 + int(bidirectional)),
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.left_pad = left_pad
        self.transform = nn.Sequential(
            nn.Linear(2 * type_embed_dim + hidden_size, decoder_hidden_size, bias=False),
            nn.Tanh(),
        )
        self.output_units = decoder_hidden_size

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().reshape(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        return sorted_len.tolist(), fwd_order, bwd_order

    def forward(self, src_tokens, src_lengths, **kwargs):
        """head_tokens, tail_tokens, body_tokens: bsz, path_num, seqlen"""
        heads, bodies, tails = src_tokens  # [B, N, ?]

        head_embed = self.subtoken_embed(heads).sum(dim=-2)  # [bsz, path_num, embed_dim]
        tail_embed = self.subtoken_embed(tails).sum(dim=-2)  # [bsz, path_num, embed_dim]

        # embed tokens
        body_embed = self.node_embed(bodies)  # bsz, path_num, seqlen, embed_dim
        bsz, path_num, seqlen, embed_dim = body_embed.size()
        body_embed = body_embed.view(-1, seqlen, embed_dim)
        # body_embed = F.dropout(body_embed, p=self.dropout_in, training=self.training)

        # pack embedded source tokens into a PackedSequence
        src_lengths = src_lengths.view(-1)
        src_lengths, fwd_order, bwd_order = self._get_sorted_order(src_lengths)
        # sort seq_input & hidden state by seq_lens
        body_embed = body_embed.index_select(dim=0, index=fwd_order)
        packed_x = nn.utils.rnn.pack_padded_sequence(body_embed, src_lengths, batch_first=True, enforce_sorted=False)

        # apply LSTM
        state_size = (1 + int(self.bidirectional)) * self.num_layers, bsz * path_num, \
                     self.hidden_size // (1 + int(self.bidirectional))
        h0 = body_embed.new_zeros(*state_size)
        c0 = body_embed.new_zeros(*state_size)
        _, (final_hiddens, _) = self.lstm(packed_x, (h0, c0))
        final_hiddens = torch.transpose(final_hiddens, dim0=0, dim1=1)  # [B,2,E]
        final_hiddens = final_hiddens.contiguous().view(final_hiddens.size(0), -1)  # [B,2*E]

        # we only use hidden_state of LSTM
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        # x = x.index_select(dim=1, index=bwd_order)
        final_hiddens = final_hiddens.index_select(dim=0, index=bwd_order)
        # final_cells = final_cells.index_select(dim=0, index=bwd_order)
        final_hiddens = final_hiddens.view(bsz, path_num, -1)

        # different from paper, we obey the intuition of code:
        #   https://github.com/tech-srl/code2seq/blob/a29d0c761f8eca4c27765a1ab04e44815f62bfe6/model.py#L537-L538
        x = torch.cat([head_embed, final_hiddens, tail_embed], dim=-1)

        x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.transform(x)  # [bsz, path_num, dim]
        final_hiddens = final_cells = x.sum(dim=1)

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': None,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions
