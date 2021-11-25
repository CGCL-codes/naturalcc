import torch
import torch.nn.functional as F

from ncc.data.constants import DEFAULT_MAX_TARGET_POSITIONS
from ncc.modules.base.layers import (
    Embedding,
    Linear,
    LSTM,
)
from ncc.utils import utils
from ..ncc_incremental_decoder import NccIncrementalDecoder


class PathDecoder(NccIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        self.lstm = LSTM(hidden_size, hidden_size, dropout=dropout_in, batch_first=True)
        self.fc_out = Linear(out_embed_dim, num_embeddings, bias=False)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, hidden, cell = encoder_out[:3]
            srclen = encoder_outs.size(1)
            encoder_outs = encoder_outs.transpose(0, 1)
        else:
            srclen = None

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C

        x = x.transpose(0, 1)
        # setup decoder recurrent hiddens and cells, initialized with encoder output
        prev_hiddens = [hidden for _ in range(len(self.layers))]  # [B, C]
        prev_cells = [cell for _ in range(len(self.layers))]

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = x[j]  # <BOS> has been append at the head of tgt_seq

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
