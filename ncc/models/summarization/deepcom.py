import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
)
from ncc.models import register_model
from ncc.models.ncc_model import NccEncoderDecoderModel
from ncc.modules.base.layers import (
    Embedding,
    LSTM,
    LSTMCell,
)
from ncc.modules.decoders.ncc_incremental_decoder import NccIncrementalDecoder
from ncc.modules.encoders.ncc_encoder import NccEncoder
from ncc.utils import utils


class LSTMEncoder(NccEncoder):
    """LSTM encoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units], (x.size())

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
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


class LSTMDecoder(NccIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.max_target_positions = max_target_positions

        num_embeddings = len(dictionary)
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx=dictionary.pad())
        else:
            self.embed_tokens = pretrained_embed

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=encoder_output_units if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if not self.share_input_output_embed:
            self.fc_out = nn.Linear(out_embed_dim, num_embeddings)

    def attention(self, hidden, encoder_out, encoder_mask):
        """
        hidden: [batch_size, hidden_size] 80, 512
        encoder_out: [batch_size, src_len, hidden_size] 16, 80, 512
        encoder_mask: [src_len, batch_size] 80, 16
        """
        # influence of src_hidden(j) over tgt_hidden_i
        # assert hidden.size(0) == encoder_out.size(0), \
        #     ('hidden', hidden.size(), 'encoder_out', encoder_out.size(), 'encoder_mask', encoder_mask.size())
        # print(('hidden', hidden.size(), 'encoder_out', encoder_out.size(), 'encoder_mask', encoder_mask.size()))
        a_ij = torch.bmm(encoder_out.transpose(0, 1), hidden.unsqueeze(dim=-1)).squeeze(dim=-1)  # [batch_size, src_len]
        # print(a_ij.size(), encoder_mask.size())
        if encoder_mask is None:
            a_ij_softmax = a_ij.softmax(dim=-1)
        else:
            a_ij_softmax = a_ij.masked_fill(encoder_mask.t(), float('-inf')).softmax(dim=-1)
        a_ij_softmax = a_ij_softmax.unsqueeze(dim=1)  # [batch_size, 1, src_len]
        # [batch_size, 1, src_len] x [batch_size, src_len, hidden_size] => [batch_size, 1, hidden_size]
        t_i = torch.bmm(a_ij_softmax, encoder_out.transpose(0, 1)).squeeze(dim=1)
        return t_i

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
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # setup zero cells, since there is no encoder
        num_layers = len(self.layers)
        prev_hiddens = [x.new_zeros(bsz, self.hidden_size) for i in range(num_layers)]
        prev_cells = [x.new_zeros(bsz, self.hidden_size) for i in range(num_layers)]

        outs = []
        for j in range(seqlen):
            input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            attn_out = self.attention(prev_hiddens[-1], encoder_outs, encoder_padding_mask)

            attn_out = F.dropout(attn_out, p=self.dropout_out, training=self.training)

            # save final output
            outs.append(attn_out)

        # collect outputs across time steps
        x = torch.stack(outs, dim=0)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        return x, None

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        x = self.fc_out(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


@register_model('deepcom')
class DeepCom(NccEncoderDecoderModel):
    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        # base_architecture(args)

        max_source_positions = args['model']['max_source_positions'] \
            if args['model']['max_source_positions'] else DEFAULT_MAX_SOURCE_POSITIONS
        max_target_positions = args['model']['max_target_positions'] \
            if args['model']['max_target_positions'] else DEFAULT_MAX_TARGET_POSITIONS

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args['model']['encoder_embed']:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_embed_path'], task.source_dictionary, args['model']['encoder_embed_dim'])
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args['model']['encoder_embed_dim'], task.source_dictionary.pad()
            )

        if args['model']['share_all_embeddings']:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args['model']['decoder_embed_path'] and (
                args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args['model']['share_decoder_input_output_embed'] = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args['model']['decoder_embed']:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args['model']['decoder_embed'],
                    task.target_dictionary,
                    args['model']['decoder_embed_dim']
                )
        # one last double check of parameter combinations
        if args['model']['share_decoder_input_output_embed'] and (
            args['model']['decoder_embed_dim'] != args['model']['decoder_out_embed_dim']):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args['model']['encoder_freeze_embed']:
            pretrained_encoder_embed.weight.requires_grad = False
        if args['model']['decoder_freeze_embed']:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args['model']['encoder_embed_dim'],
            hidden_size=args['model']['encoder_hidden_size'],
            num_layers=args['model']['encoder_layers'],
            dropout_in=args['model']['encoder_dropout_in'],
            dropout_out=args['model']['encoder_dropout_out'],
            bidirectional=bool(args['model']['encoder_bidirectional']),
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args['model']['decoder_embed_dim'],
            hidden_size=args['model']['decoder_hidden_size'],
            out_embed_dim=args['model']['decoder_out_embed_dim'],
            num_layers=args['model']['decoder_layers'],
            dropout_in=args['model']['decoder_dropout_in'],
            dropout_out=args['model']['decoder_dropout_out'],
            encoder_output_units=args['model']['encoder_embed_dim'],
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            max_target_positions=max_target_positions
        )
        return cls(encoder, decoder)
