from ncc.data.constants import DEFAULT_MAX_SOURCE_POSITIONS
from .lstm_encoder import LSTMEncoder
from .path_encoder import PathEncoder
from ..ncc_encoder import NccEncoder


class MultiModalitiesEncoder(NccEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        args,
        dictionary,
        pretrained_embed=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        self.max_source_positions = max_source_positions
        self.args = args
        self.dictionary = dictionary

        if 'code' in self.args['task']['source_lang']:
            self.code_encoder = LSTMEncoder(
                dictionary=dictionary['code'],
                embed_dim=args['model']['encoder_embed_dim'],
                hidden_size=args['model']['encoder_hidden_size'],
                num_layers=args['model']['encoder_layers'],
                dropout_in=args['model']['encoder_dropout_in'],
                dropout_out=args['model']['encoder_dropout_out'],
                bidirectional=bool(args['model']['encoder_bidirectional']),
                pretrained_embed=pretrained_embed['code'],
                max_source_positions=max_source_positions
            )

        if 'path' in self.args['task']['source_lang']:
            self.path_encoder = PathEncoder(
                dictionary=dictionary['path'],
                node_dictionary=dictionary['node'],
                embed_dim=args['model']['encoder_embed_dim'],
                hidden_size=args['model']['encoder_hidden_size'],
                num_layers=args['model']['encoder_layers'],
                dropout_in=args['model']['encoder_dropout_in'],
                dropout_out=args['model']['encoder_dropout_out'],
                bidirectional=bool(args['model']['encoder_bidirectional']),
                pretrained_embed=pretrained_embed['path'],
                max_source_positions=max_source_positions
            )

        if 'bin_ast' in self.args['task']['source_lang']:
            pass

        self.output_units = args['model']['encoder_hidden_size']
        if bool(args['model']['encoder_bidirectional']):
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        code_encoder_out, ast_encoder_out, path_encoder_out = None, None, None
        if 'code' in self.args['task']['source_lang']:
            code_encoder_out = self.code_encoder(src_tokens['code'], src_lengths=src_lengths['code'])

        if 'ast' in self.args['task']['source_lang']:
            # ast_encoder_out = self.ast_encoder()
            pass

        if 'path' in self.args['task']['source_lang']:
            path_encoder_out = self.path_encoder(src_tokens['path'], src_lengths=src_lengths['path'])

        return {
            'code': code_encoder_out,
            'ast': ast_encoder_out,
            'path': path_encoder_out
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
