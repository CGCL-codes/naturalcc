from ncc.models.ncc_model import NccEncoderDecoderModel
from ncc.modules.embedding import Embedding
# from ncc.modules.code2vec.lstm_encoder import LSTMEncoder
from ncc.modules.code2vec.nary_tree_lstm import NaryTreeLSTMEncoder
from ncc.modules.seq2seq.lstm_decoder import LSTMDecoder
from ncc.models import register_model
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('nary_tree2seq')
class NaryTree2SeqModel(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        # base_architecture(args)

        if args['model']['encoder_layers'] != args['model']['decoder_layers']:
            raise ValueError('--encoder-layers must match --decoder-layers')

        max_source_positions = args['model']['max_source_positions'] if args['model']['max_source_positions'] \
            else DEFAULT_MAX_SOURCE_POSITIONS
        max_target_positions = args['model']['max_target_positions'] if args['model']['max_target_positions'] \
            else DEFAULT_MAX_TARGET_POSITIONS

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

        encoder = NaryTreeLSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args['model']['encoder_embed_dim'],
            hidden_size=args['model']['encoder_hidden_size'],
            num_layers=args['model']['encoder_layers'],
            dropout_in=args['model']['encoder_dropout_in'],
            dropout_out=args['model']['encoder_dropout_out'],
            bidirectional=bool(args['model']['encoder_bidirectional']),
            left_pad=args['task']['left_pad_source'],
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
            attention=args['model']['decoder_attention'],
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args['model']['share_decoder_input_output_embed'],
            adaptive_softmax_cutoff=(
                args['model']['adaptive_softmax_cutoff']
                if args['criterion'] == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions
        )
        return cls(encoder, decoder)

    def forward(self, src_graphs, src_masks, src_roots, src_nodes,
                prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_graphs, src_roots, src_nodes, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features
