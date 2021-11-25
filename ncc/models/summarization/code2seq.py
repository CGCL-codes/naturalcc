from ncc.utils import utils
from ncc.models import register_model
from ncc.models.ncc_model import NccEncoderDecoderModel

from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS
)

from ncc.modules.base.layers import Embedding
from ncc.modules.encoders.base import PathEncoder
from ncc.modules.decoders.base import PathDecoder

@register_model('code2seq')
class Code2Seq(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
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

        # subtoken
        if args['model']['encoder_path_embed']:
            pretrained_encoder_path_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_path_embed'], task.source_dictionary, args['model']['encoder_path_embed_dim'])
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_path_embed = Embedding(
                num_embeddings, args['model']['encoder_path_embed_dim'],
                padding_idx=task.source_dictionary.pad()
            )
        # type
        if args['model']['encoder_terminals_embed']:
            pretrained_encoder_terminals_embed = load_pretrained_embedding_from_file(
                args['model']['encoder_terminals_embed'],
                task.type_dict, args['model']['encoder_terminals_embed_dim'])
        else:
            num_embeddings = len(task.type_dict)
            pretrained_encoder_terminals_embed = Embedding(
                num_embeddings, args['model']['encoder_terminals_embed_dim'],
                padding_idx=task.type_dict.pad()
            )
        # decoder
        if args['model']['decoder_embed']:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args['model']['decoder_embed'],
                task.target_dictionary, args['model']['decoder_embed_dim'])
        else:
            num_embeddings = len(task.target_dictionary)
            pretrained_decoder_embed = Embedding(
                num_embeddings, args['model']['decoder_embed_dim'],
                padding_idx=task.target_dictionary.pad()
            )

        if args['model']['encoder_path_freeze_embed']:
            pretrained_encoder_path_embed.weight.requires_grad = False
        if args['model']['encoder_terminals_freeze_embed']:
            pretrained_encoder_terminals_embed.weight.requires_grad = False
        if args['model']['decoder_freeze_embed']:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = PathEncoder(
            dictionary=task.source_dictionary,
            node_dictionary=task.type_dict,
            embed_dim=args['model']['encoder_path_embed_dim'],
            type_embed_dim=args['model']['encoder_terminals_embed_dim'],
            hidden_size=args['model']['encoder_hidden_size'],
            decoder_hidden_size=args['model']['decoder_hidden_size'],
            num_layers=args['model']['encoder_layers'],
            dropout_in=args['model']['encoder_dropout_in'],
            dropout_out=args['model']['encoder_dropout_out'],
            bidirectional=bool(args['model']['encoder_bidirectional']),
            pretrained_embed=pretrained_encoder_path_embed,
            pretrained_terminals_embed=pretrained_encoder_terminals_embed,
            max_source_positions=max_source_positions
        )
        decoder = PathDecoder(
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
