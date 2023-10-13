from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from ncc.models import register_model
from ncc.modules.base.layers import Embedding
from ncc.modules.decoders.base import FairTransformerDecoder
from ncc.modules.encoders.base import FairseqTransformerEncoder
from ncc.utils import utils
from .transformer import Transformer


@register_model('fairseq_transformer')
class FairseqTransformer(Transformer):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        if args['model']['max_source_positions'] is None:
            args['model']['max_source_positions'] = DEFAULT_MAX_SOURCE_POSITIONS
        if args['model']['max_target_positions'] is None:
            args['model']['max_target_positions'] = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args['model']['share_all_embeddings']:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args['model']['encoder_embed_dim'] != args['model']['decoder_embed_dim']:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args['model']['decoder_embed_path'] and (
                args['model']['decoder_embed_path'] != args['model']['encoder_embed_path']
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = encoder_embed_tokens
            args['model']['share_decoder_input_output_embed'] = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                src_dict, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
            )
            decoder_embed_tokens = cls.build_embedding(
                tgt_dict, args['model']['decoder_embed_dim'], args['model']['decoder_embed_path']
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @staticmethod
    def build_embedding(dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FairseqTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return FairTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
