# -*- coding: utf-8 -*-

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.encoders.retrieval.self_attn_encoder import SelfAttnEncoder


@register_model('simple_self_attn')
class SimpleSelfAttn(NccRetrievalModel):
    def __init__(self, args, src_encoders, tgt_encoders):
        super().__init__(src_encoders, tgt_encoders)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, dropout: Float_t = 0.1, pooling: String_t = None"""
        src_encoders = SelfAttnEncoder(
            dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
            token_types=args['model']['code_token_types'], max_positions=args['model']['code_max_tokens'],
            self_attn_layers=args['model']['self_attn_layers'], attention_heads=args['model']['attention_heads'],
            ffn_embed_dim=args['model']['ffn_embed_dim'], activation_fn=args['model']['activation_fn'],
            dropout=args['model']['code_dropout'], pooling=args['model']['code_pooling'],
        )
        tgt_encoders = SelfAttnEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            token_types=args['model']['query_token_types'], max_positions=args['model']['query_max_tokens'],
            self_attn_layers=args['model']['self_attn_layers'], attention_heads=args['model']['attention_heads'],
            ffn_embed_dim=args['model']['ffn_embed_dim'], activation_fn=args['model']['activation_fn'],
            dropout=args['model']['query_dropout'], pooling=args['model']['query_pooling'],
        )
        return cls(args, src_encoders, tgt_encoders)

    def forward(self,
                src_tokens, src_tokens_mask, src_tokens_len,
                tgt_tokens, tgt_tokens_mask, tgt_tokens_len,
                **kwargs,
                ):
        src_embed = self.src_encoders(src_tokens, src_tokens_mask, src_tokens_len)
        tgt_embed = self.tgt_encoders(tgt_tokens, tgt_tokens_mask, tgt_tokens_len)
        return src_embed, tgt_embed
