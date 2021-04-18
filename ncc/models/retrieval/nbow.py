# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.retrieval.nbow_encoder import NBOWEncoder


@register_model('nbow')
class NBOW(NccRetrievalModel):
    def __init__(self, args, src_encoders, tgt_encoders):
        super().__init__(src_encoders, tgt_encoders)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, dropout: Float_t = 0.1, pooling: String_t = None"""
        src_encoders = nn.ModuleDict()
        for lang in args['dataset']['langs']:
            src_encoders[lang] = NBOWEncoder(
                dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
                dropout=args['model']['code_dropout'], pooling=args['model']['code_pooling'],
                lang=args['task']['source_lang'],
            )
        tgt_encoders = NBOWEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            dropout=args['model']['query_dropout'], pooling=args['model']['query_pooling'],
            lang=args['task']['target_lang'],
        )
        return cls(args, src_encoders, tgt_encoders)

    def forward(self, tgt_tokens, tgt_tokens_mask, tgt_tokens_len, **kwargs):
        src_embed = [self.src_encoders[lang](**kw) for lang, kw in kwargs.items()]
        src_embed = torch.cat(src_embed, dim=0)
        tgt_embed = self.tgt_encoders(tgt_tokens, tgt_tokens_mask, tgt_tokens_len)
        return src_embed, tgt_embed
