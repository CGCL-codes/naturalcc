# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.retrieval.rnn_encoder import RNNEncoder


@register_model('birnn')
class BiRNN(NccRetrievalModel):
    def __init__(self, args, src_encoders, tgt_encoders):
        super().__init__(src_encoders, tgt_encoders)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, out_channels: Sequence_t, kernel_size: Sequence_t,"""
        src_encoders = nn.ModuleDict()
        for lang in args['dataset']['langs']:
            src_encoders[lang] = RNNEncoder(
                dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
                rnn_cell=args['model']['code_rnn_cell'], rnn_hidden_dim=args['model']['code_hidden_dim'],
                rnn_num_layers=args['model']['code_rnn_layers'],
                rnn_bidirectional=args['model']['code_rnn_bidirectional'],

                max_tokens=args['dataset']['code_max_tokens'], dropout=args['model']['dropout'],
                pooling=args['model']['code_pooling'], rnn_dropout=args['model']['code_rnn_dropout'],
            )
        tgt_encoders = RNNEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            rnn_cell=args['model']['query_rnn_cell'], rnn_hidden_dim=args['model']['query_hidden_dim'],
            rnn_num_layers=args['model']['query_rnn_layers'],
            rnn_bidirectional=args['model']['query_rnn_bidirectional'],

            max_tokens=args['dataset']['query_max_tokens'], dropout=args['model']['dropout'],
            pooling=args['model']['query_pooling'], rnn_dropout=args['model']['query_rnn_dropout'],
        )
        return cls(args, src_encoders, tgt_encoders)

    def forward(self, tgt_tokens, tgt_tokens_mask, tgt_tokens_len, **kwargs):
        src_embed = [self.src_encoders[lang](**kw) for lang, kw in kwargs.items()]
        src_embed = torch.cat(src_embed, dim=0)
        tgt_embed = self.tgt_encoders(tgt_tokens, tgt_tokens_mask, tgt_tokens_len)
        return src_embed, tgt_embed
