# -*- coding: utf-8 -*-

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.encoders.retrieval import Conv1dResEncoder


@register_model('simple_conv1d_res')
class SimpleConv1dRes(NccRetrievalModel):
    def __init__(self, args, src_encoders, tgt_encoders):
        super().__init__(src_encoders, tgt_encoders)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        """dictionary, embed_dim: Int_t, out_channels: Sequence_t, kernel_size: Sequence_t,"""
        src_encoders = Conv1dResEncoder(
            dictionary=task.source_dictionary, embed_dim=args['model']['code_embed_dim'],
            out_channels=args['model']['code_layers'], kernel_size=args['model']['code_kernel_size'],
            max_tokens=args['dataset']['code_max_tokens'],
            dropout=args['model']['code_dropout'], residual=args['model']['code_residual'],
            activation_fn=args['model']['code_activation_fn'], padding=args['model']['code_paddding'],
            pooling=args['model']['code_pooling'], position_encoding=args['model']['code_position_encoding'],
        )
        tgt_encoders = Conv1dResEncoder(
            dictionary=task.target_dictionary, embed_dim=args['model']['query_embed_dim'],
            out_channels=args['model']['query_layers'], kernel_size=args['model']['query_kernel_size'],
            max_tokens=args['dataset']['query_max_tokens'],
            dropout=args['model']['query_dropout'], residual=args['model']['query_residual'],
            activation_fn=args['model']['query_activation_fn'], padding=args['model']['query_paddding'],
            pooling=args['model']['query_pooling'], position_encoding=args['model']['query_position_encoding'],
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
