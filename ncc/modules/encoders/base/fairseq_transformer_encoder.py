# -*- coding: utf-8 -*-


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.base.layer_norm import LayerNorm
from .fairseq_transformer_encoder_layer import FaieseqTransformerEncoderLayer
from .transformer_encoder import TransformerEncoder
from ..ncc_encoder import (
    EncoderOut,
)
from ...positional_embedding import (
    PositionalEmbedding,
)


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class FairseqTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model'].get("encoder_layerdrop", 0.)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args['model']['max_source_positions'],
                embed_dim,
                self.padding_idx,
                learned=args['model']['encoder_learned_pos'],
            )
            if not args['model']['no_token_positional_embeddings']
            else None
        )

        if args['model'].get("layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args['model']['encoder_layers'])]
        )
        self.num_layers = len(self.layers)

        if args['model']['encoder_normalize_before']:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return FaieseqTransformerEncoderLayer(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False, **kwargs):
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
