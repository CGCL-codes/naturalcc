import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .neural_transformer_encoder_layer import NeuralTransformerEncoderLayer
from .transformer_encoder import TransformerEncoder
from ..ncc_encoder import (
    EncoderOut,
)
from ...positional_embedding import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
)


class NeuralTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        offset_positions_by_padding = args['model'].get('offset_positions_by_padding', False)
        if args['model']['encoder_positional_embeddings']:
            # Option 1
            if args['model']['encoder_position_encoding_version'] == 'ncc_sinusoidal':
                self.embed_positions = SinusoidalPositionalEmbedding(
                    self.embed_dim,
                    padding_idx=self.padding_idx if offset_positions_by_padding else None,
                    init_size=args['model']['max_source_positions'] + self.padding_idx + 1 \
                        if offset_positions_by_padding else args['model']['max_source_positions'],
                )
            # Option 2
            elif args['model']['encoder_position_encoding_version'] == 'ncc_learned':
                num_embeddings = args['model']['max_source_positions']
                if offset_positions_by_padding:
                    num_embeddings += self.padding_idx + 1
                m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim, padding_idx=None)
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
                self.embed_positions = m
        else:
            self.embed_positions = None

        if args['model']['layernorm_embedding']:
            self.layernorm_embedding = nn.LayerNorm(self.embed_dim)  # LayerNorm(self.embed_dim, export=export) TODO
        else:
            self.layernorm_embedding = None

        self.layer_wise_attention = args['model']['layer_wise_attention']
        self.layers = nn.ModuleList(
            [NeuralTransformerEncoderLayer(args) for _ in range(args['model']['encoder_layers'])]
        )
        self.num_layers = len(self.layers)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        encoder_states = [] if return_all_hiddens else None
        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )
