import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.modules.attention.mutlihead_attention import MultiheadAttention
from ncc.modules.base.layer_norm import LayerNorm
from .ncc_transformer_encoder_layer import NccTransformerEncoderLayer
from ..ncc_encoder import (
    NccEncoder,
    EncoderOut,
)
from ...positional_embedding import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class NccTransformerEncoder(NccEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        num_segments: int = 1,
        offset_positions_by_padding: bool = False,  # True,
        # apply_bert_init: bool = False,
        # freeze_embeddings: bool = False,
        # n_trans_layers_to_freeze: int = 0,
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout = args['model']['dropout']
        self.encoder_layerdrop = args['model']['encoder_layerdrop']

        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = dictionary.pad()  # embed_tokens.padding_idx TODO
        # self.vocab_size = vocab_size
        self.max_source_positions = args['model']['max_source_positions']

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args['model']['no_scale_embedding'] else math.sqrt(self.embed_dim)

        offset_positions_by_padding = args['model'].get('offset_positions_by_padding', True)
        if args['model']['encoder_positional_embeddings']:
            self.embed_positions = None
        else:
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
                m = LearnedPositionalEmbedding(num_embeddings, self.embed_dim,
                                               padding_idx=self.padding_idx if offset_positions_by_padding else None)
                nn.init.normal_(m.weight, mean=0, std=self.embed_dim ** -0.5)
                if self.padding_idx is not None:
                    nn.init.constant_(m.weight[self.padding_idx], 0)
                self.embed_positions = m

        self.num_segments = num_segments
        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embed_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )
        self.layers = nn.ModuleList([NccTransformerEncoderLayer(args) for _ in range(args['model']['encoder_layers'])])

        self.num_layers = len(self.layers)
        if args['model']['encoder_normalize_before']:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None
        if args['model']['layernorm_embedding']:
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

    def forward_embedding(self, src_tokens, positions, segment_labels, padding_mask):
        x = self.embed_tokens(src_tokens)

        if self.embed_scale is not None:
            x = embed = x * self.embed_scale

        if self.embed_positions is not None:
            if self.args['model']['encoder_position_encoding_version'] == 'contracode':
                x += self.embed_positions(src_tokens)
            elif self.args['model']['encoder_position_encoding_version'] == 'ncc_sinusoidal':
                x += self.embed_positions(src_tokens, positions=positions)
            elif self.args['model']['encoder_position_encoding_version'] == 'ncc_learned':
                x += self.embed_positions(src_tokens)

        if self.num_segments > 1 and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        # TODO, position里面如果已经dropout了，这里就没必要了
        # TODO, not all positional encodings have dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor = None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x, encoder_embedding = self.forward_embedding(src_tokens, positions, segment_labels, encoder_padding_mask)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        # if not last_state_only:
        #     encoder_states.append(x)

        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, src_mask=None, src_key_padding_mask=encoder_padding_mask)
            if not last_state_only:
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

