from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ncc.modules.base.activations import get_activation
from ncc.modules.base.layer_norm import LayerNorm
from ncc.modules.base.layers import Linear

from .transformer_encoder_layer import TransformerEncoderLayer


class FaieseqTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['model']['encoder_embed_dim']
        if args['model']['multihead_attention_version'] == 'pytorch':
            from ncc.modules.attention.pytorch_multihead_attention import PytorchMultiheadAttention
            self.self_attn = PytorchMultiheadAttention(self.embed_dim, args['model']['encoder_attention_heads'],
                                                       dropout=args['model']['attention_dropout'])
        elif args['model']['multihead_attention_version'] == 'ncc':
            from ncc.modules.attention.ncc_multihead_attention import NccMultiheadAttention
            self.self_attn = NccMultiheadAttention(
                self.embed_dim,
                args['model']['encoder_attention_heads'],
                dropout=args['model']['attention_dropout'],
            )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args['model']['dropout']
        self.activation_fn = get_activation(
            activation_string=args['model'].get('activation_fn', 'relu')
        )
        self.activation_dropout = args['model']['activation_dropout']
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = args['model']['relu_dropout']
        self.normalize_before = args['model']['encoder_normalize_before']
        self.fc1 = Linear(self.embed_dim, args['model']['encoder_ffn_embed_dim'])
        self.fc2 = Linear(args['model']['encoder_ffn_embed_dim'], self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
