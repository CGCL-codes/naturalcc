# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ncc.modules.base.activations import get_activation
from ncc.modules.base.initializers import (
    trunc_normal,
    xavier_uniform,
)
from ncc.modules.base.layer_norm import LayerNorm
from ncc.modules.base.layers import (
    Embedding,
    Linear,
    Parameter,
)
from ncc.modules.encoders.ncc_encoder import NccEncoder
from ncc.utils.pooling1d import pooling1d


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = Linear(
            self.kdim, embed_dim, bias=bias,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.v_proj = Linear(
            self.vdim, embed_dim, bias=bias,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.q_proj = Linear(
            embed_dim, embed_dim, bias=bias,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=bias,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.add_zero_attn = add_zero_attn

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = key_padding_mask.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = (
            q.contiguous()
                .view(bsz, tgt_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
        )
        k = (
            k.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
        )
        v = (
            v.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
        )

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(dim=1).to(torch.bool), float("-inf")
            )
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        attn = torch.matmul(attn_probs, v).transpose(1, 2)
        attn = attn.contiguous().view(bsz * tgt_len, -1)

        attn = self.out_proj(attn)
        return attn, None


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, attention_heads, dropout, ffn_embed_dim, activation_fn):
        super().__init__()
        self.dropout = dropout
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=attention_heads, dropout=dropout,
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.fc1 = Linear(
            embed_dim, ffn_embed_dim,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.fc2 = Linear(
            ffn_embed_dim, embed_dim,
            weight_initializer=trunc_normal(mean=.0, std=.02),
        )
        self.ff_layer_norm = LayerNorm(embed_dim)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -int('inf'))

        residual = x
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.ff_layer_norm(x)
        return x


class SelfAttnEncoder(NccEncoder):
    def __init__(self,
                 dictionary, embed_dim, token_types, max_positions,
                 self_attn_layers, attention_heads, ffn_embed_dim, activation_fn,
                 dropout, **kwargs,
                 ):
        super(SelfAttnEncoder, self).__init__(dictionary)
        # word embedding
        self.embed = Embedding(
            len(dictionary), embed_dim, padding_idx=self.dictionary.pad(),
            initializer=trunc_normal(mean=.0, std=.02),
        )
        # type embedding
        if token_types is not None:
            self.type_embed = Embedding(
                token_types, embed_dim,
                initializer=trunc_normal(mean=.0, std=.02),
            )
        else:
            self.type_embed = None
        # positional embedding
        if max_positions is not None:
            self.positional_embed = Parameter(
                1, max_positions, embed_dim,
                initializer=trunc_normal(mean=.0, std=.02),
            )
        else:
            self.positional_embed = None
        # layer norm for embedding
        self.embed_layer_norm = LayerNorm(embed_dim)
        self.dropout = dropout

        # self attn
        self.num_layers = self_attn_layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, attention_heads, dropout, ffn_embed_dim, activation_fn)
             for _ in range(self_attn_layers)]
        )

        # pooling
        pooling = kwargs.get('pooling', None)
        self.pooling = pooling1d(pooling)
        if 'weighted' in pooling:
            self.weight_layer = Linear(embed_dim, 1, bias=False, weight_initializer=xavier_uniform())
        else:
            self.weight_layer = None

    def forward_embedding(self, tokens, type_ids=None):
        # embed tokens and positions
        x = self.embed(tokens)
        if self.type_embed is not None:
            if type_ids is None:
                type_ids = tokens.new(*tokens.size()).long().fill_(0)
            x = x + self.type_embed(type_ids)
        if self.positional_embed is not None:
            x = x + self.positional_embed[:, :x.size(1), :]
        x = self.embed_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, tokens, tokens_mask=None, tokens_len=None, type_ids=None):
        if tokens_mask is None:
            tokens_mask = (tokens.ne(self.dictionary.pad())).to(tokens.device)
        if tokens_len is None:
            tokens_len = tokens_mask.sum(dim=-1)
        x = self.forward_embedding(tokens)

        # encoder layers
        bsz, src_len, embed_dim = x.size()
        key_padding_mask = tokens_mask.new(bsz, src_len, 1).fill_(1.) * tokens_mask.unsqueeze(dim=1)
        key_padding_mask = 1 - key_padding_mask.float()
        x = x.contiguous().view(-1, embed_dim)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = x.view(bsz, src_len, embed_dim)

        if self.pooling:
            x = self.pooling(
                input_emb=x, input_len=tokens_len, input_mask=tokens_mask, weight_layer=self.weight_layer
            )
        return x
