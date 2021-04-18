import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.models import register_model
from ncc.models.ncc_model import NccLanguageModel
from ncc.modules.common.activations import get_activation
from ncc.modules.common.layer_norm import LayerNorm
from ncc.modules.common.layers import (
    Embedding, Linear, )
from ncc.modules.seq2seq.ncc_decoder import NccDecoder


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        max_positions=None,
        dropout=0.0,
    ):
        super(MultiheadAttention, self).__init__()
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

        self.bias = nn.Parameter(
            torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions)
        )

        self.k_proj = Linear(self.kdim, embed_dim)
        self.v_proj = Linear(self.vdim, embed_dim)
        self.q_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, value=None, key_padding_mask=None):
        bsz, src_len, embed_dim = query.size()
        if key is None:
            key = query
        if value is None:
            value = query

        q = self.q_proj(query)
        q *= self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        # (batch, head, seq_length, head_features)
        q = q.contiguous().view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (batch, head, head_features, seq_length)
        k = k.contiguous().view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        # (batch, head, seq_length, head_features)
        v = v.contiguous().view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k)  # (batch, head, seq_length, seq_length)

        # if key_padding_mask is not None:
        #     # key_padding_mask: [bsz, src_len]
        #     src_len = key_padding_mask.size(1)
        #     key_padding_mask &= torch.triu(torch.ones(src_len, src_len), diagonal=1)
        #     # don't attend to padding symbols
        #     attn_weights = attn_weights.masked_fill(key_padding_mask.bool(), float("-inf"))

        tgt_len, src_len = attn_weights.size(-2), attn_weights.size(-1)
        attn_weights = attn_weights.masked_fill(
            (1 - self.bias[:, :, :src_len, :src_len]).bool(), float("-inf")
        )
        # b = self.bias[:, :, src_len - tgt_len:src_len, :src_len]
        # try:
        #     attn_weights = attn_weights * b - 1e10 * (1 - b)
        # except:
        #     from ipdb import set_trace
        #     set_trace()
        #     attn_weights = attn_weights * b - 1e10 * (1 - b)

        attn_weights_float = torch.softmax(attn_weights, dim=-1)
        # attn_probs = F.dropout(
        #     attn_weights_float.type_as(attn_weights),
        #     p=self.dropout,
        #     training=self.training,
        # )
        attn = torch.matmul(attn_weights_float, v)  # (batch, head, seq_length, head_features)
        # assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(dim0=1, dim1=2).contiguous().view(bsz, src_len, embed_dim)
        attn = self.out_proj(attn)
        return attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['model']['decoder_embed_dim']
        self.activation_fn = get_activation(args['model']['activation_fn'])
        self.dropout = args['model']['dropout']
        # attention
        self.in_layer_norm = LayerNorm(self.embed_dim)
        self.attention = MultiheadAttention(
            self.embed_dim,
            args['model']['decoder_attention_heads'],
            kdim=args['model']['decoder_embed_dim'],
            vdim=args['model']['decoder_embed_dim'],
            max_positions=args['model']['max_target_positions'] - 1,
            dropout=args['model']['dropout'],
        )
        # ff layers
        self.ff_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, args['model']['decoder_ffn_embed_dim'])
        self.fc2 = nn.Linear(args['model']['decoder_ffn_embed_dim'], self.embed_dim)

    def forward(self, x):
        # attention
        residual = x
        x = self.in_layer_norm(x)
        x = self.attention(query=x, key=x, value=x)
        x = residual + x
        # ff
        residual = x
        x = self.ff_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class TransformerDecoder(NccDecoder):
    def __init__(self, args, dictionary):
        super(TransformerDecoder, self).__init__(dictionary)
        self.dropout = args['model']['dropout']
        embed_dim = args['model']['decoder_embed_dim']
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args)
            for _ in range(args['model']['decoder_layers'])
        ])
        self.num_layers = args['model']['decoder_layers']
        self.out_layer_norm = LayerNorm(embed_dim)

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        **kwargs
    ):
        x = self.embed_tokens(prev_output_tokens)  # bsz, max_len, dim
        # x = F.dropout(x, p=self.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            x = layer(x)

        x = self.out_layer_norm(x)
        x = F.linear(x, self.embed_tokens.weight, bias=None)
        return [x]


@register_model('completion_gpt2')
class GPT2(NccLanguageModel):
    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args

    @classmethod
    def build_model(cls, args, config, task):
        decoder = TransformerDecoder(args, dictionary=task.target_dictionary)
        return cls(args, decoder)
