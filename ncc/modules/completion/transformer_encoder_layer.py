import torch
import torch.nn as nn
from ncc.modules.common.layer_norm import LayerNorm
from ncc.modules.attention.path_multihead_attention import PathMultiheadAttention
import torch.nn.functional as F
from ncc.utils import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        context_size: int = 1000,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        rel_vocab_size: int = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.context_size = context_size
        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = PathMultiheadAttention(
            self.embedding_dim,
            self.context_size,
            num_attention_heads,
            rel_vocab_size,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
        self,
        x: torch.Tensor,
        rel: torch.Tensor = None,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
    ):
        """
                LayerNorm is applied either before or after the self-attention/ffn
                modules similar to the original Transformer imlementation.
                """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
