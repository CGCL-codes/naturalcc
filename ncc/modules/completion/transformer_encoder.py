import torch
import torch.nn as nn
from ncc.modules.completion.transformer_encoder_layer import TransformerEncoderLayer
from ncc.modules.common.layer_norm import LayerNorm
from typing import Optional
import torch.nn.functional as F
import random


class PathLSTM(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super(PathLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.LSTM = nn.LSTM(n_embd, n_embd, batch_first=True)

    def forward(self, paths):
        embed = self.embedding(paths)  # bs, max_len, max_path_len, n_embd
        batch_size, bag_size, path_len, n_embd = embed.shape
        _, (h_n, _) = self.LSTM(embed.view(batch_size * bag_size, path_len, n_embd))
        return h_n.permute((1, 0, 2)).view((batch_size, bag_size, -1))


class TransformerEncoder(nn.Module):  # TODO: to check NccEncoder?
    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        context_size: int = 1000,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        layerdrop: float = 0.0,
        root_paths: list = None,
        rel_vocab_size: int = None,
        encoder_normalize_before: bool = False,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.num_encoder_layers = num_encoder_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.traceable = traceable

        self.embed_tokens = nn.Embedding(vocab_size, embedding_dim)
        if root_paths:
            self.path_lstm = PathLSTM(vocab_size, embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    context_size=context_size,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    rel_vocab_size=rel_vocab_size,
                    export=export,
                )
                for _ in range(self.num_encoder_layers)
            ]
        )
        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return None  # an arbitrary large number TODO

    def forward(
        self,
        tokens: torch.Tensor,
        rel=None,
        paths=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ):
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)

        if paths is not None:
            path_embeds = self.path_lstm(paths)
            x += path_embeds

        x = F.dropout(x, p=self.dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.layerdrop):
                x, _ = layer(x, self_attn_padding_mask=padding_mask)
                if not last_state_only:
                    inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep