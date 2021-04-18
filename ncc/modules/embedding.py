import torch
import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    if padding_idx is None:
        m = nn.Embedding(num_embeddings, embedding_dim)
    else:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.uniform_(m.weight, -0.5, 0.5)
    if padding_idx is None:
        pass
    else:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m
