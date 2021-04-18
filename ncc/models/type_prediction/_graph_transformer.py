import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.models import NccLanguageModel, register_model
# from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.seq2seq.ncc_decoder import NccDecoder

import numpy as np
import random


def attn_norm(weights, self_loop=False):
    """
    weights: aggregation weights from a node's neighbours
    add_eye:
    """
    weights = weights.t()
    weights = weights * (1 - torch.eye(weights.shape[0])).type_as(weights)
    if self_loop:
        weights = weights + torch.eye(weights.shape[0]).type_as(weights)
    degree = weights.sum(dim=1)
    degree_inversed = degree.pow(-1)
    degree_inversed[degree_inversed == float('inf')] = 0
    degree_inversed = degree_inversed * torch.eye(weights.shape[0]).type_as(weights)
    weights = (degree_inversed @ weights).t()
    return weights


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, edges):
        edges = (edges * F.softmax(self.weight, dim=1)).sum(dim=1)
        return edges

    def extra_repr(self) -> str:
        return 'ConV {}'.format(self.weight.size())


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first_layer=False):
        super(GraphTransformerLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = first_layer

        self.conv_query = ConvLayer(in_channels, out_channels) if self.first_layer else None
        self.conv_key = ConvLayer(in_channels, out_channels)

    def forward(self, key, query=None):
        if query is None:
            query = self.conv_query(key)
        key = self.conv_key(key)
        attn_weights = query @ key
        return attn_weights

    def extra_repr(self) -> str:
        return 'QueryConV = ({}), KeyConV = ({})'.format(self.conv_query, self.conv_key)


class GraphDecoder(NccDecoder):
    def __init__(self, args, target_dictionary=None):
        super(GraphDecoder, self).__init__(target_dictionary)
        self.args = args
        self.node_dim = args['model']['node_dim']
        self.feature_dim = args['model']['feature_dim']
        self.edge_types = args['model']['edge_types']
        self.num_channels = args['model']['num_channels']  #
        self.num_layers = args['model']['num_layers']
        self.num_class = args['model']['num_class']

        self.edge_transformer_layers = nn.ModuleList([
            GraphTransformerLayer(self.edge_types, self.num_channels, first_layer=i == 0)
            for i in range(self.num_layers)
        ])
        # map nodes' feature into certain dim
        self.node_proj = nn.Linear(self.node_dim, self.feature_dim, bias=False)
        # merge nodes' features from channels
        self.merge_layer = nn.Linear(self.num_channels * self.feature_dim, self.feature_dim)
        self.cls_layer = nn.Linear(self.feature_dim, self.num_class)

    def simple_gcn(self, nodes, edge_attn):
        new_nodes = self.node_proj(nodes)
        edge_attn = attn_norm(edge_attn, self_loop=True)
        new_nodes = edge_attn.t() @ new_nodes
        return new_nodes

    def normlization(self, attn_weights):
        attn_weights = torch.cat([
            attn_norm(attn_weights[i]).unsqueeze(0) for i in range(attn_weights.size(0))
        ], dim=0)
        return attn_weights

    def forward(self, nodes, edges):
        x, extra = self.extract_features(nodes, edges)
        x = self.output_layer(x)
        return x

    def extract_features(self, node_emb, edges):
        edges = edges.unsqueeze(0).permute(0, 3, 1, 2)  # [1, ch, n, n], ch=channels
        for layer_idx, layer in enumerate(self.edge_transformer_layers):
            if layer_idx == 0:
                edge_attn = layer(edges)
            else:
                edge_attn = self.normlization(edge_attn)
                edge_attn = layer(edges, query=edge_attn)

        # concate node features from different channels
        node_emb = torch.cat([
            F.relu(self.simple_gcn(node_emb, edge_attn[i]))
            for i in range(self.num_channels)
        ], dim=1)
        node_emb = F.relu(self.merge_layer(node_emb))
        return node_emb, None

    def output_layer(self, features, **kwargs):
        return self.cls_layer(features)

    def max_positions(self):
        raise NotImplementedError

    def max_decoder_positions(self):
        raise NotImplementedError


@register_model('graph_transformer')
class GraphTransformer(NccLanguageModel):
    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def forward_decoder(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def extract_features(self, *args, **kwargs):
        return self.decoder.extract_features(*args, **kwargs)
