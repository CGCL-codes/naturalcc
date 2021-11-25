# -*- coding: utf-8 -*-

import os

import dgl
import torch
import torch.nn as nn

from ncc.models import register_model
from ncc.models.ncc_model import (
    NccEncoder,
    NccEncoderModel,
)
from ncc.modules.base.layer_norm import (
    LayerNorm,
)
from ncc.modules.base.layers import (
    Embedding,
    Linear,
)


class MLPGNN(nn.Module):
    def __init__(self, edge_types, dim_in, dim_inner, dim_out):
        super(MLPGNN, self).__init__()

        def get_mlp():
            return nn.Sequential(
                Linear(dim_in, dim_inner, bias=False),
                nn.ReLU(),
                Linear(dim_inner, dim_out, bias=False),
            )

        self.edge_types = edge_types
        self.edges_modules = nn.ModuleDict({
            str(et): get_mlp()
            for et in edge_types
        })

    def forward(self, graph: dgl.DGLHeteroGraph, **kwargs):
        def message_func(edges):
            et = edges.canonical_etype[1]
            out = self.edges_modules[et](edges.src['data'])
            return {'m': out}

        def reduce_func(nodes):
            return {'data': torch.mean(nodes.mailbox['m'], dim=1)}

        with graph.local_scope():
            for et in graph.etypes:
                graph[et].update_all(
                    message_func=message_func,
                    reduce_func=reduce_func,
                    etype=et,
                )
        return graph


class GNNEncoder(nn.Module):
    def __init__(self, edge_types, dim_in, dim_inner, dim_out, dropout):
        super(GNNEncoder, self).__init__()
        self.mlp_gnn1 = MLPGNN(edge_types, dim_in=dim_in, dim_inner=dim_inner, dim_out=dim_out)
        self.ln1 = LayerNorm(dim_out)
        self.inner_layers = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mlp_gnn2 = MLPGNN(edge_types, dim_in=dim_in, dim_inner=dim_inner, dim_out=dim_out)
        self.ln2 = LayerNorm(dim_out)

    def forward(self, graph: dgl.DGLHeteroGraph):
        graph = self.mlp_gnn1(graph)
        graph.ndata['data'] = self.ln1(graph.ndata['data'])
        graph.ndata['data'] = self.inner_layers(graph.ndata['data'])
        graph = self.mlp_gnn2(graph)
        graph.ndata['data'] = self.ln2(graph.ndata['data'])
        return graph


class PoemEncoder(NccEncoder):
    def __init__(self, dictionary, embed_dim, embed_out, dropout,
                 edge_types,
                 # scoring/transform MLPs
                 out_dropout, dim_inner, dim_out,
                 ):
        super(PoemEncoder, self).__init__(dictionary)
        # embedding block
        if dictionary is not None:
            self.embed = Embedding(len(dictionary), embed_dim)
        else:
            self.embed = None
        self.embed_modules = nn.Sequential(
            Linear(embed_dim, embed_out, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # MLP-GNN
        self.gnn_modules = GNNEncoder(edge_types, dim_in=embed_out, dim_inner=dim_out, dim_out=embed_out, \
                                      dropout=dropout)

        # scoring MLP
        def get_mlp():
            return nn.Sequential(
                nn.Dropout(out_dropout),
                nn.Linear(embed_dim + embed_out, dim_inner, bias=False),
                nn.ReLU(),
                nn.Linear(dim_inner, dim_out, bias=False),
                nn.ReLU(),
            )

        self.score_mlp = get_mlp()
        self.transform_mlp = get_mlp()
        self.out_linear = nn.Sequential(
            nn.Linear(dim_out, 2),
            nn.Sigmoid(),
        )

    def forward(self, graph: dgl.DGLHeteroGraph):
        # embedding block
        if self.embed is not None:
            graph.ndata['data'] = self.embed(graph.ndata['data'])
        src_embed = graph.ndata['data']  # [B*N, 100]
        graph.ndata['data'] = self.embed_modules(src_embed)  # [B*N, 64]
        # MLP-GNN

        graph = self.gnn_modules(graph)
        graph.ndata['data'] = torch.cat((src_embed, graph.ndata['data']), dim=-1)  # [B*N, 164]
        with graph.local_scope():
            graph.ndata['scoring_out'] = self.score_mlp(graph.ndata['data'])
            weights = dgl.softmax_nodes(graph, 'scoring_out')
            node_embed = self.transform_mlp(graph.ndata['data'])
            graph.ndata['node_embed'] = weights * node_embed  # [B*N, 8]
            node_embed = dgl.sum_nodes(graph, 'node_embed')
        node_embed = self.out_linear(node_embed)
        return node_embed


@register_model('poem')
class POEM(NccEncoderModel):
    def __init__(self, args, encoder):
        super(POEM, self).__init__(encoder)
        self.args = args

    # @classmethod
    # def build_model(cls, args, config, task):
    #     encoder = PoemEncoder(
    #         dictionary=task.source_dictionary,
    #         embed_dim=args['model']['code_embed'],
    #     )
    #     return cls(args, encoder)


if __name__ == '__main__':
    import json
    from torch.nn.functional import binary_cross_entropy, one_hot


    def truncate_data(features, graph, MAX_LEN=50):
        features = features[:MAX_LEN]
        for i, type_edges in enumerate(graph):
            type_length = len(type_edges) - 1
            for reversed_idx, (src, tgt) in enumerate(type_edges[::-1]):
                if src >= MAX_LEN or tgt >= MAX_LEN:
                    type_edges.pop(type_length - reversed_idx)
        return features, graph


    self_loop = True
    edge_types = 11 + int(self_loop)

    x, y = [], []
    with open(os.path.join(os.path.dirname(__file__), 'toy_data.jsonl')) as reader:
        for line in reader:
            line = json.loads(line)
            node_features, adjacency_lists = \
                truncate_data(line['graph']['node_features'], line['graph']['adjacency_lists'])
            label = int(line['Property'])
            graph_matrices = {}
            # 0 for self loop
            graph_matrices[('node', '0', 'node')] = (
                torch.arange(0, len(node_features), dtype=torch.int32),
                torch.arange(0, len(node_features), dtype=torch.int32),
            )
            for idx, sub_graph in enumerate(adjacency_lists, start=1):
                if len(sub_graph) > 0:
                    srcs, dsts = zip(*sub_graph)
                    graph_matrices[('node', str(idx), 'node')] = (
                        torch.Tensor(srcs).int(), torch.Tensor(dsts).int()
                    )
            graph = dgl.heterograph(graph_matrices)
            graph.ndata['data'] = torch.Tensor(node_features).float()
            x.append(graph)
            y.append(label)
    batch = dgl.batch(x).to('cuda')
    ground_truth = torch.Tensor(y).long().cuda()
    ground_truth = one_hot(ground_truth, num_classes=2).float()
    model = PoemEncoder(dictionary=None, embed_dim=100, embed_out=64, dropout=0.1,
                        edge_types=[str(idx) for idx in range(edge_types)], out_dropout=0.01, dim_inner=64,
                        dim_out=8)
    model = model.cuda()
    out = model.forward(batch)
    print(out.size())
    loss = binary_cross_entropy(out, ground_truth)
    print(loss)
    loss.backward()
