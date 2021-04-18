import dgl
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncc.models import NccLanguageModel, register_model
from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.seq2seq.ncc_decoder import NccDecoder
from ncc.models.ncc_model import NccEncoderDecoderModel


class GatedGNN(nn.Module):
    def __init__(self, edge_types, edge_in, edge_out, hidden_in, hidden_out,
                 timesteps, backward, dropout, ):
        super(GatedGNN, self).__init__()
        self.edge_types = edge_types
        self.timesteps = timesteps
        self.backward = backward
        self.dropout = dropout
        self.edge_weights = nn.ModuleDict({
            et: nn.Linear(edge_in, edge_out, bias=False)
            for et in self.edge_types + [f'_{et}' for et in self.edge_types]
        })
        self.rnn_cell = nn.GRUCell(hidden_in, hidden_out)

    def forward(self, graph, prev_key, residual_state=None):
        # dgl._ffi.base.DGLError: [14:02:52] /opt/dgl/src/array/kernel.cc:94: Feature data can only be float32 or float64
        def edge_func(edges):
            edge_type = edges.canonical_etype[1]
            out = torch.dropout(
                self.edge_weights[edge_type](edges.dst[prev_key]),
                p=self.dropout, train=self.training,
            )
            return {'e': out}

        def message_func(edges):
            return {edges.canonical_etype[1]: edges.data['e']}

        reduce_funcs = {
            et: dgl.function.max(et, f'_{et}')  # "_" = backward
            for et in self.edge_types
        }

        prev_state = graph.ndata[prev_key]
        with graph.local_scope():
            for step in range(self.timesteps):
                for et in self.edge_types:
                    # compute edge type info with dst nodes and save them in edge
                    graph[et].apply_edges(edge_func, etype=et)
                    # aggregate info into nodes with different {edge type} keys
                    graph[et].update_all(message_func, reduce_func=reduce_funcs[et], apply_node_func=None, etype=et)
                graph_feature = torch.stack([graph.ndata[f'_{et}'] for et in self.edge_types], dim=1).max(dim=1)[0]
                if residual_state is not None:
                    graph_feature = torch.cat([residual_state, graph_feature], dim=-1)
                prev_state = self.rnn_cell(graph_feature, prev_state)
                if step < self.timesteps - 1:
                    prev_state = torch.dropout(prev_state, p=self.dropout, train=self.training)
        return prev_state


class GGNNEncoder(NccEncoder):
    """
        embedding nodes' subtokens feature, and
        aggregate node features with edges via GGNNs
    """

    def __init__(self, dictionary, embed_dim, hidden_size,
                 edge_types, edge_in, edge_out, edge_backward, timesteps, num_layers,
                 dropout, padding_idx=None):
        super().__init__(dictionary)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 1 embedding graphs
        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        self.node_embedding = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
        self.node_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, edge_in, bias=False),
            nn.Dropout(dropout),
        )
        # 2 layer GGNN
        self.ggnns = nn.ModuleList([
            GatedGNN(edge_types=edge_types, edge_in=edge_in, edge_out=edge_out,
                     hidden_in=(edge_out + hidden_size) if i > 0 else edge_out, hidden_out=hidden_size,
                     backward=edge_backward, timesteps=timesteps[i], dropout=dropout)
            for i in range(num_layers)
        ])

    def forward(self, graphs, node_tokens='subtoken', **kwargs):
        """
                nodes: all nodes' subtokens. [N, 5]
                edges: edges between nodes. (dict) {edge type: node_{i} --> node_{j}}, e.g. {0: [E, 2]}
                """
        # 1) nodes embedding
        nodes = graphs.ndata.pop(node_tokens).long()
        node_emb = self.node_embedding(nodes)  # [B, L, E]
        node_emb_len = (nodes > self.node_embedding.padding_idx).sum(dim=-1, keepdim=True)
        node_emb = node_emb.sum(dim=1) / node_emb_len  # mean(dim=1) => [B, E]
        node_emb = self.node_layer(node_emb)  # [B, E]
        graphs.ndata[0] = node_emb

        # 2) edges embedding
        for idx, ggnn in enumerate(self.ggnns, start=1):
            if idx > 1:
                residual_state = graphs.ndata[idx - 2]
            else:
                residual_state = None
            graphs.ndata[idx] = ggnn(graphs, prev_key=idx - 1, residual_state=residual_state)
        node_features = graphs.ndata[len(self.ggnns)]
        return node_features


class DenseDecoder(NccDecoder):
    def __init__(self, dictionary, hidden_size, dropout):
        super().__init__(dictionary)
        self.dropout = dropout
        num_embeddings = len(dictionary)
        # 3 classify layer
        self.cls_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_embeddings),
        )

    def forward(self, node_features, **kwargs):
        tgt_ids = kwargs.get('tgt_ids', None)
        if tgt_ids is not None:
            node_features = node_features[tgt_ids].contiguous()
        node_cls = self.cls_layers(node_features)
        return node_cls


@register_model('typilus')
class Typilus(NccEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, config, task):
        assert args['model']['edge_types'] == len(task.source_dictionary(key='edges'))
        encoder = GGNNEncoder(
            dictionary=task.source_dictionary(key='nodes'),
            embed_dim=args['model']['encoder_embed_dim'],
            hidden_size=args['model']['encoder_hidden_size'],
            edge_types=task.source_dictionary(key='edges').symbols,
            edge_in=args['model']['edge_in'],
            edge_out=args['model']['edge_out'],
            edge_backward=args['model']['edge_backward'],
            timesteps=args['model']['timesteps'],
            num_layers=args['model']['encoder_layers'],
            dropout=args['model']['encoder_dropout'],
            padding_idx=task.source_dictionary(key='nodes').pad(),
        )
        decoder = DenseDecoder(
            dictionary=task.target_dictionary(key='supernodes.annotation.type'),
            hidden_size=args['model']['edge_out'],
            dropout=args['model']['decoder_dropout'],
        )
        return cls(encoder, decoder)

    def forward(self, src_graphs, **kwargs):
        encoder_out = self.encoder(src_graphs, **kwargs)
        decoder_out = self.decoder(encoder_out, **kwargs)
        return decoder_out

    def forward_decoder(self, node_features, **kwargs):
        return self.decoder(node_features, **kwargs)

    def extract_features(self, src_graphs, **kwargs):
        encoder_out = self.encoder(src_graphs, **kwargs)
        return encoder_out
