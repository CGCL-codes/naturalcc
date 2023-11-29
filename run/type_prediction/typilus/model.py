import os
import ujson
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

SUBTOKEN_NUM = 3507
MAX_SUBTOKEN_LEN = 5
DROPOUT = 0.1
FEATURE_DIM = 64
CLASS_NUM = 65
BACKWARD = True
GNN_LAYERS = 2
TIMESTEPS = [7, 1]
EDGE_TYPES = [[f'f{i}', f'b{i}', ] for i in range(8)]
EDGE_TYPES = list(itertools.chain(*EDGE_TYPES))


def load_data(file=os.path.join(os.path.dirname(__file__), 'data_sample.json')):
    with open(file, 'r') as reader:
        data = ujson.load(reader)
    graph_dict = {}
    for idx, et in enumerate(['edges0', 'edges1', 'edges2', 'edges3', 'edges4', 'edges5', 'edges6', 'edges7']):
        src, dst = zip(*data[et])
        src, dst = torch.IntTensor(src), torch.IntTensor(dst)
        graph_dict[('node', f'f{idx}', 'node')] = (src, dst)
        graph_dict[('node', f'b{idx}', 'node')] = (dst, src)
    hetero_graph = dgl.heterograph(graph_dict)
    hetero_graph.ndata['subtokens'] = torch.IntTensor(data['nodes'])
    hetero_graph = hetero_graph
    assert hetero_graph.num_nodes('node') == len(data['nodes'])

    dropout = data['dropout']
    batch_size = data['batch_size']
    typed_annotation_node_ids = data['typed_annotation_node_ids']
    typed_annotation_target_class = data['typed_annotation_target_class']
    typed_annotation_pairs_are_equal = data['typed_annotation_pairs_are_equal']
    return hetero_graph, dropout, batch_size, typed_annotation_node_ids, typed_annotation_target_class, typed_annotation_pairs_are_equal


class GatedGNN(nn.Module):
    def __init__(self, edge_types, edge_in, edge_out, hidden_in, hidden_out,
                 timestep, backward, dropout, ):
        super(GatedGNN, self).__init__()
        self.edge_types = edge_types
        self.timestep = timestep
        self.backward = backward
        self.dropout = dropout
        self.edge_weights = nn.ModuleDict({
            et: nn.Linear(edge_in, edge_out, bias=False)
            for et in self.edge_types
        })
        self.gru_cell = nn.GRUCell(hidden_in, hidden_out)

    def forward(self, graph, prev_key, residual_state=None):
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
            et: dgl.function.max(et, f'_{et}')
            for et in self.edge_types
        }

        prev_state = graph.ndata[prev_key]
        with graph.local_scope():
            for step in range(self.timestep):
                for et in self.edge_types:
                    # compute edge type info with dst nodes and save them in edge
                    graph[et].apply_edges(edge_func, etype=et)
                    # aggregate info into nodes with different {edge type} keys
                    graph[et].update_all(message_func, reduce_func=reduce_funcs[et], apply_node_func=None, etype=et)
                graph_feature = torch.stack([graph.ndata[f'_{et}'] for et in self.edge_types], dim=1).max(dim=1)[0]
                if residual_state is not None:
                    graph_feature = torch.cat([residual_state, graph_feature], dim=-1)
                prev_state = self.gru_cell(graph_feature, prev_state)
                if step < self.timestep - 1:
                    prev_state = torch.dropout(prev_state, p=self.dropout, train=self.training)
        return prev_state


class Typilus(nn.Module):
    def __init__(self,
                 # node embedding
                 vocab_size, embed_size, padding_idx,
                 # ggnn
                 edge_types, edge_in, edge_out, hidden_size,
                 backward, layer_num, timesteps,
                 dropout,
                 # cls prediction
                 cls_num,
                 ):
        super().__init__()

        self.padding_idx = padding_idx
        self.dropout = dropout

        # embedding graphs
        self.node_embedding = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.node_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(embed_size, embed_size, bias=False),
            nn.Dropout(self.dropout),
        )
        # 2 layer GGNN
        self.ggnns = nn.ModuleList([
            GatedGNN(edge_types=edge_types, edge_in=edge_in, edge_out=edge_out,
                     hidden_in=(edge_out + hidden_size) if i > 0 else edge_out, hidden_out=hidden_size,
                     backward=backward, timestep=timesteps[i], dropout=dropout)
            for i in range(layer_num)
        ])
        # 3 classify layer
        self.cls_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, cls_num),
        )

    def forward(self, graph):
        """
        nodes: all nodes' subtokens. [N, 5]
        edges: edges between nodes. (dict) {edge type: node_{i} --> node_{j}}, e.g. {0: [E, 2]}
        """
        # 1) nodes embedding
        nodes = graph.ndata.pop('subtokens').long()
        node_emb = self.node_embedding(nodes)  # [B, L, E]
        node_emb_len = (nodes > self.node_embedding.padding_idx).sum(dim=-1, keepdim=True)
        node_emb = node_emb.sum(dim=1) / node_emb_len  # mean(dim=1) => [B, E]
        node_emb = self.node_layer(node_emb)  # [B, E]
        graph.ndata[0] = node_emb

        # 2) edges embedding
        for idx, ggnn in enumerate(self.ggnns, start=1):
            if idx > 1:
                residual_state = graph.ndata[idx - 2]
            else:
                residual_state = None
            graph.ndata[idx] = ggnn(graph, prev_key=idx - 1, residual_state=residual_state)
        last_state = graph.ndata[len(graph.ndata) - 1]
        return last_state

    def triplet_loss(self, repr, equal_ids, margin=2, eplison=1e-10):
        distance = torch.norm(repr.unsqueeze(dim=0) - repr.unsqueeze(dim=1), dim=-1, p=1)  # B x B
        max_pos_distance = (distance * equal_ids).max(dim=-1)[0]
        neg_filter = distance <= (max_pos_distance + margin).unsqueeze(dim=-1)
        pos_mask = equal_ids + torch.eye(*equal_ids.size()).type_as(distance)
        neg_filter = neg_filter * (1 - pos_mask)
        avg_neg_distance = (distance * neg_filter).sum(dim=-1) / (neg_filter.sum(dim=-1) + eplison)
        min_neg_distance = (distance + pos_mask * 99999).min(dim=-1)[0]
        pos_filter = (distance >= (min_neg_distance - margin).unsqueeze(dim=-1)).float()
        pos_filter = pos_filter * equal_ids
        avg_pos_distance = (distance * pos_filter).sum(dim=-1) / (pos_filter.sum(dim=-1) + eplison)
        triplet_loss = 0.5 * torch.relu(avg_pos_distance - min_neg_distance + margin) + \
                       0.5 * torch.relu(max_pos_distance - avg_neg_distance + margin)
        triplet_loss = triplet_loss.mean()
        return triplet_loss

    def ce_loss(self, repr, gt):
        repr = self.cls_layers(repr)
        repr = torch.log_softmax(repr, dim=-1)
        bsz = repr.size(0)
        ce_loss = F.cross_entropy(
            repr.view(bsz, -1),
            gt.view(-1),
            ignore_index=self.padding_idx,
        ) / bsz
        return ce_loss


if __name__ == '__main__':
    typilus_model = Typilus(vocab_size=SUBTOKEN_NUM, embed_size=FEATURE_DIM, padding_idx=0,
                            edge_types=EDGE_TYPES, backward=BACKWARD, layer_num=GNN_LAYERS, timesteps=TIMESTEPS,
                            edge_in=FEATURE_DIM, edge_out=FEATURE_DIM, hidden_size=FEATURE_DIM,
                            dropout=DROPOUT, cls_num=CLASS_NUM, )
    typilus_model = typilus_model

    hetero_graph, dropout, batch_size, typed_annotation_node_ids, typed_annotation_target_class, typed_annotation_pairs_are_equal = load_data()
    last_state = typilus_model(hetero_graph)
    last_state = last_state[typed_annotation_node_ids].contiguous()  # B x E

    typed_annotation_pairs_are_equal = torch.Tensor(typed_annotation_pairs_are_equal).float().type_as(last_state)
    typed_annotation_target_class = torch.Tensor(typed_annotation_target_class).long().to(last_state.device)

    triplet_loss = typilus_model.triplet_loss(last_state, equal_ids=typed_annotation_pairs_are_equal)
    print(triplet_loss)
    ce_loss = typilus_model.ce_loss(last_state, gt=typed_annotation_target_class)
    print(ce_loss)
    # loss = triplet_loss + ce_loss
    # loss.backward()
