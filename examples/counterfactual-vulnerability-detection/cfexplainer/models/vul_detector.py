import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout

from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import *
import torch_scatter


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class GlobalAddPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)
    

class GlobalMaxPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return torch_scatter.segment_csr(x, self.cumsum(global_add_pool(torch.ones_like(batch).to(batch.device), batch)))

    def cumsum(self, value):
        out = value.new_empty((value.size(0) + 1, ) + value.size()[1:])
        out[0] = 0
        torch.cumsum(value, 0, out=out[1:])
        return out


class Detector(nn.Module):
    def __init__(self, args, **kwargs):
        super(Detector, self).__init__()
        self.args = args
        
        self.linear = nn.Sequential(
            Linear(args.gnn_feature_dim_size, args.gnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
        )

        self.gnn_layers = torch.nn.ModuleList()
        for i in range(args.num_gnn_layers):
            if args.gnn_model == "GCNConv":
                gnn_layer = GCNConv(args.gnn_hidden_size, args.gnn_hidden_size)
            elif args.gnn_model == "GatedGraphConv":
                gnn_layer = GatedGraphConv(args.gnn_hidden_size, args.num_ggnn_steps, args.ggnn_aggr)
            elif args.gnn_model == "GINConv":
                mlp = Linear(args.gnn_hidden_size, args.gnn_hidden_size)
                gnn_layer = GINConv(mlp, args.gin_eps, args.gin_train_eps)
            elif args.gnn_model == "GraphConv":
                gnn_layer = GraphConv(args.gnn_hidden_size, args.gnn_hidden_size, args.gconv_aggr)
            self.gnn_layers.append(gnn_layer)
        self.relu = ReLU()
        self.dropout = Dropout(args.dropout_rate)
        self.relu_layers_index = range(args.num_gnn_layers)
        self.dropout_layers_index = range(args.num_gnn_layers)
        
        if args.graph_pooling == "sum":
            self.pool = GlobalAddPool()
        elif args.graph_pooling == "mean":
            self.pool = GlobalMeanPool()
        elif args.graph_pooling == "max":
            self.pool = GlobalMaxPool()
        else:
            raise ValueError("Invalid graph pooling type.")

        self.classifier = nn.Sequential(
            nn.Linear(args.gnn_hidden_size, args.gnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.gnn_hidden_size, args.num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, edge_index, batch=None, **kwargs):
        output = self.get_emb(x, edge_index, **kwargs)
        
        pooled_output = self.pool(output, batch)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_emb(self, x, edge_index, **kwargs):
        output = self.linear(x)
        for gnn_layer_index in range(len(self.gnn_layers)):
            if self.args.residual:
                output = output + self.gnn_layers[gnn_layer_index](output, edge_index)
            else:
                output = self.gnn_layers[gnn_layer_index](output, edge_index)
            if gnn_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if gnn_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        return output
