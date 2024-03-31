import math
import gc
import torch
from torch import Tensor
import torch.nn as nn
import copy
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.xgraph.models.utils import subgraph
from dig.xgraph.models.models import GraphSequential
from dig.xgraph.method.base_explainer import WalkBase
from typing import Tuple, List, Dict, Optional
from torch_geometric.nn import MessagePassing
from .vul_detector import GNNPool
EPS = 1e-15


class GNN_LRP(WalkBase):
    r"""
    An implementation of GNN-LRP in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.
    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)
    .. note::
            For node classification model, the :attr:`explain_graph` flag is False.
            GNN-LRP is very model dependent. Please be sure you know how to modify it for different models.
            For an example, see `benchmarks/xgraph
            <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.
    """

    def __init__(self, model: nn.Module, explain_graph=False):
        super().__init__(model=model, explain_graph=explain_graph)
        
    def extract_step(self, x: Tensor, edge_index: Tensor, detach: bool = True, split_fc: bool = False):

        layer_extractor = []
        hooks = []

        def register_hook(module: nn.Module):
            if not list(module.children()) or isinstance(module, MessagePassing):
                hooks.append(module.register_forward_hook(forward_hook))

        def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
            # input contains x and edge_index
            if detach:
                layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
            else:
                layer_extractor.append((module, input[0], output))

        # --- register hooks ---
        self.model.apply(register_hook)

        pred = self.model(x, edge_index)

        for hook in hooks:
            hook.remove()
        
        transform_steps = []
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[:3]:
            if isinstance(layer[0], nn.Linear):
                step['input'] = layer[1]
            step['module'].append(layer[0])
            step['output'] = layer[2]
            if isinstance(layer[0], nn.Dropout):
                transform_steps.append(step)

        walk_steps = []
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[3:]:
            if isinstance(layer[0], GNNPool):
                break
            if isinstance(layer[0], MessagePassing):
                step = {'input': layer[1], 'module': [], 'output': None}
            step['module'].append(layer[0])
            step['output'] = layer[2]
            if isinstance(layer[0], nn.Dropout):
                walk_steps.append(step)
                step = {'input': None, 'module': [], 'output': None}
        
        fc_steps = []
        pool_flag = False
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor[3:]:
            if isinstance(layer[0], GNNPool):
                pool_flag = True
                step = {'input': layer[1], 'module': [layer[0]], 'output': layer[2]}
                fc_steps.append(step)
                step = {'input': None, 'module': [], 'output': None}
            if pool_flag:
                if isinstance(layer[0], nn.Linear):
                    step = {'input': layer[1], 'module': [], 'output': None}
                step['module'].append(layer[0])
                step['output'] = layer[2]
                if isinstance(layer[0], nn.Dropout) or isinstance(layer[0], nn.Softmax):
                    fc_steps.append(step)

        return transform_steps, walk_steps, fc_steps

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        r"""
        Run the explainer for a specific graph instance.
        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.
        :rtype:
            (walks, edge_masks, related_predictions),
            walks is a dictionary including walks' edge indices and corresponding explained scores;
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.
        """
        super().forward(x, edge_index, **kwargs)
        labels = tuple(i for i in range(kwargs.get('num_classes')))
        self.model.eval()

        transform_steps, walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True)

        edge_index_with_loop, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)
        if not self.explain_graph:
            node_idx = kwargs.get('node_idx')
            node_idx = node_idx.reshape([1]).to(self.device)
            assert node_idx is not None
            self.subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())
            self.new_node_idx = torch.where(self.subset == node_idx)[0]

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]

        if kwargs.get('walks'):
            walks = kwargs.pop('walks')

        else:
            def compute_walk_score():

                # hyper-parameter gamma
                epsilon = 1e-30   # prevent from zero division
                gamma = [2, 1, 1]
                
                # --- record original weights of transform layer ---
                ori_transform_weights = []
                transform_gamma_modules = []
                for i, transform_step in enumerate(transform_steps):
                    modules = transform_step['module']
                    gamma_module = copy.deepcopy(modules[0])
                    if hasattr(modules[0], 'weight'):
                        ori_transform_weights.append(modules[0].weight.data)
                        gamma_ = 1
                        gamma_module.weight.data = ori_transform_weights[i] + gamma_ * ori_transform_weights[i].relu()
                    else:
                        ori_transform_weights.append(None)
                    transform_gamma_modules.append(gamma_module)

                # --- record original weights of GNN ---
                ori_gnn_weights = []
                gnn_gamma_modules = []
                # clear_probe = x
                for i, walk_step in enumerate(walk_steps):
                    modules = walk_step['module']
                    gamma_ = gamma[i] if i <= 1 else 1
                    gamma_module = copy.deepcopy(modules[0])
                    if hasattr(modules[0], 'lin'):
                        ori_gnn_weights.append(modules[0].lin.weight.data)
                        gamma_module.lin.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(modules[0], 'nn'):
                        ori_gnn_weights.append(modules[0].nn.weight.data)
                        gamma_module.nn.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(modules[0], 'lin_r'):
                        ori_gnn_weights.append(modules[0].lin_l.weight.data)
                        gamma_module.lin_l.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    elif hasattr(modules[0], 'lin_rel'):
                        ori_gnn_weights.append(modules[0].lin_rel.weight.data)
                        gamma_module.lin_rel.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    else:
                        ori_gnn_weights.append(modules[0].weight.data)
                        gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                    gnn_gamma_modules.append(gamma_module)

                # --- record original weights of fc layer ---
                ori_fc_weights = []
                fc_gamma_modules = []
                for i, fc_step in enumerate(fc_steps):
                    modules = fc_step['module']
                    gamma_module = copy.deepcopy(modules[0])
                    if hasattr(modules[0], 'weight'):
                        ori_fc_weights.append(modules[0].weight.data)
                        gamma_ = 1
                        gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                    else:
                        ori_fc_weights.append(None)
                    fc_gamma_modules.append(gamma_module)

                # --- GNN_LRP implementation ---
                for walk_indices in walk_indices_list:
                    walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                    for walk_idx in walk_indices:
                        walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                    h = x.requires_grad_(True)
                    
                    # --- transform LRP_gamma ---
                    for i, transform_step in enumerate(transform_steps):
                        modules = transform_step['module']
                        std_h = nn.Sequential(*modules)(h)

                        # --- gamma ---
                        s = transform_gamma_modules[i](h)
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht
                    
                    for i, walk_step in enumerate(walk_steps):
                        modules = walk_step['module']
                        std_h = GraphSequential(*modules)(h, edge_index)

                        # --- LRP-gamma ---
                        p = gnn_gamma_modules[i](h, edge_index)
                        q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                        # --- pick a path ---
                        mk = torch.zeros((h.shape[0], 1), device=self.device)
                        k = walk_node_indices[i + 1]
                        mk[k] = 1
                        ht = q * mk + q.detach() * (1 - mk)
                        h = ht

                    # --- FC LRP_gamma ---
                    # debug that torch.zeros(h.shape[0], dtype=torch.long, device=self.device)
                    # should be an edge_index with [num_edge, 2]
                    for i, fc_step in enumerate(fc_steps):
                        modules = fc_step['module']
                        std_h = nn.Sequential(*modules)(h) if i != 0 \
                            else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))

                        # --- gamma ---
                        s = fc_gamma_modules[i](h) if i != 0 \
                            else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht

                    if not self.explain_graph:
                        f = h[node_idx, label]
                    else:
                        f = h[0, label]
                    x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                    I = walk_node_indices[0]
                    r = x_grads[I, :] @ x[I].T
                    walk_scores.append(r.detach().clone())
                    del r, x_grads, f, h
                del ori_transform_weights, transform_gamma_modules, \
                    ori_gnn_weights, gnn_gamma_modules, \
                    ori_fc_weights, fc_gamma_modules
                gc.collect()
                torch.cuda.empty_cache()
            
            walk_scores_tensor_list = [None for i in labels]
            for label in labels:

                walk_scores = []

                compute_walk_score()
                walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}

        del transform_steps, walk_steps, fc_steps
        gc.collect()
        torch.cuda.empty_cache()

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                edge_masks = []
                hard_edge_masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    edge_mask = edge_attr.detach()
                    valid_mask = (edge_mask != -math.inf)
                    edge_mask[edge_mask == - math.inf] = edge_mask[valid_mask].min() - 1  # replace the negative inf

                    edge_masks.append(edge_mask)
                    hard_edge_masks.append(self.control_sparsity(edge_attr, kwargs.get('sparsity')).sigmoid())

                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks, **kwargs)

        return walks, edge_masks, related_preds, edge_index_with_loop
    
    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module.explain = True
                module._edge_mask = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.explain = False
