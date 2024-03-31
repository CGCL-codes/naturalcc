from math import sqrt
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.utils.loop import add_remaining_self_loops
from dig.xgraph.models.utils import subgraph
from dig.xgraph.method.utils import symmetric_edge_mask_indirect_graph
from torch_geometric.nn import MessagePassing
from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.method import PGExplainer
from torch_geometric.data import Data
from typing import Tuple, List, Dict, Optional
from .shapley import gnn_score, GnnNetsNC2valueFunc, GnnNetsGC2valueFunc, sparsity
import tqdm
import numpy as np
import time


class XPGExplainer(PGExplainer):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.

    Args:
        model (:class:`torch.nn.Module`): The target model prepared to explain
        in_channels (:obj:`int`): Number of input channels for the explanation network
        explain_graph (:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        epochs (:obj:`int`): Number of epochs to train the explanation network
        lr (:obj:`float`): Learning rate to train the explanation network
        coff_size (:obj:`float`): Size regularization to constrain the explanation size
        coff_ent (:obj:`float`): Entropy regularization to constrain the connectivity of explanation
        t0 (:obj:`float`): The temperature at the first epoch
        t1(:obj:`float`): The temperature at the final epoch
        num_hops (:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
        (default: :obj:`None`)

    .. note: For node classification model, the :attr:`explain_graph` flag is False.
      If :attr:`num_hops` is set to :obj:`None`, it will be automatically calculated by calculating the
      :class:`torch_geometric.nn.MessagePassing` layers in the :attr:`model`.

    """
    def __init__(self, **kwargs):
        super(XPGExplainer, self).__init__(**kwargs)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r""" Set the edge weights before message passing

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~PGExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~PGExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.edge_mask = None

    def explain(self,
                x: Tensor,
                edge_index: Tensor,
                embed: Tensor,
                tmp: float = 1.0,
                training: bool = False,
                **kwargs)\
            -> Tuple[float, Tensor]:
        r""" explain the GNN behavior for graph with explanation network

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            embed (:obj:`torch.Tensor`): Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
            tmp (:obj`float`): The temperature parameter fed to the sample procedure
            training (:obj:`bool`): Whether in training procedure or not

        Returns:
            probs (:obj:`torch.Tensor`): The classification probability for graph with edge mask
            edge_mask (:obj:`torch.Tensor`): The probability mask for graph edges
        """
        node_idx = kwargs.get('node_idx')
        nodesize = embed.shape[0]
        if self.explain_graph:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            f12self = torch.cat([f1, f2], dim=-1)
        else:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            self_embed = embed[node_idx].repeat(f1.shape[0], 1)
            f12self = torch.cat([f1, f2, self_embed], dim=-1)

        # using the node embedding to calculate the edge weight
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        )
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        probs = self.model(x, edge_index)

        self.__clear_masks__()
        return probs, edge_mask
    
    def train_explanation_network(self, dataset):
        r""" training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(self.device)
                    logits = self.model(data.x, data.edge_index)
                    emb = self.model.get_emb(data.x, data.edge_index)
                    emb_dict[gid] = emb.data
                    ori_pred_dict[gid] = logits.argmax(-1).data

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    data.to(self.device)
                    prob, edge_mask = self.explain(data.x, data.edge_index, embed=emb_dict[gid], tmp=tmp, training=True)
                    loss_tmp = self.__loss__(prob.squeeze(), ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob.argmax(-1).item()
                    pred_list.append(pred_label)

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')
        else:
            with torch.no_grad():
                data = dataset[0]
                data.to(self.device)
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                pred_dict = {}
                logits = self.model(data.x, data.edge_index)
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    pred_dict[node_idx] = logits[node_idx].argmax(-1).item()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                tic = time.perf_counter()
                for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                    with torch.no_grad():
                        x, edge_index, y, subset, _ = \
                            self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                        emb = self.model.get_emb(x, edge_index)
                        new_node_index = int(torch.where(subset == node_idx)[0])
                    pred, edge_mask = self.explain(x, edge_index, emb, tmp, training=True, node_idx=new_node_index)
                    loss_tmp = self.__loss__(pred[new_node_index], pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
            print(f"training time is {duration:.5}s")

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[None, List, List[Dict]]:
        r""" explain the GNN behavior for graph and calculate the metric values.
        The interface for the :class:`dig.evaluation.XCollector`.

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - top_k (:obj:`int`): The number of edges in the final explanation results

        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        # set default subgraph with 10 edges
        top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        self.__clear_masks__()
        probs = self.model(x, edge_index)
        pred_labels = probs.argmax(dim=-1)
        embed = self.model.get_emb(x, edge_index)

        if self.explain_graph:
            # original value
            probs = probs.squeeze()
            label = pred_labels
            # masked value
            _, edge_mask = self.explain(x, edge_index, embed=embed, tmp=1.0, training=False)
            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            masked_node_list = [node for node in range(data.x.shape[0]) if node in selected_nodes]
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNetsGC2valueFunc(self.model, target_class=label)

            masked_pred = gnn_score(masked_node_list, data,
                                    value_func=value_func,
                                    subgraph_building_method='zero_filling')

            maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                     subgraph_building_method='zero_filling')

            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]

        else:
            node_idx = kwargs.get('node_idx')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            # original value
            probs = probs.squeeze()[node_idx]
            label = pred_labels[node_idx]
            # masked value
            x, edge_index, _, subset, _ = self.get_subgraph(node_idx, x, edge_index)
            new_node_idx = torch.where(subset == node_idx)[0]
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explain(x, edge_index, embed, tmp=1.0, training=False, node_idx=new_node_idx)

            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            masked_node_list = [node for node in range(data.x.shape[0]) if node in selected_nodes]
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNetsNC2valueFunc(self.model,
                                             node_idx=new_node_idx,
                                             target_class=label)

            masked_pred = gnn_score(masked_node_list, data,
                                    value_func=value_func,
                                    subgraph_building_method='zero_filling')

            maskout_pred = gnn_score(maskout_nodes_list, data,
                                     value_func=value_func,
                                     subgraph_building_method='zero_filling')

            sparsity_score = sparsity(masked_node_list, data,
                                      subgraph_building_method='zero_filling')

        # return variables
        pred_mask = [edge_mask]
        related_preds = [{
            'masked': masked_pred,
            'maskout': maskout_pred,
            'origin': probs[label],
            'sparsity': sparsity_score}]
        return None, pred_mask, related_preds


class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model):
        super().__init__(model=model, explain_graph=pgexplainer.explain_graph)
        self.explainer = pgexplainer
        
    def __set_masks__(self, x: Tensor, edge_index: Tensor, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        loop_mask = edge_index[0] != edge_index[1]
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = True
                module._edge_mask = self.edge_mask
                module._loop_mask = loop_mask
                module._apply_sigmoid = True

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.explain = False
                module._edge_mask = None
                module._loop_mask = None
                module._apply_sigmoid = True
        self.node_feat_masks = None
        self.edge_mask = None

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)
            # edge_masks
            edge_masks = [edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                               for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(x, edge_index)
            with torch.no_grad():
                if self.explain_graph:
                    related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)

            self.__clear_masks__()

        else:
            node_idx = kwargs.get('node_idx')
            sparsity = kwargs.get('sparsity')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            select_edge_index = torch.arange(0, edge_index.shape[1])
            subgraph_x, subgraph_edge_index, _, subset, kwargs = \
                self.explainer.get_subgraph(node_idx, x, edge_index, select_edge_index=select_edge_index)
            select_edge_index = kwargs['select_edge_index']
            self.select_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                         device=self.device,
                                                         dtype=torch.bool)
            self.select_edge_mask.fill_(False)
            self.select_edge_mask[select_edge_index] = True
            self.hard_edge_mask = edge_index.new_empty(subgraph_edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)
            self.subset = subset
            self.new_node_idx = torch.where(subset == node_idx)[0]

            subgraph_embed = self.model.get_emb(subgraph_x, subgraph_edge_index)
            _, subgraph_edge_mask = self.explainer.explain(subgraph_x,
                                                           subgraph_edge_index,
                                                           embed=subgraph_embed,
                                                           tmp=1.0,
                                                           training=False,
                                                           node_idx=self.new_node_idx)

            # edge_masks
            edge_masks = [subgraph_edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [
                self.control_sparsity(subgraph_edge_mask, sparsity=sparsity).sigmoid()
                for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(subgraph_x, subgraph_edge_index)
            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    subgraph_x, subgraph_edge_index, hard_edge_masks, node_idx=self.new_node_idx)

            self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds, edge_index
