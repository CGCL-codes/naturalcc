import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from ncc.modules.code2vec.ncc_encoder import NccEncoder
from ncc.modules.embedding import Embedding
from ncc.utils import utils

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


class NaryTreeLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(NaryTreeLSTMCell, self).__init__()
        self.W_iou = Linear(input_size, 3 * hidden_size, bias=False)
        self.U_iou = Linear(2 * hidden_size, 3 * hidden_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * hidden_size))
        self.U_f = Linear(2 * hidden_size, 2 * hidden_size)

    def message_func(self, edges):
        # aggregate neighbors’ representations
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # tranform aggregated representations from neighbors
        # LOGGER.debug(nodes.mailbox['h'].size())
        h_cat = nodes.mailbox['h'].reshape(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).reshape(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return_iou = self.U_iou(h_cat)
        return {'iou': return_iou, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class NaryTreeLSTMEncoder(NccEncoder):
    """LSTM encoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = NaryTreeLSTMCell(
            input_size=embed_dim,
            hidden_size=hidden_size,
        )

        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (
            weight.new(batch_size, self.hidden_size).zero_().requires_grad_(),
            weight.new(batch_size, self.hidden_size).zero_().requires_grad_()
        )

    def forward(self, graph, root_ids, node_nums, enc_hidden=None):
        """
        Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        if enc_hidden is None:
            enc_hidden = self.init_hidden(graph.number_of_nodes())

        graph.register_message_func(self.lstm.message_func)
        graph.register_reduce_func(self.lstm.reduce_func)
        graph.register_apply_node_func(self.lstm.apply_node_func)

        emb_subtoken = self.embed_tokens(
            graph.ndata['x'] * graph.ndata['mask'].reshape(*graph.ndata['mask'].shape, 1)
        )
        wemb = torch.sum(emb_subtoken, dim=1)  # feed embedding
        graph.ndata['iou'] = self.lstm.W_iou(wemb) * graph.ndata['mask'].unsqueeze(-1).type_as(emb_subtoken)
        graph.ndata['h'], graph.ndata['c'] = enc_hidden

        dgl.prop_nodes_topo(graph)

        all_node_h_in_batch = graph.ndata.pop('h')
        all_node_c_in_batch = graph.ndata.pop('c')

        batch_size = root_ids.size()[0]
        root_node_h_in_batch, root_node_c_in_batch = [], []
        add_up_num_node = 0
        for _i in range(len(root_ids)):
            if _i - 1 < 0:
                add_up_num_node = 0
            else:
                add_up_num_node += node_nums[_i - 1]
            idx_to_query = root_ids[_i] + add_up_num_node
            root_node_h_in_batch.append(all_node_h_in_batch[idx_to_query])
            root_node_c_in_batch.append(all_node_c_in_batch[idx_to_query])

        root_node_h_in_batch = torch.cat(root_node_h_in_batch).reshape(batch_size, -1)
        root_node_c_in_batch = torch.cat(root_node_c_in_batch).reshape(batch_size, -1)

        tree_output = emb_subtoken.new_zeros(batch_size, max(node_nums), root_node_h_in_batch.shape[-1])
        add_up_node_num = 0
        for _i in range(batch_size):
            node_num = node_nums[_i]
            this_sample_h = all_node_h_in_batch[add_up_node_num:add_up_node_num + node_nums[_i]]. \
                reshape(node_num, -1)
            add_up_node_num += node_nums[_i]
            tree_output[_i, :node_num, :] = this_sample_h

        tree_output = tree_output.transpose(dim0=0, dim1=1)
        root_node_h_in_batch = root_node_h_in_batch.unsqueeze(dim=0)
        root_node_c_in_batch = root_node_c_in_batch.unsqueeze(dim=0)
        return {
            'encoder_out': (tree_output, root_node_h_in_batch, root_node_c_in_batch),
            'encoder_padding_mask': None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions
