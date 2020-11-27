# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

from typing import *

import torch
import dgl
import collections

import itertools
import numpy as np
from copy import deepcopy
import networkx as nx
import random

from ncc.data.dict import Dict as _Dict
from ncc.utils.constants import *


def pad_seq(seq_list: List[List], include_padding_mask=False):
    lengths = np.array([len(data_list) for data_list in seq_list])
    MAX_LENGTH = lengths.max()

    batch = np.zeros((len(seq_list), MAX_LENGTH), dtype=np.int)
    padding_mask = np.zeros((len(seq_list), MAX_LENGTH))

    for i, dat in enumerate(seq_list):
        batch[i, :len(dat)] = dat
        padding_mask[i, :len(dat)] = 1

    if include_padding_mask:
        return batch, lengths, padding_mask
    else:
        return batch, lengths


def pad_seq_given_len(seq_list: List[List], length: int, include_padding_mask=False):
    lengths = np.array([len(data_list) for data_list in seq_list])
    # MAX_LENGTH = lengths.max()

    batch = np.zeros((len(seq_list), length), dtype=np.int)
    padding_mask = np.zeros((len(seq_list), length))

    for i, dat in enumerate(seq_list):
        dat = dat[:length]
        batch[i, :len(dat)] = dat
        padding_mask[i, :len(dat)] = 1

    if include_padding_mask:
        return batch, lengths, padding_mask
    else:
        return batch, lengths


def pad_comment_sum(comment_list: List[List]) -> Tuple:
    lengths = np.array([len(data_list) for data_list in comment_list])
    MAX_LENGTH = lengths.max()

    comment = np.zeros(shape=(len(comment_list), MAX_LENGTH))
    comment_input = np.zeros(shape=(len(comment_list), MAX_LENGTH + 1))
    comment_target = np.zeros(shape=(len(comment_list), MAX_LENGTH + 1))
    comment_input[:, 0] = BOS

    for i, dat in enumerate(comment_list):
        comment[i, :len(dat)] = dat
        comment_input[i, 1:len(dat) + 1] = dat
        comment_target[i, :len(dat) + 1] = dat + [EOS]

    lengths = np.array(lengths)
    return comment, comment_input, comment_target, lengths,


def pad_comment_ast_attendgru(comment_list: List[List], length: int):
    lengths = np.array([len(data_list) for data_list in comment_list])

    comment = np.zeros(shape=(len(comment_list), length))
    comment_input = np.zeros(shape=(len(comment_list), length))
    comment_input2 = np.zeros(shape=(len(comment_list), length, length))
    comment_target = np.zeros(shape=(len(comment_list), length))
    comment_input[:, 0] = BOS

    for i, dat in enumerate(comment_list):
        dat = dat[:length - 1]
        comment[i, :len(dat)] = dat
        comment_input[i, 1:len(dat) + 1] = dat
        comment_target[i, :len(dat) + 1] = dat + [EOS]

    for i in range(length):
        comment_input2[:, i, 0:i] = comment_input[:, 0:i]

    lengths = np.array(lengths)
    return comment, comment_input2, comment_target, lengths,


def pad_comment_ret(comment_list: List[List]) -> Tuple:
    lengths = np.array([len(data_list) for data_list in comment_list])
    MAX_LENGTH = lengths.max()

    comment = np.zeros(shape=(len(comment_list), MAX_LENGTH))
    for i, dat in enumerate(comment_list):
        comment[i, :len(dat)] = dat

    lengths = np.array(lengths)
    return comment, lengths,


to_torch_long = lambda np_tensor: torch.from_numpy(np_tensor).long() if np_tensor is not None else None
to_torch_float = lambda np_tensor: torch.from_numpy(np_tensor).float() if np_tensor is not None else None


def build_graph(tree_dict: Dict, dict_code: _Dict, tree_leaf_subtoken=1) -> dgl.DGLGraph:
    #  叶子节点存的是拆开后的subtoken ，当然，如果token拆不开，那就还是一个token
    # 用来训练的.pt数据里叶子节点token保存格式是["a_hu",["a","hu"]],
    # （1）tree_leaf_subtoken为1时 本函数只将其subtoken转换成wordid ,#即保存为[和a对应的id，和hu对应的id]，比如[23,179]
    # 如果是拆不开的token，pt数据里格式是 ["imq",["imq",PAD_WORD]]
    # 那么这里将其转换为[和imq对应的id，和codesum.PAD_WORD]，比如[258,0]
    # pad到的长度由train val test整个数据集里token拆开后最大长度决定
    # （2）tree_leaf_subtoken为0时，本函数用的拆之前的token得到wordid，即比如用a_hu得到wordid
    for node in tree_dict.values():
        if type(node['children'][-1]) == list:
            MAX_LEN = len(node['children'][-1])
            break

    nx_graph = nx.DiGraph()

    def _build(nid, idx, tree):
        if type(tree[idx]['children'][-1]) != list:  # 当前非叶子节点
            children = [child for child in tree[idx]['children'] if child.startswith(NODE_FIX)]
            if nid is None:
                nx_graph.add_node(0, x=[DGLGraph_PAD_WORD for _ in range(MAX_LEN)],
                                  y=int(idx[len(NODE_FIX):]), mask=0)
                nid = 0
            for c in children:
                cid = nx_graph.number_of_nodes()
                y_value = int(c[len(NODE_FIX):])
                c_children = tree[c]["children"]
                if type(c_children[-1]) != list:  # 非叶子节点
                    nx_graph.add_node(cid, x=[DGLGraph_PAD_WORD for _ in range(MAX_LEN)], y=y_value,
                                      mask=0)
                    _build(cid, c, tree)
                else:  # 叶子节点
                    if tree_leaf_subtoken:
                        word_index = [dict_code.lookup_ind(subtoken, UNK) for subtoken in
                                      tree[c]['children'][-1]]
                    else:
                        word_index = [dict_code.lookup_ind(tree[c]['children'][0], UNK)]
                    nx_graph.add_node(cid, x=word_index, y=y_value, mask=1)
                nx_graph.add_edge(cid, nid)  # 因为 用的 DiGraph，所以这里添加的edge应该是cid指向nid，而nid是root节点的方向，cid是叶子节点的方向
        else:  # 叶子节点
            if tree_leaf_subtoken:
                word_index = [dict_code.lookup_ind(subtoken, UNK) for subtoken in
                              tree[idx]['children'][-1]]
            else:
                word_index = [dict_code.lookup_ind(tree[idx]['children'][0], UNK)]
            if nid is None:
                cid = 0
            else:
                cid = nx_graph.number_of_nodes()
            y_value = int(idx[len(NODE_FIX):])
            nx_graph.add_node(cid, x=word_index, y=y_value, mask=1)

            if nid is not None:
                nx_graph.add_edge(cid, nid)  # 因为 用的 DiGraph，所以这里添加的edge应该是cid指向nid，而nid是root节点的方向，cid是叶子节点的方向

    ROOT_INDEX = NODE_FIX + '1'
    _build(None, ROOT_INDEX, tree_dict)
    dgl_graph = dgl.DGLGraph()

    dgl_graph.from_networkx(nx_graph, node_attrs=['x', 'y', 'mask'])
    assert len(tree_dict) == dgl_graph.number_of_nodes(), Exception('build dgl tree error')
    return dgl_graph


def build_graph_with_mp(tree_dicts: List[Dict], index: int, code_dict: _Dict, tree_leaf_subtoken: bool) -> Tuple:
    tree_dicts = [build_graph(tree, code_dict, tree_leaf_subtoken) for tree in tree_dicts]
    return tree_dicts, index,


def get_root_node_info(dgl_trees: Union[Tuple, List]) -> Tuple:
    root_indices, node_nums = [None] * len(dgl_trees), [None] * len(dgl_trees)
    for ind, tree in enumerate(dgl_trees):
        topological_nodes = dgl.topological_nodes_generator(tree)
        root_ind_tree_dgldigraph = topological_nodes[-1].item()
        root_indices[ind] = root_ind_tree_dgldigraph
        all_num_node_tree_dgldigraph = tree.number_of_nodes()
        node_nums[ind] = all_num_node_tree_dgldigraph
    root_indices = np.array(root_indices)
    num_nodes = np.array(node_nums)
    return root_indices, num_nodes,


def pack_graph(graphs: Union[Tuple, List]) -> Tuple:
    # merge many dgl graphs into a huge one
    root_indices, node_nums, = get_root_node_info(graphs)
    packed_graph = dgl.batch(graphs)
    return packed_graph, root_indices, node_nums,


def tree_padding_mask(node_nums: List[int]):
    MAX_NODE_NUM = max(node_nums)
    tree_mask = np.zeros(shape=(len(node_nums), MAX_NODE_NUM))
    for ind in range(len(node_nums)):
        tree_mask[ind, :node_nums[ind]] = 1.0
    return tree_mask


def batch_to_cuda(batch: Dict) -> Dict:
    '''torch tensor to cuda'''
    to_cuda_fun = lambda variable: variable.cuda() if isinstance(variable, torch.Tensor) else variable

    for name, value in batch.items():
        if len(value) > 0:
            if name == 'ast':
                tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask = batch[name]
                AST_DGL_Batch = collections.namedtuple('AST_DGL_Batch', ['graph', 'mask', 'wordid', 'label'])
                tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
                                               mask=tree_dgl_batch.ndata['mask'].cuda(),
                                               wordid=tree_dgl_batch.ndata['x'].cuda(),
                                               label=tree_dgl_batch.ndata['y'].cuda())
                tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask = map(to_cuda_fun,
                                                                                [tree_dgl_root_index, tree_dgl_node_num,
                                                                                 tree_padding_mask])
                batch[name] = [tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask]
            else:
                if isinstance(value, torch.Tensor):
                    batch[name] = to_cuda_fun(value)
                else:
                    batch[name] = [to_cuda_fun(val) for val in value]
        else:
            pass

    # if ('method' in batch) and (len(batch['method']) > 0):
    #     batch['method'] = list(map(to_cuda_fun, batch['method']))
    # if ('tok' in batch) and (len(batch['tok']) > 0):
    #     batch['tok'] = list(map(to_cuda_fun, batch['tok']))
    # if ('ast' in batch) and (len(batch['ast']) > 0):
    #     tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask = batch['ast']
    #     AST_DGL_Batch = collections.namedtuple('AST_DGL_Batch', ['graph', 'mask', 'wordid', 'label'])
    #     tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
    #                                    mask=tree_dgl_batch.ndata['mask'].cuda(),
    #                                    wordid=tree_dgl_batch.ndata['x'].cuda(),
    #                                    label=tree_dgl_batch.ndata['y'].cuda())
    #     tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask = map(to_cuda_fun,
    #                                                                     [tree_dgl_root_index, tree_dgl_node_num,
    #                                                                      tree_padding_mask])
    #     batch['ast'] = [tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, tree_padding_mask]
    # if ('path' in batch) and (len(batch['path']) > 0):
    #     batch['path'] = list(map(to_cuda_fun, batch['path']))
    # if ('comment' in batch) and len(batch['comment']) > 0:
    #     batch['comment'][:-1] = list(map(to_cuda_fun, batch['comment'][:-1]))
    # if ('comment_fake' in batch) and (batch['comment_fake'] is not None) and len(batch['comment_fake']) > 0:
    #     batch['comment_fake'][:-1] = list(map(to_cuda_fun, batch['comment_fake'][:-1]))
    # if ('pointer' in batch) and (len(batch['pointer']) > 0):
    #     batch['pointer'][:-1] = list(map(to_cuda_fun, batch['pointer'][:-1]))
    # if 'teacher_output' in batch:
    #     batch['teacher_output'] = tuple(list(map(to_cuda_fun, batch['teacher_output'])))
    #     batch['alpha'] = batch['alpha'].cuda()
    # if ('index' in batch) and (len(batch['index']) > 0):
    #     batch['index'] = list(map(to_cuda_fun, batch['index']))
    return batch


def pad_tree_to_max(tree_list: List, MAX_NODE_LENGTH: int, ) -> List:
    if len(tree_list) == 0:
        return tree_list

    # if current language's tree sub-token length == MAX_NODE_LENGTH, return tree_list
    for node_name, node_value in tree_list[0].items():
        if len(node_value['children']) > 1 and type(node_value['children'][1]) == list:
            assert len(node_value['children'][1]) <= MAX_NODE_LENGTH
            if len(node_value['children'][1]) == MAX_NODE_LENGTH:
                return tree_list
            else:
                # begin pad sub-token
                break

    for tree in tree_list:
        for node_name, node_value in tree.items():
            if len(node_value['children']) > 1 and type(node_value['children'][1]) == list:
                node_value['children'][1] = node_value['children'][1] + \
                                            ['<blank>'] * (MAX_NODE_LENGTH - len(node_value['children'][1]))
    return tree_list


def merge_data(batch_data: List[List]) -> Dict:
    '''
    merge data from different languages, generally used for Transfer Reinforcement Learning
    :param batch_data: [data, ...] data from different languages
    :return: merged batch data
    '''
    code_modalities, pointer_gen = batch_data[0][0]['others'][-2], batch_data[0][0]['others'][-1]
    max_size = max([len(batch) for batch in batch_data])
    batch_size = 0
    for ind, batch in enumerate(batch_data):
        if len(batch) < max_size:
            # pad to max_size
            new_batch = []
            for _ in range(max_size // len(batch)):
                new_batch.extend(deepcopy(batch))
            if len(new_batch) < max_size:
                new_batch.extend(deepcopy(batch[:max_size - len(new_batch)]))
            batch_data[ind] = new_batch
        batch_size += max_size
        if 'tok' in batch:
            batch.sort(key=lambda x: len(x['tok'][0]), reverse=True)
    batch_data = list(itertools.chain(*batch_data))

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    # comment
    # comment, comment_extend_vocab, raw_comment, case_study_data, parsed_code, index, _ = \
    #     zip(*[batch['others'] for batch in batch_data])
    comment, comment_extend_vocab, raw_comment, case_study_data, index, _, _ = \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)
    if comment_extend_vocab[0] is not None:  # tuple of None
        _, _, comment_extend_vocab, _ = pad_comment_sum(comment_extend_vocab)
    else:
        comment_extend_vocab = None
    # comment to tensor
    comment, comment_input, comment_target, comment_len, comment_extend_vocab, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, comment_extend_vocab,))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]

        if pointer_gen:
            # for pointer
            pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
            code_dict_comment, _ = pad_seq(code_dict_comment)
            code_dict_comment = to_torch_long(code_dict_comment)
            store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]
        else:
            store_batch['pointer'] = [None, None, None, None]

    if 'ast' in code_modalities:
        # release tree/ast modal
        ast = [batch['ast'] for batch in batch_data]
        packed_graph, root_indices, node_nums, = pack_graph(ast)
        tree_mask = tree_padding_mask(node_nums)
        root_indices, node_nums, tree_mask = to_torch_long(root_indices), to_torch_long(node_nums), \
                                             to_torch_float(tree_mask)
        store_batch['ast'] = [packed_graph, root_indices, node_nums, tree_mask]

    if 'path' in code_modalities:
        # release ast-path modal
        head, center, tail = map(
            lambda batch_list: list(itertools.chain(*batch_list)),
            zip(*[batch['path'] for batch in batch_data]),
        )
        head, head_len, head_mask = pad_seq(head, include_padding_mask=True)
        head, head_len, head_mask = to_torch_long(head), to_torch_long(head_len), \
                                    to_torch_long(head_mask)

        center, center_len, center_mask = pad_seq(center, include_padding_mask=True)
        center, center_len, center_mask = to_torch_long(center), to_torch_long(center_len), \
                                          to_torch_long(center_mask)

        tail, tail_len, tail_mask = pad_seq(tail, include_padding_mask=True)
        tail, tail_len, tail_mask = to_torch_long(tail), to_torch_long(tail_len), \
                                    to_torch_long(tail_mask)

        store_batch['path'] = [
            head, head_len, head_mask,
            center, center_len, center_mask,
            tail, tail_len, tail_mask,
        ]

    # other info
    store_batch['index'] = torch.Tensor(index).long()
    # store_batch['parsed_code'] = parsed_code
    store_batch['case_study'] = case_study_data
    return store_batch


def output2topk(output, k):
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)
    topk_outp = topk_outp.reshape(-1, k)  # (B x T) x k
    topk_idx = topk_idx.reshape(-1, k)  # (B x T) x k
    return topk_idx, topk_outp


if __name__ == '__main__':
    print(pad_seq([[1, 2, 3], [1, ]], True))
