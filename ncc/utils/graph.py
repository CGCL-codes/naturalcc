# -*- coding: utf-8 -*-

import dgl
import networkx as nx
import numpy as np
import torch

from dataset.codesearchnet import MAX_SUB_TOKEN_LEN


def build_graph(tree_dict, dictionary, tree_leaf_subtoken=1, DGLGraph_PAD_WORD=-1) -> dgl.DGLGraph:
    #  叶子节点存的是拆开后的subtoken ，当然，如果token拆不开，那就还是一个token
    # 用来训练的.pt数据里叶子节点token保存格式是["a_hu",["a","hu"]],
    # （1）tree_leaf_subtoken为1时 本函数只将其subtoken转换成wordid ,#即保存为[和a对应的id，和hu对应的id]，比如[23,179]
    # 如果是拆不开的token，pt数据里格式是 ["imq",["imq",PAD_WORD]]
    # 那么这里将其转换为[和imq对应的id，和codesum.PAD_WORD]，比如[258,0]
    # pad到的长度由train val test整个数据集里token拆开后最大长度决定
    # （2）tree_leaf_subtoken为0时，本函数用的拆之前的token得到wordid，即比如用a_hu得到wordid
    nx_graph = nx.DiGraph()

    def _build(nid, idx, tree):
        # non-leaf node, 'children': ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]]
        if not isinstance(tree[idx]['children'][1], list):
            child_ids = tree[idx]['children']
            if nid is None:
                nx_graph.add_node(0, x=[DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN, y=int(idx), mask=0)
                # print('node={}, x={}, y={}, mask={}'.format(0, [DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN, int(idx), 0))
                nid = 0
            for idx in child_ids:
                cid = nx_graph.number_of_nodes()
                y_value = int(idx)
                if not isinstance(tree[str(idx)]['children'][1], list):  # non-leaf node
                    nx_graph.add_node(cid, x=[DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN, y=y_value, mask=0)
                    # print(
                    #     'node={}, x={}, y={}, mask={}'.format(cid, [DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN, y_value, 0))
                    _build(cid, str(idx), tree)
                else:  # leaf node
                    if tree_leaf_subtoken:
                        word_index = [dictionary.index(subtoken) for subtoken in tree[str(idx)]['children'][1]]
                    else:
                        word_index = [dictionary.index(tree[idx]['children'][0])]
                    nx_graph.add_node(cid, x=word_index, y=y_value, mask=1)
                    # print('node={}, x={}, y={}, mask={}'.format(cid, word_index, y_value, 1))
                nx_graph.add_edge(cid, nid)  # 因为用的 DiGraph，所以这里添加的edge应该是cid指向nid，而nid是root节点的方向，cid是叶子节点的方向
                # print('edge={}->{}'.format(cid, nid))
        else:  # leaf node
            if tree_leaf_subtoken:
                word_index = [dictionary.index(subtoken) for subtoken in tree[idx]['children'][-1]]
            else:
                word_index = [dictionary.index(tree[idx]['children'][0])]
            if nid is None:
                cid = 0
            else:
                cid = nx_graph.number_of_nodes()
            nx_graph.add_node(cid, x=word_index, y=int(idx), mask=1)
            # print('node={}, x={}, y={}, mask={}'.format(cid, word_index, int(idx), 1))

            if nid is not None:
                nx_graph.add_edge(cid, nid)  # 因为用的 DiGraph，所以这里添加的edge应该是cid指向nid，而nid是root节点的方向，cid是叶子节点的方向
                # print('edge={}->{}'.format(cid, nid))

    _build(None, '0', tree_dict)
    dgl_graph = dgl.DGLGraph()

    dgl_graph.from_networkx(nx_graph, node_attrs=['x', 'y', 'mask'])
    assert len(tree_dict) == dgl_graph.number_of_nodes(), Exception('build dgl tree error')
    return dgl_graph


def tree2dgl(tree_dict, dictionary, DGLGraph_PAD_WORD=-1):
    """
    if _subtoken == True, it means that we tokenize leaf node info into sub-tokens
        e.g. ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]]
    else, no tokenization. e.g. ["sub_token"]
    """
    _subtoken = False
    for node in tree_dict.values():
        if isinstance(node['children'][1], list):
            _subtoken = True
            break

    def nonleaf_node_info():
        if _subtoken:
            return [DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN
        else:
            return [DGLGraph_PAD_WORD]

    def token2idx(node_info):
        """
        node info => indices
        if _subtoken == True, ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]] => index(["sub", "token", <PAD>, <PAD>, <PAD>])
        else, ["sub_token"] => index(["sub_token"])
        """
        if _subtoken:
            return [dictionary.index(subtoken) for subtoken in node_info[-1]]
        else:
            return [dictionary.index(node_info[0])]

    """
    how to build DGL graph?
    node: 
        x: node info (if it's non-leaf nodes, padded with [-1, ...]),
        y: current node idx
        mask: if leaf node, mask=1; else, mask=0
        * if current node is the root node,
    edge: child => parent 
    """
    dgl_graph = dgl.DGLGraph()
    ids = sorted(tree_dict.keys(), key=int)

    dgl_graph.add_nodes(
        len(tree_dict),
        data={
            'x': torch.LongTensor([
                token2idx(tree_dict[idx]['children']) if isinstance(tree_dict[idx]['children'][1], list) \
                    else nonleaf_node_info()
                for idx in ids
            ]),
            'y': torch.LongTensor(range(len(tree_dict))),
            'mask': torch.LongTensor([isinstance(tree_dict[idx]['children'][1], list) for idx in ids]),
        }
    )

    for idx in ids:
        node = tree_dict[idx]
        if node['parent'] is not None:
            dgl_graph.add_edges(int(idx), int(node['parent']))
            # print('edge={}->{}'.format(int(idx), int(node['parent'])))

    return dgl_graph


def tree2nx2dgl(tree_dict, dictionary, DGLGraph_PAD_WORD=-1):
    """
    if _subtoken == True, it means that we tokenize leaf node info into sub-tokens
        e.g. ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]]
    else, no tokenization. e.g. ["sub_token"]
    """
    _subtoken = False
    for node in tree_dict.values():
        if isinstance(node['children'][1], list):
            _subtoken = True
            break

    def nonleaf_node_info():
        if _subtoken:
            return [DGLGraph_PAD_WORD] * MAX_SUB_TOKEN_LEN
        else:
            return [DGLGraph_PAD_WORD]

    def token2idx(node_info):
        """
        node info => indices
        if _subtoken == True, ["sub_token", ["sub", "token", <PAD>, <PAD>, <PAD>]] => index(["sub", "token", <PAD>, <PAD>, <PAD>])
        else, ["sub_token"] => index(["sub_token"])
        """
        if _subtoken:
            return [dictionary.index(subtoken) for subtoken in node_info[-1]]
        else:
            return [dictionary.index(node_info[0])]

    """
    how to build DGL graph?
    node: 
        x: node info (if it's non-leaf nodes, padded with [-1, ...]),
        y: current node idx
        mask: if leaf node, mask=1; else, mask=0
        * if current node is the root node,
    edge: child => parent 
    """

    nx_graph = nx.DiGraph()
    ids = sorted(tree_dict.keys(), key=int)

    for idx in ids:
        node = tree_dict[idx]

        nx_graph.add_node(
            int(idx),
            x=token2idx(tree_dict[idx]['children']) if isinstance(tree_dict[idx]['children'][1], list) \
                else nonleaf_node_info(),
            y=int(idx),
            mask=int(isinstance(tree_dict[idx]['children'][1], list))
        )
        # print('node={}, x={}, y={}, mask={}'.format(
        #     idx, token2idx(tree_dict[idx]['children']) if isinstance(tree_dict[idx]['children'][1], list) \
        #         else nonleaf_node_info(), int(idx), int(isinstance(tree_dict[idx]['children'][1], list))))
        if node['parent'] is not None:
            nx_graph.add_edge(int(idx), int(node['parent']))
            # print('edge={}->{}'.format(int(idx), int(node['parent'])))

    dgl_graph = dgl.DGLGraph()

    dgl_graph.from_networkx(nx_graph, node_attrs=['x', 'y', 'mask'])
    assert len(tree_dict) == dgl_graph.number_of_nodes(), Exception('build dgl tree error')
    return dgl_graph


def pack_graph(graphs):
    def get_root_node_info(dgl_trees):
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

    # merge many dgl graphs into a huge one
    root_indices, node_nums, = get_root_node_info(graphs)
    packed_graph = dgl.batch(graphs)
    return packed_graph, root_indices, node_nums,


if __name__ == '__main__':
    from ncc.tasks.summarization import SummarizationTask

    dict = SummarizationTask.load_dictionary(
        filename='/home/yang/.ncc/multi/summarization/data-mmap/ruby/binary_ast.dict.json'
    )

    bin_ast = {
        "0": {"type": "method", "parent": None, "children": [1, 2]},
        "1": {"type": "def_keyword", "parent": 0, "children": ["def", ["def", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "2": {"type": "TMP", "parent": 0, "children": [3, 4]},
        "3": {"type": "identifier", "parent": 2, "children": ["set", ["set", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "4": {"type": "TMP", "parent": 2, "children": [5, 10]},
        "5": {"type": "method_parameters", "parent": 4, "children": [6, 7]},
        "6": {"type": "LeftParenOp", "parent": 5, "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "7": {"type": "TMP", "parent": 5, "children": [8, 9]}, "8": {"type": "identifier", "parent": 7,
                                                                     "children": ["set_attributes",
                                                                                  ["set", "attributes", "<pad>",
                                                                                   "<pad>", "<pad>"]]},
        "9": {"type": "LeftParenOp", "parent": 7, "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "10": {"type": "TMP", "parent": 4, "children": [11, 26]},
        "11": {"type": "assignment", "parent": 10, "children": [12, 13]},
        "12": {"type": "identifier", "parent": 11,
               "children": ["old_attributes", ["old", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "13": {"type": "TMP", "parent": 11, "children": [14, 15]},
        "14": {"type": "AsgnOp", "parent": 13, "children": ["=", ["=", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "15": {"type": "method_call", "parent": 13, "children": [16, 17]},
        "16": {"type": "identifier", "parent": 15,
               "children": ["compute_attributes", ["compute", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "17": {"type": "argument_list", "parent": 15, "children": [18, 19]},
        "18": {"type": "LeftParenOp", "parent": 17,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "19": {"type": "TMP", "parent": 17, "children": [20, 25]},
        "20": {"type": "call", "parent": 19, "children": [21, 22]},
        "21": {"type": "identifier", "parent": 20,
               "children": ["set_attributes", ["set", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "22": {"type": "TMP", "parent": 20, "children": [23, 24]},
        "23": {"type": "DotOp", "parent": 22, "children": [".", [".", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "24": {"type": "identifier", "parent": 22,
               "children": ["keys", ["keys", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "25": {"type": "LeftParenOp", "parent": 19,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "26": {"type": "TMP", "parent": 10, "children": [27, 34]},
        "27": {"type": "method_call", "parent": 26, "children": [28, 29]},
        "28": {"type": "identifier", "parent": 27,
               "children": ["assign_attributes", ["assign", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "29": {"type": "argument_list", "parent": 27, "children": [30, 31]},
        "30": {"type": "LeftParenOp", "parent": 29,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "31": {"type": "TMP", "parent": 29, "children": [32, 33]},
        "32": {"type": "identifier", "parent": 31,
               "children": ["set_attributes", ["set", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "33": {"type": "LeftParenOp", "parent": 31,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "34": {"type": "TMP", "parent": 26, "children": [35, 36]},
        "35": {"type": "yield_keyword", "parent": 34,
               "children": ["yield", ["yield", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "36": {"type": "TMP", "parent": 34, "children": [37, 46]},
        "37": {"type": "ensure", "parent": 36, "children": [38, 39]},
        "38": {"type": "ensure_keyword", "parent": 37,
               "children": ["ensure", ["ensure", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "39": {"type": "method_call", "parent": 37, "children": [40, 41]},
        "40": {"type": "identifier", "parent": 39,
               "children": ["assign_attributes", ["assign", "attributes", "<pad>", "<pad>", "<pad>"]]},
        "41": {"type": "argument_list", "parent": 39, "children": [42, 43]},
        "42": {"type": "LeftParenOp", "parent": 41,
               "children": ["(", ["(", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "43": {"type": "TMP", "parent": 41, "children": [44, 45]}, "44": {"type": "identifier", "parent": 43,
                                                                          "children": ["old_attributes",
                                                                                       ["old", "attributes",
                                                                                        "<pad>", "<pad>",
                                                                                        "<pad>"]]},
        "45": {"type": "LeftParenOp", "parent": 43,
               "children": [")", [")", "<pad>", "<pad>", "<pad>", "<pad>"]]},
        "46": {"type": "end_keyword", "parent": 36,
               "children": ["end", ["end", "<pad>", "<pad>", "<pad>", "<pad>"]]}}
    nx2dgl_graph = build_graph(bin_ast, dict)
    dgl_graph = tree2dgl(bin_ast, dict)
    dgl_graph