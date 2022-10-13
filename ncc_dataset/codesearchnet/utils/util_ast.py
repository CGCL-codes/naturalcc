# -*- coding: utf-8 -*-
import sys
from copy import deepcopy
from typing import List, Dict

from dataset.codesearchnet import (
    RECURSION_DEPTH,
    NODE_TMP,
)
from ncc.data.constants import (
    PAD,
    SBT_LEFT_PARENTHESE,
    SBT_RIGHT_PARENTHESE,
)

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(RECURSION_DEPTH)  # recursion depth

from ncc.tokenizers.tokenization import split_identifier


def value2children(ast_tree: Dict) -> Dict:
    """node['value'] => node['children']"""
    for idx, node in ast_tree.items():
        value = node.get('value', None)
        if value:
            node.pop('value')
            node['children'] = [value]
    return ast_tree


def pad_leaf_node(ast_tree: Dict, max_len: int, PAD_TOKEN=PAD) -> Dict:
    '''
    pad leaf node's child into [XX, [XX, ...]]
    split token and pad it with PAD_TOKEN till reach MAX_TOKEN_LIST_LEN
    e.g. VariableName ->  [VariableName, [Variable, Name, PAD_TOKEN, PAD_TOKEN, ...]]
    '''
    for idx, node in ast_tree.items():
        if len(node['children']) == 1 and isinstance(node['children'][0], str):
            subtokens = split_identifier(node['children'][0], False)
            if len(subtokens) == 0:
                subtokens = [node['children'][0]]
            if len(subtokens) >= max_len:
                subtokens = subtokens[:max_len]
            else:
                subtokens.extend([PAD_TOKEN] * (max_len - len(subtokens)))
            node['children'].append(subtokens)
    return ast_tree


def build_sbt_tree(ast_tree: Dict, idx: str) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    '''
    if len(ast_tree[idx]['children']) == 2 and type(ast_tree[idx]['children'][1]) == list:
        token = ast_tree[idx]['type'] + '_' + ast_tree[idx]['children'][0]
        seq = [SBT_LEFT_PARENTHESE, token, SBT_RIGHT_PARENTHESE, token]
    else:
        token = ast_tree[idx]['type']
        seq = [SBT_LEFT_PARENTHESE, token]
        for child_idx in ast_tree[idx]['children']:
            seq += build_sbt_tree(ast_tree, str(child_idx))
        seq += [SBT_RIGHT_PARENTHESE, token]
    return seq


def build_sbtao_tree(ast_tree: Dict, idx: str) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    :return:
    '''
    if len(ast_tree[idx]['children']) == 2 and type(ast_tree[idx]['children'][1]) == list:
        token = ast_tree[idx]['type'] + '_' + '<other>'
        seq = [SBT_LEFT_PARENTHESE, token, SBT_RIGHT_PARENTHESE, token]
    else:
        token = ast_tree[idx]['type']
        seq = [SBT_LEFT_PARENTHESE, token]
        for child_idx in ast_tree[idx]['children']:
            seq += build_sbtao_tree(ast_tree, str(child_idx))
        seq += [SBT_RIGHT_PARENTHESE, token]
    return seq


def delete_comment_node(ast_tree: Dict) -> Dict:
    '''delete comment node and its children'''

    def delete_cur_node(node_idx, cur_node):
        # update its parent's children
        parent_idx = cur_node['parent']
        parent_node = ast_tree[parent_idx]
        del_idx = parent_node['children'].index(node_idx)
        del parent_node['children'][del_idx]
        # delete node
        ast_tree.pop(node_idx)
        return parent_idx, parent_node

    def dfs(node_idx):
        cur_node = ast_tree[node_idx]
        child_ids = cur_node.get('children', None)

        if 'comment' in cur_node['type']:
            node_idx, cur_node = delete_cur_node(node_idx, cur_node)
            while len(cur_node['children']) == 0:
                node_idx, cur_node = delete_cur_node(node_idx, cur_node)

        if child_ids is None:
            return

        for idx in child_ids:
            dfs(node_idx=idx)

    dfs(node_idx=0)
    return ast_tree


def get_root_idx(ast_tree: Dict) -> int:
    """get root node index"""
    for idx, node in ast_tree.items():
        if node['parent'] is None:
            return idx


def remove_root_with_uni_child(ast_tree: Dict) -> Dict:
    """
    delete root node with only a child
    because in such way, head node might be Program/Function/Error and its child is the code's AST
    """
    ids = sorted([idx for idx in ast_tree.keys()], key=int)
    for idx in ids:
        if (ast_tree[idx]['parent'] is None) and len(ast_tree[idx]['children']) == 1:
            child_idx = ast_tree[idx]['children'][0]
            ast_tree[child_idx]['parent'] = None
            ast_tree.pop(idx)
        else:
            break
    return ast_tree


def delete_node_with_uni_child(ast_tree: Dict, idx: int) -> Dict:
    '''
    delete nodes with single child node
    e.g. [1*NODEFIX1] ->  [1*NODEFIX2] -> ['void'] => [1*NODEFIX1] -> ['void']
    '''

    def dfs(idx):
        cur_node = ast_tree[idx]
        # get current node's children indices, if it's leaf node, ignore.
        if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
            child_ids = cur_node['children']
        else:
            return  # move to leaf node, return

        # each ast tree generally is parsed from a method, so it has a "program" root node and a "method" node
        # therefore, if current node is the root node with single child, we do not delete it
        while (len(child_ids) == 1) and (cur_node['parent'] is not None):
            # update its parent's children
            parent_node = ast_tree[cur_node['parent']]
            del_ind = parent_node['children'].index(int(idx))
            del parent_node['children'][del_ind]
            child_idx = child_ids[0]
            # update its children's parent to its parent
            ast_tree[child_idx]['parent'] = cur_node['parent']
            # update its parent's children
            parent_node['children'].insert(del_ind, child_idx)
            # delete itself
            ast_tree.pop(idx)

            # update current info
            idx = child_idx
            cur_node = ast_tree[idx]
            # get current node's children indices, if it's leaf node, ignore.
            if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
                child_ids = cur_node['children']
            else:
                return  # move to leaf node, return

        for idx in child_ids:
            dfs(idx)

    dfs(idx)
    return ast_tree


def binarize_tree(ast_tree: Dict, idx) -> Dict:
    '''ast tree -> binary ast tree'''
    last_node_idx = sorted(ast_tree.keys(), key=int)[-1]

    def dfs(idx):
        cur_node = ast_tree[idx]
        # get current node's children indices, if it's leaf node, ignore.
        if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
            child_ids = cur_node['children']
        else:
            return  # move to leaf node, return

        if len(child_ids) > 2:
            # add new node
            nonlocal last_node_idx
            last_node_idx = int(last_node_idx) + 1
            ast_tree[last_node_idx] = {'type': NODE_TMP, 'parent': idx, 'children': child_ids[1:]}
            # update node's children info
            cur_node['children'] = [child_ids[0], int(last_node_idx)]
            # update other childen nodes' parent info
            for child_idx in child_ids[1:]:
                ast_tree[child_idx]['parent'] = last_node_idx
            # update current node's children info
            # get current node's children indices, if it's leaf node, ignore.
            if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
                child_ids = cur_node['children']
            else:
                return  # move to leaf node, return

        for idx in child_ids:
            dfs(idx)

    dfs(idx)
    return ast_tree


def reset_indices(ast_tree: Dict, root_idx) -> Dict:
    '''rename ast tree's node indices with consecutive indices'''
    if sorted(list(ast_tree.keys())) == list(range(len(ast_tree))):
        return ast_tree

    # firstly, resort node index with _
    new_ast_idx = 0

    def dfs(idx):
        nonlocal new_ast_idx
        new_cur_idx, new_ast_idx = '_{}'.format(new_ast_idx), new_ast_idx + 1  # update for next node
        # cur_node = ast_tree[idx]
        # ast_tree[new_cur_idx] = deepcopy(cur_node)
        cur_node = ast_tree.pop(idx)
        ast_tree[new_cur_idx] = cur_node

        # update its parent's children
        if cur_node['parent'] is None:
            pass  # current node is root node, no need for update its children
        else:
            parent_node = ast_tree[cur_node['parent']]
            parent_node['children'][parent_node['children'].index(idx)] = new_cur_idx

        # get current node's children indices, if it's leaf node, ignore.
        if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
            # update its children nodes' parent
            for child_idx in cur_node['children']:
                ast_tree[child_idx]['parent'] = new_cur_idx

        # # 2. delete old node
        # ast_tree.pop(idx)

        # get current node's children indices, if it's leaf node, ignore.
        if not (len(cur_node['children']) == 1 and isinstance(cur_node['children'][0], str)):
            # update its children nodes' parent
            for child_idx in cur_node['children']:
                dfs(child_idx)
        else:
            return  # move to leaf node, return

    dfs(root_idx)

    # recover name: from _* => *
    node_ids = deepcopy(list(ast_tree.keys()))
    for idx in node_ids:
        node = ast_tree.pop(idx)
        # update children index
        if not (len(node['children']) == 1 and isinstance(node['children'][0], str)):
            node['children'] = [int(child_idx[1:]) for child_idx in node['children']]
        # update parent index
        if node['parent'] == None:
            pass
        else:
            try:
                node['parent'] = int(node['parent'][1:])
            except:
                node
        ast_tree[int(idx[1:])] = node  # _idx => idx

    return ast_tree


def convert(ast):
    new_ast = []
    start_idx = min(ast.keys())
    for idx in deepcopy(sorted(ast.keys(), key=int)):
        node = ast[idx]
        if 'children' in node:
            node['children'] = [child - int(start_idx) for child in node['children']]
        node.pop('parent')
        new_ast.append(node)
    ast = new_ast

    increase_by = {}  # count of how many idx to increase the new idx by:
    # each time there is a value node
    cur = 0
    for i, node in enumerate(ast):
        increase_by[i] = cur
        if "value" in node:
            cur += 1

    new_dp = []
    for i, node in enumerate(ast):
        inc = increase_by[i]
        if "value" in node:
            child = [i + inc + 1]
            if "children" in node:
                child += [n + increase_by[n] for n in node["children"]]
            new_dp.append({"type": node["type"], "children": child})
            new_dp.append({"value": node["value"]})
        else:
            if "children" in node:
                node["children"] = [n + increase_by[n] for n in node["children"]]
            new_dp.append(node)

    # sanity check
    children = []
    for node in new_dp:
        if "children" in node:
            children += node["children"]
    assert len(children) == len(set(children))
    return new_dp


def dfs_traversal(ast: List[Dict], only_leaf=False):
    dfs_seq = []
    for node in ast:
        if "value" in node:
            dfs_seq.append(node["value"])
        else:
            if not only_leaf:
                dfs_seq.append(node["type"])
    return dfs_seq


def separate_ast(ast: List[Dict], max_len: int):
    """
    Handles training / evaluation on long ASTs by splitting
    them into smaller ASTs of length max_len, with a sliding
    window of max_len / 2.

    Example: for an AST ast with length 1700, and max_len = 1000,
    the output will be:
    [[ast[0:1000], 0], [ast[500:1500], 1000], [ast[700:1700], 1500]]

    Input:
        ast : List[Dictionary]
            List of nodes in pre-order traversal.
        max_len : int

    Output:
        aug_asts : List[List[List, int]]
            List of (ast, beginning idx of unseen nodes)
    """
    half_len = int(max_len / 2)
    if len(ast) <= max_len:
        return [[ast, 0]]

    aug_asts = [[ast[:max_len], 0]]
    i = half_len
    while i < len(ast) - max_len:
        aug_asts.append([ast[i: i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])
    return aug_asts


def ast2old_version(ast: Dict, NODE_FIX='NODEFIX'):
    new_ast = {}
    for idx, node in ast.items():
        if node['parent'] is not None:
            node['parent'] = NODE_FIX + str(node['parent'])  # parent
        if isinstance(node['children'], list) and not isinstance(node['children'][1], list):
            node['children'] = [NODE_FIX + str(i) for i in node['children']]
        new_ast[NODE_FIX + str(idx)] = node
    del ast

    def increase_node_idx(node_idx, offset=1):
        return NODE_FIX + str(int(node_idx[len(NODE_FIX):]) + offset)

    for idx in reversed(list(new_ast.keys())):
        node = new_ast.pop(idx)
        idx = increase_node_idx(idx)
        if not (len(node['children']) == 2 and isinstance(node['children'][-1], list)):
            node['children'] = [increase_node_idx(child) for child in node['children']]
        new_ast[idx] = node
    return new_ast
