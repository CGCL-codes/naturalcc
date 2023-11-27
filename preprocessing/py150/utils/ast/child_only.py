import sys
from ncc.data.constants import (
    PAD,
    SBT_LEFT_PARENTHESE,
    SBT_RIGHT_PARENTHESE,
)
from ncc.data import tokenizer_funcs
from ..constants import (
    RECURSION_DEPTH,
    MAX_SUBTOKEN_LEN,
    NODE_TMP,
)
from copy import deepcopy

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(RECURSION_DEPTH)  # recursion depth


def child_value2child_only(ast):
    """node['value'] => node['children']"""
    for idx, node in ast.items():
        value = node.get('value', None)
        if value:
            node.pop('value')
            node['children'] = [value]
    return ast


def pad_leaf_nodes(ast, max_len=MAX_SUBTOKEN_LEN):
    '''
    pad leaf node's child into [XX, [XX, ...]]
    split token and pad it with PAD_TOKEN till reach MAX_TOKEN_LIST_LEN
    e.g. VariableName ->  [VariableName, [Variable, Name, PAD_TOKEN, PAD_TOKEN, ...]]
    '''
    for idx, node in ast.items():
        if len(node['children']) == 1 and isinstance(node['children'][0], str):
            subtokens = tokenizer_funcs._space_dpu_sub_tokenizer(node['children'][0])[:max_len]
            subtokens.extend([PAD] * (max_len - len(subtokens)))
            node['children'].append(subtokens)
    return ast


def ast2sbt(ast, idx):
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    '''
    if len(ast[idx]['children']) == 2 and type(ast[idx]['children'][1]) == list:
        token = ast[idx]['type'] + '_' + ast[idx]['children'][0]
        seq = [SBT_LEFT_PARENTHESE, token, SBT_RIGHT_PARENTHESE, token]
    else:
        token = ast[idx]['type']
        seq = [SBT_LEFT_PARENTHESE, token]
        for child_idx in ast[idx]['children']:
            seq += ast2sbt(ast, str(child_idx))
        seq += [SBT_RIGHT_PARENTHESE, token]
    return seq


def get_root(ast):
    """get root node index"""
    for idx, node in ast.items():
        if node['parent'] is None:
            return idx


def delete_root_with_unichild(ast):
    """
    delete root node with only a child
    because in such way, head node might be Program/Function/Error and its child is the code's AST
    """
    for idx in sorted([idx for idx in ast.keys()], key=int):
        if (ast[idx]['parent'] is None) and len(ast[idx]['children']) == 1:
            child_idx = ast[idx]['children'][0]
            ast[str(child_idx)]['parent'] = None
            ast.pop(idx)
        else:
            break
    return ast


def delete_nodes_with_unichild(ast):
    '''
    delete nodes with single child node
    e.g. [1*NODEFIX1] ->  [1*NODEFIX2] -> ['void'] => [1*NODEFIX1] -> ['void']
    '''

    def _dfs(idx):
        node = ast[idx]
        # get current node's children indices, if it's leaf node, ignore.
        if not (len(node['children']) == 1 and isinstance(node['children'][0], str)):
            child_ids = node['children']
        else:
            return  # move to leaf node, return

        # each ast tree generally is parsed from a method, so it has a "program" root node and a "method" node
        # therefore, if current node is the root node with single child, we do not delete it
        while (len(child_ids) == 1) and (node['parent'] is not None):
            # update its parent's children
            parent_node = ast[str(node['parent'])]
            del_idx = parent_node['children'].index(int(idx))
            parent_node['children'].pop(del_idx)
            child_idx = child_ids[0]
            # update its children's parent to its parent
            ast[str(child_idx)]['parent'] = node['parent']
            # update its parent's children
            parent_node['children'].insert(del_idx, child_idx)
            # delete itself
            ast.pop(idx)

            # update current info
            idx = str(child_idx)
            node = ast[idx]
            # get current node's children indices, if it's leaf node, ignore.
            if not (len(node['children']) == 1 and isinstance(node['children'][0], str)):
                child_ids = node['children']
            else:
                return  # move to leaf node, return

        for idx in child_ids:
            _dfs(str(idx))

    idx = get_root(ast)
    _dfs(idx)
    return ast


def ast2bin_ast(ast):
    '''ast tree -> binary ast tree'''
    last_node_idx = sorted(ast.keys(), key=int)[-1]

    def _dfs(idx):
        node = ast[idx]
        # get current node's children indices, if it's leaf node, ignore.
        if not (len(node['children']) == 1 and isinstance(node['children'][0], str)):
            child_ids = node['children']
        else:
            return  # move to leaf node, return

        if len(child_ids) > 2:
            # add new node
            nonlocal last_node_idx
            last_node_idx = str(int(last_node_idx) + 1)
            ast[last_node_idx] = {'type': NODE_TMP, 'parent': idx, 'children': child_ids[1:]}
            # update node's children info
            node['children'] = [child_ids[0], int(last_node_idx)]
            # update other childen nodes' parent info
            for child_idx in child_ids[1:]:
                ast[str(child_idx)]['parent'] = last_node_idx
            # update current node's children info
            # get current node's children indices, if it's leaf node, ignore.
            if not (len(node['children']) == 1 and isinstance(node['children'][0], str)):
                child_ids = node['children']
            else:
                return  # move to leaf node, return

        for idx in child_ids:
            _dfs(str(idx))

    idx = get_root(ast)
    _dfs(idx)
    return ast


def reset_indices(ast):
    '''rename ast tree's node indices with consecutive indices'''
    if sorted(list(ast.keys())) == list(range(len(ast))):
        return ast

    # firstly, resort node index with a prefix "_", e.g. 0 => "_0"
    _idx = 0

    def _dfs(idx, _parent_idx):
        nonlocal _idx
        _new_idx, _idx = f'_{_idx}', _idx + 1  # update for next node
        node = ast.pop(str(idx))
        ast[_new_idx] = node

        # update its parent's children
        if node['parent'] is None:
            pass  # current node is root node, no need for update its children
        else:
            parent_node = ast[_parent_idx]
            # update its index in its parent node
            parent_node['children'][parent_node['children'].index(idx)] = _new_idx
            # update parent index
            node['parent'] = _parent_idx

        if isinstance(node['children'][0], int):  # non-leaf nodes, traverse its children nodes
            # update its children nodes' parent
            for child_idx in node['children']:
                _dfs(child_idx, _parent_idx=_new_idx)
        else:
            return

    root_idx = get_root(ast)
    _dfs(root_idx, _parent_idx=None)

    # recover name: from _* => *
    node_ids = deepcopy(list(ast.keys()))
    for idx in node_ids:
        node = ast.pop(idx)
        # update children index
        if len(node['children']) > 1:
            node['children'] = [int(child_idx[1:]) for child_idx in node['children']]
        # update parent index
        if node['parent'] == None:
            pass
        else:
            node['parent'] = int(node['parent'][1:])
        ast[int(idx[1:])] = node  # _idx => idx
    return ast
