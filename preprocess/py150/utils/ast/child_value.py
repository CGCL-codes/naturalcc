import sys
from copy import deepcopy
from ncc.data.constants import (
    PAD,
    SBT_LEFT_PARENTHESE,
    SBT_RIGHT_PARENTHESE,
)
from preprocess.utils.constants import (
    RECURSION_DEPTH,
    MAX_SUBTOKEN_LEN,
)

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(RECURSION_DEPTH)  # recursion depth


def get_root(ast) -> int:
    """get root node index"""
    for idx, node in ast.items():
        if node['parent'] is None:
            return idx


def reset_indices(ast):
    '''rename ast tree's node indices with consecutive indices'''
    if sorted(list(ast.keys())) == list(range(len(ast))):
        return ast

    # firstly, resort node index with a prefix "_", e.g. 0 => "_0"
    _idx = 0

    def _dfs(idx, _parent_idx):
        nonlocal _idx
        _new_idx, _idx = f'_{_idx}', _idx + 1  # update for next node
        node = ast.pop(idx)
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

        if 'children' in node:  # non-leaf nodes, traverse its children nodes
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
        if 'children' in node:
            node['children'] = [int(child_idx[1:]) for child_idx in node['children']]
        # update parent index
        if node['parent'] == None:
            pass
        else:
            node['parent'] = int(node['parent'][1:])
        ast[int(idx[1:])] = node  # _idx => idx
    return ast


def ast2sbt(ast, idx):
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    '''
    idx = str(idx)
    if 'value' in ast[idx]:
        token = [ast[idx]['type'], ast[idx]['value']]
        seq = [SBT_LEFT_PARENTHESE, token, SBT_RIGHT_PARENTHESE, token]
    else:
        token = ast[idx]['type']
        seq = [SBT_LEFT_PARENTHESE, token]
        for child_idx in ast[idx]['children']:
            seq += ast2sbt(ast, child_idx)
        seq += [SBT_RIGHT_PARENTHESE, token]
    return seq
