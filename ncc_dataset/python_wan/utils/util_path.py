# -*- coding: utf-8 -*-
'''
ref: https://github.com/tech-srl/code2seq/blob/master/Python150kExtractor/extract.py
'''

import re
import itertools
from random import shuffle
from dataset.codesearchnet import MAX_SUB_TOKEN_LEN
from ncc import LOGGER

MAX_PATH_LENTH = 8
MAX_PATH_WIDTH = 2


def __terminals(ast, node_idx, MAX_TERMINALS=1000):
    stack, paths = [], []

    def dfs(idx):
        stack.append(idx)
        v_node = ast[idx]

        child_ids = v_node.get('children', None)
        if child_ids is None:
            # add leaf node's value
            paths.append((stack.copy(), v_node['value']))
        else:
            # converse non-leaf node
            if idx == 0:
                # add root node
                paths.append((stack.copy(), v_node['type']))
            if len(paths) >= MAX_TERMINALS:
                """some ast are too large, therefore we stop finding terminals"""
                return
            for child_idx in child_ids:
                dfs(child_idx)
        stack.pop()

    dfs(node_idx)
    return paths


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def __raw_tree_paths(ast, node_idx=1):
    tnodes = __terminals(ast, node_idx)

    tree_paths = []
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
        iterable=tnodes,
        r=2,
    ):
        prefix, lca, suffix = __merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= MAX_PATH_LENTH) \
            and (abs(len(prefix) - len(suffix)) <= MAX_PATH_WIDTH):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


def __collect_sample(ast, MAX_PATH: int):
    def _tokenize(s):
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        tokenized = pattern.sub("_", s).lower().split("_")
        return list(filter(None, tokenized))[:MAX_SUB_TOKEN_LEN]

    tree_paths = __raw_tree_paths(ast)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        start = _tokenize(start)
        finish = _tokenize(finish)

        try:
            connector = [ast[connector[0]]['value']] + \
                        [ast[v]['type'] for v in connector[1:-1]] + \
                        [ast[connector[-1]]['value']]
        except:
            # error path, skip it
            continue

        contexts.append([start, connector, finish])
    try:
        assert len(contexts) > 0, Exception('ast\'s path is None')
        if len(contexts) > MAX_PATH:
            shuffle(contexts)
            contexts = contexts[:MAX_PATH]
        return contexts
    except Exception as err:
        LOGGER.error(err)
        LOGGER.error(ast)
        return None


def ast_to_path(ast_tree, MAX_PATH: int):
    return __collect_sample(ast_tree, MAX_PATH)


