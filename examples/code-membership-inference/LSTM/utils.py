#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import multiprocessing as mp

from tqdm import tqdm


def line_positions(file_path):
    with open(file_path) as f:
        while True:
            pos = f.tell()
            if f.readline():
                yield pos
            else:
                break


def get_number_of_lines(fobj):
    nol = sum(1 for _ in fobj)
    fobj.seek(0)
    return nol


def file_tqdm(f):
    return tqdm(f, total=get_number_of_lines(f))


def parallelize(iterable, f, f_args=(), worker_init=None, n_cores=None):
    if n_cores == 1:
        return _mp_iterate_over(f, iterable, f_args)
    if n_cores is None:
        n_cores = int(mp.cpu_count())
    lst = list(iterable)
    chunksize = math.ceil(len(lst) / n_cores)
    with mp.Pool(processes=n_cores, initializer=worker_init) as pool:
        jobs = [
            pool.apply_async(
                _mp_iterate_over, (f, lst[i * chunksize : (i + 1) * chunksize], f_args)
            )
            for i in range(n_cores)
        ]
        multiple_results = [job.get() for job in jobs]
        results = flatten(multiple_results)
    return results


def _mp_iterate_over(f, lst, f_args):
    return [f(x, *f_args) for x in lst]


def flatten(list_of_lists):
    return [x for xs in list_of_lists for x in xs]


########################################################################
# generating dataset utils


def get_dfs(ast, only_leaf=False):
    dp = []
    for node in ast:
        if "value" in node:
            dp.append(node["value"])
        else:
            if not only_leaf:
                dp.append(node["type"])
    return dp


def separate_dps(ast, max_len):
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
        aug_asts.append([ast[i : i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])
    return aug_asts


def get_ancestors(ast):
    ancestors = {0: []}
    node2parent = {0: 0}
    for i, node in enumerate(ast):
        if "children" in node:
            for child in node["children"]:
                node2parent[child] = i
        ancestors[i] = [i] + ancestors[node2parent[i]]
    return ancestors


def get_terminal_nodes(ast):
    terminal_nodes = [i for i, node in enumerate(ast) if "children" not in node]
    return terminal_nodes

    
def tokenize(s):
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    tokenized = pattern.sub("_", s).lower().split("_")
    return list(filter(None, tokenized))[:5]



