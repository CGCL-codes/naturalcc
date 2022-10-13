# -*- coding: utf-8 -*-


import itertools
import math
import os
from multiprocessing import Pool, cpu_count

import numpy as np

from dataset.avatar.translation import (
    ATTRIBUTES_DIR, MODES
)
from dataset.avatar.translation.probing.diversity import (
    edit_distance, lexical_distance, )
from ncc.utils.file_ops import json_io


def multi_edit_distance(java_trees, python_trees):
    distances = []
    for java_tree, python_tree in zip(java_trees, python_trees):
        if python_tree is not None and java_tree is not None:
            distances.append(edit_distance(java_tree, python_tree))
    return distances


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Avatar dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument("--topk", "-k", default=1, type=int, )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()

    ldiversity = []
    for mode in MODES:
        java_file = os.path.join(ATTRIBUTES_DIR, f"top{args.topk}", "java", f"{mode}.code")
        python_file = os.path.join(ATTRIBUTES_DIR, f"top{args.topk}", "python", f"{mode}.code")

        with open(java_file, 'r') as java_reader, open(python_file, 'r') as python_reader:
            for java, python in zip(java_reader, python_reader):
                java = json_io.json_loads(java)
                python = json_io.json_loads(python)
                ldiversity.append(lexical_distance(java, python))
    # Lexical distance: 16.75
    print(f"Lexical distance: {np.asarray(ldiversity).mean():.2f}")

    eddiversity = []
    for mode in MODES:
        java_file = os.path.join(ATTRIBUTES_DIR, f"top{args.topk}", "java", f"{mode}.edtree")
        python_file = os.path.join(ATTRIBUTES_DIR, f"top{args.topk}", "python", f"{mode}.edtree")

        with open(java_file, 'r') as reader:
            java_edtrees = [json_io.json_loads(line) for line in reader]

        with open(python_file, 'r') as reader:
            python_edtrees = [json_io.json_loads(line) for line in reader]

        assert len(java_edtrees) == len(python_edtrees)
        print(f"{mode}: {len(java_edtrees)}")

        # num_workers = cpu_count()
        num_workers = 40
        params = []
        interval = math.ceil(len(java_edtrees) / num_workers)
        for idx in range(num_workers):
            params.append(
                (
                    java_edtrees[idx * interval:(idx + 1) * interval],
                    python_edtrees[idx * interval:(idx + 1) * interval],
                )
            )

        with Pool(processes=num_workers) as mpool:
            result = [
                mpool.apply_async(multi_edit_distance, params[idx])
                for idx in range(num_workers)
            ]
            result = [res.get() for res in result]

        result = list(itertools.chain(*result))
        eddiversity.extend(result)

    # ED distance: 314.21
    print(f"ED distance: {np.asarray(eddiversity).mean():.2f}")
