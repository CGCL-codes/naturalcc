# -*- coding: utf-8 -*-

import os
import shutil
from multiprocessing import Pool, cpu_count

from preprocessing.py150 import (
    RAW_DIR,
    ATTRIBUTES_DIR,
    MODES,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def convert(ast):
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


def ast_fn(filename, dest_filename, idx, start=0, end=-1):
    dest_filename = dest_filename + str(idx)
    with file_io.open(filename, "r", encoding="UTF-8") as reader, open(dest_filename, 'w') as writer:
        reader.seek(start)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = json_io.json_loads(line)
            ast = convert(line)
            print(json_io.json_dumps(ast), file=writer)
            line = file_io.safe_readline(reader)


def process(src_filename, tgt_filename, num_workers=cpu_count(), func=None):
    _src_filename = os.path.expanduser(src_filename)
    _tgt_filename = os.path.expanduser(tgt_filename)
    modality = tgt_filename.split('.')[-1]
    offsets = file_io.find_offsets(_src_filename, num_workers)

    # # for debug
    # idx = 0
    # func(_src_filename, _tgt_filename, idx, offsets[idx], offsets[idx + 1])

    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                func,
                (_src_filename, _tgt_filename, idx, offsets[idx], offsets[idx + 1])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]

    def _concate(_tgt_filename, num_workers, tgt_filename):
        src_filenames = [_tgt_filename + str(idx) for idx in range(num_workers)]
        with file_io.open(tgt_filename, 'w') as writer:
            for _src_fl in src_filenames:
                with file_io.open(_src_fl, 'r') as reader:
                    shutil.copyfileobj(reader, writer)
                PathManager.rm(_src_fl)

    _concate(_tgt_filename, num_workers, tgt_filename)


if __name__ == '__main__':
    # old ast => new ast
    for file, mode in zip(['python100k_train.json', 'python50k_eval.json'], MODES):
        file = os.path.join(RAW_DIR, file)
        PathManager.mkdir(ATTRIBUTES_DIR)
        tgt_file = os.path.join(ATTRIBUTES_DIR, f'{mode}.ast')
        process(src_filename=file, tgt_filename=tgt_file, num_workers=cpu_count(), func=ast_fn)
