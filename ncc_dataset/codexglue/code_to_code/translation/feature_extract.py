# -*- coding: utf-8 -*-


import os
import argparse
import itertools
from multiprocessing import Pool, cpu_count
from ncc_dataset.codexglue.code_to_code.translation import (
    LANGUAGES,
    ATTRIBUTES_DIR,
    LOGGER,
    MODES,
    LIBS_DIR,
)

from collections import Counter
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.file_io import (
    safe_readline,
    find_offsets,
)
from ncc.utils.path_manager import PathManager
from ncc.data import constants
from ncc_dataset.codexglue.parser._parser import CodeParser

import sys

sys.setrecursionlimit(999)


def ast_to_dfs(ast):
    heights = {}

    def get_height(ast, idx, h):
        heights[idx] = h
        if 'value' in ast[idx]:
            heights[idx] = h
        else:
            for child in ast[idx]['children']:
                get_height(ast, str(child), h + 1)

    dfs, dfs_height = [], []
    get_height(ast, idx='0', h=0)
    for idx, node in ast.items():
        dfs.append(node['type'])
        dfs_height.append(heights[idx])
    return dfs, dfs_height


def ast2edtree(ast):
    def _dfs(idx):
        if "value" in ast[idx]:
            return "{" + ast[idx]['type'] + "}"
        else:
            tmp = ""
            for child in ast[idx]['children']:
                tmp += _dfs(str(child))
            return "{" + ast[idx]['type'] + tmp + "}"

    return _dfs("0")


class AttrFns:
    """build your defined function for attributes"""

    @staticmethod
    def ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing
        parser = CodeParser(SO_FILE=os.path.join(kwargs['so_dir'], f"{kwargs['lang']}.so"), LANGUAGE=kwargs['lang'])

        dest_filename = f"{dest_filename}{idx}"
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                line = json_io.json_loads(line)
                ast = parser.parse_raw_ast(code=line, MAX_AST_SIZE=99999999999, append_index=True)
                print(json_io.json_dumps(ast), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def dfs_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = f"{dest_filename}{idx}"
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast is not None:
                    dfs, _ = ast_to_dfs(ast)
                else:
                    dfs = None
                print(json_io.json_dumps(dfs), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def edtree_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = f"{dest_filename}{idx}"
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast is not None:
                    edtree = ast2edtree(ast)
                else:
                    edtree = None
                print(json_io.json_dumps(edtree), file=writer)
                line = safe_readline(reader)


def process(src_filename, tgt_filename, num_workers=cpu_count(), **kwargs):
    modality = tgt_filename.split('.')[-1]
    attr_fn = getattr(AttrFns, '{}_fn'.format(modality))
    offsets = find_offsets(src_filename, num_workers)

    # for debug
    idx = 0
    attr_fn(src_filename, tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])

    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                attr_fn,
                (src_filename, tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]

    def _cat_and_remove(tgt_filename, num_workers):
        with file_io.open(tgt_filename, 'w') as writer:
            for idx in range(num_workers):
                src_filename = tgt_filename + str(idx)
                with file_io.open(src_filename, 'r') as reader:
                    PathManager.copyfileobj(reader, writer)
                PathManager.rm(src_filename)

    _cat_and_remove(tgt_filename, num_workers)


if __name__ == '__main__':
    """
    This script is to generate new attributes of code snippet.
    """
    parser = argparse.ArgumentParser(description="Download python_wan dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--languages", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of attributes",
    )
    parser.add_argument(
        "--so_dir", "-s", default=LIBS_DIR, type=str, help="library directory of so file",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=[
            'ast', 'dfs',
            # 'edtree'
        ],
        # default=[ ],
        type=str, nargs='+', help="attrs: raw_ast, ...",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    dest_raw_attrs = {
        'ast': 'code',
        'dfs': 'ast',
        'edtree': 'ast',
    }

    for lang, mode in itertools.product(args.languages, MODES):
        for tgt_attr in args.attrs:
            src_attr = dest_raw_attrs[tgt_attr]
            src_filename = os.path.join(args.attributes_dir, lang, f"{mode}.{src_attr}")
            if PathManager.exists(src_filename):
                tgt_filename = os.path.join(args.attributes_dir, lang, f"{mode}.{tgt_attr}")
                LOGGER.info('Generating {}'.format(tgt_filename))
                process(src_filename, tgt_filename, num_workers=args.cores, lang=lang, so_dir=args.so_dir)
            else:
                LOGGER.info('{} does exist'.format(src_filename))
