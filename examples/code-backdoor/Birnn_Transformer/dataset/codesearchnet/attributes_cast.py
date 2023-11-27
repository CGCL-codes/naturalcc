# -*- coding: utf-8 -*-

import os
import argparse
import itertools
import shutil
from multiprocessing import Pool, cpu_count

from dataset.codesearchnet import (
    LANGUAGES, MODES,
    RAW_DIR, ATTRIBUTES_DIR,
    LOGGER,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def flatten_attrs(raw_file, flatten_dir, lang, mode, attrs):
    def _get_file_info(filename):
        """get mode and file index from file name"""
        filename = os.path.split(filename)[-1]
        filename = filename[:str.rfind(filename, '.jsonl.gz')]
        _, _, idx = filename.split('_')
        return idx

    idx = _get_file_info(raw_file)
    attr_writers = {}
    for attr in attrs:
        attr_dir = os.path.join(flatten_dir, lang, mode, attr)
        PathManager.mkdir(attr_dir)
        attr_file = os.path.join(attr_dir, '{}.jsonl'.format(idx))
        attr_writers[attr] = file_io.open(attr_file, 'w')

    with file_io.open(raw_file, 'r') as reader:
        for line in reader:
            code_snippet = json_io.json_loads(line)
            for attr, info in code_snippet.items():
                if attr in attr_writers:
                    print(json_io.json_dumps(info), file=attr_writers[attr])


def flatten(raw_dir, lang, mode, flatten_dir, attrs, num_cores):
    """flatten attributes of raw data"""
    LOGGER.info('Cast attributes({}) of {}-{} dataset'.format(attrs, lang, mode))
    with Pool(num_cores) as mpool:
        result = [
            mpool.apply_async(
                flatten_attrs,
                (raw_file, flatten_dir, lang, mode, set(attrs))
            )
            for raw_file in PathManager.ls(os.path.join(raw_dir, lang, mode, '*.jsonl.gz'))
        ]
        result = [res.get() for res in result]


def merge_attr_files(flatten_dir, lang, mode, attrs):
    """shell cat"""

    def _merge_files(src_files, tgt_file):
        with file_io.open(tgt_file, 'w') as writer:
            for src_fl in src_files:
                with file_io.open(src_fl, 'r') as reader:
                    shutil.copyfileobj(reader, writer)

    def _get_file_idx(filename):
        filename = os.path.split(filename)[-1]
        idx = int(filename[:str.rfind(filename, '.json')])
        return idx

    for attr in attrs:
        attr_files = PathManager.ls(os.path.join(flatten_dir, lang, mode, attr, '*.jsonl'))
        attr_files = sorted(attr_files, key=_get_file_idx)
        assert len(attr_files) > 0, RuntimeError('Attribute({}) files do not exist.'.format(attr))
        dest_file = os.path.join(flatten_dir, lang, '{}.{}'.format(mode, attr))
        _merge_files(attr_files, dest_file)
    PathManager.rm(os.path.join(flatten_dir, lang, mode))


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--languages", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--raw_dataset_dir", "-r", default=RAW_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of attributes directory",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name'],
        type=str, nargs='+',
        help="attrs: code, code_tokens, docstring, docstring_tokens, func_name",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)
    for lang, mode in itertools.product(args.languages, MODES):
        flatten(raw_dir=args.raw_dataset_dir, lang=lang, mode=mode, flatten_dir=args.attributes_dir, attrs=args.attrs,
                num_cores=args.cores)
        merge_attr_files(flatten_dir=args.attributes_dir, lang=lang, mode=mode, attrs=args.attrs)
