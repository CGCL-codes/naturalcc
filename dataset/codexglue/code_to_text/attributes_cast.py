# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import Pool, cpu_count

from dataset.codexglue.code_to_text import (
    LANGUAGES, RAW_DIR, ATTRIBUTES_DIR,
)
from ncc import LOGGER
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def flatten_attrs(raw_file, flatten_dir, lang, attrs):
    def _get_file_info(filename):
        """get mode and file index from file name"""
        filename = os.path.split(filename)[-1]
        mode = filename[:str.rfind(filename, '.jsonl')]
        return mode

    mode = _get_file_info(raw_file)
    attr_writers = {}
    for attr in attrs:
        attr_file = os.path.join(flatten_dir, lang, f'{mode}.{attr}')
        PathManager.mkdir(os.path.dirname(attr_file))
        attr_writers[attr] = file_io.open(attr_file, 'w')
    print('raw_file: ', raw_file)
    with file_io.open(raw_file, 'r') as reader:
        for line in reader:
            code_snippet = json_io.json_loads(line)
            for attr, info in code_snippet.items():
                if attr in attr_writers:
                    print(json_io.json_dumps(info), file=attr_writers[attr])


def flatten(raw_dir, lang, flatten_dir, attrs, num_cores):
    """flatten attributes of raw data"""
    LOGGER.info('Flatten the attributes({}) of {} raw dataset'.format(attrs, lang))

    with Pool(num_cores) as mpool:
        result = [
            mpool.apply_async(
                flatten_attrs,
                (raw_file, flatten_dir, lang, set(attrs))
            )
            for raw_file in PathManager.ls(os.path.join(raw_dir, lang, '*.jsonl'))
        ]
        result = [res.get() for res in result]


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
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name'],
        type=str, nargs='+',
        help="attrs: code, code_tokens, docstring",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    for lang in args.languages:
        flatten(raw_dir=args.raw_dataset_dir, lang=lang, flatten_dir=args.attributes_dir, attrs=args.attrs,
                num_cores=args.cores)
