# -*- coding: utf-8 -*-

import os
import jsonlines
import ujson
import argparse
import itertools
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count

try:
    from dataset.csn_feng import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        LOGGER,
    )
except ImportError:
    from . import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        LOGGER,
    )


def flatten_attrs(raw_file, flatten_dir, lang, attrs):
    def _get_file_info(filename):
        """get mode and file index from file name"""
        filename = os.path.split(filename)[-1]
        mode = filename[:str.rfind(filename, '.jsonl')]
        return mode

    _flatten_dir = os.path.expanduser(flatten_dir)
    mode = _get_file_info(raw_file)
    attr_writers = {}
    for attr in attrs:
        attr_file = os.path.join(_flatten_dir, lang, '{}.{}'.format(mode, attr))
        os.makedirs(os.path.dirname(attr_file), exist_ok=True)
        attr_writers[attr] = open(attr_file, 'w')

    with open(raw_file, 'r') as reader:
        for line in jsonlines.Reader(reader):
            for attr, info in line.items():
                if attr in attr_writers:
                    print(ujson.dumps(info, ensure_ascii=False), file=attr_writers[attr])


def flatten(raw_dir, lang, flatten_dir, attrs, num_cores=None):
    """flatten attributes of raw data"""
    if num_cores is None:
        num_cores = cpu_count()
    num_cores = min(num_cores, cpu_count())

    LOGGER.info('Flatten the attributes({}) of {} raw dataset at {}'.format(attrs, lang, flatten_dir))
    _raw_dir = os.path.expanduser(raw_dir)
    with Pool(num_cores) as mpool:
        result = [
            mpool.apply_async(flatten_attrs, (raw_file, flatten_dir, lang, set(attrs)))
            for raw_file in glob(os.path.join(_raw_dir, lang, '*.jsonl'))
        ]
        result = [res.get() for res in result]


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    parser = argparse.ArgumentParser(description="Flatten CodeSearchNet(feng) dataset(s)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_DIR, type=str,
        help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'docstring', 'docstring_tokens'],
        type=str, nargs='+',
        help="attrs: code, code_tokens, docstring, docstring_tokens, func_name, original_string, index",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    for lang in args.language:
        flatten(raw_dir=args.dataset_dir, lang=lang, flatten_dir=args.flatten_dir, attrs=args.attrs,
                num_cores=args.cores)
