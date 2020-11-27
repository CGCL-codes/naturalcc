# -*- coding: utf-8 -*-

import os
import gzip
import ujson
import argparse
import itertools
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count

try:
    from dataset.csn import (
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
        filename = filename[:str.rfind(filename, '.jsonl.gz')]
        _, mode, idx = filename.split('_')
        return mode, idx

    _flatten_dir = os.path.expanduser(flatten_dir)
    mode, idx = _get_file_info(raw_file)
    attr_writers = {}
    for attr in attrs:
        attr_dir = os.path.join(_flatten_dir, lang, mode, attr)
        os.makedirs(attr_dir, exist_ok=True)
        attr_file = os.path.join(attr_dir, '{}.json'.format(idx))
        attr_writers[attr] = open(attr_file, 'w')

    with gzip.GzipFile(raw_file, 'r') as reader:
        for line in reader:
            code_snippet = ujson.loads(line)
            for attr, info in code_snippet.items():
                if attr in attr_writers:
                    print(ujson.dumps(info, ensure_ascii=False), file=attr_writers[attr])


def flatten(raw_dir, lang, flatten_dir, attrs, num_cores=None):
    """flatten attributes of raw data"""
    if num_cores is None:
        num_cores = cpu_count()
    num_cores = min(num_cores, cpu_count())

    LOGGER.info('Flatten the attributes({}) of {} raw dataset'.format(attrs, lang))
    _raw_dir = os.path.expanduser(raw_dir)
    with Pool(num_cores) as mpool:
        result = [
            mpool.apply_async(flatten_attrs, (raw_file, flatten_dir, lang, set(attrs)))
            for raw_file in glob(os.path.join(_raw_dir, lang, '*.jsonl.gz'))
        ]
        result = [res.get() for res in result]


def merge_attr_files(flatten_dir, lang, attrs):
    """shell cat"""

    def _merge_files(src_files, tgt_file):
        with open(tgt_file, 'w') as writer:
            for src_fl in src_files:
                with open(src_fl, 'r') as reader:
                    shutil.copyfileobj(reader, writer)

    def _get_file_idx(filename):
        filename = os.path.split(filename)[-1]
        idx = int(filename[:str.rfind(filename, '.json')])
        return idx

    _flatten_dir = os.path.expanduser(flatten_dir)
    for mode in ['train', 'valid', 'test']:
        for attr in attrs:
            attr_files = glob(os.path.join(_flatten_dir, lang, mode, attr, '*.json'))
            attr_files = sorted(attr_files, key=_get_file_idx)
            assert len(attr_files) > 0, RuntimeError('Attribute files({}) do not exist.'.format(attr_files))
            dest_file = os.path.join(_flatten_dir, lang, '{}.{}'.format(mode, attr))
            _merge_files(attr_files, dest_file)


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    parser = argparse.ArgumentParser(description="Flatten CodeSearchNet dataset(s)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_DIR, type=str, help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name'],
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
        merge_attr_files(flatten_dir=args.flatten_dir, lang=lang, attrs=args.attrs)
