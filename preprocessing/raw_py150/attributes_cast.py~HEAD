# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from multiprocessing import Pool, cpu_count

from preprocessing.raw_py150 import (
    MODES,
    RAW_DIR, ATTRIBUTES_DIR,
)
from preprocessing.raw_py150.utils import parse_file
from ncc import LOGGER
from ncc.data.constants import PAD
from ncc.utils.file_ops import (
    file_io,
    json_io,
)


def flatten_attrs(raw_file, flatten_dir, mode, attrs, start=0, end=-1):
    attr_writers = {}
    for attr in attrs:
        attr_file = os.path.join(flatten_dir, '{}.{}'.format(mode, attr))
        os.makedirs(os.path.dirname(attr_file), exist_ok=True)
        attr_writers[attr] = file_io.open(attr_file, 'w')

    with file_io.open(raw_file, 'r') as reader:
        reader.seek(start)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            filename = os.path.join(os.path.dirname(raw_file), line.strip())
            # tokens, types = parse_file(filename)
            try:
                tokens, types = parse_file(filename)
                # replace None with [PAD] for type dictionary build
                types = [PAD if t is None else t for t in types]
            except Exception as err:
                # print(err)
                # print(f'parsing {filename} error')
                line = file_io.safe_readline(reader)
                continue
            print(json_io.json_dumps(tokens), file=attr_writers['code_tokens'])
            print(json_io.json_dumps(types), file=attr_writers['code_types'])
            line = file_io.safe_readline(reader)


def flatten(raw_file, flatten_dir, mode, attrs, num_cores=None):
    """flatten attributes of raw data"""
    LOGGER.info('Flatten the attributes({}) of raw dataset at {}'.format(attrs, raw_file))
    if num_cores > 1:
        offsets = file_io.find_offsets(raw_file, num_cores)
        with Pool(num_cores) as mpool:
            result = [
                mpool.apply_async(flatten_attrs, \
                                  (raw_file, flatten_dir, f'{mode}{idx}', attrs, offsets[idx], offsets[idx + 1]))
                for idx in range(num_cores)
            ]
            result = [res.get() for res in result]
        for attr in attrs:
            attr_file = os.path.join(flatten_dir, '{}.{}'.format(mode, attr))
            with file_io.open(attr_file, 'w') as writer:
                for idx in range(num_cores):
                    src_file = os.path.join(flatten_dir, '{}.{}'.format(mode + str(idx), attr))
                    with file_io.open(src_file, 'r') as reader:
                        shutil.copyfileobj(reader, writer)
                    os.remove(src_file)
    else:
        flatten_attrs(raw_file, flatten_dir, mode, attrs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse raw_py150")
    parser.add_argument(
        "--raw_dataset_dir", "-r", default=RAW_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code_tokens', 'code_types'],
        type=str, nargs='+',
        help="attrs: tokens, types",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()

    files = ['python100k_train.txt', 'python50k_eval.txt']
    for mode, file in zip(MODES, files):
        src_file = os.path.join(args.raw_dataset_dir, file)
        flatten(raw_file=src_file, flatten_dir=args.attributes_dir, mode=mode, attrs=args.attrs, num_cores=args.cores)
