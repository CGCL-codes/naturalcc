# -*- coding: utf-8 -*-

import argparse
import os
from collections import (
    Counter,
)
from glob import glob
from multiprocessing import cpu_count

import ujson
from tqdm import tqdm

try:
    from preprocess.codesearchnet_feng.proj_oriented import (
        LANGUAGES, MODES,
        RAW_DATA_DIR,
        RAW_PROJ_DATA_DIR, LIBS_DIR, FLATTEN_PROJ_DATA_DIR,
    )
except ImportError:
    from . import (
        LANGUAGES, MODES,
        RAW_DATA_DIR,
        RAW_PROJ_DATA_DIR, LIBS_DIR, FLATTEN_PROJ_DATA_DIR,
    )


def file_lines(reader):
    num = sum(1 for _ in reader)
    reader.seek(0)
    return num


def split_projs(data, projs, train=0.5, valid=0.2):
    train_projs, valid_projs, test_projs = projs[:int(len(projs) * train)], \
                                           projs[int(len(projs) * train):int(len(projs) * (train + valid))], \
                                           projs[int(len(projs) * (train + valid)):]
    train_projs, valid_projs, test_projs = set(train_projs), set(valid_projs), set(test_projs)
    train_data, valid_data, test_data = [], [], []
    for line in data:
        if line['repo'] in train_projs:
            train_data.append(line)
        elif line['repo'] in valid_projs:
            valid_data.append(line)
        else:
            test_data.append(line)
    return train_data, valid_data, test_data


if __name__ == '__main__':
    """
    This script is to split datasets into project-oriented datasets
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--proj_dataset_dir", "-p", default=RAW_PROJ_DATA_DIR, type=str,
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
        files = glob(os.path.join(RAW_DATA_DIR, lang, '*'))

        data = []
        for file in files:
            with open(file, 'r') as reader:
                for line in tqdm(reader, total=file_lines(reader)):
                    line = ujson.loads(line)
                    data.append(line)
        proj_nums = Counter([line['repo'] for line in data])

        save_proj_names = set([proj for proj, proj_num in proj_nums.items() if 30 <= proj_num <= 100])
        save_data = [line for line in data if line['repo'] in save_proj_names]
        save_proj_nums = Counter([line['repo'] for line in save_data])
        print(f'before: {len(data)}, after: {len(save_data)}')

        save_proj_names = sorted(list(save_proj_names))
        train_data, valid_data, test_data = split_projs(save_data, save_proj_names)

        os.makedirs(os.path.join(RAW_PROJ_DATA_DIR, lang), exist_ok=True)
        with open(os.path.join(RAW_PROJ_DATA_DIR, lang, 'train.jsonl'), 'w') as writer:
            for line in train_data:
                print(ujson.dumps(line), file=writer)
        with open(os.path.join(RAW_PROJ_DATA_DIR, lang, 'valid.jsonl'), 'w') as writer:
            for line in valid_data:
                print(ujson.dumps(line), file=writer)
        with open(os.path.join(RAW_PROJ_DATA_DIR, lang, 'test.jsonl'), 'w') as writer:
            for line in test_data:
                print(ujson.dumps(line), file=writer)
