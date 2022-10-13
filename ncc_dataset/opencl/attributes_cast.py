# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import re

import pandas as pd

from ncc_dataset.opencl import (
    LANGUAGES, RAW_DIR, ATTRIBUTES_DIR,
    MODES,
)
from ncc_dataset.opencl.inst2vec import inst2vec_preprocess
from ncc_dataset.opencl.inst2vec import task_utils
from ncc import LOGGER
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def flatten(raw_file, dst_dir, mode):
    """flatten attributes of raw data"""
    data_frame = pd.read_csv(raw_file)
    attrs = data_frame.columns.values.tolist()[1:-1]
    LOGGER.info('Cast attributes({}) of OpenCL-{} dataset'.format(attrs, lang))
    for attr in attrs:
        dst_file = os.path.join(dst_dir, f"{mode}.{attr}")
        data = getattr(data_frame, attr).values.tolist()
        with file_io.open(dst_file, 'w') as writer:
            for line in data:
                print(json_io.json_dumps(line), file=writer)


def code_tokenization(src_file):
    from clgen._atomizer import GreedyAtomizer
    from clgen._langs import Language

    with open(src_file, 'r') as reader:
        src_codes = reader.readlines()
    opencl_lang = Language.from_str('opencl')
    atomizer = GreedyAtomizer.from_text(opencl_lang, text='\n'.join(src_codes))

    dst_file = f"{src_file}_tokens"
    with open(dst_file, 'w') as writer:
        for code in src_codes:
            code = json_io.json_loads(code)
            code_tokens = atomizer.atomize(code)
            code_tokens = [atomizer.atoms[idx] for idx in code_tokens]
            print(json_io.json_dumps(code_tokens), file=writer)


def xfg(src_dir, languages, dst_dir):
    xfg_src_files = PathManager.ls(os.path.join(src_dir, "kernels_ir", '*.ll'))

    filenames = []
    ir_data = []
    for filename in xfg_src_files:
        filenames.append(os.path.basename(filename)[:-3])
        with open(filename, 'r') as reader:
            lines = reader.read().splitlines()
        ir_data.append(lines)
    # convert list to dict
    filenames = {name: idx for idx, name in enumerate(filenames)}

    processed_data, _ = inst2vec_preprocess.preprocess(ir_data)
    processed_data, _ = task_utils.inline_struct_types_txt(processed_data, ir_data)
    processed_data = task_utils.abstract_statements_from_identifiers_txt(processed_data)

    for idx, lines in enumerate(processed_data):
        processed_data[idx] = [
            line
            for line in lines if not re.match(r'((?:<label>:)?(<LABEL>):|; <label>:<LABEL>)', line)
        ]

    for lang in languages:
        raw_file = os.path.join(src_dir, f'{lang}.csv')
        # read raw csv file to load corresponding benchmarks
        data_frame = pd.read_csv(raw_file)
        benchmarks = data_frame["benchmark"].values.tolist()
        datasets = data_frame["dataset"].values.tolist()
        del data_frame

        # write
        dst_file = os.path.join(dst_dir, lang, f'train.xfg')
        with open(dst_file, 'w') as writer:
            for idx, (bm, ds) in enumerate(zip(benchmarks, datasets)):
                if bm[:3] == "npb":
                    bm += f'_{ds}'
                xfg = processed_data[filenames[bm]]
                print(json_io.json_dumps(xfg), file=writer)


if __name__ == '__main__':
    """
    This script is to flatten attributes of opencl dataset
    """
    parser = argparse.ArgumentParser(description="cast attributes of OpenCL dataset")
    parser.add_argument(
        "--languages", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--raw_dataset_dir", "-r", default=RAW_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of attributes directory",
    )
    args = parser.parse_args()
    # print(args)

    for lang, mode in itertools.product(args.languages, MODES):
        raw_file = os.path.join(args.raw_dataset_dir, f'{lang}.csv')
        dst_dir = os.path.join(args.attributes_dir, lang)
        PathManager.mkdir(dst_dir)
        flatten(raw_file, dst_dir, mode)

        code_tokenization(
            src_file=os.path.join(dst_dir, f'{mode}.src')
        )

    # xfg -> inst2vec
    xfg(src_dir=args.raw_dataset_dir, languages=args.languages, dst_dir=args.attributes_dir)
