# -*- coding: utf-8 -*-

import os
import argparse
import itertools
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count
from ncc.utils.file_ops import (
    file_io, json_io
)
from ncc.utils.path_manager import PathManager
from preprocessing.codexglue.parser._parser import CodeParser

from preprocessing.codexglue.code_to_code.translation import (
    LANGUAGES, MODES,
    RAW_DIR, LIBS_DIR, ATTRIBUTES_DIR,
    LOGGER,
)
from ncc.tokenizers.tokenization import SPACE_SPLITTER

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
        "--dataset_dir", "-d", default=RAW_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=ATTRIBUTES_DIR, type=str,
        help="data directory of flatten attribute",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['code', 'code_tokens', 'code_types', 'ast'],
        type=str, nargs='+',
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    for mode in MODES:
        src_files = [os.path.join(args.dataset_dir, f"{mode}.{lang}") for lang in args.languages]
        src_readers = [file_io.open(file, 'r') for lang, file in zip(args.languages, src_files)]

        for lang in args.languages:
            PathManager.mkdir(os.path.join(args.flatten_dir, lang))
        dst_files = [os.path.join(args.flatten_dir, lang, f"{mode}.code") for lang in args.languages]
        dst_writers = {lang: file_io.open(file, 'w') for lang, file in zip(args.languages, dst_files)}

        for lines in zip(*src_readers):
            lines = list(map(lambda line: SPACE_SPLITTER.sub(" ", line.strip()), lines))
            for lang, line in zip(args.languages, lines):
                print(json_io.json_dumps(line.strip()), file=dst_writers[lang])
