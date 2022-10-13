# -*- coding: utf-8 -*-

import argparse
from multiprocessing import cpu_count

from ncc import LOGGER

try:
    from ncc_dataset.codesearchnet_feng.proj_oriented import (
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

if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--language", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--dataset_dir", "-d", default=RAW_PROJ_DATA_DIR, type=str, help="raw dataset download directory",
    )
    parser.add_argument(
        "--flatten_dir", "-f", default=FLATTEN_PROJ_DATA_DIR, type=str,
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
    LOGGER.info(args)
