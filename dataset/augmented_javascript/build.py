# -*- coding: utf-8 -*-


import os
import wget
import argparse
import zipfile
import shutil
from tree_sitter import Language

try:
    from dataset.augmented_javascript import (
        RAW_DATA_DIR, LIBS_DIR, LOGGER,
    )
except ImportError:
    from . import (
        RAW_DATA_DIR, LIBS_DIR, LOGGER,
    )


def download_file(url, local):
    """download raw data files from amazon and lib files from github.com"""
    _local = os.path.expanduser(local)
    os.makedirs(os.path.dirname(_local), exist_ok=True)
    if os.path.exists(_local):
        LOGGER.info('File {} exists, ignore it. If you want to overwrite it, pls delete it firstly.'.format(local))
    else:
        LOGGER.info('Download {} from {}'.format(local, url))
        wget.download(url=url, out=_local)


def build_so(lib_dir, lang):
    """build so file for certain language with Tree-Sitter"""
    _lib_dir = os.path.expanduser(lib_dir)
    lib_file, _lib_file = os.path.join(lib_dir, '{}.zip'.format(lang)), os.path.join(_lib_dir, '{}.zip'.format(lang))
    if os.path.exists(_lib_file):
        LOGGER.info('Tree-Sitter so file for {} does not exists, compiling.'.format(lib_file))
        # decompress Tree-Sitter library
        with zipfile.ZipFile(_lib_file, 'r') as zip_file:
            zip_file.extractall(path=_lib_dir)
        so_file, _so_file = os.path.join(lib_dir, '{}.so'.format(lang)), os.path.join(_lib_dir, '{}.so'.format(lang))
        LOGGER.info('Building Tree-Sitter compile file {}'.format(so_file))
        Language.build_library(
            # your language parser file, we recommend buidl *.so file for each language
            _so_file,
            # Include one or more languages
            [os.path.join(_lib_dir, 'tree-sitter-{}-master'.format(lang))],
        )
    else:
        LOGGER.info('Tree-Sitter so file for {} exists, ignore it.'.format(lib_file))


if __name__ == '__main__':
    """
    This script is
        1) to download one language dataset from CodeSearchNet and the corresponding library
        2) to decompress raw data file and Tree-Sitter libraries
        3) to compile Tree-Sitter libraries into *.so file

    # ====================================== CodeSearchNet ====================================== #
    Tree-Sitter: AST generation tools, TreeSitter repositories from Github can be updated, therefore their size is capricious
       # language   # URL
       Java:        https://codeload.github.com/tree-sitter/tree-sitter-java/zip/master
       Javascript:  https://codeload.github.com/tree-sitter/tree-sitter-javascript/zip/master
       PHP:         https://codeload.github.com/tree-sitter/tree-sitter-php/zip/master
       GO:          https://codeload.github.com/tree-sitter/tree-sitter-go/zip/master
       Ruby:        https://codeload.github.com/tree-sitter/tree-sitter-ruby/zip/master
       Python:      https://codeload.github.com/tree-sitter/tree-sitter-python/zip/master
    """
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--libs_dir", "-b", default=LIBS_DIR, type=str, help="tree-sitter library directory",
    )
    args = parser.parse_args()
    # print(args)

    lang = 'javascript'
    # download Tree-Sitter AST parser libraries
    lib_url = 'https://codeload.github.com/tree-sitter/tree-sitter-{}/zip/master'.format(lang)
    lib_filename = os.path.join(args.libs_dir, f'{lang}.zip')
    download_file(url=lib_url, local=lib_filename)
    # compiling Tree-Sitter so file
    build_so(lib_dir=args.libs_dir, lang=lang)
