# -*- coding: utf-8 -*-

import argparse
import os
import itertools
import shutil

from preprocess.codexglue.code_to_text import (
    LANGUAGES, MODES,
)
from ncc import tasks
from ncc.data import (
    Dictionary,
    indexed_dataset,
)
from ncc.utils.file_ops.yaml_io import recursive_expanduser
from ncc.utils.file_ops import file_io
from ncc.utils.path_manager import PathManager

if __name__ == '__main__':
    task = tasks.get_task('multilingual_denoising')
    base_dir = recursive_expanduser('~/ncc_data/codexglue/code_to_text/multilingual_denoising/data-mmap')

    dict_file = os.path.join(base_dir, 'dict.jsonl')
    vocab = task.load_dictionary(dict_file)

    for mode in MODES:
        dst_file = os.path.join(base_dir, 'docstring', f"{mode}.docstring.spm")
        PathManager.mkdir(os.path.dirname(dst_file))
        # mmap
        ds = indexed_dataset.make_builder(f'{dst_file}.mmap', impl='mmap', vocab_size=len(vocab))
        for lang in LANGUAGES:
            src_file = os.path.join(base_dir, lang, f"{mode}.docstring.spm")
            ds.merge_file_(src_file)
        ds.finalize(f'{dst_file}.idx')
        # # raw
        # with file_io.open(ds, 'w') as writer:
        #     for lang in LANGUAGES:
        #         src_file = os.path.join(base_dir, lang, f"{mode}.docstring.spm")
        #         with open(src_file, 'r') as reader:
        #             shutil.copyfileobj(reader, writer)
