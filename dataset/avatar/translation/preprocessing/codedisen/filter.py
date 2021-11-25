# -*- coding: utf-8 -*-

import os
import itertools

from dataset.avatar.translation import (
    LANGUAGES,
    DATASET_DIR,
    ATTRIBUTES_DIR,
    MODES,
)
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.path_manager import PathManager

if __name__ == '__main__':
    from ncc.data.dictionary import TransformersDictionary

    vocab = TransformersDictionary.from_pretrained('microsoft/graphcodebert-base')

    for topk in [1, 3, 5]:

        attributes = ['code', 'ast', 'dfs']
        dst_dir = os.path.join(DATASET_DIR, 'codedisen', 'data')
        for lang in LANGUAGES:
            PathManager.mkdir(os.path.join(dst_dir, f"top{topk}", lang))
        for mode in MODES:
            readers = [
                file_io.open(os.path.join(ATTRIBUTES_DIR, f"top{topk}", lang, f"{mode}.{attr}"), 'r')
                for lang in LANGUAGES for attr in attributes
            ]
            writers = [
                file_io.open(os.path.join(dst_dir, f"top{topk}", lang, f"{mode}.{attr}"), 'w')
                for lang in LANGUAGES for attr in attributes
            ]
            writers += [
                file_io.open(os.path.join(dst_dir, f"top{topk}", lang, f"{mode}.code_tokens"), 'w')
                for lang in LANGUAGES
            ]
            for lines in zip(*readers):
                lines = list(map(json_io.json_loads, lines))
                if all(line is not None for line in lines):
                    src_ast, tgt_ast = lines[1], lines[len(attributes) + 1]
                    # src_code_tokens = tokenize(src_ast)
                    # tgt_code_tokens = tokenize(tgt_ast)
                    src_code_tokens = [node['value'] for _, node in src_ast.items() if 'value' in node]
                    tgt_code_tokens = [node['value'] for _, node in tgt_ast.items() if 'value' in node]
                    lines += [src_code_tokens, tgt_code_tokens]
                    for line, writer in zip(lines, writers):
                        print(json_io.json_dumps(line), file=writer)
