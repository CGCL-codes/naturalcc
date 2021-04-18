# -*- coding: utf-8 -*-

import os

from dataset.codesearchnet import (
    MODES,
    RAW_DIR, ATTRIBUTES_DIR,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def cast_code(raw_code_file, refined_code_file, dst_file):
    with file_io.open(raw_code_file, 'r') as raw_reader:
        raw_codes = {}
        for line in raw_reader:
            raw_code = line
            raw_code = raw_code[raw_code.find('def '):]
            func_name = raw_code[:raw_code.find('(')][4:].strip()
            raw_codes[func_name] = line.rstrip('\n')

    PathManager.mkdir(os.path.dirname(dst_file))
    with file_io.open(refined_code_file, 'r') as refined_reader, file_io.open(dst_file, 'w') as writer:
        for line in refined_reader:
            func_name = line[line.find('def '):].split()[1]
            raw_code = raw_codes[func_name]
            print(raw_code, file=writer)


def cast_code_tokens(src_file, dst_file):
    with file_io.open(src_file, 'r') as reader, file_io.open(dst_file, 'w') as writer:
        for line in reader:
            print(json_io.json_dumps(line.split()), file=writer)


def cast_docstring(src_file, dst_file):
    with file_io.open(src_file, 'r') as reader, file_io.open(dst_file, 'w') as writer:
        for line in reader:
            print(json_io.json_dumps(line.rstrip('\n')), file=writer)


def cast_docstring_tokens(src_file, dst_file):
    with file_io.open(src_file, 'r') as reader, file_io.open(dst_file, 'w') as writer:
        for line in reader:
            docstring_tokens = line.split()
            print(json_io.json_dumps(docstring_tokens), file=writer)


if __name__ == '__main__':
    for mode in MODES:
        # cast code
        raw_code_file = os.path.join(RAW_DIR, 'code.json')
        refined_code_file = os.path.join(RAW_DIR, mode, 'code.original')
        dst_file = os.path.join(ATTRIBUTES_DIR, '{}.code'.format(mode))
        cast_code(raw_code_file, refined_code_file, dst_file)

        # cast code_tokens
        src_file = os.path.join(RAW_DIR, mode, 'code.original_subtoken')
        dst_file = os.path.join(ATTRIBUTES_DIR, '{}.code_tokens'.format(mode))
        cast_code_tokens(src_file, dst_file)

        # cast docstring
        src_file = os.path.join(RAW_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(ATTRIBUTES_DIR, '{}.docstring'.format(mode))
        cast_docstring(src_file, dst_file)

        # cast docstring_tokens
        src_file = os.path.join(RAW_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(ATTRIBUTES_DIR, '{}.docstring_tokens'.format(mode))
        cast_docstring_tokens(src_file, dst_file)
