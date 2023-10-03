# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import re
from multiprocessing import Pool, cpu_count

import ujson
from preprocess.stackoverflow.py2x import parse_csharp_code

from preprocess.stackoverflow import (
    ATTRIBUTES_DIR,
    MODES,
)
from ncc.tokenizers import tokenization


def parse_docstring(docstring):
    docstring = docstring.strip()
    tokens = re.findall(r"[\w]+|[^\s\w]", docstring)
    tokens = [token for token in tokens if len(token) > 0]
    return tokens


def parse_python_code(code):
    # remove comments in a code line
    code_lines = code.split('\n')
    code_lines = [line[:str.find(line, '#')] if '#' in line else line for line in code_lines]
    code_lines = filter(lambda line: len(line) > 0, code_lines)
    code = '\n'.join(code_lines)

    tokens = tokenization._space_dpu_sub_tokenizer(code)
    return tokens


def merge_file(src_files, tgt_file):
    import shutil
    with open(tgt_file, 'w', encoding='utf8') as writer:
        for src_fl in src_files:
            with open(src_fl, 'r', encoding='utf8') as reader:
                shutil.copyfileobj(reader, writer)
            os.remove(src_fl)


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def get_tokenizer(lang, modality):
    if modality == 'code':
        if lang == 'csharp':
            return parse_csharp_code
        else:  # python
            return parse_python_code
    else:
        return parse_docstring


def parse_fn(raw_filename, dst_filename, lang, modality, start=0, end=-1):
    token_fn = get_tokenizer(lang, modality)
    with open(raw_filename, 'r', encoding='UTF-8') as reader, open(dst_filename, 'w') as writer:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = ujson.loads(line)
            tokens = token_fn(line)
            print(ujson.dumps(tokens, ensure_ascii=False), file=writer)
            line = safe_readline(reader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flatten StackOverflow C#/Python datasets")
    parser.add_argument(
        "--languages", "-l", type=str, nargs='+', help="languages constain [{}]".format(['csharp', 'python']),
        default=['python', 'csharp', ],
    )
    parser.add_argument(
        "--dataset_dir", "-d", type=str, help="flatten dataset directory",
        default=ATTRIBUTES_DIR,
    )
    args = parser.parse_args()

    modalities = ['code', 'docstring', ]
    num_workers = cpu_count()

    with Pool(num_workers) as mpool:
        for lang, modality in itertools.product(args.languages, modalities):
            modes = MODES
            if os.path.exists(os.path.join(args.dataset_dir, lang, f'dev.{modality}')):
                modes.append('dev')
            if os.path.exists(os.path.join(args.dataset_dir, lang, f'eval.{modality}')):
                modes.append('eval')
            for mode in modes:
                raw_filename = os.path.join(args.dataset_dir, lang, '{}.{}'.format(mode, modality))
                dst_filename = raw_filename + '_tokens'
                offsets = find_offsets(raw_filename, num_workers)

                if num_workers > 1:
                    result = [
                        mpool.apply_async(
                            parse_fn,
                            (raw_filename, dst_filename + str(idx), lang, modality, offsets[idx], offsets[idx + 1])
                        )
                        for idx in range(num_workers)
                    ]
                    result = [res.get() for res in result]
                    merge_file([dst_filename + str(idx) for idx in range(num_workers)], dst_filename)
                else:
                    parse_fn(raw_filename, dst_filename, lang, modality, offsets[0], offsets[-1])
