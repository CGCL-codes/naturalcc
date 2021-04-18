import os
import re
import ujson
import argparse
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from dataset import LOGGER
from dataset.augmented_javascript import DATASET_DIR


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


_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        LOGGER.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def get_leaf_tokens(in_file, out_file, start, end):
    with open(in_file, 'r', encoding='utf8') as reader, open(out_file, 'w', encoding='utf8') as writer:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            tree = ujson.loads(line)
            leaf_tokens = []
            for node in tree:
                if 'value' in node:
                    leaf_tokens.append(normalize_program(node['value']))
            print(' '.join(leaf_tokens), file=writer)
            line = safe_readline(reader)


def merge_file(src_files, tgt_file):
    import shutil
    with open(tgt_file, 'w', encoding='utf8') as writer:
        for src_fl in src_files:
            with open(src_fl, 'r', encoding='utf8') as reader:
                shutil.copyfileobj(reader, writer)
            os.remove(src_fl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--in_files", "-i",
        default=os.path.join(DATASET_DIR, 'codebert/traverse_roberta/filter/javascript/train.ast'),
        type=str, nargs='+', help="data directory of flatten attribute(load)",
    )
    parser.add_argument(
        "--out_file", "-o",
        default=os.path.join(DATASET_DIR, 'codebert/traverse_roberta/filter/javascript/train.ast.leaf_token'),
        type=str, help="out file to save tree node type names",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    args.in_files = [os.path.expanduser(args.in_files)] if type(args.in_files) == str \
        else [os.path.expanduser(file) for file in args.in_files]
    args.out_file = os.path.expanduser(args.out_file)

    tmp_files = []
    for in_file in args.in_files:
        LOGGER.info('Multi-processing with {}'.format(in_file))
        offsets = find_offsets(in_file, args.cores)
        out_file = ['{}.tmp{}'.format(in_file, worker_id) for worker_id in range(args.cores)]
        tmp_files.extend(out_file)
        with Pool(args.cores) as mpool:
            result = [
                mpool.apply_async(
                    get_leaf_tokens,
                    (in_file, out_file[idx], offsets[idx], offsets[idx + 1])
                )
                for idx in range(args.cores)
            ]
            result = [res.get() for res in result]

    merge_file(src_files=tmp_files, tgt_file=args.out_file)
