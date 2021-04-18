import os
import ujson
import argparse
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count

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


def get_non_leaf_node(filename, start, end):
    node_couter = Counter()
    with open(filename, 'r') as reader:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            tree = ujson.loads(line)
            tokens = []
            for node in tree:
                if 'children' in node:
                    tokens.append(node['type'])
            node_couter.update(tokens)
            line = safe_readline(reader)
    return node_couter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--in_files", "-i",
        default=os.path.join(DATASET_DIR, 'codebert/traverse_roberta/filter/javascript/train.ast'),
        type=str, nargs='+', help="data directory of flatten attribute(load)",
    )
    parser.add_argument(
        "--out_file", "-o",
        default=os.path.join(DATASET_DIR, 'codebert/traverse_roberta/filter/javascript/.ast.node_types'),
        type=str, help="out file to save tree node type names",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    args.in_files = [os.path.expanduser(args.in_files)] if type(args.in_files) == str \
        else [os.path.expanduser(file) for file in args.in_files]
    args.out_file = os.path.expanduser(args.out_file)

    for in_file in args.in_files:
        offsets = find_offsets(in_file, args.cores)
        with Pool(args.cores) as mpool:
            result = [
                mpool.apply_async(
                    get_non_leaf_node,
                    (in_file, offsets[idx], offsets[idx + 1])
                )
                for idx in range(args.cores)
            ]
            result = [res.get() for res in result]
    dict = Counter(itertools.chain(*[res.elements() for res in result]))
    with open(args.out_file, 'w') as writer:
        for token, freq in sorted(dict.items(), key=lambda key_value: key_value[-1], reverse=True):
            print(ujson.dumps([token, freq], ensure_ascii=False), file=writer)
