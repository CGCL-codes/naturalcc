# -*- coding: utf-8 -*-

import argparse
import os
from multiprocessing import cpu_count

from dataset.avatar.translation import (
    LANGUAGES, MODES,
    RAW_DIR, ATTRIBUTES_DIR,
)
from ncc.utils.file_ops import (
    file_io, json_io
)
from ncc.utils.path_manager import PathManager


def revert_code(code: str):
    """
    code = "def getInt ( ) : return int ( input ( ) ) NEW_LINE def getIntList ( ) : return [ int ( x ) for x in input ( ) . split ( ) ] NEW_LINE def dmp ( x ) : NEW_LINE INDENT global debug NEW_LINE if debug : NEW_LINE INDENT print ( x ) NEW_LINE DEDENT DEDENT def probC ( ) : NEW_LINE INDENT N , T = getIntList ( ) NEW_LINE Ts = getIntList ( ) NEW_LINE dmp ( ( N , T , Ts ) ) NEW_LINE total = Ts [ 0 ] NEW_LINE for i in range ( 1 , N ) : NEW_LINE INDENT total += min ( T , Ts [ i ] - Ts [ i - 1 ] ) NEW_LINE DEDENT return total + T NEW_LINE DEDENT debug = False NEW_LINE print ( probC ( ) ) NEW_LINE"
    code = revert_code(code)
    print(code)
    """
    codelines = code.split("NEW_LINE")
    new_codelines = []
    tab_shifts = 0
    for line in codelines:
        # INDENT
        right_tab = line.count("INDENT")
        if right_tab > 0:
            tab_shifts += right_tab
            line = line.replace("INDENT", "").strip(' ')
        # DEDENT
        left_tab = line.count("DEDENT")
        if left_tab > 0:
            tab_shifts -= left_tab
            line = line.replace("DEDENT", "").strip(' ')

        if tab_shifts > 0:
            line = "\t" * tab_shifts + line.strip(' ')
        new_codelines.append(line.strip(' '))
    code = '\n'.join(new_codelines)
    return code


if __name__ == '__main__':
    """
    This script is to flatten attributes of code_search_net dataset
            Examples: 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'func_name', 'original_string', 'index',
    """
    parser = argparse.ArgumentParser(description="Avatar dataset(s) or Tree-Sitter Library(ies)")
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
    parser.add_argument("--topk", "-k", default=5, type=int, )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    # for mode in MODES:
    #     for lang in LANGUAGES:
    #         # code tokens
    #         src_file = os.path.join(args.dataset_dir, f"{mode}.java-python.{lang}")
    #         tgt_file = os.path.join(args.flatten_dir, lang, f"{mode}.code_tokens")
    #         PathManager.mkdir(os.path.dirname(tgt_file))
    #         with file_io.open(src_file, 'r') as reader, file_io.open(tgt_file, 'w') as writer:
    #             for line in reader:
    #                 code_tokens = line.strip().split()
    #                 print(json_io.json_dumps(code_tokens), file=writer)
    #         # code
    #         tgt_file = os.path.join(args.flatten_dir, lang, f"{mode}.code")
    #         with file_io.open(src_file, 'r') as reader, file_io.open(tgt_file, 'w') as writer:
    #             for line in reader:
    #                 raw_code = revert_code(line) if lang == 'python' else line.strip(' ')
    #                 print(json_io.json_dumps(raw_code), file=writer)

    # problem-solution (topk5):
    #     train: java X python
    #     valid: 1 X 1
    #     tests: 1 X 1

    for mode in MODES:
        code_num = 0
        TOPK = 1 if mode != "train" else args.topk
        file = os.path.join(args.dataset_dir, f"{mode}.jsonl")

        id_file = os.path.join(args.flatten_dir, f"top{args.topk}", f"{mode}.id")
        print(id_file)

        jv_code = os.path.join(args.flatten_dir, f"top{args.topk}", 'java', f"{mode}.code")
        jv_raw_code = os.path.join(args.flatten_dir, f"top{args.topk}", 'java', f"{mode}.raw_code")
        PathManager.mkdir(os.path.dirname(jv_code))

        py_code = os.path.join(args.flatten_dir, f"top{args.topk}", 'python', f"{mode}.code")
        py_raw_code = os.path.join(args.flatten_dir, f"top{args.topk}", 'python', f"{mode}.raw_code")
        PathManager.mkdir(os.path.dirname(py_code))

        with file_io.open(file, 'r') as reader, file_io.open(id_file, 'w') as id_writer, \
            file_io.open(jv_code, 'w') as jv_code_writer, file_io.open(jv_raw_code, 'w') as jv_raw_writer, \
            file_io.open(py_code, 'w') as py_code_writer, file_io.open(py_raw_code, 'w') as py_raw_writer:
            for line in reader:
                line = json_io.json_loads(line)
                id, jv_codes, py_codes = line['id'], line['java'][:TOPK], line['python'][:TOPK]

                problem_index = []
                for jv_idx, jv_code in enumerate(jv_codes, start=0):
                    # java
                    jv_raw_code = jv_code.strip()
                    for py_idx, py_code in enumerate(py_codes, start=0):
                        # python
                        py_raw_code = revert_code(py_code)
                        py_code = py_code.strip()
                        # java
                        print(json_io.json_dumps(jv_raw_code), file=jv_raw_writer)
                        print(json_io.json_dumps(jv_code), file=jv_code_writer)
                        # print(jv_raw_code, file=jv_raw_writer)
                        # print(jv_code, file=jv_code_writer)
                        # python
                        print(json_io.json_dumps(py_raw_code), file=py_raw_writer)
                        print(json_io.json_dumps(py_code), file=py_code_writer)
                        # print(py_raw_code, file=py_raw_writer)
                        # print(py_code, file=py_code_writer)
                        problem_index.append([jv_idx, py_idx])
                print(json_io.json_dumps(problem_index), file=id_writer)
