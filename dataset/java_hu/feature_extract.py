# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import random
import re
from multiprocessing import Pool, cpu_count

from dataset.java_hu import (
    LANGUAGES, LIBS_DIR, ATTRIBUTES_DIR,
    LOGGER,
    MAX_SUB_TOKEN_LEN, MODES
)
from dataset.python_wan.parser._parser import CodeParser
from dataset.python_wan.utils import (
    util_ast,
    util_path,
    util_traversal,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.file_ops.file_io import (
    safe_readline,
    find_offsets,
)
from ncc.utils.path_manager import PathManager


class AttrFns:
    """build your defined function for attributes"""

    @staticmethod
    def code_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        PathManager.mkdir(os.path.dirname(dest_filename))
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code = json_io.json_loads(line)
                print(json_io.json_dumps(code), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def docstring_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        PathManager.mkdir(os.path.dirname(dest_filename))
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                docstring = json_io.json_loads(line)
                print(json_io.json_dumps(docstring), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def code_tokens_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        PathManager.mkdir(os.path.dirname(dest_filename))
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code_tokens = json_io.json_loads(line)
                if code_tokens:
                    # filter comment in code_tokens, eg. //***\n /* */\n
                    code_tokens = [token for token in code_tokens
                                   if not (str.startswith(token, '//') or str.startswith(token, '#') or \
                                           (str.startswith(token, '/*') and str.endswith(token, '*/')))
                                   ]

                    if not all(str.isascii(token) for token in code_tokens):
                        code_tokens = None
                    if code_tokens is None or len(code_tokens) < 1:
                        code_tokens = None
                else:
                    code_tokens = None

                print(json_io.json_dumps(code_tokens), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def docstring_tokens_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        PathManager.mkdir(os.path.dirname(dest_filename))
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                docstring_tokens = json_io.json_loads(line)
                if docstring_tokens:
                    docstring_tokens = [
                        token for token in docstring_tokens \
                        if not (re.match(r'[\-|\*|\=|\~]{2,}', token) or re.match(r'<.*?>', token))
                    ]
                    if not all(str.isascii(token) for token in docstring_tokens):
                        docstring_tokens = None
                    if (docstring_tokens is None) or not (3 < len(docstring_tokens) <= 50):
                        docstring_tokens = None
                else:
                    docstring_tokens = None
                print(json_io.json_dumps(docstring_tokens), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def raw_ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        """code => raw_ast"""
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing
        lang = kwargs.get('lang')
        so_dir = kwargs.get('so_dir')

        so_filename = os.path.join(os.path.expanduser(so_dir), '{}.so'.format(lang))
        parser = CodeParser(so_filename, lang)
        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                code = json_io.json_loads(line)
                if code:
                    raw_ast = parser.parse_raw_ast(code)
                else:
                    raw_ast = None
                print(json_io.json_dumps(raw_ast), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                raw_ast = json_io.json_loads(line)
                if raw_ast:
                    ast = util_ast.convert(raw_ast)
                else:
                    ast = None
                print(json_io.json_dumps(ast), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def path_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast:
                    paths = util_path.ast_to_path(ast)
                    print(json_io.json_dumps(paths), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbt_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast:
                    ast = util_ast.value2children(ast)
                    padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                    root_idx = util_ast.get_root_idx(padded_ast)
                    sbt = util_ast.build_sbt_tree(padded_ast, idx=root_idx)
                else:
                    sbt = None
                print(json_io.json_dumps(sbt), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def sbtao_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast:
                    ast = util_ast.value2children(ast)
                    padded_ast = util_ast.pad_leaf_node(ast, MAX_SUB_TOKEN_LEN)
                    root_idx = util_ast.get_root_idx(padded_ast)
                    sbt = util_ast.build_sbtao_tree(padded_ast, idx=root_idx)
                else:
                    sbt = None
                print(json_io.json_dumps(sbt), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def binary_ast_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast:
                    try:
                        ast = util_ast.value2children(ast)
                        ast = util_ast.remove_root_with_uni_child(ast)
                        root_idx = util_ast.get_root_idx(ast)
                        ast = util_ast.delete_node_with_uni_child(ast, idx=root_idx)
                        root_idx = util_ast.get_root_idx(ast)
                        bin_ast = util_ast.binarize_tree(ast, idx=root_idx)  # to binary ast tree
                        root_idx = util_ast.get_root_idx(ast)
                        bin_ast = util_ast.reset_indices(bin_ast, root_idx)  # reset node indices
                        bin_ast = util_ast.pad_leaf_node(bin_ast, MAX_SUB_TOKEN_LEN)
                    except RecursionError:
                        LOGGER.error('RecursionError, ignore this tree')
                        bin_ast = None
                    except Exception as err:
                        LOGGER.error(err)
                        bin_ast = None
                else:
                    bin_ast = None
                print(json_io.json_dumps(bin_ast), file=writer)
                line = safe_readline(reader)

    @staticmethod
    def traversal_fn(filename, dest_filename, idx, start=0, end=-1, *args):
        kwargs = args[0][0]  # canot feed dict parameters in multi-processing

        dest_filename = dest_filename + str(idx)
        with file_io.open(filename, "r") as reader, file_io.open(dest_filename, 'w') as writer:
            reader.seek(start)
            line = safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                ast = json_io.json_loads(line)
                if ast:
                    ast_traversal = util_traversal.get_dfs(ast)
                else:
                    ast_traversal = None
                print(json_io.json_dumps(ast_traversal), file=writer)
                line = safe_readline(reader)


def process(src_filename, tgt_filename, num_workers=cpu_count(), **kwargs):
    modality = tgt_filename.split('.')[-1]
    attr_fn = getattr(AttrFns, '{}_fn'.format(modality))
    offsets = find_offsets(src_filename, num_workers)

    # # for debug
    # idx = 0
    # attr_fn(src_filename, tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])

    with Pool(num_workers) as mpool:
        result = [
            mpool.apply_async(
                attr_fn,
                (src_filename, tgt_filename, idx, offsets[idx], offsets[idx + 1], [kwargs])
            )
            for idx in range(num_workers)
        ]
        result = [res.get() for res in result]

    def _cat_and_remove(tgt_filename, num_workers):
        with file_io.open(tgt_filename, 'w') as writer:
            for idx in range(num_workers):
                src_filename = tgt_filename + str(idx)
                with file_io.open(src_filename, 'r') as reader:
                    PathManager.copyfileobj(reader, writer)
                PathManager.rm(src_filename)

    _cat_and_remove(tgt_filename, num_workers)


if __name__ == '__main__':
    """
    This script is to generate new attributes of code snippet.
    """
    parser = argparse.ArgumentParser(description="Download python_wan dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--languages", "-l", default=LANGUAGES, type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    )
    parser.add_argument(
        "--attributes_dir", "-d", default=ATTRIBUTES_DIR, type=str, help="data directory of attributes",
    )
    parser.add_argument(
        "--so_dir", "-s", default=LIBS_DIR, type=str, help="library directory of so file",
    )
    parser.add_argument(
        "--attrs", "-a",
        default=['raw_ast', 'ast', ],
        type=str, nargs='+', help="attrs: raw_ast, ...",
    )
    parser.add_argument(
        "--cores", "-c", default=cpu_count(), type=int, help="cpu cores for flatten raw data attributes",
    )
    args = parser.parse_args()
    # print(args)

    """
    a mapping to generate new attributes of code snippet.
    Examples:
        "raw_ast" <= "code",    # raw_ast, an AST contains all info of a code, e.g. comment, single root node, ...
        "ast" <= "raw_ast",     # ast, saving leaf nodes into "value" nodes and non-leaf nodes into "children" nodes
        "path" <= "ast",        # path, a path from a leaf node to another leaf node 
        "sbt" <= "raw_ast",     # sbt, a depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "sbtao" <= "sbt'",       # sbtao, an improved depth first traversal path of an AST, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "binary_ast" <= "raw_ast", # bin_ast, an sophisticated binary AST, remove nodes with single child, tokenize leaf node and padding with <PAD>(for DGL Lib.)
        "traversal" <= "ast",   #
    """

    dest_raw_attrs = {
        'new.code_tokens': 'code_tokens',
        'new.docstring_tokens': 'docstring_tokens',
        'raw_ast': 'code',
        'ast': 'raw_ast',
        'path': 'ast',
        'sbt': 'raw_ast',
        'sbtao': 'raw_ast',
        'binary_ast': 'raw_ast',
        'traversal': 'ast',
    }

    for lang, mode in itertools.product(args.languages, MODES):
        for tgt_attr in args.attrs:
            src_attr = dest_raw_attrs[tgt_attr]
            src_filename = os.path.join(args.attributes_dir, '{}.{}'.format(mode, src_attr))
            if os.path.exists(src_filename):
                tgt_filename = os.path.join(args.attributes_dir, '{}.{}'.format(mode, tgt_attr))
                LOGGER.info('Generating {}'.format(tgt_filename))
                process(src_filename, tgt_filename, num_workers=args.cores, lang=lang, so_dir=args.so_dir)
            else:
                LOGGER.info('{} does exist'.format(src_filename))
