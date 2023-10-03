# -*- coding: utf-8 -*-

import itertools
import os
from collections import Counter

import torch

from preprocess.clcdsa import (
    MODES,
    ATTRIBUTES_DIR,
)
from ncc import LOGGER
from ncc.data import constants
from ncc.data import indexed_dataset
from ncc.data.dictionary import (
    Dictionary,
)
from ncc.data.dictionary import (
    TransformersDictionary,
)
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager

import sentencepiece as spm
from preprocess.clcdsa.plbart import (
    SPM_VOCAB_FILE,
)


def bfs_to_dfs(bfs_ast):
    dfs_ast = []

    def dfs(idx, parent_idx):
        node = bfs_ast[str(idx)]
        if 'value' in node:
            dfs_ast[parent_idx]['children'].append(len(dfs_ast))
            dfs_ast.append({"type": node['type'], 'parent': parent_idx, 'value': node['value']})
        else:
            if parent_idx is not None:
                dfs_ast[parent_idx]['children'].append(len(dfs_ast))
            dfs_ast.append({"type": node['type'], 'parent': parent_idx, 'children': []})
            parent_idx = len(dfs_ast) - 1
            for child in node['children']:
                dfs(child, parent_idx)

    dfs(idx=0, parent_idx=None)
    return dfs_ast


def main(args):
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])

    vocab = spm.SentencePieceProcessor()
    vocab.load(SPM_VOCAB_FILE)

    def tokenization(tokens):
        for idx, tok in enumerate(tokens):
            if len(tok) != 0:
                tokens[idx] = vocab.encode(tok, out_type=str)
        return tokens

    def ast_to_graph(ast):
        nodes, tokens, adjacence = [], [], [[] for _ in range(len(ast))]
        for idx, node in enumerate(ast):
            nodes.append(node['type'])
            if 'children' in node:
                tokens.append([])
                for child in node['children']:
                    adjacence[idx].append(child)
                    adjacence[child].append(idx)
            elif 'value' in node:
                tokens.append(node['value'])
            else:
                raise NotImplementedError

        tokens = tokenization(tokens)

        depth = {0: 1}  # 0 for pad
        for idx, node in enumerate(ast[1:], start=1):
            depth[idx] = depth[node['parent']] + 1
        depth = list(depth.values())

        assert len(nodes) == len(tokens) == len(adjacence) == len(depth)
        return nodes, tokens, adjacence, depth

    def save_token_dict():
        src_file = os.path.join(os.path.dirname(SPM_VOCAB_FILE), 'dict.txt')
        tgt_file = os.path.join(args['preprocess']['destdir'], 'dict.jsonl')
        # Dictionary.text_to_jsonl(src_file, tgt_file)
        vocab = Dictionary()
        with file_io.open(src_file, 'r') as reader:
            for line in reader:
                token, num = line.strip().split()
                vocab.add_symbol(token, eval(num))
        vocab.save(tgt_file)
        return vocab

    token_dict = save_token_dict()

    def save_node_dict():
        src_file = PathManager.expanduser("~/clcdsa/astbert/data-mmap/node.jsonl")
        dict = Dictionary.load(src_file)
        tgt_file = os.path.join(args['preprocess']['destdir'], 'node.jsonl')
        PathManager.mkdir(os.path.dirname(tgt_file))
        dict.save(tgt_file)
        return dict

    node_dict = save_node_dict()

    def save_lang_dict():
        src_file = PathManager.expanduser("~/clcdsa/astbert/data-mmap/lang.jsonl")
        dict = Dictionary.load(src_file)
        tgt_file = os.path.join(args['preprocess']['destdir'], 'lang.jsonl')
        PathManager.mkdir(os.path.dirname(tgt_file))
        dict.save(tgt_file)
        return dict

    lang_dict = save_lang_dict()

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    lang = args['preprocess']['lang']
    for mode in MODES:
        src_file = f"{args['preprocess'][f'{mode}pref']}.ast"

        node_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.node")
        PathManager.mkdir(os.path.dirname(node_file))
        node_dataset = indexed_dataset.make_builder(f"{node_file}.mmap", impl='mmap', vocab_size=len(node_dict))

        depth_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.depth")
        depth_dataset = indexed_dataset.make_builder(f"{depth_file}.mmap", impl='mmap')

        code_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code")
        code_dataset = indexed_dataset.make_builder(f"{code_file}.bin", impl='bin', dtype=str)

        adjacence_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.adjacence")
        adjacence_dataset = indexed_dataset.make_builder(f"{adjacence_file}.bin", impl='bin')

        code_tokens_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code_tokens")
        code_tokens_dataset = indexed_dataset.make_builder(f"{code_tokens_file}.bin", impl='bin')

        with file_io.open(src_file, 'r') as reader:
            for idx, line in enumerate(reader):
                line = json_io.json_loads(line)
                ast = bfs_to_dfs(line)
                nodes, tokens, adjacence, depth = ast_to_graph(ast)
                # save node into mmap dataset
                nodes = torch.IntTensor([node_dict.index(tok) for tok in nodes])
                node_dataset.add_item(nodes)
                # save depth into mmap dataset
                depth = torch.IntTensor(depth)
                depth_dataset.add_item(depth)
                # code
                code = ''.join(itertools.chain(*tokens)).replace(constants.SPM_SPACE, ' ').strip()
                code_dataset.add_item(code)
                # tokens
                tokens = [[token_dict.index(tok) for tok in toks] if len(toks) > 0 else [] for toks in tokens]
                code_tokens_dataset.add_item(tokens)
                # adjacence
                for adj in adjacence:
                    assert adj == sorted(adj)
                adjacence_dataset.add_item(adjacence)

        node_dataset.finalize(f"{node_file}.idx")
        depth_dataset.finalize(f"{depth_file}.idx")
        code_dataset.finalize(f"{code_file}.idx")
        code_tokens_dataset.finalize(f"{code_tokens_file}.idx")
        adjacence_dataset.finalize(f"{adjacence_file}.idx")

        # proj indices
        with file_io.open(f"{args['preprocess'][f'{mode}pref']}.proj", 'r') as reader:
            projs = [json_io.json_loads(line) for line in reader]
        proj_indices = Counter(projs)
        proj_indices = [proj_num for idx, proj_num in proj_indices.items()]
        proj_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.proj")
        proj_dataset = indexed_dataset.make_builder(f"{proj_file}.seq", impl='seq')
        proj_dataset.add_item(torch.IntTensor(proj_indices))
        proj_dataset.finalize(f"{proj_file}.idx")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/cpp'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
