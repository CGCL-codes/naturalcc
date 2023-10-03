# -*- coding: utf-8 -*-

import itertools
import os
from multiprocessing import Pool

import torch

from preprocess.codexglue.code_to_code.translation import (
    DATASET_DIR,
    MODES,
)
from ncc import LOGGER
from ncc.data import indexed_dataset
from ncc.data.dictionary import (
    Dictionary,
)
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.file_io import find_offsets
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager


def binarize_tokens(args, filename: str, dict, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    with file_io.open(filename, 'r') as reader:
        reader.seek(offset)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = json_io.json_loads(line)
            code_tokens = dict.subtokenize(line)
            code_tokens = torch.IntTensor(dict.tokens_to_indices(code_tokens))
            ds.add_item(code_tokens)
            line = reader.readline()
    ds.finalize('{}.idx'.format(in_file))


def binarize_dfs(args, filename: str, dict, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    with file_io.open(filename, 'r') as reader:
        reader.seek(offset)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = json_io.json_loads(line)
            dfs = torch.IntTensor([dict.index(tok) for tok in line])
            ds.add_item(dfs)
            line = reader.readline()
    ds.finalize('{}.idx'.format(in_file))


def main(args):
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])

    from ncc.data.dictionary import TransformersDictionary
    vocab = TransformersDictionary.from_pretrained('microsoft/graphcodebert-base')

    file = os.path.join(args['preprocess']['destdir'], 'dfs.jsonl')
    node_dict = Dictionary.load(file)

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    num_workers = args['preprocess']['workers']
    src_lang, tgt_lang = args['preprocess']['src_lang'], args['preprocess']['tgt_lang']

    # code tokens => code tokens
    for mode, lang in itertools.product(MODES, [src_lang, tgt_lang]):
        data_dir = str.replace(args['preprocess'][f'{mode}pref'], '*', lang)
        src_file = f"{data_dir}.code_tokens"
        PathManager.mkdir(os.path.join(args['preprocess']['destdir'], lang))
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code_tokens")
        offsets = find_offsets(src_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(dst_file, worker_id)
                pool.apply_async(
                    binarize_tokens,
                    (
                        args,
                        src_file,
                        vocab,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                )
            pool.close()

        ds_file = '{}.mmap'.format(dst_file)
        ds = indexed_dataset.make_builder(ds_file, impl="mmap", vocab_size=len(vocab))
        end = offsets[1]
        with file_io.open(src_file, 'r') as reader:
            reader.seek(0)
            line = file_io.safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                line = json_io.json_loads(line)
                code_tokens = vocab.subtokenize(line)
                code_tokens = torch.IntTensor(vocab.tokens_to_indices(code_tokens))
                ds.add_item(code_tokens)
                line = reader.readline()

        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(dst_file, worker_id)
                ds.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        ds.finalize('{}.idx'.format(dst_file))

    # code => code
    for mode, lang in itertools.product(MODES, [src_lang, tgt_lang]):
        data_dir = str.replace(args['preprocess'][f'{mode}pref'], '*', lang)
        src_file = f"{data_dir}.code"
        PathManager.mkdir(os.path.join(args['preprocess']['destdir'], lang))
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code")

        ds_file = '{}.bin'.format(dst_file)
        ds = indexed_dataset.make_builder(ds_file, impl="bin", vocab_size=len(vocab))
        with open(src_file, 'r') as reader:
            for line in reader:
                line = json_io.json_loads(line)
                ds.add_item(line)
        ds.finalize('{}.idx'.format(dst_file))

    # dfs => dfs
    for mode, lang in itertools.product(MODES, [src_lang, tgt_lang]):
        data_dir = str.replace(args['preprocess'][f'{mode}pref'], '*', lang)
        src_file = f"{data_dir}.dfs"
        PathManager.mkdir(os.path.join(args['preprocess']['destdir'], lang))
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.dfs")
        offsets = find_offsets(src_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(dst_file, worker_id)
                pool.apply_async(
                    binarize_dfs,
                    (
                        args,
                        src_file,
                        node_dict,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                )
            pool.close()

        ds_file = '{}.mmap'.format(dst_file)
        ds = indexed_dataset.make_builder(ds_file, impl="mmap", vocab_size=len(vocab))
        end = offsets[1]
        with file_io.open(src_file, 'r') as reader:
            reader.seek(0)
            line = file_io.safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                line = json_io.json_loads(line)
                dfs = torch.IntTensor([node_dict.index(tok) for tok in line])
                ds.add_item(dfs)
                line = reader.readline()

        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(dst_file, worker_id)
                ds.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        ds.finalize('{}.idx'.format(dst_file))


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/top1'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
