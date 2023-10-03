# -*- coding: utf-8 -*-

import itertools
import os
from multiprocessing import Pool

import sentencepiece as spm
import torch

from preprocess.avatar.translation import (
    MODES,
    BPE_DIR,
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


def binarize(args, in_file: str, out_file: str, vocab, token_dict, offset: int, end: int):
    ds = indexed_dataset.make_builder(f"{out_file}.mmap", impl='mmap', vocab_size=len(vocab))
    with file_io.open(in_file, 'r') as reader:
        reader.seek(offset)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = json_io.json_loads(line)
            code_tokens = vocab.encode(line, out_type=str)
            code_tokens = torch.IntTensor([token_dict.index(token) for token in code_tokens])
            ds.add_item(code_tokens)
            line = reader.readline()
    ds.finalize(f'{out_file}.idx')


def main(args):
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])

    SPM_VOCAB_FILE = os.path.join(BPE_DIR, 'plbart', "sentencepiece.bpe.model")
    vocab = spm.SentencePieceProcessor()
    vocab.load(SPM_VOCAB_FILE)

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

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    num_workers = args['preprocess']['workers']
    src_lang, tgt_lang = args['preprocess']['src_lang'], args['preprocess']['tgt_lang']

    for lang, mode in itertools.product([src_lang, tgt_lang], MODES):
        # cp id
        src_id = args['preprocess'][f'{mode}pref'].replace('*', '') + ".id"
        tgt_id = os.path.join(args['preprocess']['destdir'], f"{mode}.id")
        PathManager.copy(src_id, tgt_id)

        src_file = args['preprocess'][f'{mode}pref'].replace('*', lang) + ".code"
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code_tokens")
        PathManager.mkdir(os.path.dirname(dst_file))

        offsets = find_offsets(src_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(dst_file, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        src_file,
                        prefix,
                        vocab,
                        token_dict,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                )
            pool.close()

        ds = indexed_dataset.make_builder(f"{dst_file}.mmap", impl='mmap', vocab_size=len(vocab))
        end = offsets[1]

        with file_io.open(src_file, 'r') as reader:
            reader.seek(0)
            line = file_io.safe_readline(reader)
            while line:
                if end > 0 and reader.tell() > end:
                    break
                line = json_io.json_loads(line)
                code_tokens = vocab.encode(line, out_type=str)
                code_tokens = torch.IntTensor([token_dict.index(token) for token in code_tokens])
                ds.add_item(code_tokens)
                line = reader.readline()

        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(dst_file, worker_id)
                ds.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        ds.finalize(f"{dst_file}.idx")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/topk5-o2o'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
