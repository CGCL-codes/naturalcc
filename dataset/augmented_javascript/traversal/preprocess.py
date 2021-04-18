#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import os
from multiprocessing import Pool
from ncc.utils.pathos_pool import PPool
from collections import (namedtuple, OrderedDict, Counter)
from ncc.data import (Dictionary, indexed_dataset)
from ncc.utils.util_file import load_yaml
from ncc import tasks
from ncc.data.tools.binarizer import Binarizer
from tqdm import tqdm
import sentencepiece as spm
from ncc import LOGGER
import pickle
import re
import itertools
import torch
import ujson
from dataset.csn.utils import util_traversal

_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        LOGGER.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class CombineDicionary(object):
    def __init__(self, spm_dict, token_dict):
        self.spm_dict = spm_dict
        self.token_dict = token_dict
        self.tokens = [self.spm_dict.id_to_piece(idx) for idx in range(len(self.spm_dict))]
        self.token2idx = {self.spm_dict.id_to_piece(idx): idx for idx in range(len(self.spm_dict))}
        for token, idx in self.token_dict.indices.items():
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
                self.tokens.append(token)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.nspecial = 9

    def encode_spm(self, string):
        string = normalize_program(string)
        tokens = self.spm_dict.EncodeAsPieces(string)
        return tokens

    def encode_node(self, string):
        tokens = [string]
        return tokens

    def tokens2ids(self, tokens):
        try:
            tokens = [self.token2idx[token] for token in tokens]
        except Exception as err:
            print(tokens)
            # print(self.token2idx)
            raise err
        tokens = torch.IntTensor(tokens)
        return tokens

    def save(self, file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as writer:
            for token in self.tokens[self.nspecial:]:
                print(ujson.dumps([token, 1], ensure_ascii=False), file=writer)

    def __len__(self):
        return len(self.tokens)


def get_dfs(ast, spm_dict, to_ids=True):
    """get token(namely, type node or value node) list of a ast"""
    dp = []
    for node in ast:
        if 'value' in node:
            string = node['value']
            id = spm_dict.piece_to_id(string)
            if spm_dict.unk_id() == id:
                # if id = [UNK], spm_dict does not contain such token, use BPE instead
                if to_ids:
                    tokens = spm_dict.encode_as_ids(string)
                else:
                    tokens = spm_dict.encode_as_pieces(string)
            else:
                # if id != [UNK], spm_dict does contain such token, directly view it as token
                if to_ids:
                    tokens = [id]
                else:
                    tokens = [string]
        else:
            string = node['type']
            if to_ids:
                tokens = [spm_dict.piece_to_id(string)]
            else:
                tokens = [string]
        dp.append(tokens)
    return dp


def dump_traversal(filename, combine_dict, consumer, offset, end):
    with open(filename, "r", encoding="utf-8") as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            line = ujson.loads(line)
            dfs_traversal = get_dfs(line, combine_dict)
            dfs_traversal = list(itertools.chain(*dfs_traversal))
            dfs_traversal = torch.IntTensor(dfs_traversal)
            consumer(dfs_traversal)
            line = safe_readline(f)


def binarize(args, filename: str, combine_dict, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], \
                                      vocab_size=len(combine_dict))

    def consumer(tensor):
        ds.add_item(tensor)

    dump_traversal(filename, combine_dict, consumer, offset, end)
    ds.finalize('{}.idx'.format(in_file))


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".json"

    target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    # load node/spm dictionary
    src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    src_sp = spm.SentencePieceProcessor()
    src_sp.load(args['preprocess']['src_sp'])

    for lang in args['preprocess']['source_lang']:
        src_dict.save_json(dict_path(lang))  # save spm dict to ncc.dictionary

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab, input_file, output_file, attr: str, num_workers: int):
        """make binary dataset"""
        # split a file into different parts
        # if use multi-processing, we first process 2nd to last file
        # 1.txt -> 10 processor, 0(p0)(0-99), 100(p1)(100-199), ...
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_file, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                )
            pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'],
                                          vocab_size=len(vocab))
        dump_traversal(input_file, vocab, consumer=lambda t: ds.add_item(t), offset=0, end=offsets[1])

        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(output_file, worker_id)
                ds.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        ds.finalize('{}.idx'.format(output_file))

    def make_dataset(vocab, sp, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            out_file = dest_path(output_prefix, lang=lang)
            offsets = Binarizer.find_offsets(input_prefix, num_workers)

            def _dump_traversal(in_file, sp, out_file, offset, end):
                with open(in_file, "r", encoding="utf-8") as reader, open(out_file, 'w', encoding="utf-8") as writer:
                    reader.seek(offset)
                    # next(f) breaks f.tell(), hence readline() must be used
                    line = safe_readline(reader)
                    while line:
                        if end > 0 and reader.tell() > end:
                            break
                        line = ujson.loads(line)
                        dfs_traversal = get_dfs(line, sp, to_ids=False)
                        dfs_traversal = list(itertools.chain(*dfs_traversal))
                        print(ujson.dumps(dfs_traversal, ensure_ascii=False), file=writer)
                        line = safe_readline(reader)

            with PPool(num_workers) as pool:
                params = []
                for worker_id in range(num_workers):
                    prefix = "{}{}".format(out_file, worker_id)
                    params.append((input_prefix,
                                   sp,
                                   prefix,
                                   offsets[worker_id],
                                   offsets[worker_id + 1],))
                pool.feed(_dump_traversal, params)

            def merge_file(src_files, tgt_file):
                import shutil
                with open(tgt_file, 'w', encoding='utf8') as writer:
                    for src_fl in src_files:
                        with open(src_fl, 'r', encoding='utf8') as reader:
                            shutil.copyfileobj(reader, writer)
                        os.remove(src_fl)

            merge_file(src_files=["{}{}".format(out_file, worker_id) for worker_id in range(num_workers)],
                       tgt_file=out_file)

        elif args['preprocess']['dataset_impl'] == "mmap":
            out_file = dest_path(output_prefix, lang=lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            make_binary_dataset(sp, input_prefix, out_file, lang, num_workers)

    def make_all(lang, vocab, sp):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, sp, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, sp, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, sp, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    for lang in args['preprocess']['source_lang']:
        make_all(lang, src_dict, src_sp)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing code_search_net dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", default='preprocess.traversal', type=str,
        help="load python_wan/tokenization/config/{yaml_file}.yml for train",
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), '../config', '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
