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
import ujson

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


def spm_encode(in_file, sp, out_file, start=0, end=0):
    with open(in_file, 'r', encoding='utf8') as reader, open(out_file, 'w', encoding='utf8') as writer:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break

            example = ujson.loads(line)
            program = normalize_program(example)
            program = sp.EncodeAsPieces(program)
            print(ujson.dumps(program, ensure_ascii=False), file=writer)

            line = safe_readline(reader)


def tokenizer(sp):
    def _tokenizer(program):
        program = normalize_program(program)
        program = sp.EncodeAsPieces(program)

    return _tokenizer


def binarize(args, filename: str, dict: Dictionary, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_bpe(filename, dict=None, sp=dict, consumer=consumer, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


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
        return dest_path("dict", lang) + ".txt"

    target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    if args['preprocess']['joined_dictionary']:
        assert not args['preprocess']['srcdict'] or not args['preprocess']['tgtdict'], \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        elif args['preprocess']['tgtdict']:
            src_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        else:
            LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
            exit()

        tgt_dict = src_dict

        # Load sentencepiece (sp) module
        if args['preprocess']['src_sp']:
            src_sp = spm.SentencePieceProcessor()
            src_sp.load(args['preprocess']['src_sp'])
        elif args['preprocess']['tgt_sp']:
            src_sp = spm.SentencePieceProcessor()
            src_sp.load(args['preprocess']['tgt_sp'])
        else:
            LOGGER.error('Please assign the sentencepiece model path.')
            exit()
        tgt_sp = src_sp

    else:
        if args['preprocess']['srcdict'] and args['preprocess']['src_sp']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
            src_sp = spm.SentencePieceProcessor()
            src_sp.load(args['preprocess']['src_sp'])
        else:
            LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
            exit()

        if target:
            if args['preprocess']['tgtdict'] and args['preprocess']['tgt_sp']:
                tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
                tgt_sp = spm.SentencePieceProcessor()
                tgt_sp.load(args['preprocess']['tgt_sp'])
            else:
                # assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
                # tgt_dict = build_dictionary([train_path(args['preprocess']['target_lang'])], tgt=True)
                LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
                exit()
        else:
            tgt_dict = None
            tgt_sp = None
    for lang in args['preprocess']['source_lang']:
        src_dict.save_json(dict_path(lang))  # save spm dict to ncc.dictionary

    # 2. ***************build dataset********************
    def make_binary_dataset(sp, input_file, output_file,
                            attr: str, num_workers: int):
        """make binary dataset"""
        n_seq_tok = [0, 0]
        replaced = Counter()  # save un-recorded tokens

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

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
                        sp,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(sp))
        merge_result(
            Binarizer.binarize_bpe(
                input_file, dict=None, sp=sp, consumer=lambda t: ds.add_item(t),
                offset=0, end=offsets[1],
            )
        )
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

        LOGGER.info(
            "[{}] {}: {} sents, {} tokens, BPE node replacement".format(
                attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
            )
        )

    def make_dataset(vocab, sp, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            out_file = dest_path(output_prefix, lang=lang)
            offsets = Binarizer.find_offsets(input_prefix, num_workers)

            with PPool(num_workers) as pool:
                params = []
                for worker_id in range(num_workers):
                    prefix = "{}{}".format(out_file, worker_id)
                    params.append((input_prefix,
                                   sp,
                                   prefix,
                                   offsets[worker_id],
                                   offsets[worker_id + 1],))
                pool.feed(spm_encode, params)

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
    if target:
        for lang in args['preprocess']['target_lang']:
            make_all(lang, tgt_dict, tgt_sp)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing code_search_net dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", default='preprocess', type=str,
        help="load python_wan/tokenization/config/{yaml_file}.yml for train",
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
