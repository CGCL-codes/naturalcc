#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
from typing import Dict
import os
from multiprocessing import Pool
from collections import (namedtuple, Counter)
from ncc.data import (Dictionary, indexed_dataset)
from ncc.utils.util_file import load_yaml
from ncc import tasks
from ncc.data.tools.binarizer import Binarizer
from dataset.csn.utils.util import normalize_program
import sentencepiece as spm
import ujson
from ncc import LOGGER


def binarize(args: Dict, filename: str, dict: Dictionary, out_file_prefix: str, attr: str,
             offset: int, end: int):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(out_file_prefix)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_bpe(filename, dict, consumer, offset=offset, end=end)
    ds.finalize('{}.idx'.format(out_file_prefix))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

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

    # 1. build vocabulary from bpe directory
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
    # exit()
    # 2. ***************build dataset********************
    def make_binary_dataset(vocab: Dictionary, input_file, output_file,
                            attr: str, num_workers: int):
        """make binary dataset"""
        LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab) - 1))
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
                        vocab,
                        prefix,
                        attr,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
        merge_result(
            Binarizer.binarize_bpe(
                input_file, vocab, lambda t: ds.add_item(t), offset=0, end=offsets[1]
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
            "[{}] {}: {} sents, {} tokens, BPE no replaced token".format(
                attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
            )
        )

    def make_dataset(vocab, sp, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == 'raw':
            with open(file_name(input_prefix, lang), 'rb') as input_file, open(dest_path(output_prefix, lang), 'w', encoding="utf-8") as output_file:
                for line in input_file.readlines()[0: 100]:  # TODO only for debug
                    line = ujson.loads(line)
                    line = normalize_program(line)
                    line = sp.EncodeAsPieces(line)
                    output_file.write(ujson.dumps(line) + '\n')
        else:
            in_file = file_name(input_prefix, lang)
            out_file = dest_path(output_prefix, lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            make_binary_dataset(vocab, in_file, out_file, lang, num_workers)

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

    # # 2. build dataset
    make_all(args['preprocess']['source_lang'], src_dict, src_sp)
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict, tgt_sp)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_codebert.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
