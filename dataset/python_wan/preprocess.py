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
import shutil
from multiprocessing import Pool
from collections import Counter
from ncc.data import (
    Dictionary,
    indexed_dataset,
)
from ncc.data.tools.binarizer import Binarizer
from ncc import (
    LOGGER, tasks
)
from ncc.utils.yaml import load_yaml
from ncc.data.tokenizers.tokenizer_funcs import list_tokenizer


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=list_tokenizer,
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def valid_path(lang):
        return "{}{}".format(args['preprocess']['validpref'], ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path(lang, "dict") + ".json"

    target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    if args['preprocess']['only_train']:
        LOGGER.info('Generating dictionaries with Train data files.')
    else:
        LOGGER.info('Generating dictionaries with Train/Validation data files.')

    if args['preprocess']['joined_dictionary']:
        assert not args['preprocess']['srcdict'] or not args['preprocess']['tgtdict'], \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        elif args['preprocess']['tgtdict']:
            src_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
            filenames = [train_path(args['preprocess']['source_lang']), train_path(args['preprocess']['target_lang'])]
            if not args['preprocess']['only_train']:
                filenames.extend( \
                    [valid_path(args['preprocess']['source_lang']), valid_path(args['preprocess']['target_lang'])])
            src_dict = task.build_dictionary(
                filenames,
                tokenize_func=list_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['thresholdsrc'],
                # set max len for joint dictionaries
                nwords=max(args['preprocess']['nwordssrc'], args['preprocess']['nwordstgt']),
                # padding_factor=args['preprocess']['padding_factor'],
            )
        tgt_dict = src_dict

    else:
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

            filenames = [train_path(args['preprocess']['source_lang'])]
            if not args['preprocess']['only_train']:
                filenames.append(valid_path(args['preprocess']['source_lang']))
            src_dict = task.build_dictionary(
                filenames,
                tokenize_func=list_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['thresholdsrc'],
                nwords=args['preprocess']['nwordssrc'],
                # padding_factor=args['preprocess']['padding_factor'],
            )
        if target:
            if args['preprocess']['tgtdict']:
                tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
            else:
                assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
                filenames = [train_path(args['preprocess']['target_lang'])]
                if not args['preprocess']['only_train']:
                    filenames.append(valid_path(args['preprocess']['target_lang']))
                tgt_dict = task.build_dictionary(
                    filenames,
                    tokenize_func=list_tokenizer,
                    workers=args['preprocess']['workers'],
                    threshold=args['preprocess']['thresholdsrc'],
                    nwords=args['preprocess']['nwordstgt'],
                    # padding_factor=args['preprocess']['padding_factor'],
                )
        else:
            tgt_dict = None

    src_dict.save_json(dict_path(args['preprocess']['source_lang']))  # save spm dict to ncc.dictionary
    if target and tgt_dict is not None:
        tgt_dict.save_json(dict_path(args['preprocess']['target_lang']))

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab: Dictionary, input_file, output_file, num_workers: int):
        """make binary dataset"""
        # LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab) - 1))
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
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=list_tokenizer, offset=0, end=offsets[1], append_eos=False,
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
            "{}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                # attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            in_file = file_name(input_prefix, lang)
            out_dir = args['preprocess']['destdir']
            os.makedirs(out_dir, exist_ok=True)
            LOGGER.info('Copying {} into {}'.format(in_file, out_dir))
            shutil.copy(src=in_file, dst=args['preprocess']['destdir'])
        else:
            in_file = file_name(input_prefix, lang)
            out_file = dest_path(output_prefix, lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            make_binary_dataset(vocab, in_file, out_file, num_workers)

    def make_all(lang, vocab):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    # for attr in args['preprocess']['source_lang']:
    make_all(args['preprocess']['source_lang'], src_dict)
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/preprocess',
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
