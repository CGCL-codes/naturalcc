# !/usr/bin/env python3
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
from ncc import tasks
from collections import Counter
from ncc.data import (
    Dictionary,
    indexed_dataset,
)
from ncc.data.retrieval.retrieval_binarizer import RetrievalBinarizer as Binarizer
from ncc.utils.yaml import load_yaml
from ncc import LOGGER
from ncc.data.retrieval import tokenizers


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenizers.sub_tokenizer,
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

    def bpe_dict_path(lang):
        return dest_path(lang + '.bpe', "dict") + ".json"

    target = not args['preprocess']['only_source']

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        src_dict = task.build_dictionary(
            [train_path(args['preprocess']['source_lang'])],
            tokenize_func=tokenizers.sub_tokenizer,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'],
        )
    if target:
        if args['preprocess']['tgtdict']:
            tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
            tgt_dict = task.build_bpe_dictionary(
                [train_path(args['preprocess']['target_lang'])],
                tokenize_func=tokenizers.lower_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=0,
                nwords=args['preprocess']['nwordstgt'],
            )
    else:
        tgt_dict = None

    if args['preprocess']['source_bpe']:
        src_dict_name = bpe_dict_path(args['preprocess']['source_lang'])
    else:
        src_dict_name = dict_path(args['preprocess']['source_lang'])
    src_dict.save_json(src_dict_name)
    if args['preprocess']['target_bpe']:
        tgt_dict_name = bpe_dict_path(args['preprocess']['target_lang'])
    else:
        tgt_dict_name = dict_path(args['preprocess']['target_lang'])
    tgt_dict.save_json(tgt_dict_name)

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
        # if num_workers > 1:
        #     # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
        #     pool = Pool(processes=num_workers - 1)
        #     for worker_id in range(1, num_workers):
        #         prefix = "{}{}".format(output_file, worker_id)
        #         pool.apply_async(
        #             binarize,
        #             (
        #                 args,
        #                 input_file,
        #                 vocab,
        #                 prefix,
        #                 offsets[worker_id],
        #                 offsets[worker_id + 1]
        #             ),
        #             callback=merge_result
        #         )
        #     pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
        if 'code_tokens_wo_func' in os.path.basename(output_file):
            bin_out = Binarizer.binarize_wo_func(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=tokenizers.string_sub_tokenizer, offset=0, end=offsets[1], append_eos=False,
            )
        elif 'code_tokens' in os.path.basename(input_file):
            bin_out = Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=tokenizers.sub_tokenizer, offset=0, end=offsets[1], append_eos=False,
            )
        elif 'docstring_tokens' in os.path.basename(input_file):
            bin_out = Binarizer.binarize_bpe(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=tokenizers.lower_tokenizer, offset=0, end=offsets[1], append_eos=False,
            )
        elif 'func_name' in os.path.basename(input_file):
            bin_out = Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=tokenizers.func_name_tokenizer, offset=0, end=offsets[1], append_eos=False,
            )

        merge_result(bin_out)
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, is_bpe=False):
        if args['preprocess']['dataset_impl'] == "raw":
            in_file = file_name(input_prefix, lang)
            out_dir = args['preprocess']['destdir']
            os.makedirs(out_dir, exist_ok=True)
            LOGGER.info('Copying {} into {}'.format(in_file, out_dir))
            shutil.copy(src=in_file, dst=args['preprocess']['destdir'])
        else:
            if lang == 'code_tokens_wo_func':
                in_file = file_name(input_prefix, 'code_tokens')
                out_file = dest_path(output_prefix, lang)
                num_workers = 1  # not support multi-processing
            else:
                in_file = file_name(input_prefix, lang)
                out_file = dest_path(output_prefix, lang)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            if is_bpe:
                out_file += '.bpe'
            make_binary_dataset(vocab, in_file, out_file, num_workers)

    def make_all(lang, vocab, is_bpe=False):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'], is_bpe=is_bpe)
        if args['preprocess']['validpref']:
            make_dataset(vocab, args['preprocess']['validpref'], "valid", lang,
                         num_workers=args['preprocess']['workers'], is_bpe=is_bpe)
        if args['preprocess']['testpref']:
            make_dataset(vocab, args['preprocess']['testpref'], "test", lang, num_workers=args['preprocess']['workers'],
                         is_bpe=is_bpe)

    make_all(args['preprocess']['source_lang'], src_dict, is_bpe=args['preprocess']['source_bpe'])
    make_all("code_tokens_wo_func", src_dict, is_bpe=args['preprocess']['source_bpe'])  # func_name
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict, is_bpe=args['preprocess']['target_bpe'])
        make_all("func_name", tgt_dict, is_bpe=args['preprocess']['target_bpe'])  # func_name


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generating raw/bin CodeSearchNet dataset(s) for retrieval task")
    parser.add_argument(
        "--yaml_file", "-f", default='config/ruby', type=str,
        help="load {yaml_file}.yml for train",
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
