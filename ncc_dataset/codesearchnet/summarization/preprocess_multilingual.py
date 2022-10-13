#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import itertools
import logging
import os
import shutil
from collections import (
    Counter,
    OrderedDict,
)
from collections import namedtuple
from multiprocessing import Pool, cpu_count
from typing import Dict

import ujson
from dgl.data.utils import save_graphs

from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    Dictionary,
    indexed_dataset,
)
from ncc.data.tools.binarizer import Binarizer
from ncc.tokenizers import tokenization
from ncc.utils import graph_utils
from ncc.utils.file_ops.yaml_io import load_yaml

logger = logging.getLogger(__name__)


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str, attr: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenization.json_tokenizer,
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def binarize_dgl(args: Dict, filename: str, dict: Dictionary, in_file: str, offset: int, end: int):
    """binarize function for multi-processing"""
    graphes = []
    with open(filename, "r", encoding="utf-8") as reader:
        reader.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            ast = ujson.loads(line)
            graph = graph_utils.build_graph(ast, dict)
            graphes.append(graph)
            line = reader.readline()
    save_graphs(in_file + '.bin', graphes)


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path(modality):
        train_files = []
        for lang, value in args['preprocess']['dataprefs'].items():
            train_files.append(
                "{}{}".format(args['preprocess']['dataprefs'][lang]['trainpref'], ("." + modality) if modality else "")
            )
        return train_files

    def build_dictionary(filenames, modality, src=False, tgt=False):
        """
        ['code_tokens', 'docstring_tokens', 'path', 'sbt', 'sbtao', 'binary_ast', 'traversal']
        """
        assert src ^ tgt
        if modality in ['binary_ast']:
            tokenize_func = tokenization.json_tokenizer
        elif modality in ['code_tokens', 'docstring_tokens', 'path', 'path.terminals', 'sbt', 'sbtao', 'traversal']:
            tokenize_func = tokenization.json_tokenizer
        else:
            raise NotImplementedError("{}".format(modality))

        return task.build_dictionary(
            filenames,
            tokenize_func=tokenize_func,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'] if src else args['preprocess']['nwordstgt'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    def build_vocab_dict(args):
        """Build vocabulary (dictionary) for source and target domain"""
        LOGGER.info('Build vocabularies...')
        # task = tasks.get_task(args['preprocess']['task'])
        src_dicts = OrderedDict()

        def load_dict(modality):
            modality_dict_filename = os.path.join(
                args['preprocess']['destdir'], 'data-{}'.format(args['preprocess']['dataset_impl']),
                '{}.dict.json'.format(modality))
            os.makedirs(os.path.dirname(modality_dict_filename), exist_ok=True)
            if os.path.exists(modality_dict_filename):
                LOGGER.info('Loading {} dict from {}'.format(modality, modality_dict_filename))
                modality_dict = Dictionary.load_json(modality_dict_filename)
            else:
                modality_dict = build_dictionary(train_path(modality), modality, src=True)
                LOGGER.info('Saving {} dict at {}'.format(modality, modality_dict_filename))
                modality_dict.save_json(modality_dict_filename)
            return modality_dict

        if args['preprocess']['joined_dictionary']:
            modalities = args['preprocess']['source_lang'] + [args['preprocess']['target_lang']]
            modalities = sorted(list(itertools.filterfalse(lambda modality: modality is None, modalities)))
            joined_dictionary_filename = os.path.join(args['preprocess']['destdir'],
                                                      '{}.dict.txt'.format('_'.join(modalities)))
            if os.path.exists(joined_dictionary_filename):
                LOGGER.info('Loading joint dict from {}'.format(joined_dictionary_filename))
                joined_dictionary = Dictionary.load_json(joined_dictionary_filename)
            else:
                joined_dictionary = build_dictionary(
                    [train_path(modality) for modality in modalities], modalities, src=True
                )
                LOGGER.info('Saving joint dict at {}'.format(joined_dictionary_filename))
                joined_dictionary.save_json(joined_dictionary_filename)

            for modality in modalities:
                src_dicts[modality] = joined_dictionary
            tgt_dict = joined_dictionary
        else:
            # src dict
            for modality in args['preprocess']['source_lang']:
                src_dicts[modality] = load_dict(modality)

            # tgt dict
            if args['preprocess']['target_lang']:
                tgt_dict = load_dict(args['preprocess']['target_lang'])
            else:
                tgt_dict = None

        return src_dicts, tgt_dict

    # 1. build vocabulary
    src_dicts, tgt_dict = build_vocab_dict(args)

    # 2. ***************build dataset********************
    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, impl, lang, modality):
        return os.path.join(args['preprocess']['destdir'], 'data-{}'.format(impl), lang, file_name(prefix, modality))

    def make_binary_dataset(dict: Dictionary, input_file, output_file,
                            attr: str, num_workers: int):
        """make binary dataset"""
        LOGGER.info("[{}] Dictionary: {} types".format(attr, len(dict) - 1))
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
                        dict,
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
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))
        merge_result(
            Binarizer.binarize(
                input_file, dict, lambda t: ds.add_item(t),
                tokenize=tokenization.json_tokenizer, offset=0, end=offsets[1], append_eos=False,
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
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                attr,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                dict.unk_word,
            )
        )

    def make_graph_bin_dataset(dict: Dictionary, input_file, output_file, num_workers):
        offsets = Binarizer.find_offsets(input_file, num_workers)
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers)
            for worker_id in range(num_workers):
                prefix = "{}{}".format(output_file, worker_id)
                pool.apply_async(
                    binarize_dgl,
                    (
                        args,
                        input_file,
                        dict,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                )
            pool.close()
        else:
            prefix = "{}0".format(output_file)
            binarize_dgl(args, input_file, dict, prefix, 0, -1)

    def make_dataset(vocab, input_prefix, output_prefix, lang, modality, num_workers=1):
        in_file = file_name(input_prefix, modality)
        out_file = dest_path(output_prefix, args['preprocess']['dataset_impl'], lang, modality)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if args['preprocess']['dataset_impl'] == "raw":
            logger.info('Copying {} into {}'.format(in_file, out_file))
            shutil.copy(src=in_file, dst=out_file)
        else:
            if modality == 'binary_ast':
                make_graph_bin_dataset(vocab, in_file, out_file, num_workers)
            else:
                make_binary_dataset(vocab, in_file, out_file, modality, num_workers)

    def make_all(modality, vocab, lang, data_prefs):
        num_workers = min(args['preprocess']['workers'], cpu_count())
        if data_prefs['trainpref']:
            make_dataset(vocab, data_prefs['trainpref'], "train", lang, modality, num_workers=num_workers)
        if data_prefs['validpref']:
            for k, validpref in enumerate(data_prefs['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, modality, num_workers=num_workers)
        if data_prefs['testpref']:
            for k, testpref in enumerate(data_prefs['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, modality, num_workers=num_workers)

    def build_dataset(args: Dict, src_dicts: Dict[str, Dictionary], tgt_dict: Dictionary):
        """build dataset for modal"""
        for modality, src_dict in src_dicts.items():
            LOGGER.info('Building dataset for {}'.format(modality))
            for lang, data_prefs in args['preprocess']['dataprefs'].items():
                make_all(modality, src_dict, lang, data_prefs)

    # 2. build dataset
    build_dataset(args, src_dicts, tgt_dict)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_multilingual.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    """
    nohup python -m dataset.csn.summarization.preprocess_multilingual > log 2>&1 &
    """
    cli_main()
