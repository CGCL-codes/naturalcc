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
import dgl
import torch
import ujson
import shutil
from tqdm import tqdm
from multiprocessing import Pool
from preprocess.typilus.utils import (
    TokenEmbedder,
    ignore_type_annotation,
)
from preprocess.typilus.preprocess import typilus_tokenizers
from ncc import tasks
from collections import Counter
from ncc.data import (
    # Dictionary,
    indexed_dataset,
)
from ncc.tokenizers import tokenization
# from ncc.data.constants import PLACEHOLDER
from ncc.data import constants
from ncc.data.type_prediction.typilus.typilus_binarizer import TypilusBinarizer as Binarizer
from ncc.data.type_prediction.typilus.typilus_dictionary import TypilusDictionary as Dictionary
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc import LOGGER


def file_lines(reader):
    num = sum(1 for line in reader)
    reader.seek(0)
    return num


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str, lang,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    tokenize = getattr(typilus_tokenizers, f'{lang}_binarizer_tokenizer')
    res = Binarizer.binarize_nodes(filename, dict, consumer, tokenize=tokenize,
                                   append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def binarize_edges(args: Dict, filename: str, dict: Dictionary, in_file: str, out_files: Dict, lang,
                   offset: int, end: int):
    ds_files, ds = {}, {}
    for edeg_type, out_file in out_files.items():
        ds_files[edeg_type] = {
            'src': '{}.mmap'.format(out_file),
            'tgt': '{}.mmap'.format(out_file),
        }
        ds[edeg_type] = {
            'src': indexed_dataset.make_builder( \
                ds_files[edeg_type]['src'], impl=args['preprocess']['dataset_impl'], vocab_size=len(dict)),
            'tgt': indexed_dataset.make_builder( \
                ds_files[edeg_type]['tgt'], impl=args['preprocess']['dataset_impl'], vocab_size=len(dict)),
        }

    def consumer(tensors):
        for edeg_type, tensor in tensors.items():
            ds[edeg_type].add_item(tensor)

    res = Binarizer.binarize_edges(filename, dict, consumer, tokenize=ujson.loads, offset=offset, end=end)
    for edeg_type, d in ds.items():
        d.finalize('{}.idx'.format(in_file))
    return res


def binarize_annotation(args: Dict, filename: str, dict: Dictionary, node_file: str, info_file: str, lang,
                        offset: int, end: int, tokenize, is_train, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_node_file = '{}.mmap'.format(node_file)
    ds_node = indexed_dataset.make_builder(ds_node_file, impl=args['preprocess']['dataset_impl'])
    ds_info_file = '{}.mmap'.format(info_file)
    ds_info = indexed_dataset.make_builder(ds_info_file, impl=args['preprocess']['dataset_impl'],
                                           vocab_size=len(dict))
    info_json = '{}.json'.format(info_file)
    info_writer = open(info_json, 'w')

    def consumer(ids, info, raw_info):
        ids = torch.Tensor(ids).int()
        ds_node.add_item(ids)
        ds_info.add_item(info)
        print(ujson.dumps(raw_info), file=info_writer)

    res = Binarizer.binarize_supernodes(filename, dict, consumer, tokenize=tokenize,
                                        append_eos=append_eos, offset=offset, end=end,
                                        is_train=is_train)
    ds_node.finalize('{}.idx'.format(node_file))
    ds_info.finalize('{}.idx'.format(info_file))
    info_writer.close()
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
        """
        $prefix.$lang
        """
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        """ $destdir/$prefix.$lang """
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        """ $lang -> $destdir/$lang.dict.json """
        return dest_path(lang, "dict") + ".json"

    for idx, lang in enumerate(args['preprocess']['langs']):
        if not args['preprocess']['dicts'][idx] and os.path.exists(dict_path(lang)):
            raise FileExistsError(dict_path(lang))

    if args['preprocess']['only_train']:
        LOGGER.info('Generating dictionaries with Train data files.')
    else:
        LOGGER.info('Generating dictionaries with Train/Validation data files.')

    lang_dicts = {}
    for idx, lang in enumerate(args['preprocess']['langs']):
        if args['preprocess']['dicts'][idx]:
            # lang_dicts[lang] = task.load_typilus_dictionary(args['preprocess']['dicts'][idx], lang=lang)
            lang_dicts[lang] = task.load_dictionary(args['preprocess']['dicts'][idx])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

            filenames = [train_path(lang)]
            if not args['preprocess']['only_train']:
                filenames.append(valid_path(lang))

            tokenizer = None
            pad, bos, eos, unk, extra_special_symbols = None, None, None, None, None
            if lang == 'nodes':
                tokenizer = getattr(typilus_tokenizers, f'{lang}_tokenizer')
                pad, unk = constants.PAD, constants.UNK
                extra_special_symbols = [TokenEmbedder.INT_LITERAL, TokenEmbedder.FLOAT_LITERAL,
                                         TokenEmbedder.STRING_LITERAL]
            elif lang == 'edges':
                tokenizer = getattr(typilus_tokenizers, f'{lang}_tokenizer')
            elif str.startswith(lang, 'supernodes'):
                tokenizer = getattr(typilus_tokenizers, f'{lang.split(".")[-1]}_tokenizer')
                filenames = [str.rsplit(f, '.', 1)[0] for f in filenames]
                unk = constants.UNK
            else:
                raise NotImplementedError

            print(filenames)
            lang_dict = task.build_dictionary(
                filenames,
                tokenize_func=tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['thresholds'][idx],
                nwords=args['preprocess']['nwords'][idx],
                padding_factor=args['preprocess']['padding_factor'],
                pad=pad, bos=bos, eos=eos, unk=unk, extra_special_symbols=extra_special_symbols,
            )
            # lang_dict.save_json(dict_path(lang))
            lang_dict.save(dict_path(lang))
            lang_dicts[lang] = lang_dict

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab: Dictionary, lang, input_file, output_file, num_workers: int):
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
                        lang,
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
            Binarizer.binarize_nodes(
                input_file, vocab, lambda t: ds.add_item(t),
                tokenize=getattr(typilus_tokenizers, f'{lang}_binarizer_tokenizer'),
                offset=0, end=offsets[1], append_eos=False,
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

    def make_supernodes_binary_dataset(vocab: Dictionary, lang, input_file, node_file, info_file, num_workers: int):
        """make binary dataset"""
        # LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()  # save un-recorded tokens

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        is_train = os.path.basename(input_file).split('.')[0] == 'train'
        tokenize = getattr(typilus_tokenizers, f'{lang.split(".")[-1]}_binarizer_tokenizer')

        # split a file into different parts
        # if use multi-processing, we first process 2nd to last file
        # 1.txt -> 10 processor, 0(p0)(0-99), 100(p1)(100-199), ...
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            # p1-pN -> (1 bin-txt, 1 idx), (N bin-txt, N idx)
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                node_prefix = "{}{}".format(node_file, worker_id)
                info_prefix = "{}{}".format(info_file, worker_id)
                pool.apply_async(
                    binarize_annotation,
                    (
                        args,
                        input_file,
                        vocab,
                        node_prefix,
                        info_prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        tokenize,
                        is_train,
                    ),
                    callback=merge_result
                )
            pool.close()
        # process 1th file, if multi-processing available. If not, process all file
        # p0 -> 0,end
        ds_node_file = '{}.mmap'.format(node_file)
        ds_node = indexed_dataset.make_builder(ds_node_file, impl=args['preprocess']['dataset_impl'])
        ds_info_file = '{}.mmap'.format(info_file)
        ds_info = indexed_dataset.make_builder(ds_info_file, impl=args['preprocess']['dataset_impl'],
                                               vocab_size=len(vocab))
        info_writer = open('{}.json'.format(info_file), 'w')

        def consumer(ids, info, raw_info):
            ds_node.add_item(torch.IntTensor(ids))
            ds_info.add_item(info)
            print(ujson.dumps(raw_info), file=info_writer)

        merge_result(
            Binarizer.binarize_supernodes(
                input_file, vocab, consumer, tokenize, offset=0, end=offsets[1], append_eos=False, is_train=is_train,
            )
        )

        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(node_file, worker_id)
                ds_node.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

                temp_file_path = "{}{}".format(info_file, worker_id)
                ds_info.merge_file_(temp_file_path)
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

                # append json
                temp_file_path = "{}{}.json".format(info_file, worker_id)
                with open(temp_file_path, 'r') as reader:
                    shutil.copyfileobj(reader, info_writer)
                os.remove(temp_file_path)
        ds_node.finalize('{}.idx'.format(node_file))
        ds_info.finalize('{}.idx'.format(info_file))
        info_writer.close()

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
            if lang == 'edges':
                # this method may meet a problem: while merge graphs via dgl.batch, it will return Error because some
                # heterogenuous graphs do not have all edges such that cannot merge graph
                graphs = []
                with open(in_file, "r", encoding="utf-8") as reader:
                    for line in tqdm(reader, total=file_lines(reader)):
                        edges = vocab.encode_edges_line(line=line, line_tokenizer=ujson.loads)
                        graph = dgl.heterograph(edges, idtype=torch.long)
                        graphs.append(graph)
                dgl.data.utils.save_graphs(out_file, graphs)
            elif str.startswith(lang, 'supernodes'):
                in_file = in_file.rsplit('.', 1)[0]
                out_file_node = f'{out_file}.node'
                out_file_type = f'{out_file}.type'
                make_supernodes_binary_dataset(vocab, lang, in_file, out_file_node, out_file_type, num_workers)
            else:
                make_binary_dataset(vocab, lang, in_file, out_file, num_workers)

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

    for idx, (lang, lang_dict) in enumerate(lang_dicts.items()):
        make_all(lang, lang_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='typilus'
    )
    args = parser.parse_args()
    LOGGER.info(args)
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)

def dataset_binarize():
    cli_main()

if __name__ == "__main__":
    cli_main()
