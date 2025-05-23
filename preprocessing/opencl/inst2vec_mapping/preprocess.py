import os
import pickle
import shutil
from collections import Counter
from multiprocessing import Pool
from typing import Dict

from preprocessing.opencl import (
    INST2VEC_VOCAB,
    INST2VEC_STMTS,
)
from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    indexed_dataset,
)
from ncc.data.constants import UNK
from ncc.data.mapping.xfg_dictionary import XFGDicionary
from ncc.data.tools.binarizer import Binarizer
from ncc.tokenizers import tokenization
from ncc.utils.file_ops.yaml_io import load_yaml


def xfg_tokenization(line, **kwargs):
    line = tokenization.json_tokenizer(line)
    cutoff_stmts = kwargs['cutoff_stmts']

    tokens = []
    for stmt in line:
        # check whether this is an unknown
        if stmt in cutoff_stmts:
            stmt = UNK
        tokens.append(stmt)
    return tokens


def binarize(args: Dict, filename: str, dict: XFGDicionary, in_file: str, attr: str,
             offset: int, end: int, cutoff_stmts=None, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, dict, consumer,
        tokenize=tokenization.json_tokenizer,
        append_eos=append_eos, offset=offset, end=end,
        cutoff_stmts=cutoff_stmts,
    )
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
        return dest_path(lang, "dict") + ".jsonl"

    # 1. build vocabulary
    LOGGER.info('Build vocabularies...')

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"
        src_dict = task.build_dictionary(
            [],
            tokenize_func=tokenization.json_tokenizer,
            workers=args['preprocess']['workers'],
            padding_factor=args['preprocess']['padding_factor'],
            pad=None, bos=None, eos=None,
        )
        with open(INST2VEC_VOCAB, 'rb') as reader:
            vocab = pickle.load(reader)
            for key, _ in list(vocab.items())[:-1]:
                src_dict.add_symbol(key)

    LOGGER.info('dict_path: {}'.format(dict_path(args['preprocess']['source_lang'])))
    src_dict.save(dict_path(args['preprocess']['source_lang']))

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab: XFGDicionary, input_file, output_file,
                            attr: str, num_workers: int):
        """make binary dataset"""
        LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()  # save un-recorded tokens

        with open(INST2VEC_STMTS, 'rb') as reader:
            cutoff_stmts = pickle.load(reader)
            cutoff_stmts = set(cutoff_stmts)

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
                        offsets[worker_id + 1],
                        cutoff_stmts
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
                tokenize=xfg_tokenization,
                offset=0, end=offsets[1], append_eos=False,
                cutoff_stmts=cutoff_stmts,
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
            make_binary_dataset(vocab, in_file, out_file, lang, num_workers)

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

    make_all(args['preprocess']['source_lang'], src_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='nvidia'
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
