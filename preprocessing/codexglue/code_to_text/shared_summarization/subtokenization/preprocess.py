import os
from collections import Counter
from multiprocessing import Pool
from typing import Dict

from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    Dictionary,
    indexed_dataset,
)
from ncc.data.tools.binarizer import Binarizer
from ncc.tokenizers import tokenization
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, dict, consumer, tokenize=tokenization.dpu_sub_tokenizer,
                             append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])

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
        return dest_path(lang, "dict") + ".jsonl"

    target = not args['preprocess']['only_source']

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
                tokenize_func=tokenization.dpu_sub_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['threshold'],
                # set max len for joint dictionaries
                nwords=max(args['preprocess']['nwordssrc'], args['preprocess']['nwordstgt']),
            )
        tgt_dict = src_dict

    else:
        if args['preprocess']['srcdict']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        else:
            assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

            filenames = PathManager.ls(train_path(args['preprocess']['source_lang']))
            if not args['preprocess']['only_train']:
                filenames.extend(
                    PathManager.ls(valid_path(args['preprocess']['source_lang']))
                )
            src_dict = task.build_dictionary(
                filenames,
                tokenize_func=tokenization.dpu_sub_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['thresholdsrc'],
                nwords=args['preprocess']['nwordssrc'],
                padding_factor=args['preprocess']['padding_factor'],
            )
        if target:
            if args['preprocess']['tgtdict']:
                tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
            else:
                assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
                filenames = PathManager.ls(train_path(args['preprocess']['target_lang']))
                if not args['preprocess']['only_train']:
                    filenames.extend(
                        PathManager.ls(valid_path(args['preprocess']['target_lang']))
                    )
                tgt_dict = task.build_dictionary(
                    filenames,
                    tokenize_func=tokenization.dpu_sub_tokenizer,
                    workers=args['preprocess']['workers'],
                    threshold=args['preprocess']['thresholdtgt'],
                    nwords=args['preprocess']['nwordstgt'],
                    padding_factor=args['preprocess']['padding_factor'],
                )
        else:
            tgt_dict = None

    src_dict.save(dict_path(args['preprocess']['source_lang']))  # save spm dict to ncc.dictionary
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args['preprocess']['target_lang']))

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
                tokenize=tokenization.dpu_sub_tokenizer, offset=0, end=offsets[1], append_eos=False,
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, out_file=None, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            raise NotImplementedError
        else:
            in_file = file_name(input_prefix, lang)
            if out_file is None:
                out_file = dest_path(output_prefix, lang)
            PathManager.mkdir(os.path.dirname(out_file))
            make_binary_dataset(vocab, in_file, out_file, num_workers)

    def make_all(lang, vocab):
        for l in os.listdir(args['preprocess']['trainpref'].split('*')[0]):
            # copy shared dict into each languages
            out_dir = os.path.join(args['preprocess']['destdir'], l)
            PathManager.mkdir(out_dir)
            dst_dict = os.path.join(out_dir, f'{lang}.dict.jsonl')
            PathManager.copy(dict_path(lang), dst_dict)

            if args['preprocess']['trainpref']:
                out_file = os.path.join(out_dir, f"train.{lang}")
                make_dataset(vocab, args['preprocess']['trainpref'].replace('*', l), "train", lang,
                             out_file=out_file, num_workers=args['preprocess']['workers'])
            if args['preprocess']['validpref']:
                out_file = os.path.join(out_dir, f"valid.{lang}")
                make_dataset(vocab, args['preprocess']['validpref'].replace('*', l), 'valid', lang,
                             out_file=out_file, num_workers=args['preprocess']['workers'])
            if args['preprocess']['testpref']:
                out_file = os.path.join(out_dir, f"test.{lang}")
                make_dataset(vocab, args['preprocess']['testpref'].replace('*', l), 'test', lang,
                             out_file=out_file, num_workers=args['preprocess']['workers'])

    make_all(args['preprocess']['source_lang'], src_dict)
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/shared'
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
