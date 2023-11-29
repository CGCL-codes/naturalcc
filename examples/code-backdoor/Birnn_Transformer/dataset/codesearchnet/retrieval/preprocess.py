import os
from collections import Counter
from glob import glob
from multiprocessing import Pool
from typing import Dict

from ncc import LOGGER
from ncc import tasks
from ncc.data import indexed_dataset
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval import tokenizers
from ncc.data.retrieval.hybrid.hybrid_retrieval_binarizer import HybridRetrievalBinarizer as Binarizer
from ncc.utils.file_ops.file_io import find_offsets
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str, tokenizer, use_func: bool,
             offset: int, end: int, func_offset: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, dict, consumer,
        tokenize=tokenizer, use_func=use_func, offset=offset, end=end, func_offset=func_offset, append_eos=append_eos,
        min_func_len=args['preprocess']['min_func_len'],
    )

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
        return dest_path(lang, "dict") + ".jsonl"

    target = not args['preprocess']['only_source']

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        data_files = train_path(args['preprocess']['source_lang'])
        data_files = PathManager.ls(data_files)

        src_dict = task.build_bpe_dictionary(
            data_files,
            tokenize_func=tokenizers.sub_tokenizer,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'],
            padding_factor=args['preprocess']['padding_factor'],
            bos=None, eos=None, bpe_portion=args['preprocess']['source_bpe_portion'],
        )
    if target:
        if args['preprocess']['tgtdict']:
            tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        else:
            data_files = train_path(args['preprocess']['target_lang'])
            if '*' in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]

            assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
            tgt_dict = task.build_bpe_dictionary(
                data_files,
                tokenize_func=tokenizers.lower_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=0,
                nwords=args['preprocess']['nwordstgt'],
                padding_factor=args['preprocess']['padding_factor'],
                bos=None, eos=None, bpe_portion=args['preprocess']['target_bpe_portion'],
            )
    else:
        tgt_dict = None

    src_dict.save(dict_path(args['preprocess']['source_lang']))
    tgt_dict.save(dict_path(args['preprocess']['target_lang']))

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab: Dictionary, input_file, output_file, use_func, num_workers: int):
        """make binary dataset"""
        # LOGGER.info("[{}] Dictionary: {} types".format(attr, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()  # save un-recorded tokens

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        offsets = find_offsets(input_file, num_chunks=num_workers)
        func_offsets = None
        modality = input_file.split('.')[-1]
        if modality == 'code_tokens':
            tokenizer = tokenizers.list_tokenizer
            if use_func:
                func_offsets = Binarizer.find_func_offsets(input_file, offsets=offsets)
        elif modality == 'func_name':
            tokenizer = tokenizers.func_name_tokenizer
        elif modality == 'docstring_tokens':
            tokenizer = tokenizers.lower_tokenizer
        else:
            raise NotImplementedError(modality)

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
                        tokenizer,
                        use_func and (modality == 'code_tokens'),
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        func_offsets[worker_id] if func_offsets else 0,
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
                tokenize=tokenizer, use_func=use_func and (modality == 'code_tokens'),
                offset=offsets[0], end=offsets[1], func_offset=func_offsets[0] if func_offsets else 0, append_eos=False,
                min_func_len=args['preprocess']['min_func_len'],
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

    def make_dataset(vocab, input_prefix, output_prefix, lang, use_func=False, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            raise NotImplementedError
        else:
            in_files = file_name(input_prefix, lang)
            if '*' in in_files:
                in_files = glob(in_files)
            else:
                in_files = [in_files]
            for in_file in in_files:
                if lang == 'code_tokens':
                    out_file = dest_path(output_prefix, f'{str.split(in_file, os.sep)[-2]}.{lang + ".wo_func"}') \
                        if use_func == True else dest_path(output_prefix, f'{str.split(in_file, os.sep)[-2]}.{lang}')
                else:
                    out_file = dest_path(output_prefix, f'{str.split(in_file, os.sep)[-2]}.{lang}')
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                make_binary_dataset(vocab, in_file, out_file, use_func, num_workers)

    def make_all(lang, vocab, use_func=False):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, args['preprocess']['trainpref'], "train", lang,
                         num_workers=args['preprocess']['workers'], use_func=use_func)
        if lang in ['code_tokens', 'docstring_tokens'] and not use_func:
            if args['preprocess']['validpref']:
                make_dataset(vocab, args['preprocess']['validpref'], "valid", lang,
                             num_workers=args['preprocess']['workers'], use_func=use_func)
            if args['preprocess']['testpref']:
                make_dataset(vocab, args['preprocess']['testpref'], "test", lang,
                             num_workers=args['preprocess']['workers'], use_func=use_func)

    make_all(args['preprocess']['source_lang'], src_dict)
    make_all(args['preprocess']['source_lang'], src_dict, use_func=True)
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict)
        make_all('func_name', tgt_dict)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", help="load {yaml_file}.yml for train", type=str,
        default='config/python'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
