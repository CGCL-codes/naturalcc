import os
import shutil
from collections import Counter
from multiprocessing import Pool
from typing import Dict

from dataset.raw_py150.utils import (
    separate_list,
)
from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    indexed_dataset,
)
from ncc.data.completion.completion_binarizer import CompletionBinarizer as Binarizer
from ncc.data.completion.completion_dictionary import CompletionDictionary as Dictionary
from ncc.tokenizers.tokenization import json_tokenizer
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.file_ops.yaml_io import load_yaml


def binarize(args: Dict, filename: str, dict: Dictionary, in_file: str,
             offset: int, end: int, append_eos: bool = False):
    """binarize function for multi-processing"""
    ds_file = f'{in_file}.mmap'
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))
    ext_ds = indexed_dataset.make_builder(f"{in_file}.ext", impl='seq')

    def consumer(data, start_idx):
        ds.add_item(data)
        ext_ds.add_item(start_idx)

    def seperate_tokenize(line):
        line = json_io.json_loads(line)
        tokens = separate_list(line, args['preprocess']['max_len'])
        return tokens

    res = Binarizer.binarize_seperate(filename, dict, consumer, tokenize=seperate_tokenize,
                                      append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    ext_ds.finalize()
    return res


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    def train_path(lang):
        return "{}.{}".format(args['preprocess']['trainpref'], lang)

    def valid_path(lang):
        return "{}{}".format(args['preprocess']['validpref'], ("." + lang) if lang else "")

    def file_name(prefix, lang):
        return f"{prefix}.{lang}"

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path(lang, "dict") + ".jsonl"

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
                tokenize_func=json_tokenizer,
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

            filenames = [train_path(args['preprocess']['source_lang'])]
            if not args['preprocess']['only_train']:
                filenames.append(valid_path(args['preprocess']['source_lang']))
            src_dict = task.build_dictionary(
                filenames,
                tokenize_func=json_tokenizer,
                workers=args['preprocess']['workers'],
                threshold=args['preprocess']['threshold'],
                nwords=args['preprocess']['nwordssrc'],
                padding_factor=args['preprocess']['padding_factor'],
                bos=None, eos=None,
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
                    tokenize_func=json_tokenizer,
                    workers=args['preprocess']['workers'],
                    threshold=args['preprocess']['threshold'],
                    nwords=args['preprocess']['nwordstgt'],
                    padding_factor=args['preprocess']['padding_factor'],
                    bos=None, eos=None,
                )
        else:
            tgt_dict = None

    src_dict.save(dict_path(args['preprocess']['source_lang']))  # save spm dict to ncc.dictionary
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args['preprocess']['target_lang']))

    # 2. ***************build dataset********************
    def seperate_tokenize(line):
        line = json_io.json_loads(line)
        tokens = separate_list(line, args['preprocess']['max_len'])
        return tokens

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
        offsets = file_io.find_offsets(input_file, num_workers)
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
        ds_file = f'{output_file}.mmap'
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
        ext_ds = indexed_dataset.make_builder(f'{output_file}.ext', impl='seq')

        def consumer(data, start_idx):
            ds.add_item(data)
            ext_ds.add_item(start_idx)

        merge_result(
            Binarizer.binarize_seperate(
                input_file, vocab, consumer,
                tokenize=seperate_tokenize, offset=0, end=offsets[1], append_eos=False,
            )
        )
        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(output_file, worker_id)
                ds.merge_file_(temp_file_path)
                ext_ds.merge_file_(f"{temp_file_path}.ext")
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(f"{temp_file_path}.ext"))
        ds.finalize('{}.idx'.format(output_file))
        ext_ds.finalize()
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
