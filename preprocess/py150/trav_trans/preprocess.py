import itertools
import os
from collections import Counter
from multiprocessing import Pool

from preprocess.py150 import py150_util
from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    indexed_dataset,
    constants,
)
from ncc.data.completion.completion_binarizer import CompletionBinarizer as Binarizer
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager
from ncc.utils.pathos_pool import PPool

MAX_BATCH_SIZE = 5000


def tokenize_func(line):
    dp = []
    for node in json_io.json_loads(line):
        if "value" in node:
            dp.append(node["value"])
        else:
            dp.append(node["type"])
    return dp


def type_tokenize_func(line):
    ast = json_io.json_loads(line)
    code_types = []
    idx = 0
    while idx < len(ast):
        if ast[idx].get('type', None) in {"attr", "Num", "NameLoad", "NameStore", "NameParam"}:
            code_types.extend([constants.PAD, ast[idx]['type']])
            idx += 2
        else:
            code_types.append(constants.PAD)
            idx += 1
    return code_types


def binarize(args, filename: str, dict, in_file: str, lang, offset: int, end: int, append_eos: bool = False):
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))
    ext_ds = indexed_dataset.make_builder(f'{in_file}.ext', impl='seq')

    def consumer(tensor, start_idx):
        ds.add_item(tensor)
        ext_ds.add_item(start_idx)

    def string2dfs(line):
        line = json_io.json_loads(line)
        asts = py150_util.separate_dps(line, args['preprocess']['max_len'])
        ast_dfs = [[py150_util.get_dfs(ast), ext] for ast, ext in asts if len(ast) > 1]
        return ast_dfs

    def string2type_dfs(line):
        type_dfs = type_tokenize_func(line)
        type_dfs = py150_util.separate_dps(type_dfs, args['preprocess']['max_len'])
        type_dfs = [[dfs, ext] for dfs, ext in type_dfs if len(dfs) > 1]
        return type_dfs

    tokenize = string2dfs if lang == 'ast' else string2type_dfs
    res = Binarizer.binarize_seperate(filename, dict, consumer, tokenize=tokenize,
                                      append_eos=append_eos, offset=offset, end=end)
    ds.finalize('{}.idx'.format(in_file))
    ext_ds.finalize()
    return res


# TODO: Don't abstract it. Try to be consistent with Fairseq.
def main(args):
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])
    # 1. ***************build vocabulary***************
    task = tasks.get_task(args['preprocess']['task'])

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path(lang, "dict") + ".jsonl"

    def string2dfs(line):
        line = json_io.json_loads(line)
        asts = py150_util.separate_dps(line, args['preprocess']['max_len'])
        ast_dfs = [[py150_util.get_dfs(ast), ext] for ast, ext in asts if len(ast) > 1]
        return ast_dfs

    def string2type_dfs(line):
        type_dfs = type_tokenize_func(line)
        type_dfs = py150_util.separate_dps(type_dfs, args['preprocess']['max_len'])
        type_dfs = [[dfs, ext] for dfs, ext in type_dfs if len(dfs) > 1]
        return type_dfs

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def valid_path(lang):
        return "{}{}".format(args['preprocess']['validpref'], ("." + lang) if lang else "")

    target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    if args['preprocess']['only_train']:
        LOGGER.info('Generating dictionaries with Train data files.')
    else:
        LOGGER.info('Generating dictionaries with Train/Validation data files.')

    if args['preprocess']['srcdict']:
        src_dict = task.load_dictionary(args['preprocess']['srcdict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        filenames = [train_path(args['preprocess']['source_lang'])]
        if not args['preprocess']['only_train']:
            filenames.append(valid_path(args['preprocess']['source_lang']))
        src_dict = task.build_dictionary(
            filenames,
            tokenize_func=tokenize_func,
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
            # code_types are from ast
            filenames = [train_path(args['preprocess']['source_lang'])]
            if not args['preprocess']['only_train']:
                filenames.append(valid_path(args['preprocess']['source_lang']))
            tgt_dict = task.build_dictionary(
                filenames,
                tokenize_func=type_tokenize_func,
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
    def make_binary_dataset(vocab, input_file, output_file, lang, num_workers):
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        offsets = file_io.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(num_workers - 1)
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
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result
                )
            pool.close()

        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
        ext_ds = indexed_dataset.make_builder(f'{output_file}.ext', impl='seq')

        def consumer(data, start_idx):
            ds.add_item(data)
            ext_ds.add_item(start_idx)

        tokenize = string2dfs if lang == 'ast' else string2type_dfs
        merge_result(
            Binarizer.binarize_seperate(
                input_file, vocab, consumer,
                tokenize=tokenize, offset=0, end=offsets[1], append_eos=False,
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
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            # TODO: parse json to txt file, one line one traversal, please help me parallize it.
            """
            because only 1 thread is allowed to write file, we have to use multi-processing for deal with data
            and merge results from CPUs into a block and then dumps such block. 
            """

            def _func(line):
                line = py150_util.separate_dps(json_io.json_loads(line.strip()), args['preprocess']['n_ctx'])
                line = [py150_util.get_dfs(ast) + [ext] for ast, ext in line if len(ast) > 1]
                # line = [json.dumps([py150_utils.get_dfs(ast), ext]) for ast, ext in line if len(ast) > 1]
                return line

            with PPool() as thread_pool:
                with file_io.open(file_name(input_prefix, lang), 'r') as f, \
                    file_io.open(dest_path(output_prefix, lang), 'w') as fout:
                    def _write(result):
                        for res in itertools.chain(*result):
                            print(json_io.json_dumps(res), file=fout)

                    batch_data = []
                    for line in f:
                        batch_data.append(line)
                        if len(batch_data) >= MAX_BATCH_SIZE:
                            result = thread_pool.feed(_func, batch_data, one_params=True)
                            _write(result)
                            del batch_data
                            batch_data = []

                    if len(batch_data) > 0:
                        result = thread_pool.feed(_func, batch_data, one_params=True)
                        _write(result)
                        del batch_data
        else:
            if lang == 'code_types':
                in_file = file_name(input_prefix, 'ast')
            else:
                in_file = file_name(input_prefix, lang)
            out_file = dest_path(output_prefix, lang)
            PathManager.mkdir(os.path.dirname(out_file))
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
    if target:
        make_all(args['preprocess']['target_lang'], tgt_dict)

    # # 3. ***************generate ids********************
    # def generate_ids(input_prefix, output_prefix, lang):
    #     def _func(line):
    #         line = py150_utils.separate_dps(json_io.json_loads(line.strip()), args['preprocess']['n_ctx'])
    #         tmp = []
    #         for ast, _ in line:
    #             if len(ast) > 1:
    #                 ids = {}
    #                 if args['preprocess']['id_type'] in {"leaf", "all"}:
    #                     ids.update(py150_utils.get_leaf_ids(ast))
    #                 if args['preprocess']['id_type'] in {"value", "all"}:
    #                     ids.update(py150_utils.get_value_ids(ast))
    #                 if args['preprocess']['id_type'] in {"type", "all"}:
    #                     ids.update(py150_utils.get_type_ids(ast))
    #                 tmp.append(ids)
    #         return tmp
    #
    #     with PPool() as thread_pool:
    #         with open(file_name(input_prefix, lang), "r", encoding="utf-8") as f, \
    #             open(dest_path(output_prefix, 'ids'), "w") as fout:
    #             def _write(result):
    #                 for res in itertools.chain(*result):
    #                     print(json.dumps(res), file=fout)
    #
    #             batch_data = []
    #             for line in f:
    #                 batch_data.append(line)
    #                 if len(batch_data) >= MAX_BATCH_SIZE:
    #                     result = thread_pool.feed(_func, batch_data, one_params=True)
    #                     _write(result)
    #                     del batch_data
    #                     batch_data = []
    #
    #             if len(batch_data) > 0:
    #                 result = thread_pool.feed(_func, batch_data, one_params=True)
    #                 _write(result)
    #                 del batch_data
    #
    # def make_all_ids():
    #     if args['preprocess']['trainpref']:
    #         generate_ids(args['preprocess']['trainpref'], "train", args['preprocess']['source_lang'])
    #     if args['preprocess']['validpref']:
    #         generate_ids(args['preprocess']['validpref'], "valid", args['preprocess']['source_lang'])
    #     if args['preprocess']['testpref']:
    #         generate_ids(args['preprocess']['testpref'], "test", args['preprocess']['source_lang'])
    #
    # make_all_ids()


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/preprocess'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
