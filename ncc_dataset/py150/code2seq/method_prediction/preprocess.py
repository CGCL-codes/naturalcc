import os
from collections import Counter
from multiprocessing import Pool

from ncc import LOGGER
from ncc import tasks
from ncc.data import (
    indexed_dataset,
)
from ncc.data.summarization.path_binarizer import PathSummarizationBinarizer
from ncc.data.tools.binarizer import Binarizer
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager


def binarize(args, filename: str, vocab, aux_dict, in_file: str, lang, tokenize, max_path_num: int,
             offset: int, end: int, append_eos: bool = False):
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
    if lang == 'method_path':
        sz_ds_file = '{}.sz.mmap'.format(in_file)
        sz_ds = indexed_dataset.make_builder(sz_ds_file, impl=args['preprocess']['dataset_impl'],
                                             vocab_size=len(vocab))
    else:
        sz_ds = None

    def consumer(tensor, size=None):
        ds.add_item(tensor)
        if size is not None:
            sz_ds.add_item(size)

    if sz_ds is None:
        res = Binarizer.binarize(filename, vocab, consumer, tokenize=tokenize,
                                 append_eos=append_eos, offset=offset, end=end, )
        ds.finalize('{}.idx'.format(in_file))
    else:
        res = PathSummarizationBinarizer.path_binarizer(filename, vocab, consumer, tokenize=tokenize,
                                                        append_eos=append_eos, offset=offset, end=end,
                                                        type_dict=aux_dict, max_path_num=max_path_num, )
        ds.finalize('{}.idx'.format(in_file))
        sz_ds.finalize('{}.sz.idx'.format(in_file))
    return res


def subtoken_tokenize(line, **kwargs):
    line = json_io.json_loads(line)
    paths = line.split(' ')[1:]
    subtokens = []
    for p in paths:
        head, _, tail = p.split(',')
        subtokens.extend(head.split('|'))
        subtokens.extend(tail.split('|'))
    return subtokens


def type_tokenize(line, **kwargs):
    line = json_io.json_loads(line)
    paths = line.split(' ')[1:]
    subtokens = []
    for p in paths:
        _, body, _ = p.split(',')
        subtokens.extend(body.split('|'))
    return subtokens


def method_tokenize(line, **kwargs):
    line = json_io.json_loads(line)
    method = line.split(' ')[0].split('|')
    return method


def path_tokenize(line, **kwargs):
    line = json_io.json_loads(line)
    paths = line.split(' ')[1:]
    paths = paths[:kwargs['max_path_num']]
    # do not sample paths randomly to keep our generated datasets to be the same
    # if len(paths) > kwargs['max_path_num']:
    #     paths = np.random.choice(paths, kwargs['max_path_num'], replace=False).tolist()
    heads, bodies, tails = [], [], []
    for p in paths:
        head, body, tail = p.split(',')
        heads.append(head.split('|'))
        bodies.append(body.split('|'))
        tails.append(tail.split('|'))
    return heads, bodies, tails


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

    def train_path(lang):
        return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def valid_path(lang):
        return "{}{}".format(args['preprocess']['validpref'], ("." + lang) if lang else "")

    if args['preprocess']['only_train']:
        LOGGER.info('Generating dictionaries with Train data files.')
    else:
        LOGGER.info('Generating dictionaries with Train/Validation data files.')

    if args['preprocess']['subtokendict']:
        subtoken_dict = task.load_dictionary(args['preprocess']['subtokendict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        filenames = [train_path(args['preprocess']['source_lang'])]
        if not args['preprocess']['only_train']:
            filenames.append(valid_path(args['preprocess']['source_lang']))
        subtoken_dict = task.build_dictionary(
            filenames,
            tokenize_func=subtoken_tokenize,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['threshold'],
            nwords=args['preprocess']['nwordssubtoken'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    if args['preprocess']['typedict']:
        type_dict = task.load_dictionary(args['preprocess']['typedict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        filenames = [train_path(args['preprocess']['source_lang'])]
        if not args['preprocess']['only_train']:
            filenames.append(valid_path(args['preprocess']['source_lang']))
        type_dict = task.build_dictionary(
            filenames,
            tokenize_func=type_tokenize,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['threshold'],
            nwords=args['preprocess']['nwordstype'],
            padding_factor=args['preprocess']['padding_factor'],
            bos=None, eos=None,
        )

    if args['preprocess']['methoddict']:
        method_dict = task.load_dictionary(args['preprocess']['methoddict'])
    else:
        assert args['preprocess']['trainpref'], "--trainpref must be set if --srcdict is not specified"

        filenames = [train_path(args['preprocess']['source_lang'])]
        if not args['preprocess']['only_train']:
            filenames.append(valid_path(args['preprocess']['source_lang']))
        method_dict = task.build_dictionary(
            filenames,
            tokenize_func=method_tokenize,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['threshold'],
            nwords=args['preprocess']['nwordsmethod'],
            padding_factor=args['preprocess']['padding_factor'],
        )

    subtoken_dict.save(dict_path('subtoken'))
    type_dict.save(dict_path('type'))
    method_dict.save(dict_path('method'))

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab, aux_dict, input_file, output_file, lang, max_path_num, num_workers):
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        tokenize = path_tokenize if lang == 'method_path' else method_tokenize
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
                        aux_dict,
                        prefix,
                        lang,
                        tokenize,
                        max_path_num,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result
                )
            pool.close()

        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))
        if lang == 'method_path':
            sz_ds_file = '{}.sz.mmap'.format(output_file)
            sz_ds = indexed_dataset.make_builder(sz_ds_file, impl=args['preprocess']['dataset_impl'],
                                                 vocab_size=len(vocab))
        else:
            sz_ds = None

        def consumer(tensor, size=None):
            ds.add_item(tensor)
            if size is not None:
                sz_ds.add_item(size)

        if sz_ds is None:
            merge_result(
                Binarizer.binarize(
                    input_file, vocab, consumer,
                    tokenize=tokenize, offset=0, end=offsets[1], append_eos=False,
                    max_path_num=max_path_num,
                )
            )
        else:
            merge_result(
                PathSummarizationBinarizer.path_binarizer(
                    input_file, vocab, consumer,
                    tokenize=tokenize, offset=0, end=offsets[1], append_eos=False, type_dict=aux_dict,
                    max_path_num=max_path_num,
                )
            )
        if num_workers > 1:
            # p1-pN
            pool.join()
            # merge sub-processors' index and data files into final files and delete them.
            for worker_id in range(1, num_workers):
                temp_file_path = "{}{}".format(output_file, worker_id)
                ds.merge_file_(temp_file_path)
                if sz_ds is not None:
                    sz_ds.merge_file_(f"{temp_file_path}.sz")
                # idx, txt
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
                if sz_ds is not None:
                    os.remove(indexed_dataset.data_file_path(f"{temp_file_path}.sz"))
                    os.remove(indexed_dataset.index_file_path(f"{temp_file_path}.sz"))
        ds.finalize('{}.idx'.format(output_file))
        if sz_ds is not None:
            sz_ds.finalize('{}.sz.idx'.format(output_file))
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

    def make_dataset(vocab, aux_dict, input_prefix, output_prefix, lang, max_path_num, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            raise NotImplementedError
        else:
            in_file = file_name(input_prefix, 'method_path')
            out_file = dest_path(output_prefix, lang)
            PathManager.mkdir(os.path.dirname(out_file))
            make_binary_dataset(vocab, aux_dict, in_file, out_file, lang, max_path_num, num_workers)

    def make_all(lang, vocab, aux_dict=None):
        if args['preprocess']['trainpref']:
            max_path_num = args['preprocess']['train_path_num']
            make_dataset(vocab, aux_dict, args['preprocess']['trainpref'], "train", lang, max_path_num,
                         num_workers=args['preprocess']['workers'])
        if args['preprocess']['validpref']:
            max_path_num = args['preprocess']['eval_path_num']
            for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, aux_dict, validpref, outprefix, lang, max_path_num,
                             num_workers=args['preprocess']['workers'])
        if args['preprocess']['testpref']:
            max_path_num = args['preprocess']['eval_path_num']
            for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, aux_dict, testpref, outprefix, lang, max_path_num,
                             num_workers=args['preprocess']['workers'])

    make_all(args['preprocess']['source_lang'], subtoken_dict, type_dict)
    make_all('method', method_dict)


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
