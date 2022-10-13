import os
from collections import Counter
from multiprocessing import Pool

from ncc import LOGGER
from ncc import tasks
from ncc.data import indexed_dataset
from ncc.data.completion.completion_binarizer import CompletionBinarizer as Binarizer
from ncc.tokenizers import tokenization
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager


def string2tokens(line):
    line = tokenization.json_tokenizer(line)
    return [
        [line, 0]
    ]


def binarize(args, filename, dict, in_file, offset, end, append_eos=False):
    """binarize function for multi-processing"""
    ds_file = '{}.mmap'.format(in_file)
    ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(dict))

    def consumer(data, _):
        ds.add_item(data)

    res = Binarizer.binarize_seperate(filename, dict, consumer, tokenize=string2tokens,
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
        return dest_path(lang, "dict") + ".jsonl"

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))

    if args['preprocess']['only_train']:
        LOGGER.info('Generating dictionaries with Train data files.')
    else:
        LOGGER.info('Generating dictionaries with Train/Validation data files.')

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
            tokenize_func=tokenization.json_tokenizer,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['threshold'],
            nwords=args['preprocess']['nwordssrc'],
            padding_factor=args['preprocess']['padding_factor'],
            bos=None, eos=None,
        )

    src_dict.save(dict_path(args['preprocess']['source_lang']))  # save spm dict to ncc.dictionary
    # copy shared dict into each language's data directory
    for d in PathManager.ls(os.path.dirname(args['preprocess']['trainpref'])):
        lang = os.path.basename(d)
        src_dict.save(
            os.path.join(args['preprocess']['destdir'], lang, f"{args['preprocess']['source_lang']}.dict.jsonl")
        )

    # 2. ***************build dataset********************
    def make_binary_dataset(vocab, input_file, output_file, num_workers):
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
        ds_file = '{}.mmap'.format(output_file)
        ds = indexed_dataset.make_builder(ds_file, impl=args['preprocess']['dataset_impl'], vocab_size=len(vocab))

        def consumer(data, _):
            ds.add_item(data)

        merge_result(
            Binarizer.binarize_seperate(
                input_file, vocab, consumer,
                tokenize=string2tokens, offset=0, end=offsets[1], append_eos=False,
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
            raise NotImplementedError
        else:
            languages = [
                os.path.basename(d)
                for d in PathManager.ls(os.path.dirname(input_prefix))
            ]
            for l in languages:
                in_file = file_name(input_prefix, lang)
                in_file = str.replace(in_file, '*', l)
                out_file = dest_path(os.path.join(l, output_prefix), lang)
                PathManager.mkdir(os.path.dirname(out_file))
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


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/csn_feng'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
