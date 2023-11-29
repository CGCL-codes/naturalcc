import os
import shutil
from multiprocessing import Pool

import dgl
import ujson
from dgl.data.utils import (load_graphs, save_graphs)

from ncc import LOGGER
from ncc import tasks
from ncc.tokenizers import tokenization
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.file_io import (
    safe_readline,
    find_offsets,
)
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.graph_utils import (
    tree2dgl,
)


def bin_ast_tokenizer(line, **kwargs):
    line = json_io.json_loads(line)
    if line is None:
        return []
    else:
        return tokenization._bin_ast_tokenizer(line)


def build_dgl_graph(vocab, input_file, output_file, start=0, end=-1):
    graph_batch = []
    with open(input_file, 'r') as reader:
        reader.seek(start)
        line = safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            ast = ujson.loads(line)
            if ast is None:
                graph = dgl.DGLGraph()
            else:
                graph = tree2dgl(ast, vocab)
            graph_batch.append(graph)
            line = safe_readline(reader)
    save_graphs(output_file, graph_batch)


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

        filenames = [train_path(args['preprocess']['source_lang'])]
        if not args['preprocess']['only_train']:
            filenames.append(valid_path(args['preprocess']['source_lang']))
        src_dict = task.build_dictionary(
            filenames,
            tokenize_func=bin_ast_tokenizer,
            workers=args['preprocess']['workers'],
            threshold=args['preprocess']['thresholdsrc'],
            nwords=args['preprocess']['nwordssrc'],
            padding_factor=args['preprocess']['padding_factor'],
        )
        src_dict.save(dict_path(args['preprocess']['source_lang']))

    # 2. ***************build dataset********************
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
            offsets = find_offsets(in_file, num_workers)
            with Pool(num_workers) as mpool:
                results = [
                    mpool.apply_async(
                        build_dgl_graph,
                        (vocab, in_file, f'{out_file}{worker_id}.mmap', offsets[worker_id], offsets[worker_id + 1]),
                    )
                    for worker_id in range(num_workers)
                ]
                results = [res.get() for res in results]
            graph_batch = []
            for worker_id in range(num_workers):
                sub_file = f'{out_file}{worker_id}.mmap'
                glist, _ = load_graphs(sub_file)
                graph_batch.extend(glist)
                os.remove(sub_file)
            save_graphs(f'{out_file}.mmap', graph_batch)

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
        default='./config/python'
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
