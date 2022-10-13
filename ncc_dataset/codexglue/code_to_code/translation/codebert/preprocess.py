# -*- coding: utf-8 -*-

import itertools
import os

from ncc_dataset.clcdsa import (
    MODES,
)
from ncc import LOGGER
from ncc.data.dictionary import (
    TransformersDictionary,
)
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager
from run.translation.codebert.config.codetrans import config


def main(args):
    # task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])
    vocab = TransformersDictionary.from_pretrained('microsoft/codebert-base', do_lower_case=False)

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    def parse_source_input(code):
        code_tokens = vocab.tokenize(code)
        # truncating
        code_tokens = code_tokens[:config.MAX_SOURCE_LENGTH - 2]
        source_tokens = [vocab.cls_token] + code_tokens + [vocab.sep_token]
        source_ids = vocab.convert_tokens_to_ids(source_tokens)
        source_size = len(source_tokens)
        source_mask = [1] * source_size
        padding_length = config.MAX_SOURCE_LENGTH - len(source_ids)
        source_ids += [vocab.pad()] * padding_length
        source_mask += [0] * padding_length
        return [source_ids, source_mask, source_size]

    def parse_target_input(code):
        target_tokens = vocab.tokenize(code)[:config.MAX_TARGET_LENGTH - 2]
        target_tokens = [vocab.cls_token] + target_tokens + [vocab.sep_token]
        target_ids = vocab.convert_tokens_to_ids(target_tokens)
        target_size = len(target_ids)
        target_mask = [1] * target_size
        padding_length = config.MAX_TARGET_LENGTH - len(target_ids)
        target_ids += [vocab.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        return [target_ids, target_mask, target_size]

    src_lang, tgt_lang = args['preprocess']['src_lang'], args['preprocess']['tgt_lang']
    for lang, mode in itertools.product([src_lang, tgt_lang], MODES):
        src_file = args['preprocess'][f'{mode}pref'].replace('*', lang) + ".code"
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.pkl")
        PathManager.mkdir(os.path.dirname(dst_file))
        with file_io.open(src_file, 'r') as reader:
            keys = ['code', 'src_tokens', 'src_masks', 'src_sizes', 'tgt_tokens', 'tgt_masks', 'tgt_sizes']
            data = {key: [] for key in keys}
            for line in reader:
                src_code = json_io.json_loads(line)
                # src_code = SPACE_SPLITTER.sub(" ", line)
                # source_ids, source_mask
                src_line = parse_source_input(src_code)
                # target_ids, target_mask
                tgt_line = parse_target_input(src_code)
                for key, src in zip(keys, [src_code] + src_line + tgt_line):
                    data[key].append(src)
            file_io.open(dst_file, mode='wb', data=data)


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
