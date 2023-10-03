# -*- coding: utf-8 -*-

import os
from collections import Counter

import sentencepiece as spm
import torch

from preprocess.clcdsa import (
    MODES,
)
from preprocess.clcdsa.plbart import (
    SPM_VOCAB_FILE,
)
from ncc import LOGGER
from ncc.data.dictionary import (
    Dictionary,
)
from ncc.tokenizers.tokenization import SPACE_SPLITTER
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager
from ncc.data import indexed_dataset


def main(args):
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])
    vocab = spm.SentencePieceProcessor()
    vocab.load(SPM_VOCAB_FILE)

    def save_dict():
        src_file = os.path.join(os.path.dirname(SPM_VOCAB_FILE), 'dict.txt')
        tgt_file = os.path.join(args['preprocess']['destdir'], 'dict.jsonl')
        # Dictionary.text_to_jsonl(src_file, tgt_file)
        vocab = Dictionary()
        with file_io.open(src_file, 'r') as reader:
            for line in reader:
                token, num = line.strip().split()
                vocab.add_symbol(token, eval(num))
        vocab.save(tgt_file)
        return vocab

    dictionary = save_dict()

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    lang = args['preprocess']['lang']
    for mode in MODES:
        file = f"{args['preprocess'][f'{mode}pref']}.code"
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.code")
        PathManager.mkdir(os.path.dirname(dst_file))
        dataset = indexed_dataset.make_builder(f"{dst_file}_tokens.mmap", impl='mmap', vocab_size=len(vocab))
        PathManager.mkdir(os.path.dirname(dst_file))
        with file_io.open(file, 'r') as reader:
            data = {'code': []}
            for line in reader:
                line = json_io.json_loads(line)
                code = SPACE_SPLITTER.sub(" ", line)
                data['code'].append(code)
                code_tokens = vocab.encode(code, out_type=str)
                code_tokens = torch.IntTensor([dictionary.index(token) for token in code_tokens])
                # code_tokens = torch.IntTensor(vocab.encode_as_ids(code))
                dataset.add_item(code_tokens)
            dataset.finalize(f"{dst_file}_tokens.idx")
            # proj indices
            # cp id
            data['proj_indices'] = [1] * len(data['code'])
            file_io.open(f"{dst_file}.pkl", mode='wb', data=data)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
