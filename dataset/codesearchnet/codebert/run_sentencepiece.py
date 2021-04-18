#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import sentencepiece as spm
from ncc import LOGGER
from ncc.data import constants
from dataset.codesearchnet.utils.codebert_utils import get_special_symbols
from dataset.augmented_javascript.utils.util import normalize_program
from dataset.augmented_javascript.utils.jsonl_dataset import normalize_docstring
import tqdm
from dataset.codesearchnet.utils.codebert_utils import vocab2dict
import ujson


def make_corpus(input_files, corpus_file, corpus_modalities):
    for modality in corpus_modalities:
        if modality == 'code':
            with open(corpus_file['code'], 'w') as writer:
                for input_file in input_files['code']:
                    with open(input_file, 'r', encoding='UTF-8') as reader:
                        for line in reader:
                            function = ujson.loads(line)
                            function = normalize_program(function)
                            writer.write(function + '\n')

        elif modality == 'docstring':
            with open(corpus_file['docstring'], 'w') as writer:
                for input_file in input_files['docstring']:
                    with open(input_file, 'r', encoding='UTF-8') as reader:
                        for line in reader:
                            docstring = ujson.loads(line)
                            docstring = normalize_docstring(docstring)
                            writer.write(docstring + '\n')


def spm_train(input: str, model_prefix: str, vocab_size: int, character_coverage=0.9995, model_type='unigram', special_symbols=None):
    special_symbols = ','.join(special_symbols)
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} --unk_piece=[UNK] " \
              f"--pad_piece=[PAD] --user_defined_symbols={special_symbols} --hard_vocab_limit=false"
    LOGGER.info(command)
    # exit()
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    # python -m dataset.csn.codebert.run_sentencepiece --src-dir ~/.ncc/CodeSearchNet/flatten --tgt-dir ~/.ncc/CodeSearchNet/codebert/ --vocab-size 50000 --model-type bpe --model-prefix codesearchnet
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str,
                        default='~/.ncc/code_search_net/flatten',
                        help='source data')
    parser.add_argument("--language", type=str, help='sentencepiece tokenizer for language')
    # parser.add_argument("--corpus_modalities", type=list, help='sentencepiece tokenizer for modalities')
    parser.add_argument("--tgt-dir", type=str,
                        default='~/.ncc/CodeSearchNet/codebert/',
                        help='save dir for sentencepiece bpe models or save files')
    # parser.add_argument("--bpe-dir", type=str, default='wordpiece_bpe', help='wordpiece_bpe modal save direction')
    parser.add_argument("--model-type", type=str, default='unigram', help='source data')
    parser.add_argument("--model-prefix", type=str, default='codesearchnet', help='source data')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()
    args.src_dir = os.path.expanduser(args.src_dir)
    args.tgt_dir = os.path.expanduser(args.tgt_dir)

    corpus_modalities = ['code', 'docstring']
    input_files = {
        modality: [
            os.path.join(args.src_dir, args.language, '{}.{}'.format(mode, modality))
            for mode in constants.MODES
        ]
        for modality in corpus_modalities
    }
    corpus_file = {
        modality: os.path.join(args.src_dir, args.language, 'corpus.{}'.format(modality)) for modality in corpus_modalities
    }

    # 1. make corpus
    make_corpus(input_files, corpus_file, corpus_modalities)
    special_symbols = ['[CLS]', '[SEP]', '[MASK]', '[EOL]', '[URL]'] # get_special_symbols(args)
    # 2. spm_train
    corpus_file = ','.join(corpus_file.values())
    model_prefix = os.path.join(args.tgt_dir, args.language, '{}_{}'.format(args.model_prefix, args.language))
    spm_train(corpus_file, model_prefix=model_prefix, vocab_size=args.vocab_size, model_type=args.model_type, special_symbols=special_symbols)
    # vocab2dict(vocab_file='{}.vocab'.format(model_prefix))

    # for modality in args.modalities:
    #     for input_file, output_file in zip(args.input_files[modality], args.output_files[modality]):
    #         LOGGER.info('write {} into {}'.format(input_file, output_file))
    #         write_bpe_files(args, [input_file], [output_file])
