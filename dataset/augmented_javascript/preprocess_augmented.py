#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""
import os
from multiprocessing import Pool
from collections import (namedtuple, OrderedDict, Counter)
from ncc.data import (Dictionary, indexed_dataset)
from ncc.utils.util_file import load_yaml
from ncc import tasks
from ncc.data.tools.binarizer import Binarizer
import sentencepiece as spm
from ncc import LOGGER
from tqdm import tqdm
import pickle
import re
import ujson

_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        LOGGER.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def main(args):
    task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir for {} task'.format(args['preprocess']['task']))
    os.makedirs(args['preprocess']['destdir'], exist_ok=True)

    # def train_path(lang):
    #     return "{}{}".format(args['preprocess']['trainpref'], ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args['preprocess']['destdir'], file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    # target = not args['preprocess']['only_source']

    if not args['preprocess']['srcdict'] and os.path.exists(dict_path(args['preprocess']['source_lang'])):
        raise FileExistsError(dict_path(args['preprocess']['source_lang']))
    # if target and not args['preprocess']['tgtdict'] and os.path.exists(dict_path(args['preprocess']['target_lang'])):
    #     raise FileExistsError(dict_path(args['preprocess']['target_lang']))

    if args['preprocess']['joined_dictionary']:
        pass
        # assert not args['preprocess']['srcdict'] or not args['preprocess']['tgtdict'], \
        #     "cannot use both --srcdict and --tgtdict with --joined-dictionary"
        # if args['preprocess']['srcdict']:
        #     src_dict = task.load_dictionary(args['preprocess']['srcdict'])
        # elif args['preprocess']['tgtdict']:
        #     src_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        # else:
        #     LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
        #     exit()
        #
        # tgt_dict = src_dict
        #
        # # Load sentencepiece (sp) module
        # if args['preprocess']['src_sp']:
        #     src_sp = spm.SentencePieceProcessor()
        #     src_sp.load(args['preprocess']['src_sp'])
        # elif args['preprocess']['tgt_sp']:
        #     src_sp = spm.SentencePieceProcessor()
        #     src_sp.load(args['preprocess']['tgt_sp'])
        # else:
        #     LOGGER.error('Please assign the sentencepiece model path.')
        #     exit()
        # tgt_sp = src_sp

    else:
        if args['preprocess']['srcdict'] and args['preprocess']['src_sp']:
            src_dict = task.load_dictionary(args['preprocess']['srcdict'])
            src_sp = spm.SentencePieceProcessor()
            src_sp.load(args['preprocess']['src_sp'])
        else:
            LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
            exit()

        # if target:
        #     if args['preprocess']['tgtdict'] and args['preprocess']['tgt_sp']:
        #         tgt_dict = task.load_dictionary(args['preprocess']['tgtdict'])
        #         tgt_sp = spm.SentencePieceProcessor()
        #         tgt_sp.load(args['preprocess']['tgt_sp'])
        #     else:
        #         # assert args['preprocess']['trainpref'], "--trainpref must be set if --tgtdict is not specified"
        #         # tgt_dict = build_dictionary([train_path(args['preprocess']['target_lang'])], tgt=True)
        #         LOGGER.error('Please run sentencepiece to generate the model and vocab files first.')
        #         exit()
        # else:
        #     tgt_dict = None
        #     tgt_sp = None

    # 2. ***************build dataset********************
    def make_dataset(vocab, sp, input_prefix, output_prefix, lang, min_alternatives=2, num_workers=1):
        if args['preprocess']['dataset_impl'] == "raw":
            examples = pickle.load(open(input_prefix, 'rb'))
            # examples = ujson.load(open(input_prefix, 'rb'))
            # [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]
            examples = list(map(sorted, map(list, examples)))
            examples = list(filter(lambda ex: len(ex) >= min_alternatives, examples))
            output_file = dest_path(output_prefix + '.sp.json', lang=None)
            LOGGER.info('Writing data in {}'.format(output_file))
            with open(output_file, 'w', encoding="utf-8") as output_file:
                for example in tqdm(examples):
                    programs = []
                    for program in example:
                        program = normalize_program(program)
                        program = sp.EncodeAsPieces(program)
                        programs.append(program)
                    print(ujson.dumps(programs, ensure_ascii=False), file=output_file)

    def make_all(lang, vocab, sp):
        if args['preprocess']['trainpref']:
            make_dataset(vocab, sp, args['preprocess']['trainpref'], "javascript_augmented_debug", lang,
                         num_workers=args['preprocess']['workers'])
        # if args['preprocess']['validpref']:
        #     for k, validpref in enumerate(args['preprocess']['validpref'].split(",")):
        #         outprefix = "valid{}".format(k) if k > 0 else "valid"
        #         make_dataset(vocab, sp, validpref, outprefix, lang, num_workers=args['preprocess']['workers'])
        # if args['preprocess']['testpref']:
        #     for k, testpref in enumerate(args['preprocess']['testpref'].split(",")):
        #         outprefix = "test{}".format(k) if k > 0 else "test"
        #         make_dataset(vocab, sp, testpref, outprefix, lang, num_workers=args['preprocess']['workers'])

    make_all(args['preprocess']['source_lang'], src_dict, src_sp)
    # if target:
    #     make_all(args['preprocess']['target_lang'], tgt_dict, tgt_sp)


def cli_main():
    Argues = namedtuple('Argues', 'yaml')
    args_ = Argues('preprocess_augmented.yml')  # train_sl
    LOGGER.info(args_)
    yaml_file = os.path.join(os.path.dirname(__file__), 'config', args_.yaml)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
