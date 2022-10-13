# -*- coding: utf-8 -*-

import argparse
import os
from copy import deepcopy

import ujson

from ncc_dataset.stackoverflow import (
    RAW_DIR, ATTRIBUTES_DIR,
    LANGUAGES, MODES,
)
from ncc import LOGGER
from ncc.utils.path_manager import PathManager

PathManager.mkdir(RAW_DIR)
PathManager.mkdir(ATTRIBUTES_DIR)


def flatten(lang):
    modes = deepcopy(MODES)
    # dev
    if os.path.exists(os.path.join(RAW_DIR, lang, 'dev.txt')):
        modes.append('dev')
    # eval
    if os.path.exists(os.path.join(RAW_DIR, lang, 'eval.txt')):
        modes.append('eval')

    for mode in modes:
        raw_filename = os.path.join(RAW_DIR, lang, '{}.txt'.format(mode))
        code_filename = os.path.join(ATTRIBUTES_DIR, lang, '{}.code'.format(mode))
        docstring_filename = os.path.join(ATTRIBUTES_DIR, lang, '{}.docstring'.format(mode))
        id_filename = os.path.join(ATTRIBUTES_DIR, lang, '{}.id'.format(mode))  # docstring id for dev/eval
        os.makedirs(os.path.dirname(code_filename), exist_ok=True)
        LOGGER.info(
            'Flatten {} into code({}) and docstring({}).'.format(raw_filename, code_filename, docstring_filename))
        with open(raw_filename, 'r') as reader, \
            open(code_filename, 'w') as code_writer, open(docstring_filename, 'w') as docstring_writer, \
            open(id_filename, 'w') as id_writer:
            for idx, line in enumerate(reader):
                """example: [\d+]\t[\d+]\t[docstring]\t[code]\t0\n"""
                try:
                    parsed_line = line.rstrip('\n').split('\t')
                    assert len(parsed_line) == 5, AssertionError(idx, line)
                except AssertionError:
                    continue
                id, docstring, code = parsed_line[1].strip(), parsed_line[2].strip(), parsed_line[3].strip()
                docstring, code = map(lambda string: string.replace('\\r\\n', '\n').replace('\\n', '\n'),
                                      (docstring, code,))
                print(ujson.dumps(docstring, ensure_ascii=False), file=docstring_writer)
                print(ujson.dumps(code, ensure_ascii=False), file=code_writer)
                print(ujson.dumps(id, ensure_ascii=False), file=id_writer)

    # dev.ref
    dev_ref_file = os.path.join(RAW_DIR, lang, 'dev.ref.txt')
    if os.path.exists(dev_ref_file):
        docstring_filename = os.path.join(ATTRIBUTES_DIR, lang, 'dev.ref.docstring')  # docstring for dev/eval
        id_filename = os.path.join(ATTRIBUTES_DIR, lang, 'dev.ref.id')  # docstring id for dev/eval
        with open(dev_ref_file, 'r') as reader, \
            open(docstring_filename, 'w') as docstring_writer, open(id_filename, 'w') as id_writer:
            for idx, line in enumerate(reader):
                """example: [\d+]\t[\d+]\t[docstring]\t[code]\t0\n"""
                try:
                    parsed_line = line.rstrip('\n').split('\t')
                    assert len(parsed_line) == 2, AssertionError(idx, line)
                except AssertionError:
                    continue
                id, ref = parsed_line[0].strip(), parsed_line[1].strip()
                ref = ref.replace('\\r\\n', '\n').replace('\\n', '\n')
                print(ujson.dumps(ref, ensure_ascii=False), file=docstring_writer)
                print(ujson.dumps(id, ensure_ascii=False), file=id_writer)

    # eval.ref
    eval_ref_file = os.path.join(RAW_DIR, lang, 'eval.ref.txt')
    if os.path.exists(eval_ref_file):
        docstring_filename = os.path.join(ATTRIBUTES_DIR, lang, 'eval.ref.docstring')  # docstring for dev/eval
        id_filename = os.path.join(ATTRIBUTES_DIR, lang, 'eval.ref.id')  # docstring id for dev/eval
        with open(eval_ref_file, 'r') as reader, \
            open(docstring_filename, 'w') as docstring_writer, open(id_filename, 'w') as id_writer:
            for idx, line in enumerate(reader):
                """example: [\d+]\t[\d+]\t[docstring]\t[code]\t0\n"""
                try:
                    parsed_line = line.rstrip('\n').split('\t')
                    assert len(parsed_line) == 2, AssertionError(idx, line)
                except AssertionError:
                    continue
                id, ref = parsed_line[0].strip(), parsed_line[1].strip()
                ref = ref.replace('\\r\\n', '\n').replace('\\n', '\n')
                print(ujson.dumps(ref, ensure_ascii=False), file=docstring_writer)
                print(ujson.dumps(id, ensure_ascii=False), file=id_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flatten StackOverflow C#/Python/SQL datasets")
    parser.add_argument(
        "--language", "-l", type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
        default=LANGUAGES,
    )
    parser.add_argument(
        "--dataset_dir", "-d", type=str, help="raw dataset download directory",
        default=RAW_DIR,
    )
    parser.add_argument(
        "--ATTRIBUTES_DIR", "-f", type=str, help="data directory of flatten attribute",
        default=ATTRIBUTES_DIR,
    )
    args = parser.parse_args()

    for lang in args.language:
        flatten(lang)
