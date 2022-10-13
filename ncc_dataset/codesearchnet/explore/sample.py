# -*- coding: utf-8 -*-


import os

import ujson

from dataset.codesearchnet import LANGUAGES

MODALITIES = [
    'code', 'docstring',
    'code_tokens', 'docstring_tokens',
    'raw_ast'
]
CUR_DIR = os.path.dirname(__file__)
MAX_LENGTH = 1000

writer = open('info.txt', 'w')

mode = 'train'
for lang in LANGUAGES:
    print('[{}]'.format(lang), file=writer)
    for modality in MODALITIES:
        with open(os.path.expanduser('~/code_search_net/flatten/{}/{}.{}'.format(lang, mode, modality)),
                  'r') as reader:
            line = reader.readline()
            data = ujson.loads(line)

            if modality == 'raw_ast':
                print('{}[legnth={}]:'.format(modality, len(data)), file=writer)
            else:
                print('{}:'.format(modality), file=writer)

            print(data, file=writer)
            print('\n', file=writer)
