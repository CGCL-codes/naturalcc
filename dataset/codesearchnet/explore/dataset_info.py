# -*- coding: utf-8 -*-

import os
import ujson
import itertools
from collections import Counter

from dataset.codesearchnet.utils.constants import LANGUAGES, MODES

CUR_DIR = os.path.dirname(__file__)
MODALITIES = [
    'code', 'docstring',
    'code_tokens', 'docstring_tokens',
    'raw_ast'
]

for lang, mode, modality in itertools.product(LANGUAGES, MODES, MODALITIES):
    file = '~/.ncc/code_search_net/flatten/{}/{}.{}'.format(lang, mode, modality)
    file = os.path.expanduser(file)
    with open(file, 'r', encoding='UTF-8') as reader:
        counter = Counter([len(ujson.loads(line)) for line in reader])
    info_file = '{}-{}-{}.json'.format(lang, mode, modality)
    info_file = os.path.join(CUR_DIR, info_file)
    print(info_file)
    with open(info_file, 'w') as writer:
        for length, count in sorted(counter.items(), key=lambda info: info[-1], reverse=True):
            print(ujson.dumps([length, count]), file=writer)
