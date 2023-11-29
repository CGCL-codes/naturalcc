# -*- coding: utf-8 -*-

import itertools
import os

from preprocessing.codesearchnet import (
    ATTRIBUTES_DIR,
    LANGUAGES,
    MODES,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)

CUR_DIR = os.path.dirname(__file__)
MODALITIES = [
    'code_tokens', 'docstring_tokens',
]


def average(arr):
    return round(sum(arr) / len(arr), 2)


for lang, modality in itertools.product(LANGUAGES, MODALITIES):
    print(lang, modality)
    modality_info = {}
    for mode in MODES:
        file = os.path.join(ATTRIBUTES_DIR, lang, "{}.{}".format(mode, modality))
        with file_io.open(file, 'r') as reader:
            modality_info[mode] = [len(json_io.json_loads(line)) for line in reader]
    for mode in MODES:
        print(f"{mode}: {average(modality_info[mode])}")
    print(f"all: {average(list(itertools.chain(*modality_info.values())))}")
