# -*- coding: utf-8 -*-

import itertools
import os

from preprocessing.codesearchnet_feng import (
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
    'ast',
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
        print(
            f"{mode}, avg: {average(modality_info[mode])}, max: {max(modality_info[mode])}, min: {min(modality_info[mode])}")
    lst = list(itertools.chain(*modality_info.values()))
    print(f"all avg: {average(lst)}, max: {max(lst)}, min: {min(lst)}")
