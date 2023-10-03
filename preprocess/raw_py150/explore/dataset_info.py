# -*- coding: utf-8 -*-

import os
import ujson
import itertools
from collections import Counter
from ncc.utils.file_ops import (
    file_io,
    json_io,
)

from preprocess.raw_py150 import (
    ATTRIBUTES_DIR,
    MODES,
)

CUR_DIR = os.path.dirname(__file__)
MODALITIES = [
    'code_tokens', 'code_types',
]


def average(arr):
    return round(sum(arr) / len(arr), 2)


for modality in MODALITIES:
    print(modality)
    modality_info = {}
    for mode in MODES:
        file = os.path.join(ATTRIBUTES_DIR, "{}.{}".format(mode, modality))
        with file_io.open(file, 'r') as reader:
            modality_info[mode] = [len(json_io.json_loads(line)) for line in reader]
            modality_info[mode] = [l for l in modality_info[mode] if l > 1]
    for mode in MODES:
        print(
            f"{mode}, len: {len(modality_info[mode])}, avg: {average(modality_info[mode])}, max: {max(modality_info[mode])}, min: {min(modality_info[mode])}")
    lst = list(itertools.chain(*modality_info.values()))
    print(f"all len: {len(lst)}, avg: {average(lst)}, max: {max(lst)}, min: {min(lst)}")
