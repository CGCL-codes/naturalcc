# -*- coding: utf-8 -*-

import os

from preprocess.codexglue import DATASET_DIR
from ncc import (
    __BPE_DIR__,
    __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES, MODES,
)

DATASET_NAME = 'code_to_text'
DATASET_DIR = os.path.join(DATASET_DIR, DATASET_NAME)
RAW_DIR = os.path.join(DATASET_DIR, RAW)
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, ATTRIBUTES)
BPE_DIR = __BPE_DIR__
LIBS_DIR = __TREE_SITTER_LIBS_DIR__

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript']

PATH_NUM = 300


__all__ = [
    "DATASET_NAME",
    "RAW_DIR", "ATTRIBUTES_DIR",
    "BPE_DIR", "LIBS_DIR",
    "LANGUAGES", "MODES",
]
