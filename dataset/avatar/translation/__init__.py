# -*- coding: utf-8 -*-
import os

from ncc import (
    __NCC_DIR__, __BPE_DIR__, __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES, MODES,
)

DATASET_NAME = 'avatar'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME, "translation")

RAW_DIR = os.path.join(DATASET_DIR, RAW)
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, ATTRIBUTES)
BPE_DIR = __BPE_DIR__
LIBS_DIR = __TREE_SITTER_LIBS_DIR__

LANGUAGES = ['java', 'python']

__all__ = (
    DATASET_NAME,
    RAW_DIR, ATTRIBUTES_DIR,
    BPE_DIR, LIBS_DIR,
    LANGUAGES, MODES,
)
