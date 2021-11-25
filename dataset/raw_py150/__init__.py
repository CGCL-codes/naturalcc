# -*- coding: utf-8 -*-

import os
from ncc import (
    __NCC_DIR__,
    __BPE_DIR__, __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES,
)

DATASET_NAME = 'raw_py150'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, RAW)
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, ATTRIBUTES)
BPE_DIR = __BPE_DIR__
LIBS_DIR = __TREE_SITTER_LIBS_DIR__

MODES = ['train', 'test']

__all__ = (
    "DATASET_NAME",
    "RAW_DIR", "BPE_DIR", "LIBS_DIR", "ATTRIBUTES_DIR",
    "MODES",
)
