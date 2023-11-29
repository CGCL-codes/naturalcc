# -*- coding: utf-8 -*-

import os
from ncc import (
    __NCC_DIR__,
    __BPE_DIR__, __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES,
)

DATASET_NAME = 'opencl'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, RAW)
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, ATTRIBUTES)
BPE_DIR = __BPE_DIR__
LIBS_DIR = __TREE_SITTER_LIBS_DIR__

# Although define amd/nvidia as `LANGUAGES` is incorrect, it will make OpenCL APIs consistent with others.
LANGUAGES = ['amd', 'nvidia']
MODES = ['train']

INST2VEC_EMBEDDING = os.path.join(RAW_DIR, 'inst2vec.pkl')
INST2VEC_VOCAB = os.path.join(RAW_DIR, 'vocabulary', 'dic_pickle')
INST2VEC_STMTS = os.path.join(RAW_DIR, 'vocabulary', 'cutoff_stmts_pickle')

__all__ = (
    "DATASET_NAME",
    "RAW_DIR", "ATTRIBUTES_DIR",
    "BPE_DIR", "LIBS_DIR",
    "LANGUAGES", "MODES",
    # inst2vec
    "INST2VEC_EMBEDDING", "INST2VEC_VOCAB", "INST2VEC_STMTS",
)
