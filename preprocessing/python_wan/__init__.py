# -*- coding: utf-8 -*-

import os
from ncc import (
    __NCC_DIR__,
    __BPE_DIR__, __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES,
)
from . import dataset_download

DATASET_NAME = 'python_wan'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, RAW)
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, ATTRIBUTES)
BPE_DIR = __BPE_DIR__
LIBS_DIR = __TREE_SITTER_LIBS_DIR__

LANGUAGES = ['python']
MODES = ['train', 'valid', 'test']

RECURSION_DEPTH = 999  # dfs recursion limitation
# path modality
PATH_NUM = 200  # path modality number
# sbt modality
NODE_TMP = 'TMP'

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])
MAX_COMMENT_TOKEN_LIST_LEN = 25
MAX_CODE_TOKEN_LEN = 70
NO_METHOD = '<NO_METHOD>'

__all__ = (
    "DATASET_NAME",
    "RAW_DIR", "LIBS_DIR", "ATTRIBUTES_DIR",
    "LANGUAGES", "MODES",

    "RECURSION_DEPTH", "NODE_TMP",
)
