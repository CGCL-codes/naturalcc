# -*- coding: utf-8 -*-


import os
from ncc import (
    __CACHE_DIR__,
    __BPE_DIR__, __LIBS_DIR__,
    LOGGER,
)

DATASET_NAME = 'java_hu'
DATASET_DIR = os.path.join(__CACHE_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, 'raw')
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, 'attributes')
BPE_DIR = __BPE_DIR__
LIBS_DIR = __LIBS_DIR__

LANGUAGES = ['java']
MODES = ['train', 'valid', 'test']

RECURSION_DEPTH = 999  # dfs recursion limitation
# path modality
PATH_NUM = 300  # path modality number
# sbt modality
MAX_SUB_TOKEN_LEN = 5  # we only consider the first 5 sub-tokens from tokenizer
SBT_PARENTHESES = ['(_SBT', ')_SBT']
# for binary-AST
NODE_TMP = 'TMP'

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])

__all__ = (
    DATASET_NAME,
    RAW_DIR, LIBS_DIR, ATTRIBUTES_DIR,
    LANGUAGES, MODES,
    LOGGER,

    RECURSION_DEPTH, MAX_SUB_TOKEN_LEN, NODE_TMP,
)
