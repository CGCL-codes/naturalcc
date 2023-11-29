# -*- coding: utf-8 -*-

import os
from ncc import (
    __CACHE_DIR__,
    __BPE_DIR__, __LIBS_DIR__,
    LOGGER,
)

DATASET_NAME = 'codesearchnet'
DATASET_DIR = os.path.join(__CACHE_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, 'raw')
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, 'attributes')
DEDUPE_DIR = os.path.join(DATASET_DIR, 'dedupe')
BPE_DIR = __BPE_DIR__
LIBS_DIR = __LIBS_DIR__

LANGUAGES = ['ruby', 'python', 'java', 'go', 'php', 'javascript']
MODES = ['train', 'valid', 'test']

RECURSION_DEPTH = 999  # dfs recursion limitation
# path modality
PATH_NUM = 300  # path modality number
# sbt modality
MAX_SUB_TOKEN_LEN = 5  # we only consider the first 5 sub-tokens from tokenizer
NODE_TMP = 'TMP'

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])
MAX_COMMENT_TOKEN_LIST_LEN = 25
MAX_CODE_TOKEN_LEN = 70
NO_METHOD = '<NO_METHOD>'

__all__ = (
    DATASET_NAME,
    RAW_DIR, ATTRIBUTES_DIR, DEDUPE_DIR,
    BPE_DIR, LIBS_DIR,
    LANGUAGES, MODES,
    LOGGER,

    RECURSION_DEPTH, PATH_NUM, MAX_SUB_TOKEN_LEN,
    MEANINGLESS_TOKENS, COMMENT_END_TOKENS,
    MAX_CODE_TOKEN_LEN,
    MAX_COMMENT_TOKEN_LIST_LEN,
    NO_METHOD,
)
