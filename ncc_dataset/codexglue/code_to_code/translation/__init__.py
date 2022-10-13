# -*- coding: utf-8 -*-
import os
from ncc import (
    __CACHE_DIR__,
    __BPE_DIR__, __LIBS_DIR__,
    LOGGER,
)
from dataset.codexglue.code_to_code import (
    DATASET_DIR
)

DATASET_NAME = "translation"
DATASET_DIR = os.path.join(DATASET_DIR, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, 'raw')
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, 'attributes')
BPE_DIR = __BPE_DIR__
LIBS_DIR = __LIBS_DIR__

LANGUAGES = ['java', 'csharp']
MODES = ['train', 'valid', 'test']

# sbt modality
MAX_SUB_TOKEN_LEN = 5  # we only consider the first 5 sub-tokens from tokenizer
MAX_TOKEN_LEN = 256

OP_FILES = os.path.join(os.path.dirname(__file__), 'parser/operators.json')

__all__ = (
    DATASET_NAME,
    RAW_DIR, ATTRIBUTES_DIR,
    BPE_DIR, LIBS_DIR,
    LANGUAGES, MODES,
    LOGGER,

    MAX_SUB_TOKEN_LEN,
)
