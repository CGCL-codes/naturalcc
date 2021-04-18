# -*- coding: utf-8 -*-

import os
from ncc import (
    __CACHE_DIR__,
    __BPE_DIR__, __LIBS_DIR__,
    LOGGER,
)

DATASET_NAME = 'py150'
DATASET_DIR = os.path.join(__CACHE_DIR__, DATASET_NAME)

RAW_DIR = os.path.join(DATASET_DIR, 'raw')
ATTRIBUTES_DIR = os.path.join(DATASET_DIR, 'attributes')
BPE_DIR = __BPE_DIR__
LIBS_DIR = __LIBS_DIR__

MODES = ['train', 'test']

__all__ = (
    DATASET_NAME,
    RAW_DIR, BPE_DIR, LIBS_DIR, ATTRIBUTES_DIR,
    MODES, LOGGER,
)
