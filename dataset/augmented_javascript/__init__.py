# -*- coding: utf-8 -*-

import os
from ncc import (
    __CACHE_DIR__,
)

DATASET_NAME = 'augmented_javascript'
DATASET_DIR = os.path.join(__CACHE_DIR__, DATASET_NAME)
RAW_DATA_DIR = os.path.join(DATASET_DIR, 'raw')
LIBS_DIR = os.path.join(DATASET_DIR, 'libs')
FLATTEN_DIR = os.path.join(DATASET_DIR, 'flatten')
REFINE_DIR = os.path.join(DATASET_DIR, 'refine')
FILTER_DIR = os.path.join(DATASET_DIR, 'filter')

RAW_DATA_DIR_TYPE_PREDICTION = os.path.join(DATASET_DIR, 'type_prediction/raw')

MODES = ['train', 'valid', 'test']

__all__ = (
    DATASET_NAME,
    RAW_DATA_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR,
    RAW_DATA_DIR_TYPE_PREDICTION,
    MODES,
)
