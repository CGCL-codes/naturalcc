# -*- coding: utf-8 -*-

import os
from dataset import (
    HOSTNAME, USERNAME, DEFAULT_DIR, LIBS_DIR, LOGGER,
)

DATASET_NAME = 'py150'
DATASET_DIR = os.path.join(DEFAULT_DIR, DATASET_NAME)
RAW_DATA_DIR = os.path.join(DATASET_DIR, 'raw')
FLATTEN_DIR = os.path.join(DATASET_DIR, 'flatten')
REFINE_DIR = os.path.join(DATASET_DIR, 'refine')
FILTER_DIR = os.path.join(DATASET_DIR, 'filter')

LANGUAGES = ['python', ]
MODES = ['train', 'test']

__all__ = (
    DATASET_NAME,
    RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR, REFINE_DIR, FILTER_DIR,
    LANGUAGES, MODES,
)
