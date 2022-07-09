# -*- coding: utf-8 -*-

import os
from ncc import (
    __NCC_DIR__,
    __BPE_DIR__, __TREE_SITTER_LIBS_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES,
)

DATASET_NAME = 'avatar'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME)