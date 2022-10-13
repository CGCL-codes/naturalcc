# -*- coding: utf-8 -*-

import os

from dataset.codesearchnet_feng import (
    DATASET_NAME,
    DATASET_DIR,
    RAW_DIR,
    LIBS_DIR,
    LANGUAGES, MODES,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES,
)

PROJ = 'proj'
RAW_PROJ_DIR = os.path.join(DATASET_DIR, PROJ, RAW)
ATTRIBUTES_PROJ_DIR = os.path.join(DATASET_DIR, PROJ, ATTRIBUTES)

__all__ = (
    DATASET_NAME,
    RAW_DIR,
    RAW_PROJ_DIR, LIBS_DIR, ATTRIBUTES_PROJ_DIR,
    LANGUAGES, MODES,
)
