# -*- coding: utf-8 -*-
import os
from ncc import LOGGER
import gdown
from dataset.codexglue.code_to_text import RAW_DIR
from ncc.utils.path_manager import PathManager

from ncc import __NCC_DIR__

# codesearchnet(feng) dataset
DATASET_DIR = os.path.join(__NCC_DIR__, "demo")
DATASET_URL = "https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h"
out_file = os.path.join(DATASET_DIR, "Cleaned_CodeSearchNet.zip")
if not PathManager.exists(out_file):
    gdown.download(DATASET_URL, output=out_file)
LOGGER.info(f"Dataset has been downloaded at {out_file}")

# inflating data

import zipfile

# DATA_DIR = os.path.join(DATASET_DIR, "completion")
with zipfile.ZipFile(out_file, "r") as writer:
    writer.extractall(path=DATASET_DIR)

LOGGER.info(f"Inflating data at {DATASET_DIR}")

#
