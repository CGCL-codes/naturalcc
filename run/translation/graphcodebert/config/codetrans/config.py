# -*- coding: utf-8 -*-

from preprocess.codexglue.code_to_code.translation import DATASET_DIR

SEED = 123456

MAX_SOURCE_LENGTH = 320
MAX_TARGET_LENGTH = 256

TRAIN_EPOCHS = 100
LR = 5e-5
LOG_INTERVAL = 500
SAVE_INTERVAL = 5
EARLY_STOP = 10

# train
TRAIN_BATCH_SIZE = 4
DEV_BATCH_SIZE = 32

# # debug
# TRAIN_BATCH_SIZE = 2
# DEV_BATCH_SIZE = 2

DATA_PATH = f"{DATASET_DIR}/graphcodebert/data-mmap"
SRC_KEYS = ['src_tokens', 'src_positions', 'dfg2code', 'dfg2dfg', 'src_masks']
TGT_KEYS = ['tgt_tokens', 'tgt_masks']
