# -*- coding: utf-8 -*-

from preprocessing.avatar import DATASET_DIR

SEED = 123456

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

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

DATA_PATH = {
    k: f"{DATASET_DIR}/translation/top{k}/codebert/data-mmap"
    for k in [1, 3, 5]
}
SRC_KEYS = ['src_tokens', 'src_masks']
TGT_KEYS = ['tgt_tokens', 'tgt_masks']
