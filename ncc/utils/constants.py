# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

LANGUAGES = ['python', 'java', 'go', 'php', 'ruby', 'javascript']
MODES = ['train', 'valid', 'test']

POS_INF = 999999999999999999
NEG_INF = -POS_INF
EPS_ZERO = 1e-12

PAD = 0  # must be 0, can't be changed
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'

NODE_FIX = 'NODEFIX'
DGLGraph_PAD_WORD = -1

CODE_MODALITIES = ['seq', 'sbt', 'tree', 'cfg']

METRICS = ['bleu', 'meteor', 'rouge', 'cider']

# POP_KEYS = ['repo', 'path', 'language', 'original_string', 'partition', 'sha', 'url']

MEANINGLESS_TOKENS = set(['(', ')', '[', ']', '{', '}', ';', '@', '#', ':', '()', '<>', '{}'])
COMMENT_END_TOKENS = set(['{', '[', '('])
SBT_PARENTHESES = ['(_SBT', ')_SBT']

MAX_COMMENT_TOKEN_LIST_LEN = 25
MAX_RAW_COMMENT_LEN = 4
MAX_CODE_TOKEN_LEN = 70  # if the length is bigger than this threshold, skip this code snippet

MAX_TOKEN_SIZE = 50000

# NODE_FIX = 'NODEFIX'  # 1 -> inf
ROOT_NODE_NAME = NODE_FIX + str(1)
NODE_TMP = 'TMP'
# PAD_WORD = '<PAD>'

PAD_TOKEN_IND = 0
UNK_TOKEN_IND = 1
VALID_VOCAB_START_IND = 2
