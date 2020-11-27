# -*- coding: utf-8 -*-

MODES = ['train', 'valid', 'test']

PAD = "[PAD]"
EOS = "</s>"
UNK = "[UNK]"
BOS = "<s>"

MASK = '[MASK]'
SEP = '[SEP]'
URL = '[URL]'
EOL = '[EOL]'  # end of line for code

# for code bert
S_SEP = '[S_SEP]'  # statement seperator
CLS = '[CLS]'
STATEMENT_SEPS = [S_SEP, CLS]
T_MASK = '[T_MASK]'  # token mask

# for path bert
H_SEP = '[H_SEP]'
T_SEP = '[T_SEP]'
P_SEP = '[P_SEP]'  # path seperator
PATH_SEPS = [H_SEP, T_SEP, P_SEP]
LN_MASK = '[LN_MASK]'  # leaf node mask
IN_MASK = '[IN_MASK]'  # intermediate node mask

# for unilm
S2S_SEP = '[S2S_SEP]'
S2S_BOS = '[S2S_BOS]'

# sentencepiece space tag for bep encoding
SP_SPACE = '▁'

# only for code modality in bert
INSERTED = '_inserted'

INF = 99999999
