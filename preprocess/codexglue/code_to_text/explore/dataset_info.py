# -*- coding: utf-8 -*-

import itertools
import os

from preprocess.codexglue.code_to_text import (
    MODES,
    LANGUAGES,
    ATTRIBUTES_DIR,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)

MODALITIES = [
    # 'code', 'docstring',
    'code_tokens', 'docstring_tokens',
    # 'raw_ast'
]

for lang, modality in itertools.product(LANGUAGES, MODALITIES):
    file_lens = []
    for mode in MODES:
        file = os.path.join(ATTRIBUTES_DIR, lang, f"{mode}.{modality}")
        with file_io.open(file, 'r') as reader:
            file_lens.extend([len(json_io.json_loads(line)) for line in reader])
    print(lang, modality, round(sum(file_lens) / len(file_lens), 2))
