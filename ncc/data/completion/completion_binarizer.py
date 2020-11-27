# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import torch
from collections import Counter


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


from ncc.data.tools.binarizer import Binarizer


class CompletionBinarizer(Binarizer):

    @staticmethod
    def binarize_seperate(
        filename,
        dict,  # Ditionary
        consumer,
        tokenize=None,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        def replaced_consumer(word, idx):
            """save un-recorded token"""
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids_ext = dict.encode_line(
                    line=line,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                for ids, ext in ids_ext:
                    ntok += len(ids)
                    consumer(ids, torch.IntTensor([ext]))
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }
