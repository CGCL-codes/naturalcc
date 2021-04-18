# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import re
import os
import torch
import ujson
from collections import Counter
from dataset.csn.utils.util import normalize_program
from ncc.data.tools.binarizer import Binarizer
import dgl


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class TypilusBinarizer(Binarizer):
    @staticmethod
    def binarize_nodes(
        filename,
        dict,  # Ditionary
        consumer,
        tokenize=None,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        # def replaced_consumer(word, idx):
        #     """save un-recorded token"""
        #     if idx == dict.unk_index and word != dict.unk_word:
        #         replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_nodes_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        # consumer=replaced_consumer,
                        consumer=None,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()

        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }


    @staticmethod
    def binarize_supernodes(
        filename,
        dict,  # Ditionary
        consumer,
        tokenize=None,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        **kwargs,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        # def replaced_consumer(word, idx):
        #     """save un-recorded token"""
        #     if idx == dict.unk_index and word != dict.unk_word:
        #         replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                node_ids, ids, raw_data = dict.encode_supernodes_line(
                    line=line,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    # consumer=replaced_consumer,
                    consumer=None,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    **kwargs,
                )
                nseq += 1
                ntok += len(ids)
                consumer(node_ids, ids, raw_data)
                line = f.readline()

        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }
