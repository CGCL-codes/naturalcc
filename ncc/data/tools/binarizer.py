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
from ncc.utils.file_ops.file_io import (
    safe_readline,
)

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_string(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dict,  # Ditionary
        consumer,
        tokenize=tokenize_string,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
        **kwargs,
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
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                        **kwargs,
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
    def binarize_bpe(
        filename,
        dict,
        consumer,
        reverse_order=False,
        offset=0,
        end=-1,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                line = ujson.loads(line)
                line = ' '.join(line) if isinstance(line, list) else line
                ids = dict.encode_ids(line)
                if reverse_order:
                    words = list(reversed(words))
                ids = torch.IntTensor(ids)

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
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def binarize_trav_trans(
        filename,
        dicts,  # (token_dict, mask_dict)
        consumer,  # (data, ext, ids, )
        tokenize=tokenize_string,
        offset=0,
        end=-1,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        token_dict, mask_dict = dicts
        replaced = Counter()  # un-recorded tokens

        def replaced_consumer(word, idx):
            """save un-recorded token"""
            if idx == token_dict.unk_index and word != token_dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                for data, ext, ids, mask in tokenize(line):
                    data = token_dict.encode_list(data, add_if_not_exist=False, consumer=replaced_consumer)
                    ext = torch.IntTensor([ext])
                    if ids:
                        for key, value in ids.items():
                            if len(value) == 0:
                                ids[key] = torch.IntTensor([-1])
                            else:
                                ids[key] = torch.IntTensor(value)
                    if mask:
                        mask = mask_dict.encode_list(mask, add_if_not_exist=False)

                    consumer(data, ext, ids, mask)
                    nseq += 1
                    ntok += len(data)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }
