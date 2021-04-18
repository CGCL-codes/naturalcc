# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import *

import torch

from ncc.utils.file_ops import file_io
from ncc.utils.file_ops.file_io import safe_readline


class HybridRetrievalBinarizer:
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=None,
        use_func=False,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        func_offset=0,
        already_numberized=False,
        **kwargs,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        def replaced_consumer(word, idx):
            """save un-recorded token"""
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with file_io.open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            if use_func:
                func_reader = file_io.open(filename[:str.rfind(filename, '.')] + '.func_name', 'r')
                func_reader.seek(func_offset)
            line = safe_readline(f)
            func_name = safe_readline(func_reader) if use_func else None
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
                        func_name=func_name,
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
                func_name = safe_readline(func_reader) if use_func else None
        if use_func:
            func_reader.close()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0
        with file_io.open(filename, "r") as f:
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
    def find_func_offsets(filename, offsets):
        func_filename = filename[:str.rfind(filename, '.')] + '.func_name'
        count = 1
        func_offsets = [0 for _ in range(len(offsets))]
        with file_io.open(filename, "r", encoding="utf-8") as f, \
            file_io.open(func_filename, "r", encoding="utf-8") as func:
            line, _ = f.readline(), func.readline()
            while line:
                if f.tell() == offsets[count]:
                    func_offsets[count] = func.tell()
                    count += 1
                line, _ = f.readline(), func.readline()
        return func_offsets
