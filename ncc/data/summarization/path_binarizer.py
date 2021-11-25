# -*- coding: utf-8 -*-

import torch
import itertools
from collections import Counter
from typing import *

from ncc.data.tools.binarizer import Binarizer
from ncc.utils.file_ops import file_io


class PathSummarizationBinarizer(Binarizer):

    @staticmethod
    def path_binarizer(
        filename,
        subtoken_dict,
        consumer,
        tokenize=None,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        type_dict=None,
        **kwargs
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        def binarization(parts, dict):
            part_sizes = [len(p) for p in parts]
            parts = list(itertools.chain(*parts))
            parts = torch.Tensor([dict.index(token) for token in parts]).long()
            parts = parts.split(part_sizes, dim=0)
            return parts

        def encode_path(
            line,
        ):
            heads, bodies, tails = tokenize(line, max_path_num=kwargs['max_path_num'])
            heads = binarization(heads, subtoken_dict)
            bodies = binarization(bodies, type_dict)
            tails = binarization(tails, subtoken_dict)
            paths, path_sizes = [], []
            for head, body, tail in zip(heads, bodies, tails):
                paths.extend([head, body, tail])
                path_sizes.extend([len(head), len(body), len(tail)])
            paths = torch.cat(paths, dim=0)
            path_sizes = torch.Tensor(path_sizes).long()
            assert len(paths) == path_sizes.sum().item()
            return paths, path_sizes

        with file_io.open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = file_io.safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                paths, path_sizes = encode_path(line)
                ntok += len(paths)
                consumer(paths, path_sizes)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }
