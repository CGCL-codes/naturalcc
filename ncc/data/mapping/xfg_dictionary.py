# -*- coding: utf-8 -*-


import itertools
from collections import Counter
from typing import *

from multiprocessing import Pool

import torch

from ncc.data import constants
from ncc.data.constants import INF
from ncc.data.dictionary import Dictionary


class XFGDicionary(Dictionary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        **kwargs
    ):
        words = line_tokenizer(line, vocab=self, **kwargs) if line_tokenizer is not None else line
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
                # assert idx != self.unk_index, (line, word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_xfg_to_dictionary(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        **kwargs,
    ):
        words = line_tokenizer(line, **kwargs) if line_tokenizer is not None else line
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
        ids = list(itertools.chain(*ids))
        ids = torch.IntTensor(ids)
        return ids

    @staticmethod
    def add_xfg_to_dictionary(filename: str, dict, tokenize: Any, eos_word: Optional[str], num_workers: int):
        def merge_result(counter: Counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        XFGDicionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                XFGDicionary._add_xfg_to_dictionary(
                    filename, tokenize, eos_word
                )
            )
