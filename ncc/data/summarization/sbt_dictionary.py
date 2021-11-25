# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import os
import json
from collections import Counter
from multiprocessing import Pool
import itertools

import torch
from ncc.data.tools.binarizer import safe_readline
from ncc.data.tools import data_utils
from ncc.data import constants
from ncc.utils.file_io import PathManager
from ncc.utils import tokenizer  # import tokenize_line
import json
from ncc.utils import py150_utils

from ..dictionary import Dictionary


class SBTDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad=constants.PAD,
        eos=constants.EOS,
        unk=constants.UNK,
        bos=constants.BOS,
        extra_special_symbols=None,
    ):
        super(SBTDictionary, self).__init__(pad, eos, unk, bos, extra_special_symbols)

    def sbt_index(self, type, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        elif type in self.indices:
            return self.indices[type]
        else:
            return self.unk()

    @staticmethod
    def _add_tok_to_dictionary_single_worker(
        filename: str, tokenize: Any,
        eos_word: Optional[str], worker_id: int = 0, num_workers: int = 1
    ):
        type_counter, counter = Counter(), Counter()
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                sbt_tokens, tokens = tokenize(line)
                type_counter.update(sbt_tokens)
                counter.update(tokens)
                if eos_word is not None:
                    counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return type_counter, counter

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def add_type_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        idx = len(self.symbols)
        self.indices[word] = idx
        self.symbols.append(word)
        self.count.append(n)
        return idx

    @staticmethod
    def add_token_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(type_counter, counter):
            for w, c in sorted(type_counter.items()):
                dict.add_type_symbol(w, max(c, 9999999))
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        SBTDictionary._add_tok_to_dictionary_single_worker,
                        (filename, tokenize, None, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(*r.get())
        else:
            merge_result(
                *SBTDictionary._add_tok_to_dictionary_single_worker(
                    filename, tokenize, None
                )
            )

    def encode_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line) if line_tokenizer is not None else line
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if isinstance(word, str):
                idx = self.index(word)
                if consumer is not None:
                    consumer(word, idx)
            else:
                idx = self.sbt_index(*word)
                if consumer is not None:
                    consumer(word[-1], idx)
            # assert idx != self.unk(), (word) # unk types/tokens still remain
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids
