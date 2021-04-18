# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import torch

from ncc.data import constants
from ncc.data.dictionary import Dictionary
from ncc.data.retrieval.word_bpe_dictionary import WordBpeDicionary


class HybridRetrievalDictionary(object):
    """Hybrid retrieval dictionary, composed of subtoken and bpe dictionaries"""

    def __init__(self, subtoken_dict=None, bpetoken_dict=None):
        self.subtoken_dict = subtoken_dict
        self._subtoken_len = 0 if self.subtoken_dict is None else len(self.subtoken_dict)
        self.bpetoken_dict = bpetoken_dict
        self._bpetoken_len = 0 if self.bpetoken_dict is None else len(self.bpetoken_dict)

    def __len__(self):
        return self._subtoken_len + self._bpetoken_len

    def __getitem__(self, idx):
        if idx < self.__len__():
            if idx < self._subtoken_len:
                return self.subtoken_dict.symbols[idx]
            elif idx < self._subtoken_len + self._bpetoken_len:
                return self.bpetoken_dict.symbols[idx - self._bpetoken_len]
        return constants.UNK

    def __eq__(self, other):
        return (self.subtoken_dict is not None and self.subtoken_dict.indices == other.subtoken_dict.indices) and \
               (self.bpetoken_dict is not None and self.bpetoken_dict.indices == other.bpetoken_dict.indices)

    def __contains__(self, sym):
        return (self.subtoken_dict is not None and sym in self.subtoken_dict.indices) and \
               (self.bpetoken_dict is not None and sym in self.bpetoken_dict.indices)

    def unk(self):
        if self.subtoken_dict:
            return self.subtoken_dict.unk()
        else:
            return None

    @property
    def unk_word(self):
        if self.subtoken_dict:
            return self.subtoken_dict.unk_word
        else:
            return None

    def pad(self):
        if self.subtoken_dict:
            return self.subtoken_dict.pad()
        else:
            return None

    def eow(self):
        if self.bpetoken_dict:
            return self.bpetoken_dict.eow()
        else:
            return None

    def sow(self):
        if self.bpetoken_dict:
            return self.bpetoken_dict.sow()
        else:
            return None

    @classmethod
    def load(cls, f):
        subtoken_dict = Dictionary.load(f)
        splitted_filenames = f.rsplit('.', 2)
        bpe_f = '.'.join([splitted_filenames[0], 'bpe'] + splitted_filenames[-2:])
        bpetoken_dict = WordBpeDicionary.load(bpe_f)
        return cls(subtoken_dict, bpetoken_dict)

    def save(self, f):
        self.subtoken_dict.save(f)
        splitted_filenames = f.rsplit('.', 2)
        bpe_f = '.'.join([splitted_filenames[0], 'bpe'] + splitted_filenames[-2:])
        self.bpetoken_dict.save(bpe_f)

    def index(self, word):
        if word in self.subtoken_dict:
            subtokens = [word]
        else:
            subtokens = self.bpe_tokenize(word)
        subtoken_ids = []
        for token in subtokens:
            if token in self.subtoken_dict:
                subtoken_ids.append(self.subtoken_dict.index(token))
            elif token in self.bpetoken_dict:
                subtoken_ids.append(self.bpetoken_dict.index(token) + self._subtoken_len)
            else:
                subtoken_ids.append(self.subtoken_dict.unk())
        return subtoken_ids

    def encode_line(
        self,
        line,
        line_tokenizer,
        func_name,
        **kwargs
    ):
        words = line_tokenizer(line, func_name=func_name, min_func_len=kwargs.get('min_func_len', None)) \
            if line_tokenizer is not None else line
        ids = []
        for i, word in enumerate(words):
            idx = self.index(word)
            ids.extend(idx)
        ids = torch.Tensor(ids).long()
        return ids

    def bpe_tokenize(self, word: str) -> List[str]:
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.bpetoken_dict.ngram_max])
        sw_tokens = [self.bpetoken_dict.sow_word]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpetoken_dict:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.bpetoken_dict.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.bpetoken_dict.unk_word)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.bpetoken_dict.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.bpetoken_dict.eow_word)
        return sw_tokens
