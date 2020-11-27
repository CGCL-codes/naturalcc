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

from ncc.data.dictionary import Dictionary

_SOW = '<sow>'
_EOW = '<eow>'


class RetrievalDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sow_word = _SOW
        self.sow_index = self.add_symbol(_SOW)  # start of (bpe) word
        self.eow_word = _EOW
        self.eow_index = self.add_symbol(_EOW)  # end of (bpe) word
        self.nspecial += 2
        self.ngram_min = kwargs.get('ngram_min', 2)
        self.ngram_max = kwargs.get('ngram_max', 8)

    def sow(self):
        return self.sow_index

    def eow(self):
        return self.eow_index

    def byte_pair_counts(self, words):
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """

        def count_tokens(words):
            """ Count tokens into a BPE vocab """
            token_counts = Counter(words)
            return {' '.join(token): count for token, count in token_counts.items()}

        for token, count in count_tokens(words).items():
            bp_counts = Counter()  # type: Counter
            sub_tokens = token.split(' ')
            joined_tokens = ''.join(sub_tokens)
            token_offsets = [0]
            length = 0
            for ngram in sub_tokens:
                bp_counts[ngram] += count
                length += len(ngram)
                token_offsets += [length]
            for ngram_size in range(self.ngram_min, min(self.ngram_max, len(sub_tokens)) + 1):
                for i in range(len(sub_tokens) - ngram_size + 1):
                    bp_counts[joined_tokens[token_offsets[i]:token_offsets[i + ngram_size]]] += count

            yield bp_counts

    def learn_bpe_vocab(self, words, bpe_vocab_size):
        def trim_vocab(n, vocab):
            """  Deletes all pairs below 10 * vocab size to prevent memory problems """
            pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
            pairs_to_trim = [pair for pair, count in pair_counts[n:]]
            for pair in pairs_to_trim:
                del vocab[pair]

        vocab = Counter()
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            vocab.update(byte_pair_count)
            if (idx + 1) % 10000 == 0:
                trim_vocab(10 * bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:bpe_vocab_size]
        return sorted_bpe_counts

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
        ids = []
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids.extend(idx)
        if append_eos:
            ids.append(self.eos_index)
        ids = torch.Tensor(ids).long()
        return ids

    def bpe_tokenize(self, word: str) -> List[str]:
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.sow_word]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.unk_word)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.eow_word)
        return sw_tokens

    def index(self, word):
        def _index(sym):
            assert isinstance(sym, str)
            if sym in self.indices:
                return self.indices[sym]
            return self.unk_index

        subtokens = self.bpe_tokenize(word)
        subtoken_ids = [_index(token) for token in subtokens]
        return subtoken_ids

    @staticmethod
    def _add_sub_tok_to_dictionary_single_worker(
        filename: str, tokenize: Any,
        eos_word: Optional[str], worker_id: int = 0, num_workers: int = 1
    ) -> Counter:
        counter = Counter()
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
                tokens = tokenize(line)
                counter.update(tokens)
                if eos_word is not None:
                    counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_sub_token_to_dictionary(filename, dict, tokenize=None, num_workers=1):

        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        RetrievalDictionary._add_sub_tok_to_dictionary_single_worker,
                        (filename, tokenize, None, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                RetrievalDictionary._add_sub_tok_to_dictionary_single_worker(
                    filename, tokenize, None
                )
            )

    @staticmethod
    def add_bpe_token_to_dictionary(filename, dict, vocab_size, tokenize=None, num_workers=1):

        def merge_result(counter):
            # merge subtoken counters into a bpe counter
            # from ncc.data.retrieval.tokenizers import learn_bpe_vocab
            counter = dict.learn_bpe_vocab(counter, vocab_size)

            for tok, freq in sorted(counter):
                dict.add_symbol(tok, freq)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        RetrievalDictionary._add_sub_tok_to_dictionary_single_worker,
                        (filename, tokenize, None, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()

            words = Counter()
            for r in results:
                words += r.get()
            merge_result(words)
        else:
            merge_result(
                RetrievalDictionary._add_sub_tok_to_dictionary_single_worker(
                    filename, tokenize, None
                )
            )

    def encode_bpe_line(
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
        # nwords = len(words)
        # ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
        ids = []

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids.append(idx)
        if append_eos:
            ids.append(self.eos_index)
        ids = list(itertools.chain(*ids))
        ids = torch.IntTensor(ids)
        return ids
