# -*- coding: utf-8 -*-


import itertools
from collections import Counter
from typing import *

import torch

from ncc.data import constants
from ncc.data.constants import INF
from ncc.data.dictionary import Dictionary


class WordBpeDicionary(Dictionary):
    def __init__(
        self,
        sow=constants.SOW,
        eow=constants.EOW,
        ngram_min=2,
        ngram_max=8,
    ):
        super().__init__(
            pad=None,
            bos=None,
            eos=None,
            unk=None,
            extra_special_symbols=None,
        )
        if sow is not None:
            self.sow_word = sow
            self.sow_index = self.add_symbol(sow, n=INF)  # start of (bpe) word
        if eow is not None:
            self.eow_word = eow
            self.eow_index = self.add_symbol(eow, n=INF)  # end of (bpe) word
        self.nspecial += 2
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def sow(self):
        return self.sow_index

    def eow(self):
        return self.eow_index

    @classmethod
    def load(cls, f):
        d = cls(sow=None, eow=None)
        d.add_from_file(f)
        if constants.SOW in d.indices:
            d.sow_index = d.indices[constants.SOW]
            d.sow_word = constants.SOW
        if constants.EOW in d.indices:
            d.eow_index = d.indices[constants.EOW]
            d.eow_word = constants.EOW
        return d

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
        for w, c in sorted_bpe_counts:
            self.add_symbol(w, c)

    def add_bpe_token_to_dictionary(
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
