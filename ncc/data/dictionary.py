# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool
from typing import (
    Optional,
    Any,
)

import torch

from ncc.data import constants
from ncc.data.tools import data_utils
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.file_ops.file_io import safe_readline
from ncc.utils.path_manager import PathManager


def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad=constants.PAD,
        bos=constants.BOS,
        eos=constants.EOS,
        unk=constants.UNK,
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        if pad is not None:
            self.pad_index = self.add_symbol(pad, n=constants.INF)
        else:
            self.pad_index = -1
        if bos is not None:
            self.bos_index = self.add_symbol(bos, n=constants.INF)
        else:
            self.bos_index = -1
        if eos is not None:
            self.eos_index = self.add_symbol(eos, n=constants.INF)
        else:
            self.eos_index = -1
        if unk is not None:
            self.unk_index = self.add_symbol(unk, n=constants.INF)
        else:
            self.unk_index = -1
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s, n=constants.INF)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        trunc_eos=False,
    ):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, trunc_eos=trunc_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = " ".join(
            token_string(i)
            for i in tensor
            if item(i) not in extra_symbols_to_ignore
        )

        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <[unk]>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

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

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=1):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))
            )
        )

        most_common = c.most_common(nwords - self.nspecial)
        for symbol, count in most_common:
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return getattr(self, 'bos_index', None)

    def pad(self):
        """Helper to get index of pad symbol"""
        return getattr(self, 'pad_index', None)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return getattr(self, 'eos_index', None)

    def unk(self):
        """Helper to get index of unk symbol"""
        return getattr(self, 'unk_index', None)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:
        [<symbol0>, <count0>]\n
        [<symbol1>, <count1>]\n
        ...
        """
        d = cls(pad=None, bos=None, eos=None, unk=None)
        d.add_from_file(f)
        # set pad/bos/eos/unk indices
        if constants.PAD in d.indices:
            d.pad_index = d.indices[constants.PAD]
            d.pad_word = constants.PAD
            d.nspecial += 1
        if constants.BOS in d.indices:
            d.bos_index = d.indices[constants.BOS]
            d.bos_word = constants.BOS
            d.nspecial += 1
        if constants.EOS in d.indices:
            d.eos_index = d.indices[constants.EOS]
            d.eos_word = constants.EOS
            d.nspecial += 1
        if constants.UNK in d.indices:
            d.unk_index = d.indices[constants.UNK]
            d.unk_word = constants.UNK
            d.nspecial += 1
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with file_io.open(f, "r") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                raw_line = json_io.json_loads(line.rstrip())
                line, field = raw_line[:-1], raw_line[-1]
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line[:-1], line[-1]
                else:
                    line = line[0]
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                            .format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdir(os.path.dirname(f))
            with file_io.open(f, "w") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print(json_io.json_dumps([k, v]), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """
        Stores dictionary into a text file
        only pad/bos/eos/unk are auto-saved/load, user defined special tokens are saved into file
        """
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols,
                ex_vals + self.count,
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        **kwargs,
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

    def encode_list(
        self,
        words,
        add_if_not_exist=True,
        consumer=None,
    ):
        """In some cases, line have been tokenized already."""
        nwords = len(words)
        ids = torch.IntTensor(nwords)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        return ids

    def encode_tok(
        self,
        line,
        # line_tokenizer,  # =tokenizer.tokenize_line
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line  # line_tokenizer(line)
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
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename: str, tokenize: Any,
        eos_word: Optional[str], worker_id: int = 0, num_workers: int = 1
    ) -> Counter:
        counter = Counter()
        with file_io.open(filename, "r") as f:
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
    def add_file_to_dictionary(filename: str, dict, tokenize: Any, eos_word: Optional[str], num_workers: int):
        def merge_result(counter: Counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, eos_word
                )
            )

    @staticmethod
    def _add_tok_to_dictionary_single_worker(
        filename: str, tokenize: Any,
        eos_word: Optional[str], worker_id: int = 0, num_workers: int = 1,
    ) -> Counter:
        counter = Counter()
        with file_io.open(filename, "r") as f:
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
    def add_token_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_tok_to_dictionary_single_worker,
                        (filename, tokenize, None, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_tok_to_dictionary_single_worker(
                    filename, tokenize, eos_word=None,
                )
            )


class TruncatedDictionary(object):
    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {},
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()


import itertools
from transformers import RobertaTokenizer


class TransformersDictionary(RobertaTokenizer):

    def pad(self):
        return self.pad_idx

    @property
    def pad_idx(self):
        return self.pad_token_id

    def bos(self):
        return self.bos_idx

    @property
    def bos_idx(self):
        return self.bos_token_id

    def eos(self):
        return self.eos_idx

    @property
    def eos_idx(self):
        return self.eos_token_id

    def unk(self):
        return self.unk_idx

    @property
    def unk_idx(self):
        return self.unk_token_id

    def cls(self):
        return self.cls_idx

    @property
    def cls_idx(self):
        return self.cls_token_id

    def sep(self):
        return self.sep_idx

    @property
    def sep_idx(self):
        return self.sep_token_id

    def mask(self):
        return self.mask_idx

    @property
    def mask_idx(self):
        return self.mask_token_id

    def subtokenize(self, tokens):
        """
        to further tokenize List[str]
        """
        tokens = [
            self.tokenize('@ ' + tok)[1:] if idx != 0 else self.tokenize(tok) \
            for idx, tok in enumerate(tokens)
        ]
        tokens = list(itertools.chain(*tokens))
        return tokens

    def tokens_to_indices(self, tokens):
        return self.convert_tokens_to_ids(tokens)

    def tokens_to_string(self, tokens):
        return self.convert_tokens_to_string(tokens)

    def indices_to_tokens(self, indices):
        return self.convert_ids_to_tokens(indices)

    def indices_to_string(self, indices):
        tokens = self.indices_to_tokens(indices)
        string = self.tokens_to_string(tokens)
        return string

    def string_to_tokens(self, string):
        return self.tokenize(string)

    def string_to_indices(self, string):
        tokens = self.string_to_tokens(string)
        indices = self.tokens_to_indices(tokens)
        return indices

    def add_symbol(self, token, special_tokens=False):
        self.add_tokens(token, special_tokens=special_tokens)
        return self.index(token)

    def index(self, string):
        return self.get_vocab()[string]

    def string(self, indices, **kwargs):
        indices = indices.tolist()
        return self.indices_to_string(indices)
