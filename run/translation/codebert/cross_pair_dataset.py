# -*- coding: utf-8 -*-

import os

import numpy as np
import torch

from ncc.data.dictionary import TransformersDictionary
from ncc.utils.file_ops import (
    file_io, json_io,
)


class CrossPairDataset:
    def __init__(
        self,
        vocab: TransformersDictionary, data_path, mode, src_lang, tgt_lang,
        cls=None, sep=None, pad=None, unk=None, dataset=None, topk=1,
    ):
        self.vocab = vocab
        self.mode = mode
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        src_file = os.path.join(data_path, src_lang, f"{mode}.pkl")
        src_data = file_io.open(src_file, 'rb')

        self.src_code = src_data['code']
        self.src_tokens = src_data['src_tokens']
        self.src_sizes = np.asarray(src_data['src_sizes'], dtype=np.int_)
        self.src_masks = src_data['src_masks']
        del src_data

        tgt_file = os.path.join(data_path, tgt_lang, f"{mode}.pkl")
        tgt_data = file_io.open(tgt_file, 'rb')

        if dataset == 'avatar':
            from dataset.avatar import RAW_DIR
            raw_file = os.path.join(RAW_DIR, "test.jsonl")
            with file_io.open(raw_file, 'r') as reader:
                tgt_code = [json_io.json_loads(line)[tgt_lang][:topk] for line in reader]
            self.tgt_code = tgt_code
        else:
            self.tgt_code = tgt_data['code']
        self.tgt_tokens = tgt_data['tgt_tokens']
        self.tgt_sizes = np.asarray(tgt_data['tgt_sizes'], dtype=np.int_)
        self.tgt_masks = tgt_data['tgt_masks']
        del tgt_data

        self.cls = vocab.cls() if cls is None else cls
        self.sep = vocab.sep() if sep is None else sep
        self.pad = vocab.pad() if pad is None else pad
        self.unk = vocab.unk() if unk is None else unk

    def __getitem__(self, index):
        src_tokens = torch.Tensor(self.src_tokens[index]).long()
        # print(self.vocab.decode(src_tokens, skip_special_tokens=True))
        src_masks = torch.Tensor(self.src_masks[index]).int()
        # assert ((src_tokens != self.pad) == src_masks).all()
        tgt_tokens = torch.Tensor(self.tgt_tokens[index]).long()
        # print(self.vocab.decode(tgt_tokens, skip_special_tokens=True))
        tgt_masks = torch.Tensor(self.tgt_masks[index]).int()
        # assert ((tgt_tokens != self.pad) == tgt_masks).all()
        return {
            'index': index,
            'src_tokens': src_tokens,
            'src_masks': src_masks,
            'tgt_tokens': tgt_tokens,
            'tgt_masks': tgt_masks,
        }

    def __len__(self):
        return len(self.src_code)

    def ordered_indices(self, shuffle=True):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if shuffle:
            indices = np.random.permutation(len(self))
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)


def collater(samples):
    # source
    src_tokens = torch.stack([s['src_tokens'] for s in samples], dim=0)
    src_masks = torch.stack([s['src_masks'] for s in samples], dim=0)
    # target
    tgt_tokens = torch.stack([s['tgt_tokens'] for s in samples], dim=0)
    tgt_masks = torch.stack([s['tgt_masks'] for s in samples], dim=0)
    index = [s['index'] for s in samples]
    return [src_tokens, src_masks, tgt_tokens, tgt_masks, index]
