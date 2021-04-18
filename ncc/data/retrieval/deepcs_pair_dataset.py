# -*- coding: utf-8 -*-


import random

import numpy as np
import torch

from ncc.data.ncc_dataset import NccDataset


def collate_tokens(values, pad_idx, size=None, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if size is None else size
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate(samples, pad_idx, max_subtoken_len):
    if len(samples) == 0:
        return {}

    def merge(key, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, max_subtoken_len['desc' if 'desc' in key else key], eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    # name
    name_tokens = merge('name')
    name_lengths = torch.LongTensor([s['name'].numel() for s in samples])
    # apiseq
    apiseq_tokens = merge('apiseq')
    apiseq_lengths = torch.LongTensor([s['apiseq'].numel() for s in samples])
    # tokens
    tok_tokens = merge('tokens')
    tok_lengths = torch.LongTensor([s['tokens'].numel() for s in samples])
    # pos desc
    pos_desc_tokens = merge('pos_desc')
    pos_desc_lengths = torch.LongTensor([s['pos_desc'].numel() for s in samples])
    # neg desc
    neg_desc_tokens = merge('neg_desc')
    neg_desc_lengths = torch.LongTensor([s['neg_desc'].numel() for s in samples])

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': None,
        'net_input': {
            # code
            'name': name_tokens,
            'name_len': name_lengths,
            'apiseq': apiseq_tokens,
            'apiseq_len': apiseq_lengths,
            'tokens': tok_tokens,
            'tokens_len': tok_lengths,
            # docstring
            'pos_desc': pos_desc_tokens,
            'pos_desc_len': pos_desc_lengths,
            'neg_desc': neg_desc_tokens,
            'neg_desc_len': neg_desc_lengths,
        },
        'target': None,
    }
    return batch


class DeepCSLanguagePairDataset(NccDataset):

    def __init__(
        self, srcs, src_sizes, src_dicts,
        tgts, tgt_sizes, tgt_dicts,
        pad=None, shuffle=True,
        src_max_tokens=None, tgt_max_tokens=None,
        **kwargs
    ):
        self.srcs = srcs
        self.src_sizes = src_sizes
        self.src_dicts = src_dicts
        self.tgts = tgts
        self.tgt_sizes = tgt_sizes
        self.tgt_dicts = tgt_dicts
        self.pad = pad
        self.shuffle = shuffle
        self.max_subtoken_len = dict(
            **{key: length for key, length in zip(srcs.keys(), src_max_tokens)},
            **{key: length for key, length in zip(tgts.keys(), tgt_max_tokens)},
        )

        self.length = None
        for src, src_dataset in self.srcs.items():
            if len(src_dataset) is not None:
                self.length = len(src_dataset)
                break

    def __getitem__(self, index):
        name = self.srcs['name'][index]
        apiseq = self.srcs['apiseq'][index]
        tokens = self.srcs['tokens'][index]
        pos_desc = self.tgts['desc'][index]
        rand_offset = random.randint(0, len(self.tgts['desc']) - 1)
        neg_desc = self.tgts['desc'][rand_offset]
        example = {
            'id': index,
            'name': name,
            'apiseq': apiseq,
            'tokens': tokens,
            'pos_desc': pos_desc,
            'neg_desc': neg_desc,
        }
        return example

    def __len__(self):
        return self.length

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, max_subtoken_len=self.max_subtoken_len)

    def num_tokens(self, index):
        """return a constant value to present no sort by length"""
        return -1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            {src: src_size[index] for src, src_size in self.src_sizes},
            {tgt: tgt_size[index] for tgt, tgt_size in self.tgt_sizes}
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        sizes_gt_0 = np.ones(len(self)).astype(bool)
        for sizes in list(self.src_sizes.values()) + list(self.tgt_sizes.values()):
            if sizes is not None:
                sizes_gt_0 &= sizes > 0
        indices = np.arange(len(self))[sizes_gt_0]
        if self.shuffle:
            np.random.shuffle(indices)
        return indices
