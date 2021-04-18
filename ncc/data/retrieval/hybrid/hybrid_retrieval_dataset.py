# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import OrderedDict

import numpy as np
import torch

from ncc.data.ncc_dataset import NccDataset
from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)


def collate_tokens(values, pad_idx, max_size=None, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_size is None else max_size
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


def collate(samples, pad_idx, labels):
    if len(samples) == 0:
        return {}

    def merge(part_samples, key):
        return collate_tokens(
            [s[key] for s in part_samples], pad_idx,
            max_size=samples[0]['src_max_size'] if key == 'source' else samples[0]['tgt_max_size'],
        )

    def preprocess_input(part_samples, key):
        input = merge(part_samples, key)
        input_mask = input.ne(pad_idx).float().to(input.device)
        input_len = input_mask.sum(-1, keepdim=True).int()
        return input, input_mask, input_len

    id = [idx['id'] for idx in samples]

    batches = OrderedDict({})
    tgt_tokens, tgt_tokens_mask, tgt_tokens_len = [], [], []
    for lbl in labels:
        lbl_batch = [s for s in samples if s['lang'] == lbl]
        if len(lbl_batch) == 0:
            continue
        lbl_src_tokens, lbl_src_tokens_mask, lbl_src_tokens_len = preprocess_input(lbl_batch, key='source')
        lbl_tgt_tokens, lbl_tgt_tokens_mask, lbl_tgt_tokens_len = preprocess_input(lbl_batch, key='target')
        batches[lbl] = {
            'tokens': lbl_src_tokens,
            'tokens_mask': lbl_src_tokens_mask,
            'tokens_len': lbl_src_tokens_len,
        }
        tgt_tokens.append(lbl_tgt_tokens)
        tgt_tokens_mask.append(lbl_tgt_tokens_mask)
        tgt_tokens_len.append(lbl_tgt_tokens_len)
    tgt_tokens = torch.cat(tgt_tokens, dim=0)
    tgt_tokens_mask = torch.cat(tgt_tokens_mask, dim=0)
    tgt_tokens_len = torch.cat(tgt_tokens_len, dim=0)

    return {
        'id': id,
        'ntokens': len(samples),
        'net_input': {
            **batches,
            'tgt_tokens': tgt_tokens,
            'tgt_tokens_mask': tgt_tokens_mask,
            'tgt_tokens_len': tgt_tokens_len,
        },
    }


class HybridRetrievalDataset(NccDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS, max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        shuffle=True, input_feeding=True,

        # for csn implementation
        src_aux=None, src_aux_sizes=None, src_aux_dict=None,
        tgt_aux=None, tgt_aux_sizes=None, tgt_aux_dict=None,
        fraction_using_func_name=0., labels=None,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding

        self.src_aux = src_aux
        self.src_aux_sizes = None if src_aux_sizes is None else np.array(src_aux_sizes)
        self.src_aux_dict = src_dict if src_aux_dict is None else src_aux_dict

        self.tgt_aux = tgt_aux
        self.tgt_aux_sizes = None if tgt_aux_sizes is None else np.array(tgt_aux_sizes)
        self.tgt_aux_dict = tgt_dict if tgt_aux_dict is None else tgt_aux_dict

        self.fraction_using_func_name = fraction_using_func_name
        self.labels = labels

    def __getitem__(self, index):
        if random.uniform(0., 1.) < self.fraction_using_func_name and \
            (self.src_aux is not None or self.tgt_aux is not None) and \
            (self.src_aux_sizes[index] > 0 or self.tgt_aux_sizes[index] > 0):
            # <code_tokens_wo_func_name, func_name>
            src_item = self.src_aux[index]
            tgt_item = self.tgt_aux[index]
            lang = self.src_aux.get_label(index)
        else:
            # <code_tokens, docstring_tokens>
            src_item = self.src[index]
            tgt_item = self.tgt[index]
            lang = self.src.get_label(index)

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'lang': lang,

            'src_max_size': self.max_source_positions,
            'tgt_max_size': self.max_target_positions,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), labels=self.labels,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return -1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def ordered_indices(self):
        indices = np.arange(len(self))
        if self.shuffle:
            random.shuffle(indices)
        return indices
