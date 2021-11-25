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


def collate_tokens(values, pad_idx, max_size=None, eos_idx=None, left_pad=False, move_eos_to_beginning=False, ):
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


def collate(samples, pad_idx, labels, sample_neg,
            max_src_lens, max_tgt_lens,
            src_modalties, tgt_modalities,
            **kwargs):
    if len(samples) == 0:
        return {}

    def merge(part_samples, key, lang, max_size):
        return collate_tokens(
            [s[key][lang] for s in part_samples], pad_idx, max_size,
        )

    def preprocess_input(part_samples, key, langs):
        inputs = OrderedDict()
        for idx, lang in enumerate(langs):
            max_len = max_src_lens[idx] if key == 'source' else max_tgt_lens[idx]
            tokens = merge(part_samples, key, lang, max_size=max_len)
            tokens_mask = tokens.ne(pad_idx).float().to(tokens.device)
            tokens_len = tokens_mask.sum(-1, keepdim=True).int()
            inputs[lang] = {
                'tokens': tokens,
                'tokens_mask': tokens_mask,
                'tokens_len': tokens_len,
            }
        return inputs

    id = [idx['id'] for idx in samples]

    src_batches, tgt_batches, neg_tgt_batches = OrderedDict(), OrderedDict(), OrderedDict()
    for lbl in labels:
        lbl_batch = [s for s in samples if s['lang'] == lbl]
        if len(lbl_batch) == 0:
            continue
        src_batches[lbl] = preprocess_input(lbl_batch, key='source', langs=src_modalties)
        tgt_batches[lbl] = preprocess_input(lbl_batch, key='target', langs=tgt_modalities)
        if sample_neg:
            neg_tgt_batches[lbl] = preprocess_input(lbl_batch, key='neg_target', langs=tgt_modalities)
    if not sample_neg:
        neg_tgt_batches = None

    return {
        'id': id,
        'ntokens': len(samples),
        'nsentences': len(samples),
        'net_input': {
            'src_batches': src_batches,
            'tgt_batches': tgt_batches,
            'neg_tgt_batches': neg_tgt_batches,
        },
    }


class MultiModalitiesRetrievalDataset(NccDataset):
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
        self, srcs, src_sizes, src_dicts,
        tgts=None, tgt_sizes=None, tgt_dicts=None,
        max_source_positions=None, max_target_positions=None,
        pad=None,
        shuffle=True, input_feeding=True,
        sample_neg=False,
        # for csn implementation
        fraction_using_func_name=0., labels=None,
    ):
        self.srcs = srcs
        self.tgts = tgts
        self.src_sizes = src_sizes
        self.tgt_sizes = tgt_sizes if tgt_sizes is not None else None
        self.src_dicts = src_dicts
        self.tgt_dicts = tgt_dicts
        self.src_modalities = list(self.src_dicts.keys())
        self.tgt_modalities = list(self.tgt_dicts.keys())
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding

        self.sample_neg = sample_neg
        self.pad = src_dicts[self.src_modalities[0]].pad()

        self.fraction_using_func_name = fraction_using_func_name
        self.labels = labels

    def __getitem__(self, index):
        # if random.uniform(0., 1.) < self.fraction_using_func_name and \
        #     (self.src_aux_sizes[index] > 0 or self.tgt_aux_sizes[index] > 0):
        # <code_tokens, docstring_tokens>
        src_items = {
            src: item[index]
            for src, item in self.srcs.items()
        }
        tgt_items = {
            tgt: item[index]
            for tgt, item in self.tgts.items()
        }
        lang = self.srcs[self.src_modalities[0]].get_label(index)

        if self.sample_neg:
            rand_offset = random.randint(0, len(self) - 1)
            neg_tgt_items = {
                tgt: item[rand_offset]
                for tgt, item in self.tgts.items()
            }
        else:
            neg_tgt_items = None

        example = {
            'id': index,
            'source': src_items,
            'target': tgt_items,
            'neg_target': neg_tgt_items,
            'lang': lang,
        }
        return example

    def __len__(self):
        return len(self.srcs[self.src_modalities[0]])

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.pad, labels=self.labels, sample_neg=self.sample_neg,
            max_src_lens=self.max_source_positions, max_tgt_lens=self.max_target_positions,
            src_modalties=self.src_modalities, tgt_modalities=self.tgt_modalities,
            fraction_prob=self.fraction_using_func_name,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return -1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            max(sz[index] for sz in self.src_sizes.values()),
            max(sz[index] for sz in self.tgt_sizes.values()) if self.tgt_sizes is not None else 0
        )

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def ordered_indices(self):
        indices = np.arange(len(self))
        if self.shuffle:
            random.shuffle(indices)
        return indices
