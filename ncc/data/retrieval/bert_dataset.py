# -*- coding: utf-8 -*-


import random
from collections import OrderedDict

import numpy as np
import torch

from ncc.data.ncc_dataset import NccDataset
from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    CLS,
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


def collate(samples, pad_idx, max_size=None):
    if len(samples) == 0:
        return {}

    def merge(part_samples, key, max_size=None):
        return collate_tokens(
            [s[key] for s in part_samples], pad_idx, max_size
        )

    id = [idx['id'] for idx in samples]
    src_tokens = merge(samples, key='item', max_size=max_size)
    src_lengths = torch.LongTensor([s['item'].numel() for s in samples])
    ntokens = src_lengths.sum().item()
    # src_masks = merge(samples, key='mask', max_size=max_size)

    return {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            # 'src_masks': src_masks,
        },
    }


class BertDataset(NccDataset):
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
        max_positions=DEFAULT_MAX_SOURCE_POSITIONS + DEFAULT_MAX_TARGET_POSITIONS,
        append_source_eos=False, append_target_eos=False,
        cls=None, pad=None, eos=None,
        shuffle=True, input_feeding=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.max_positions = max_positions
        self.append_source_eos = append_source_eos
        self.append_target_eos = append_target_eos
        self.shuffle = shuffle
        self.input_feeding = input_feeding

        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.bos() == tgt_dict.bos()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.cls = (src_dict.index(CLS) if CLS in src_dict else src_dict.index(CLS)) \
            if cls is None else cls
        self.pad = src_dict.pad() if pad is None else pad
        self.eos = src_dict.eos() if eos is None else eos

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]

        while True:
            total_length = len(src_item) + len(tgt_item)
            if total_length <= self.max_positions - 3:
                break
            elif len(src_item) > len(tgt_item):
                src_item = src_item[:-1]
            else:
                tgt_item = tgt_item[:-1]

        if self.append_source_eos:
            src_item = torch.cat((src_item, torch.Tensor([self.eos])), dim=0)
        if self.append_target_eos:
            tgt_item = torch.cat((tgt_item, torch.Tensor([self.eos])), dim=0)

        item = torch.cat((torch.Tensor([self.cls]), src_item, tgt_item), dim=0).long()
        # mask = torch.ones_like(item)

        example = {
            'id': index,
            'item': item,
            # 'mask': mask,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.pad, max_size=self.max_positions,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return -1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.max_positions

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
