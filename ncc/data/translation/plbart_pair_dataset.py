# -*- coding: utf-8 -*-

import numpy as np
import torch

from ncc.data.constants import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils


def collate(
    samples, pad_idx, bos_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        # tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class PLBartPairDataset(NccDataset):
    def __init__(
        self,
        src_dict, tgt_dict,
        src, src_sizes, src_code,
        tgt, tgt_sizes, tgt_code,
        src_lang, tgt_lang,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        shuffle=False,
        pad=None, bos=None, eos=None,
    ):
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.src = src
        self.src_sizes = src_sizes
        self.src_lang = src_lang
        self.src_code = src_code

        self.tgt = tgt
        self.tgt_sizes = tgt_sizes
        self.tgt_lang = tgt_lang
        self.tgt_code = tgt_code

        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        self.shuffle = shuffle

        assert len(src) == len(tgt)

        self.pad = pad if pad is not None else tgt_dict.pad()
        self.bos = bos if bos is not None else tgt_dict.bos()
        self.eos = eos if eos is not None else tgt_dict.eos()

        assert self.src_dict == self.tgt_dict

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.pad, bos_idx=self.bos, eos_idx=self.eos,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
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
