# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from random import randint
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
import random


def collate(samples, pad_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens([s[key] for s in samples], pad_idx)

    def preprocess_input(key):
        input = merge(key)
        input_mask = input.ne(pad_idx).float().to(input.device)
        input_len = input_mask.sum(-1, keepdim=True)
        return input, input_mask, input_len

    id = [idx['id'] for idx in samples]

    src_tokens, src_tokens_mask, src_tokens_len = preprocess_input(key='source')
    tgt_tokens, tgt_tokens_mask, tgt_tokens_len = preprocess_input(key='target')

    return {
        'id': id,
        'ntokens': tgt_tokens.size(0),

        'net_input': {
            'src_tokens': src_tokens,
            'src_tokens_mask': src_tokens_mask,
            'src_tokens_len': src_tokens_len,

            'tgt_tokens': tgt_tokens,
            'tgt_tokens_mask': tgt_tokens_mask,
            'tgt_tokens_len': tgt_tokens_len,
        },
    }


class RetrievalDataset(NccDataset):
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
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,

        # for csn implementation
        src_aux=None, src_aux_sizes=None, src_aux_dict=None,
        tgt_aux=None, tgt_aux_sizes=None, tgt_aux_dict=None,
        fraction_using_func_name=0.,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
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
        self.src_aux_sizes = np.array(src_aux_sizes)
        self.src_aux_dict = src_dict if src_aux_dict is None else src_aux_dict

        self.tgt_aux = tgt_aux
        self.tgt_aux_sizes = np.array(tgt_aux_sizes)
        self.tgt_aux_dict = tgt_dict if tgt_aux_dict is None else tgt_aux_dict

        self.fraction_using_func_name = fraction_using_func_name

    def __getitem__(self, index):
        if random.uniform(0., 1.) < self.fraction_using_func_name:
            if len(self.tgt_aux[index]) >= 12:
                # <code_tokens_wo_func_name, func_name>
                src_item = self.src_aux[index]
                tgt_item = self.tgt_aux[index]
            else:
                # <code_tokens, docstring_tokens>
                src_item = self.src[index]
                tgt_item = self.tgt[index]
        else:
            # <code_tokens, docstring_tokens>
            src_item = self.src[index]
            tgt_item = self.tgt[index]
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            # 'source_auxiliary': src_aux_item,
            # 'target_auxiliary': tgt_aux_item,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad()
        )

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
