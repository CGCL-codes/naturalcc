# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict
import numpy as np
from random import randint, shuffle, choice
# from random import random as rand
import random
import torch
from ncc.data.ncc_dataset import NccDataset
from ncc.data import constants
from ncc.data.tools.truncate import truncate_seq
from ncc import LOGGER
import sys


def collate(samples, src_dict, tgt_dict, left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    src_tokens = torch.LongTensor([s['src_tokens'] for s in samples])
    segment_labels = torch.LongTensor([s['segment_labels'] for s in samples])
    attention_mask_unilm = torch.stack([s['attention_mask_unilm'] for s in samples])
    mask_qkv = []
    for s in samples:
        if s['mask_qkv']:
            mask_qkv.append(s['mask_qkv'])
        else:
            mask_qkv.append(None)
    # mask_qkv = torch.LongTensor([s['mask_qkv'] for s in samples])
    masked_ids = torch.LongTensor([s['masked_ids'] for s in samples])
    masked_pos = torch.LongTensor([s['masked_pos'] for s in samples])
    masked_weights = torch.LongTensor([s['masked_weights'] for s in samples])

    example = {
        'net_input': {
            'src_tokens': src_tokens,
            'segment_labels': segment_labels,
            'attention_mask_unilm': attention_mask_unilm,
            'masked_pos': masked_pos,
            # 'mask_qkv': mask_qkv
        },
        'target': masked_ids,
        'ntokens': masked_weights.sum().item(),
        'nsentences': 2,
        'sample_size': masked_ids.size(0),
    }
    return example


class MaskCodeDocstringPairDataset(NccDataset):
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
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
            self, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
            align_dataset=None,
            append_bos=False, eos=None,
            s2s_special_token=False,
            pos_shift=False,
            max_pred=50,
            mask_source_words=False,
            skipgram_prb=0.0,
            skipgram_size=0.0,
            max_len=512,
            mask_prob=0.15,
            num_qkv=0,
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
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones((max_len, max_len), dtype=torch.long))

        self.shuffle = shuffle
        # self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        # self.align_dataset = align_dataset
        # if self.align_dataset is not None:
        #     assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        # self.eos = (eos if eos is not None else src_dict.eos())
        self.s2s_special_token = s2s_special_token
        self.pos_shift = pos_shift
        self.max_pred = max_pred
        self.mask_source_words = mask_source_words
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_prob = mask_prob  # masking probability
        self.num_qkv = num_qkv

    def __getitem__(self, index):
        # => tensor([1 3 54 654]), self.src.lines[index]=>str('▁Returns ▁a ▁hash ▁in ▁the ...')
        src_item = self.src[index]
        tgt_item = self.tgt[index] if self.tgt is not None else None

        src_item, tgt_item, _, _ = truncate_seq(src_item, tgt_item, self.max_len - 3,
                                                self.max_source_positions, self.max_target_positions)
        # TODO: below operators should be considered into truncate_seq, as src/tgt seq has been truncated already
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        if self.pos_shift:
            tgt_item = torch.cat([torch.LongTensor(self.tgt_dict.index(constants.S2S_BOS)), tgt_item])

        # Add Special Tokens
        # if self.s2s_special_token:
        #     item = ['[S2S_CLS]'] + src_item + \
        #              ['[S2S_SEP]'] + tgt_item + ['[SEP]']
        # else:
        #     item = ['[CLS]'] + src_item + ['[SEP]'] + tgt_item + ['[SEP]']
        if self.s2s_special_token:
            item = torch.cat([src_item, torch.LongTensor([self.src_dict.index(constants.S2S_SEP)]), tgt_item,
                              torch.LongTensor([self.src_dict.index(constants.SEP)])])
        else:
            # <CLS> + S1 + <SEP> + S2 + <SEP>
            item = torch.cat([
                torch.LongTensor([self.src_dict.index(constants.CLS)]),
                src_item,
                torch.LongTensor([self.src_dict.index(constants.S_SEP)]),
                tgt_item,
                torch.LongTensor([self.src_dict.index(constants.S_SEP)]),
            ])

        # TODO: assign segment ids to each code statement
        segment_ids = [4] * (len(src_item) + 2) + [5] * (len(tgt_item) + 1)

        if self.pos_shift:  # pos_shift is set to True only when fine-tuning
            n_pred = min(self.max_pred, len(tgt_item))
            masked_pos = [len(src_item) + 1 + i for i in range(len(tgt_item))]
            masked_weights = [1] * n_pred
            masked_ids = tgt_item.tolist()[1:] + [self.src_dict.index(constants.SEP)]
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tgt_item)
            if self.mask_source_words:
                effective_length += len(src_item)
            n_pred = min(self.max_pred, max(1, int(round(effective_length * self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(item.tolist()):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(src_item) + 2) and (tk != self.tgt_dict.index(constants.S2S_BOS)):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(src_item) + 1) and (tk != self.src_dict.index(constants.CLS)) \
                        and (not self.src_dict.symbols[tk].startswith('<SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)
            masked_pos = cand_pos[:n_pred]

            masked_ids = [item.tolist()[pos] for pos in masked_pos]
            for pos in masked_pos:
                if random.random() < 0.8:  # 80%
                    item[pos] = self.src_dict.index(constants.T_MASK)  # '[MASK]'    #
                elif random.random() < 0.5:  # 10%
                    # get random word
                    item[pos] = randint(0, len(self.src_dict) - 1)

            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1] * len(masked_ids)

        # Token Indexing: the item has converted into ids
        # input_ids = self.indexer(tokens)
        input_ids = item.tolist()

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([self.tgt_dict.pad_index] * n_pad)
        segment_ids.extend([self.tgt_dict.pad_index] * n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0] * (len(src_item) + 1) + [1] * (len(tgt_item) + 1)
            mask_qkv.extend([0] * n_pad)
        else:
            mask_qkv = None
        # sys.exit()
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(src_item) + 2].fill_(1)
        second_st, second_end = len(src_item) + 2, len(src_item) + len(tgt_item) + 3
        input_mask[second_st:second_end, second_st:second_end]. \
            copy_(self._tril_matrix[:second_end - second_st, :second_end - second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([self.src_dict.pad_index] * n_pad)
            if masked_pos is not None:
                masked_pos.extend([0] * n_pad)
            if masked_weights is not None:
                masked_weights.extend([0] * n_pad)

        example = {
            'src_tokens': input_ids,  # list
            'segment_labels': segment_ids,  # list
            'attention_mask_unilm': input_mask,  # LongTensor
            'mask_qkv': mask_qkv,  # list
            'masked_ids': masked_ids,  # list
            'masked_pos': masked_pos,  # list
            'masked_weights': masked_weights,  # list
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
        # return collate(
        #     samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
        #     left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        #     input_feeding=self.input_feeding,
        # )
        return collate(
            samples, src_dict=self.src_dict, tgt_dict=self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            # input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index] + self.tgt_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

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
