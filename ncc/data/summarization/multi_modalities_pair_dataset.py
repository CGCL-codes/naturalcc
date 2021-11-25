# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch
import numpy as np
import random
from ncc import LOGGER
from collections import OrderedDict
from ncc.data.tools import data_utils
from ncc.data.ncc_dataset import NccDataset
from dataset.csn import PATH_NUM


def collate(
    samples,
    pad_idx, eos_idx, src_modalities, left_pad_source=True, left_pad_target=False,
    input_feeding=True, **kwargs,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        if key == 'path':
            random_ids = random.sample(range(PATH_NUM), kwargs.get('num_sample'))
            batch_terminals = [s['path.terminals'] for s in samples]
            head = [[terminals[2 * idx] for idx in random_ids] for terminals in batch_terminals]
            tail = [[terminals[2 * idx + 1] for idx in random_ids] for terminals in batch_terminals]
            batch_paths = [s['path'] for s in samples]
            body = [[path[idx] for idx in random_ids] for path in batch_paths]
            return data_utils.collate_paths(
                head, body, tail,
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )

    id = [s['id'] for s in samples]
    src_tokens, src_lengths = OrderedDict(), OrderedDict()
    # source tokens
    for modality in src_modalities:
        if modality == 'path' and 'path.terminals' in src_modalities:
            # handle path and path.terminals together
            src_tokens[modality] = merge('path', left_pad=left_pad_source)
            src_lengths[modality] = (src_tokens[modality][1] != pad_idx).sum(-1)
        elif modality == 'path.terminals':
            pass
        else:
            src_tokens[modality] = merge(modality, left_pad=left_pad_source)
            src_lengths[modality] = torch.LongTensor([s[modality].numel() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True
            )
    else:
        ntokens = sum(len(s['code']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,  # OrderedDict
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class MultiModalitiesPairDataset(NccDataset):
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
        self, src, src_sizes, src_dicts,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=True,
        align_dataset=None,
        append_bos=False, eos=None,
        **kwargs
    ):
        src_modalities = list(src_dicts.keys())
        if tgt_dict is not None:
            for modality in src_modalities:
                assert src_dicts[modality].pad() == tgt_dict.pad()
                assert src_dicts[modality].eos() == tgt_dict.eos()
                assert src_dicts[modality].unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_modalities = src_modalities
        self.src_sizes = src_sizes  # the src_sizes has already been narray
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dicts = src_dicts
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dicts[src_modalities[0]].eos()
        self.length = OrderedDict()
        for modality in self.src_modalities:
            if modality == 'path':
                self.length[modality] = len(self.src[modality]) // PATH_NUM
            elif modality == 'path.terminals':
                self.length[modality] = len(self.src[modality]) // PATH_NUM // 2
            else:
                self.length[modality] = len(self.src[modality])
        self.num_sample = kwargs.get('num_sample', None)

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = OrderedDict()
        for modality in self.src_modalities:
            if modality == 'path':
                src_item[modality] = [
                    self.src[modality][i] for i in range(index * PATH_NUM, (index + 1) * PATH_NUM)
                ]
            elif modality == 'path.terminals':
                src_item[modality] = [
                    self.src[modality][i] for i in range(index * 2 * PATH_NUM, (index + 1) * 2 * PATH_NUM)
                ]
            else:
                src_item[modality] = self.src[modality][index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dicts['code'].eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dicts['code'].bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dicts['code'].bos()
            if self.src[index][-1] != bos:
                src_item['code'] = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dicts['code'].eos()
            if self.src[index][-1] == eos:
                src_item['code'] = self.src[index][:-1]

        example = {
            'id': index,
            **src_item,
            'target': tgt_item,

        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        """some modality data may be larger than others, therefore we employ the min one"""
        return min(self.length.values())

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
            samples, pad_idx=self.src_dicts[self.src_modalities[0]].pad(), eos_idx=self.eos,
            src_modalities=self.src_modalities,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            num_sample=self.num_sample,  # for path
        )

    def num_tokens(self, index):
        """
        Return the number of tokens in a sample. This value is used to enforce ``--max-tokens`` during batching.
        However, in multi-modalities encoder, sort or filter data by the size of each modality data are meaningless.
        """
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return max(0, self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """
        Return an example's size as a float or tuple. This value is used when filtering a dataset with ``--max-positions``.
        However, in multi-modalities encoder, sort or filter data by the size of each modality data are meaningless.
        """
        # return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return (0, self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

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
