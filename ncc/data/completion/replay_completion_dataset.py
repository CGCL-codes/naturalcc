import random

import numpy as np

from ncc.data.constants import DEFAULT_MAX_TARGET_POSITIONS
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
from ncc.data.completion.completion_dataset import CompletionDataset


def collate(samples, pad_idx, unk_idx, attrs=None):
    # no need for left padding
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
        )

    src_tokens = merge('source')
    tgt_tokens = merge('target')

    attr_masks = {attr: [] for attr in attrs} if attrs is not None else None

    extends = []
    max_len = src_tokens.size(-1)
    for i, s in enumerate(samples):
        extends.append(s['extend'])
        if attr_masks is not None:
            for attr in attrs:
                attr_masks[attr].append(s['attr_masks'][attr] + max_len * i)
    if attrs:
        for attr in attrs:
            attr_masks[attr] = np.concatenate(attr_masks[attr], axis=0)

    ntokens = sum(sum(s['target'][s['extend']:] != pad_idx) for s in samples).item()

    batch = {
        'id': [s['id'] for s in samples],
        'net_input': {
            'src_tokens': src_tokens,
        },
        'target': tgt_tokens,
        'attr_masks': attr_masks,
        'extends': extends,
        'ntokens': ntokens,
    }
    return batch


class ReplayCompletionDataset(CompletionDataset):

    def __init__(
        self,
        tgt, tgt_sizes,
        prev_tgts, prev_tgt_sizes,
        tgt_dict, extends=None,
        attrs=None, attr_indices=None, attr_dict=None,
        attrs_mapping=None, reversed_attrs_mapping=None,
        left_pad_source=False, left_pad_target=False,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        shuffle=True,
    ):
        self.tgt = tgt
        self.tgt_sizes = np.array(tgt_sizes)

        self.prev_tgts = prev_tgts
        self.prev_tgt_sizes = prev_tgt_sizes
        self.prev_tgt_indices = []
        for prev_tgt, prev_tgt_szs in zip(self.prev_tgts, self.prev_tgt_sizes):
            indices = np.arange(len(prev_tgt))
            indices = indices[prev_tgt_szs > 1]
            self.prev_tgt_indices.append(indices)

        self.tgt_dict = tgt_dict

        self.extends = extends
        self.attrs = attrs
        self.attr_indices = attr_indices
        self.attr_dict = attr_dict
        self.attrs_mapping = attrs_mapping
        self.reversed_attrs_mapping = reversed_attrs_mapping
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_target_positions = max_target_positions

        self.shuffle = shuffle

        self.pad = self.tgt_dict.pad()
        self.unk = self.tgt_dict.unk()

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS
        # and remove EOS from end of src sentence if it exists.
        # This is useful when we use existing datasets for opposite directions
        #   i.e., when we want to use tgt_dataset as src_dataset and vice versa
        src_item = self.tgt[index][:-1]
        tgt_item = self.tgt[index][1:]

        extend = 0 if self.extends is None else self.extends[index].item()
        if self.attrs_mapping:
            # do not move attr_masks into cuda
            attr_masks = {attr: [] for attr in self.attrs}
            for idx, attr_idx in enumerate(self.attr_indices[index].tolist()[1:][extend:], start=extend):
                if attr_idx in self.reversed_attrs_mapping:
                    attr_masks[self.reversed_attrs_mapping[attr_idx]].append(idx)
            for attr in self.attrs:
                attr_masks[attr] = np.array(attr_masks[attr])
        else:
            attr_masks = None

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,

            'attr_masks': attr_masks,
            'extend': extend,
        }
        return example

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, unk_idx=self.unk, attrs=self.attrs)

    def get_prev_task_batches(self, batch_size):
        def get_prev_item(task_idx, index):
            src_item = self.prev_tgts[task_idx][index][:-1]
            tgt_item = self.prev_tgts[task_idx][index][1:]

            extend = 0 if self.extends is None else self.extends[index].item()
            if self.attrs_mapping:
                # do not move attr_masks into cuda
                attr_masks = {attr: [] for attr in self.attrs}
                for idx, attr_idx in enumerate(self.attr_indices[index].tolist()[1:][extend:], start=extend):
                    if attr_idx in self.reversed_attrs_mapping:
                        attr_masks[self.reversed_attrs_mapping[attr_idx]].append(idx)
                for attr in self.attrs:
                    attr_masks[attr] = np.array(attr_masks[attr])
            else:
                attr_masks = None

            example = {
                'id': index,
                'source': src_item,
                'target': tgt_item,

                'attr_masks': attr_masks,
                'extend': extend,
            }
            return example

        prev_batches = []
        for task_idx, (prev_tgt, prev_ids) in enumerate(zip(self.prev_tgts, self.prev_tgt_indices)):
            np.random.shuffle(prev_ids)
            prev_batches.append(
                self.collater([get_prev_item(task_idx, idx) for idx in prev_ids[:batch_size]])
            )
        return prev_batches
