import random
import torch
import numpy as np

from ncc.data.constants import DEFAULT_MAX_TARGET_POSITIONS
from ncc.data.ncc_dataset import NccDataset
from ncc.data.tools import data_utils
from .completion_dataset import CompletionDataset


def collate(samples, pad_idx, unk_idx, attrs=None, distill_topk=None):
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

    batch['distill_indices'] = [idx for idx, s in enumerate(samples) if s['distill']]
    batch['prev_indices'] = [idx for idx, s in enumerate(samples) if not s['distill']]
    return batch


class LifelongKDCompletionDataset(CompletionDataset):

    def __init__(
        self,
        kd_indices,
        *args, **kwargs,
    ):
        self.kd_indices = kd_indices
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        example = super().__getitem__(index)
        example['distill'] = self.kd_indices[index]
        return example

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, unk_idx=self.unk, attrs=self.attrs)
