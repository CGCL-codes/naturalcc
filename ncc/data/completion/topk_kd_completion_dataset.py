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

    teacher_out_ids, teacher_out, prev_ids = [], [], []
    for i, s in enumerate(samples):
        if s.get('topk_idx', None) is not None:
            teacher_out_ids.append(i)
            teacher_out.append(s)
        else:
            prev_ids.append(i)
    batch['prev_ids'] = prev_ids

    if len(teacher_out_ids) > 0:  # contain TeacherOut
        sizes = max(s['topk_idx'].size(0) for s in teacher_out), distill_topk
        out_ids = teacher_out[0]['topk_idx'].new(len(teacher_out_ids), *sizes).fill_(pad_idx).long()
        out_probs = teacher_out[0]['topk_prob'].new(len(teacher_out_ids), *sizes).fill_(pad_idx)

        for i, s in enumerate(teacher_out):
            topk_idx = s['topk_idx'].long()
            out_ids[i, :topk_idx.size(0), :].copy_(topk_idx[:, :distill_topk])
            topk_prob = s['topk_prob']
            out_probs[i, :topk_prob.size(0), :].copy_(topk_prob[:, :distill_topk])
        batch['teacher_out_ids'] = teacher_out_ids
        batch['out_ids'] = out_ids
        batch['out_probs'] = out_probs
    return batch


class TopkKDCompletionDataset(CompletionDataset):

    def __init__(
        self,
        topk_ids=None, topk_probs=None, topk=None, distill_topk=None,
        *args, **kwargs,
    ):
        kwargs['shuffle'] = False
        super().__init__(*args, **kwargs)
        self.topk_ids = topk_ids
        self.topk_probs = topk_probs
        self.topk = topk
        self.distill_topk = distill_topk
        assert bool(topk_ids and topk_probs and topk and distill_topk)

    def __getitem__(self, index):
        example = super().__getitem__(index)
        if self.topk_ids is not None:
            example['topk_idx'] = None if self.topk_ids[index] is None \
                else self.topk_ids[index].view(-1, self.topk)
        if self.topk_probs is not None:
            example['topk_prob'] = None if self.topk_probs[index] is None \
                else self.topk_probs[index].view(-1, self.topk)
        return example

    def collater(self, samples):
        return collate(samples, pad_idx=self.pad, unk_idx=self.unk, attrs=self.attrs, distill_topk=self.distill_topk)
