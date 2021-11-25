# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

from ncc import LOGGER
from ncc.data.ncc_dataset import NccDataset


def collate(samples):
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = torch.stack([s['source'] for s in samples], dim=0)
    src_lengths = torch.LongTensor([s['source_len'] for s in samples])
    source_aux = torch.stack([s['source_aux'] for s in samples])

    target = torch.stack([s['target'] for s in samples])
    ntokens = sum(len(s['target']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_aux': source_aux,
        },
        'target': target,
    }

    return batch


class LanguagePairDataset(NccDataset):

    def __init__(
        self, src, src_sizes, src_dict, src_aux=None,
        tgt=None, tgt_sizes=None, tgt_dict=None, tgt_aux=None,
        left_pad_source=True, pad=None,
        max_source_positions=1024,
        shuffle=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_aux = src_aux
        self.tgt_aux = tgt_aux
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.pad = pad if pad is not None else self.src_dict.pad()
        self.max_source_positions = max_source_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        source_len = src_item.numel()
        pad_tensor = torch.Tensor([self.pad] * (self.max_source_positions - source_len))
        src_item = torch.cat(
            tensors=((pad_tensor, src_item) if self.left_pad_source else (src_item, pad_tensor)),
            dim=-1,
        )
        src_item = src_item.long()

        if self.src_aux is not None:
            source_aux = torch.Tensor([v[index] for v in self.src_aux.values()])
        else:
            source_aux = None
        # if self.tgt_aux is not None:
        #     target_aux = torch.Tensor([v[index] for v in self.tgt_aux.values()])
        # else:
        #     target_aux = None

        example = {
            'id': index,
            'source': src_item,
            'source_len': source_len,
            'source_aux': source_aux,
            'target': tgt_item,
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
        return collate(samples)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        sizes_gt_0 = (self.src_sizes > 0) & (self.tgt_sizes > 0)
        indices = super(LanguagePairDataset, self).ordered_indices()[sizes_gt_0]
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

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
