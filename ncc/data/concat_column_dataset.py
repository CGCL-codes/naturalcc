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
from ncc import LOGGER
import sys


class ConcatColumnDataset(NccDataset):

    def __init__(
        self, datasets,
    ):
        super(ConcatColumnDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.real_sizes = [len(d) for d in self.datasets]

    def __getitem__(self, index):
        # src_item = self.src[index] # => tensor([1 3 54 654]), self.src.lines[index]=>str('▁Returns ▁a ▁hash ▁in ▁the ...')
        # tgt_item = self.tgt[index] if self.tgt is not None else None
        return torch.cat([dataset[index] for dataset in self.datasets])


    def __len__(self):
        return self.real_sizes[-1]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return sum([len(dataset[index]) for dataset in self.datasets])

    @property
    def sizes(self):
        # _dataset_sizes = []
        # for ds, sr in zip(self.datasets, self.sample_ratios):
        #     if isinstance(ds.sizes, np.ndarray):
        #         _dataset_sizes.append(np.tile(ds.sizes, sr))
        #     else:
        #         # Only support underlying dataset with single size array.
        #         assert isinstance(ds.sizes, list)
        #         _dataset_sizes.append(np.tile(ds.sizes[0], sr))
        # return np.concatenate(_dataset_sizes)
        # return np.sum(np.concatenate([np.array(dataset.sizes) for dataset in self.datasets], 1), 1)
        return np.sum([np.array(dataset.sizes) for dataset in self.datasets], 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.argsort(self.sizes)

    @property
    def supports_prefetch(self):
        # return (
        #     getattr(self.src, 'supports_prefetch', False)
        #     and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        # )
        return all([getattr(dataset, 'supports_prefetch', False) for dataset in self.datasets])

    def prefetch(self, indices):
        # self.src.prefetch(indices)
        # if self.tgt is not None:
        #     self.tgt.prefetch(indices)
        # if self.align_dataset is not None:
        #     self.align_dataset.prefetch(indices)
        for dataset in self.datasets:
            dataset.prefetch(indices)
