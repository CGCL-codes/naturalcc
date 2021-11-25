# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ncc.data.wrappers.base_wrapper_dataset import BaseWrapperDataset


class TruncateDataset(BaseWrapperDataset):

    def __init__(self, dataset, truncation_length, truncate_prefix=1):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset
        """
        if truncate_prefix == 1, array => array[:truncation_length]
        else truncate_prefix: 0,  array => array[-truncation_length:]
        """
        self.truncate_prefix = truncate_prefix

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > self.truncation_length:
            if self.truncate_prefix:
                item = item[:self.truncation_length]
            else:
                item = item[-self.truncation_length:]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)
