import numpy as np

from ncc.data.wrappers.base_wrapper_dataset import BaseWrapperDataset
from ncc.data.constants import INF


class PlaceholderDataset(BaseWrapperDataset):
    """
    ```portion``` ahead part of dataset
    """

    def __init__(self, placeholder=None, length=INF):
        super().__init__(None)
        self.placeholder = placeholder
        self.length = length
        self._sizes = np.zeros(length, dtype=np.int32)

    def __getitem__(self, index):
        return self.placeholder

    def __len__(self):
        return self.length

    @property
    def sizes(self):
        return self._sizes

    def size(self, index):
        return 0

    def ordered_indices(self):
        return np.arange(len(self))
