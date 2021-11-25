# -*- coding: utf-8 -*-
import numpy as np
import torch

from ncc.data.indexed_dataset import (
    IndexedCachedDataset,
    IndexedDatasetBuilder,
)


class TeacherOutDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.mmap'
        idx_path = prefix + '.idx'
        builder = TeacherOutDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item


class TeacherOutDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))
