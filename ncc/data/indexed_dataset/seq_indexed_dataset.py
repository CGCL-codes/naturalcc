# -*- coding: utf-8 -*-
import shutil
from functools import lru_cache

import numpy as np

from ncc.data.constants import DEFAULT_MAX_TARGET_POSITIONS
from ncc.utils.file_ops import file_io
from ncc.utils.path_manager import PathManager
from .mmap_indexed_dataset import MMapIndexedDatasetBuilder
from ..ncc_dataset import NccDataset


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def seq_file_path(prefix_path):
    return prefix_path + '.seq'


class SeqIndexedDataset(NccDataset):
    _HDR_MAGIC = b'SEQIDX\x00\x00'
    _dtype = np.int32

    def __init__(self, path):
        self.path = path
        self.read_data(path)

    def read_data(self, path):
        with file_io.open(index_file_path(path), mode='rb') as stream:
            magic_test = stream.read(8)
            assert self._HDR_MAGIC == magic_test, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            buffer = stream.read()
            self._data = np.frombuffer(buffer, dtype=self._dtype)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._data[i]

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path))

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        assert self._HDR_MAGIC == other._HDR_MAGIC
        return self._data == other._data

    def truncate(self, start=0, end=None):
        if end is None:
            end = len(self)
        self._data = self._data[start:end]

    def append(self, new_data):
        self._data = np.concatenate([self._data, new_data._data])

    def clip(self, min_position=0, max_position=DEFAULT_MAX_TARGET_POSITIONS):
        self._data = np.clip(self._data, min_position, max_position)


class SeqIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=SeqIndexedDataset._dtype):
        self._data_file = file_io.open(index_file_path(out_file), 'wb')
        self._data_file.write(SeqIndexedDataset._HDR_MAGIC)
        self._data = []
        self._dtype = dtype

    def add_item(self, idx):
        self._data.append(idx)

    def merge_file_(self, another_file):
        with file_io.open(index_file_path(another_file), 'rb') as f:
            version = f.read(8)
            assert version == SeqIndexedDataset._HDR_MAGIC
            np_array = np.frombuffer(f.read(), dtype=self._dtype)
            self._data.extend(np_array.tolist())

    def finalize(self):
        np_array = np.array(self._data, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._data_file.close()


class SeqIndexedDatasetBuilder(MMapIndexedDatasetBuilder):
    def merge_file_(self, another_file):
        """merge sub file(bin/idx) for multi-processing"""
        # Concatenate index
        index = SeqIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with file_io.open(seq_file_path(another_file), 'rb') as f:
            # append sub-bin/idx files to 1st bin/idx file
            shutil.copyfileobj(f, self._data_file)

    def add_item(self, tensor):
        for t in tensor[..., None]:
            # write an array
            np_array = np.array(t.numpy(), dtype=self._dtype)  # type transform
            # bin file
            self._data_file.write(np_array.tobytes(order='C'))  # write np.array into C stream
            # idx file
            self._sizes.append(np_array.size)

    def finalize(self, index_file):
        # assert len(self._sizes) > 0, Exception('{} {}'.format(self._data_file, self._sizes))
        self._data_file.close()

        with SeqIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
