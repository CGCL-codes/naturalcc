import mmap
import os
import pickle
import shutil
import struct

import numpy as np

from ncc.utils.file_ops import file_io
from .mmap_indexed_dataset import MMapIndexedDataset


def _warmup_mmap_file(path):
    with file_io.open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def bin_file_path(prefix_path):
    return prefix_path + '.bin'


class BinaryIndexedDataset(MMapIndexedDataset):
    class Index(MMapIndexedDataset.Index):
        _HDR_MAGIC = b'BINARYIDX'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    """for with open. this init method"""
                    self._file = file_io.open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)  # self-defined format
                    self._file.write(struct.pack('<Q', 1))  # version number, occupying 8 bit
                    self._file.write(struct.pack('<B', code(dtype)))  # data type, 1 bit

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(bin_file_path(self._path))

        self._bin_buffer_mmap_dot_mmap = open(bin_file_path(self._path), 'rb')
        self._bin_buffer_mmap = mmap.mmap(self._bin_buffer_mmap_dot_mmap.fileno(), 0, access=mmap.ACCESS_READ)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    # @lru_cache(maxsize=8)
    # do not use lru_cache, because sometimes we will access memomery within a batch
    def __getitem__(self, i):
        ptr, size = self._index[i]
        obj = pickle.loads(self._bin_buffer[ptr:ptr + size].tobytes())
        # self._bin_buffer_mmap.seek(ptr)
        # obj = pickle.loads(self._bin_buffer[:size].tobytes())
        return obj

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(bin_file_path(path))
        )

    def __del__(self):
        self._bin_buffer_mmap_dot_mmap.close()
        del self._bin_buffer_mmap
        del self._index


class BinaryIndexedDatasetBuilder(object):
    """memory-mapping dataset builder with index"""

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = file_io.open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []

    def add_item(self, obj):
        # bin file
        bin_buffer = pickle.dumps(obj, protocol=4)
        self._data_file.write(bin_buffer)  # write np.array into C stream
        # idx file
        self._sizes.append(len(bin_buffer))

    def merge_file_(self, another_file):
        """merge sub file(bin/idx) for multi-processing"""
        # Concatenate index
        index = BinaryIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with file_io.open(bin_file_path(another_file), 'rb') as f:
            # append sub-bin/idx files to 1st bin/idx file
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        # assert len(self._sizes) > 0, Exception('{} {}'.format(self._data_file, self._sizes))
        self._data_file.close()

        with BinaryIndexedDataset.Index.writer(index_file, np.int64) as index:
            index.write(self._sizes)
