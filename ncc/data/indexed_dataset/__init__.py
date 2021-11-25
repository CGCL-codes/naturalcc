# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from .indexed_dataset import (
    IndexedDataset, IndexedCachedDataset, IndexedDatasetBuilder,
)
from .rawtext_indexed_dataset import RawTextIndexedDataset
from .mmap_indexed_dataset import (
    MMapIndexedDataset, MMapIndexedDatasetBuilder,
)
from .seq_indexed_dataset import (
    SeqIndexedDataset, SeqIndexedDatasetBuilder,
)
from .bin_indexed_dataset import (
    BinaryIndexedDataset, BinaryIndexedDatasetBuilder,
)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def infer_dataset_impl(path):
    if RawTextIndexedDataset.exists(path):
        return 'raw'
    elif IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    elif SeqIndexedDataset.exists(path):
        return 'seq'
    elif BinaryIndexedDataset.exists(path):
        return 'bin'
    else:
        return None


def make_builder(out_file, impl, vocab_size=None, dtype=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size) if dtype is None else dtype)
    elif impl == 'bin':
        return BinaryIndexedDatasetBuilder(out_file, dtype=np.int64)
    elif impl == 'seq':
        return SeqIndexedDatasetBuilder(out_file, dtype=np.int32)
    else:
        raise IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None, tokenizer=None):
    if impl == 'raw' and RawTextIndexedDataset.exists(path):
        assert dictionary is not None
        return RawTextIndexedDataset(path, dictionary, tokenizer=tokenizer)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path)
    elif impl == 'bin':
        return BinaryIndexedDataset(path)
    elif impl == 'seq':
        return SeqIndexedDatasetBuilder(path, dtype=np.int32)
    return None


def dataset_exists(path, impl):
    if impl == 'raw':
        return RawTextIndexedDataset.exists(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset.exists(path)
    elif impl == 'bin' and BinaryIndexedDataset.exists(path):
        return BinaryIndexedDataset.exists(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'seq' and SeqIndexedDataset.exists(path):
        return SeqIndexedDataset.exists(path)
    else:
        return False
