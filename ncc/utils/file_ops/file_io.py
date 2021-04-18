# -*- coding: utf-8 -*-

import os
import gzip
import pickle
import numpy as np
import builtins


def open(file, mode=None, data=None, **kwargs):
    name_seps = os.path.basename(file).split('.', 1)
    if len(name_seps) > 1:
        file_type = name_seps[-1]

        if file_type == 'jsonl.gz':
            return gzip.GzipFile(file, mode=mode, **kwargs)


        elif file_type in ['npy', 'txt', 'json', 'jsonl', 'jsonlines']:
            return builtins.open(file, mode=mode, **kwargs)

        elif file_type == 'mmap':
            # numpy.mmap file
            if mode == 'r':
                # read
                return np.memmap(file, dtype=kwargs.get('dtype', np.uint16), mode=mode, order='C')
            elif mode == 'w':
                # write with open
                return builtins.open(file, mode=mode, **kwargs)
            else:
                raise NotImplementedError(f"Numpy cannot handle file with {mode}")

        elif file_type == 'pkl':
            # cannot read/writer pkl line by line
            if mode == 'rb':
                with builtins.open(file, mode, **kwargs) as reader:
                    return pickle.load(reader)
            elif mode == 'wb':
                with builtins.open(file, mode, **kwargs) as writer:
                    pickle.dump(data, writer)
            else:
                raise NotImplementedError(f"pickle cannot handle file with {mode}")

    return builtins.open(file, mode=mode, **kwargs)


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets
