# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import glob
import itertools
from ncc import __DEFAULT_DIR__


class PathManager:
    """
        Wrapper for insulating OSS I/O (using Python builtin operations) from
        fvcore's PathManager abstraction (for transparently handling various
        internal backends).
        """

    @staticmethod
    def copy(src_path, dst_path):
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def diff(path1, path2, buffer_size=10240):
        if os.stat(path1) == os.stat(path2):
            return True
        with open(path1, 'rb') as reader1, open(path2, 'rb') as reader2:
            while True:
                buffer1 = reader1.read(buffer_size)
                buffer2 = reader2.read(buffer_size)
                if buffer1 != buffer2:
                    return False
                if not reader2:
                    return True

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def is_file(path):
        return os.path.isfile(path)

    @staticmethod
    def is_dir(path):
        return os.path.isdir(path) or str.startswith(path, '~/')

    @staticmethod
    def ls(path):
        if PathManager.is_dir(path):
            return glob.glob(os.path.join(path, '*'))
        else:
            if '*' in path:
                paths = []
                head, tail = path.split('*', 1)
                heads = glob.glob(head + '*')
                if len(tail) > 0:
                    paths.extend(list(itertools.chain(*
                                                      [PathManager.ls(h + tail) for h in heads]
                                                      )))
                return paths
            else:
                return [path]

    @staticmethod
    def mkdir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path):
        paths = glob.glob(path)
        if len(paths) > 0:
            for p in paths:
                if PathManager.is_file(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)

    @staticmethod
    def expanduser(path):
        if str.startswith(path, '~/'):
            return os.path.join(__DEFAULT_DIR__, path[2:])
        else:
            return path

    @staticmethod
    def copyfileobj(fsrc, fdst):
        shutil.copyfileobj(fsrc, fdst)
