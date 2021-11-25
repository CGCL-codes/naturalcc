# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os
import platform
import shutil

from ncc import __NCC_DIR__


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
        system = platform.uname().system
        if system in ['Linux', 'Unix']:
            cmd = 'ls'
        elif system in ['Windows']:
            cmd = 'dir'
        else:
            raise NotImplementedError("Unkown System")
        out = os.popen(f"{cmd} {path}")
        out = [line.rstrip('\n') for line in out.readlines()]
        return out

    @staticmethod
    def cp(src_dir, dst_dir):
        system = platform.uname().system
        if system in ['Linux', 'Unix', 'Windows']:
            cmd = 'cp'
        else:
            raise NotImplementedError("Unkown System")
        if os.path.isdir(dst_dir):
            PathManager.mkdir(dst_dir)
        else:
            PathManager.mkdir(os.path.dirname(dst_dir))
        out = os.popen(f"{cmd} -fr {src_dir} {dst_dir}")
        out = [line.rstrip('\n') for line in out.readlines()]
        return out

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
            return os.path.join(__NCC_DIR__, path[2:])
        else:
            return path

    @staticmethod
    def copyfileobj(fsrc, fdst):
        shutil.copyfileobj(fsrc, fdst)

    @staticmethod
    def is_empty(path):
        files = PathManager.ls(path)
        return len(files) == 0
