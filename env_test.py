# -*- coding: utf-8 -*-

"""
check whether your $env meet NaturalCC basic libraries
"""
import importlib
import platform
from collections import namedtuple

import gpustat
import psutil
from packaging.version import parse as parse_version

from ncc import LOGGER

# name: library name
# key: import key
# version: lowest version
Library = namedtuple('Library', ['name', 'key', 'version'])

LIBRARY_INFO = [
    Library(name='Python', key='sys', version='3.5'),
    Library(name='Pytorch', key='torch', version='1.4.0'),
    Library(name='TreeSitter', key='tree_sitter', version='0.2.2'),
    Library(name='DGL', key='dgl', version='0.5.0'),
    Library(name='Cython', key='cython', version='0.29.0'),
    Library(name='Deprecated', key='deprecated', version='1.2.0'),
]


def machine_info():
    print("\n" + "=" * 10 + 'Machine information' + "=" * 10)
    system_info = platform.uname()
    print(f"OS: {system_info.system}-{system_info.version}")
    print(f"User: {system_info.node}")
    # cpu
    print(f"CPUs: {psutil.cpu_count(logical=False)}")
    print(f"Threads: {psutil.cpu_count()}")
    cpufreq = psutil.cpu_freq()
    print(f"CPU Frequency: {cpufreq.min / 1024:.2f} ~ {cpufreq.max / 1024:.2f} Ghz")
    # mem
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.total / (1024 ** 3):.2f} GB")
    # gpu
    gpustat.print_gpustat()
    import torch

    print(f"CUDA: {torch.version.cuda}")
    print(f"Cudnn: {torch.backends.cudnn.version()}")


machine_info()


def check_version(libraries):
    print("\n" + "=" * 10 + 'Python interpreter information' + "=" * 10)

    def _check(library: Library):
        lowest_version = parse_version(library.version)
        if library.name == 'Python':
            version = platform.python_version()
        else:
            try:
                lib = importlib.import_module(library.key)
            except:
                LOGGER.error(f"You do not install [{library.name}], please install it.")

            try:
                version = lib.__version__
            except Exception as err:
                LOGGER.info(f"Cannot get version of [{library.key}], please check it via \"pip list\"")
                return
        current_version = parse_version(version)
        try:
            assert current_version >= lowest_version
            print(f"[{library.name}] version({version}) >= required version({library.version}).")
        except AssertionError as err:
            LOGGER.error(
                f"{'!' * 5} [{library.name}] version({version}) < required version({library.version}). {'!' * 5}")

    for lib in libraries:
        _check(lib)


check_version(LIBRARY_INFO)
