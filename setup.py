#!/usr/bin/env python3
# Copyright (c) HUST, UTS, UIC and Salesforce.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import subprocess
import sys

from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >= 3.6 is required for natural code.')


def write_version_py():
    with open(os.path.join("ncc", "version.txt")) as f:
        version = f.read().strip()

    # write version info to ncc/version.py
    with open(os.path.join("ncc", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open('README.md', encoding='UTF-8') as f:
    readme = f.read()


if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        "ncc.libbleu",
        sources=[
            "ncc/clib/libbleu/libbleu.cpp",
            "ncc/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "ncc.data.data_utils_fast",
        sources=["ncc/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "ncc.data.token_block_utils_fast",
        sources=["ncc/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]


extensions.extend(
    [
        cpp_extension.CppExtension(
            "ncc.libbase",
            sources=[
                "ncc/clib/libbase/balanced_assignment.cpp",
            ],
        ),
        cpp_extension.CppExtension(
            'ncc.libnat',
            sources=[
                'ncc/clib/libnat/edit_dist.cpp',
            ],
        ),
    ]
)

if 'CUDA_HOME' in os.environ:
    extensions.extend(
        [
            cpp_extension.CppExtension(
                'ncc.libnat_cuda',
                sources=[
                    'ncc/clib/libnat_cuda/edit_dist.cu',
                    'ncc/clib/libnat_cuda/binding.cpp'
                ],
            ),
            cpp_extension.CppExtension(
                "ncc.ngram_repeat_block_cuda",
                sources=[
                    "ncc/clib/cuda/ngram_repeat_block_cuda.cpp",
                    "ncc/clib/cuda/ngram_repeat_block_cuda_kernel.cu",
                ],
            ),
        ]
    )

cmdclass = {"build_ext": cpp_extension.BuildExtension}

if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]

    # use CPU build of PyTorch
    dependency_links = [
        "https://download.pytorch.org/whl/cpu/torch-1.7.0%2Bcpu-cp36-cp36m-linux_x86_64.whl"
    ]
else:
    dependency_links = []


import setuptools.command.build_py

class generate_proto(setuptools.command.build_py.build_py):
    """Custom build command."""

    def run(self):
        print('running proto')
        # generate proto files
        proto_path = 'third_party/programl/programl/proto'
        # clean proto files
        subprocess.run(["rm -f {}/*.py".format(proto_path)], shell=True, )
        print(glob.glob('{}/*'.format(proto_path)))
        for file in glob.glob('{}/*'.format(proto_path)):
            args = "--proto_path=. --python_out=. --grpc_python_out=. {}".format(file)
            result = subprocess.call("python -m grpc_tools.protoc " + args, shell=True)
            print("grpc generation result for '{0}': code {1}".format(file, result))


cmdclass['generate_proto'] = generate_proto


if 'clean' in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    subprocess.run(['rm -f ncc/*.so ncc/**/*.so ncc/*.pyd ncc/**/*.pyd'], shell=True)


extra_packages = []
if os.path.exists(os.path.join("ncc", "model_parallel", "megatron", "mpu")):
    extra_packages.append("ncc.model_parallel.megatron.mpu")


def do_setup(package_data):
    setup(
        name='ncc',
        version=version,
        description='NaturalCode: A Benchmark towards Understanding the Naturalness of Source Code Comprehension',
        url='https://github.com/xcodemind/naturalcc',
        classifiers=[
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Artificial Intelligence :: Software Engineering',
        ],
        long_description=readme,
        long_description_content_type='text/markdown',
        # setup_requires=[
        #     'cython',
        #     'numpy',
        #     'setuptools>=18.0',
        # ],
        install_requires=[
            "cffi",
            "cython",
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "torch>=1.13",
            "tqdm",
            "bitarray",
            "torchaudio>=0.8.0",
            "scikit-learn",
            "packaging",
        ],
        dependency_links=dependency_links,
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                # "scripts",
                # "scripts.*",
                "tests",
                "tests.*",
                ]
            ) + extra_packages,
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        # ext_modules=extensions,
        # # test_suite='tests',
        entry_points={
            'console_scripts': [
                'ncc-run = run.main:main',
                "ncc-train = ncc_cli.train:cli_main",
                "ncc-preprocess = ncc_cli.preprocess:cli_main"
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


def get_files(path, relative_to="ncc"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    try:
        # symlink examples into ncc package so package_data accepts them
        ncc_examples = os.path.join("ncc", "examples")
        if "build_ext" not in sys.argv[1:] and not os.path.exists(ncc_examples):
            os.symlink(os.path.join("..", "examples"), ncc_examples)

        package_data = {
            "ncc": (
                get_files(ncc_examples)
                + get_files(os.path.join("ncc", "config"))
            )
        }
        do_setup(package_data)
    finally:
        if "build_ext" not in sys.argv[1:] and os.path.islink(ncc_examples):
            os.unlink(ncc_examples)
