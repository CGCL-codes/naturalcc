# -*- coding: utf-8 -*-

import argparse
import os
import tarfile

import gdown

from ncc import (
    __BPE_DIR__,
    LOGGER,
)

os.makedirs(__BPE_DIR__, exist_ok=True)

BPE_MODEL_ARCHIVE_MAP = {
    "csn": "https://drive.google.com/uc?id=1mJaRffUVvPj2R7bFp8qpCvUsGChtfqQ0",
}


def download(name):
    if name in BPE_MODEL_ARCHIVE_MAP:
        url = BPE_MODEL_ARCHIVE_MAP[name]
        LOGGER.info(f"Download {name} BPE model from {url}")
        out_file = os.path.join(__BPE_DIR__, f"{name}.tar.gz")
        gdown.download(url=url, output=out_file)
        try:
            with tarfile.open(out_file) as reader:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(reader, __BPE_DIR__)
            os.remove(out_file)
        except tarfile.ExtractError as err:
            LOGGER.error(__BPE_DIR__)
            LOGGER.warning(f"{name}.tar.gz is corrupted, please contact us.")
    else:
        raise FileExistsError(f"No {name}.tar.gz in the server. Please build your own BPE models. " \
                              f"Once they are built, you can upload them into the server.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downloading BPE models")
    parser.add_argument(
        "--names", "-n", type=str, nargs='+', help="BPE model names",
        default=list(BPE_MODEL_ARCHIVE_MAP.keys()),
    )
    args = parser.parse_args()

    for name in args.names:
        download(name)
