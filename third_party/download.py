# -*- coding: utf-8 -*-

import argparse
import gdown
import os
import tarfile

from ncc import LOGGER
from ncc.utils.path_manager import PathManager

THIRD_PARTY_LIB_ARCHIVE_MAP = {
    "pycocoevalcap": {
        'url': "https://drive.google.com/uc?id=1RkfVufVc1yPmfq3BeGzsAai5b9D6x5tu",
        'dst_dir': "pycocoevalcap/meteor",
    },
    "programl": {
        'url': "https://drive.google.com/uc?id=1OLrwwFW7NT-hrncoHOazZKngpHN_g8nu",
        'dst_dir': "programl/programl/ir/llvm/internal",
    },
}
THIRD_PART_DIR = os.path.dirname(__file__)


def download(name):
    if name in THIRD_PARTY_LIB_ARCHIVE_MAP:
        url = THIRD_PARTY_LIB_ARCHIVE_MAP[name]['url']
        LOGGER.info(f"Download {name} library from {url}")
        out_dir = os.path.join(THIRD_PART_DIR, THIRD_PARTY_LIB_ARCHIVE_MAP[name]['dst_dir'])
        out_file = os.path.join(out_dir, f"{name}.tar.gz")
        gdown.download(url=url, output=out_file)
        try:
            with tarfile.open(out_file) as reader:
                reader.extractall(out_dir)
            PathManager.rm(out_file)
        except tarfile.ExtractError as err:
            LOGGER.error(err)
            LOGGER.warning(f"{name}.tar.gz is corrupted, please contact us.")
    else:
        raise FileExistsError(f"No {name}.tar.gz in the server. Please build your own BPE models. " \
                              f"Once they are built, you can upload them into the server.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downloading BPE models")
    parser.add_argument(
        "--names", "-n", type=str, nargs='+', help="third party library names",
        default=list(THIRD_PARTY_LIB_ARCHIVE_MAP.keys()),
    )
    args = parser.parse_args()

    for name in args.names:
        download(name)
