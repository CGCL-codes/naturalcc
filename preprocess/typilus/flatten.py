import argparse
import glob
import gzip
import os

import ujson

from preprocess.typilus import (
    RAW_DIR, ATTRIBUTES_DIR,
    MODES,
)
from ncc.utils.path_manager import PathManager
from ncc.utils.logging import LOGGER
PathManager.mkdir(RAW_DIR)
PathManager.mkdir(ATTRIBUTES_DIR)


def __flatten(attrs):
    for mode in MODES:
        raw_files = sorted(glob.glob(os.path.join(RAW_DIR, mode, '*')))

        attr_writers = {}
        for attr in attrs:
            attr_file = os.path.join(ATTRIBUTES_DIR, f'{mode}.{attr}')
            os.makedirs(os.path.dirname(attr_file), exist_ok=True)
            attr_writers[attr] = open(attr_file, 'w')

        for raw_file in raw_files:
            with gzip.open(raw_file, 'r') as reader:
                for idx, line in enumerate(reader):
                    line = ujson.loads(line)
                    for attr, info in line.items():
                        if attr in attr_writers:
                            try:
                                print(ujson.dumps(info, ensure_ascii=False), file=attr_writers[attr])
                            except Exception as err:
                                print(err)
                                print(ujson.dumps(None, ensure_ascii=False), file=attr_writers[attr])


def flatten(attrs = ['nodes', 'edges', 'token-sequence', 'supernodes', 'filename']):
    if(os.path.exists(os.path.join(ATTRIBUTES_DIR, "train.code"))):
        LOGGER.info(f"The typilus dataset is already flattened, "
                    f"to re-flatten the dataset, please delete the directory '{ATTRIBUTES_DIR}'.")
        return
    LOGGER.info("Flattening the typilus dataset...")
    __flatten(attrs)
    LOGGER.info("The flatten process is finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flatten Typilus datasets")
    # parser.add_argument(
    #     "--language", "-l", type=str, nargs='+', help="languages constain [{}]".format(LANGUAGES),
    #     default=LANGUAGES,
    # )
    # parser.add_argument(
    #     "--dataset_dir", "-d", type=str, help="raw dataset download directory",
    #     default=RAW_DIR,
    # )
    # parser.add_argument(
    #     "--ATTRIBUTES_DIR", "-f", type=str, help="data directory of flatten attribute",
    #     default=ATTRIBUTES_DIR,
    # )
    parser.add_argument(
        "--attrs", "-a",
        default=['nodes', 'edges', 'token-sequence', 'supernodes', 'filename'],
        type=str, nargs='+',
    )
    args = parser.parse_args()

    __flatten(args.attrs)
