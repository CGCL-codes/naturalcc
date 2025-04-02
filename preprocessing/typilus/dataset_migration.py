import os
from ncc import (
    __NCC_DIR__,
)
from ncc.data.constants import (
    RAW, ATTRIBUTES, MODES,
)
from ncc.utils.logging import LOGGER
DATASET_NAME = 'typilus'
DATASET_DIR = os.path.join(__NCC_DIR__, DATASET_NAME)
RAW_DATA = os.path.join(DATASET_DIR, RAW)


def check_migrated():
    if(os.path.exists(os.path.join(RAW_DATA, "train"))):
        return True
    return False


def dataset_migration(typilus_path : str):
    if(typilus_path is None and not check_migrated()):
        raise AssertionError("The argument `typilus_path` is expected to be provided.")
    if(check_migrated()):
        LOGGER.info(f"The typilus graph is migrated to NaturalCC directory. \n"
                    f"To re-migrate, please delete the directory '{RAW_DATA}'")
        return
    if(not os.path.exists(os.path.join(typilus_path, 'graph-dataset-split'))):
        raise FileNotFoundError(f"The specified path '{typilus_path}' is not a typilus graph data directory.")
    LOGGER.info("Migrating the generated typilus graph to NaturalCC dataset directory...")
    os.system(f"mkdir -p '{RAW_DATA}'")
    os.system(f"cd '{typilus_path}/graph-dataset-split' && cp -r train test valid '{RAW_DATA}'")
    LOGGER.info("The migration process is finished.")

