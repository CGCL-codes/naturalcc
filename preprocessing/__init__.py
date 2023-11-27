# -*- coding: utf-8 -*-
import os
import zenodo_client

import preprocessing.typilus.make_dataset
from ncc.utils.logging import LOGGER
if ('NCC' not in os.environ):
    raise FileNotFoundError("The dataset variable $NCC is not set, please first define the variable");
else:
    print(f"NaturalCC dataset and cache path: '{os.getenv('NCC')}'");

from ncc import __NCC_DIR__
TYPILUS_GRAPH_DIR = __NCC_DIR__ + "/typilus_graph"

# This function performs 1st stage of dataset preparation, without the incorporation of ncc
# It can be used to do general-purpose dataset creation.
# from type: [raw, json] or [<dataset_name>].
# [raw]  means the data would be directly read from folder structure.
# [json] means the data is stored in a predefined json file (with the hard-coded header {"id", "code", "language", "filename"}
def make_dataset(from_type, to_type,
                 from_path, to_path=None):
    if(to_type == "typilus"):
        if(from_type == "raw"):
            from preprocessing.typilus.make_dataset import from_raw
            if(to_path == None):
                to_path = TYPILUS_GRAPH_DIR
            preprocessing.typilus.make_dataset.from_raw(from_path, to_path)




# This function performs 2nd stage of dataset preparation, mainly flatten the dataset for ncc usage.

def prepare_dataset(ds: str, **kwargs):
    if (ds == "codesearchnet"):
        from preprocessing import codesearchnet
        codesearchnet.dataset_download.download()
        codesearchnet.attributes_cast.attributes_cast()
    elif (ds == 'python_wan'):
        from preprocessing import python_wan
        python_wan.dataset_download.download()
    elif (ds == "typilus"):
        from preprocessing.typilus import dataset_migration, flatten

        if ((not dataset_migration.check_migrated()) and 'typilus_path' not in kwargs):
            raise FileNotFoundError("Please specify the argument 'typilus_path' for typilus graph path "
                                    "(the directory containing `graph-dataset-split`,"
                                    "if you don't know what it is, please refer to "
                                    "'https://github.com/typilus/typilus/tree/master/src/data_preparation'")

        dataset_migration.dataset_migration(kwargs['typilus_path'] if ('typilus_path' in kwargs) else None)
        flatten.flatten()
    else:
        raise NotImplementedError(f"The required dataset {ds} is currently not supported.")

# This function performs the final stage of dataset preparation, i.e., binarize.
def binarize_dataset(ds: str):
    if (ds == "typilus"):
        from preprocessing.typilus.preprocess import preprocess
        preprocess.dataset_binarize()
    else:
        raise NotImplementedError(f"The required dataset {ds} is currently not supported.")


