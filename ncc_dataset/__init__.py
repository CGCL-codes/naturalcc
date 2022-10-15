# -*- coding: utf-8 -*-
import os
import zenodo_client
from ncc.utils.logging import LOGGER
if ('NCC' not in os.environ):
    raise FileNotFoundError("The dataset variable $NCC is not set, please first define the variable");
else:
    print(f"NaturalCC dataset and cache path: '{os.getenv('NCC')}'");

def prepare_dataset(ds: str, **kwargs):
    if(ds == "codesearchnet"):
        from ncc_dataset import codesearchnet
        codesearchnet.dataset_download.download()
        codesearchnet.attributes_cast.attributes_cast()
    elif(ds == 'python_wan'):
        from ncc_dataset import python_wan
        python_wan.dataset_download.download()
    elif(ds == "typilus"):
        from ncc_dataset.typilus import dataset_migration, flatten

        if((not dataset_migration.check_migrated()) and 'typilus_path' not in kwargs):
            raise FileNotFoundError("Please specify the argument 'typilus_path' for typilus graph path "
                                    "(the directory containing `graph-dataset-split`,"
                                    "if you don't know what it is, please refer to "
                                    "'https://github.com/typilus/typilus/tree/master/src/data_preparation'")

        dataset_migration.dataset_migration(kwargs['typilus_path'] if('typilus_path' in kwargs) else None)
        flatten.flatten()
    else:
        raise FileNotFoundError(f"The required dataset {ds} is currently not supported.")




