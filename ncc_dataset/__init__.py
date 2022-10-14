# -*- coding: utf-8 -*-
import os
if ('NCC' not in os.environ):
    raise FileNotFoundError("The dataset variable $NCC is not set, please first define the variable");
else:
    print(f"NaturalCC dataset and cache path: '{os.getenv('NCC')}'");

def prepare_dataset(ds: str):
    if(ds == "codesearchnet"):
        from ncc_dataset.codesearchnet import dataset_download, attributes_cast
        dataset_download.download()
        attributes_cast.attributes_cast()
    else:
        raise FileNotFoundError(f"The required dataset {ds} is currently not supported.")
