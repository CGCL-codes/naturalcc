# -*- coding: utf-8 -*-

import os
from typing import Dict

import ruamel.yaml as yaml

from ncc import __NCC_DIR__
from ncc.utils.path_manager import PathManager


def recursive_expanduser(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_expanduser(value)
    elif isinstance(obj, str) and obj.startswith('~/'):
        obj = PathManager.expanduser(obj)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, str) and val.startswith('~/'):
                obj[i] = PathManager.expanduser(val)
    return obj


def contractuser(string, old_cache_name=__NCC_DIR__, new_cache_name=__NCC_DIR__):
    # linux
    string = str.replace(string, '.ncc', 'ncc_data')
    # dst_dir = string.rsplit(old_cache_name, maxsplit=1)[-1]
    # if str.startswith(dst_dir, os.path.sep):
    #     dst_dir = dst_dir[len(os.path.sep):]
    # contract_string = os.path.join('~', new_cache_name, dst_dir)
    contract_string = '~' + string.split('ncc_data', maxsplit=1)[-1]
    return contract_string


def recursive_contractuser(obj, old_cache_name=__NCC_DIR__, new_cache_name=__NCC_DIR__):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_contractuser(value, old_cache_name=old_cache_name, new_cache_name=new_cache_name)
    elif isinstance(obj, str) and (str.startswith(obj, '/')):
        obj = contractuser(obj, old_cache_name=old_cache_name, new_cache_name=new_cache_name)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            obj[i] = recursive_contractuser(val, old_cache_name=old_cache_name, new_cache_name=new_cache_name)
    return obj


def load_yaml(yaml_file: str) -> Dict:
    '''
    read yaml file
    :param yaml_file:
    '''
    with open(yaml_file, 'r', encoding='utf-8') as reader:
        args = yaml.safe_load(reader)
    recursive_expanduser(args)
    return args
