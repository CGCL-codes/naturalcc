# -*- coding: utf-8 -*-

import os

'''
util_file.py
read/load gz, json, jsonline, yaml
'''
from typing import Dict
import ruamel.yaml as yaml


def recursive_expanduser(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_expanduser(value)
    elif isinstance(obj, str) and obj.startswith('~/'):
        obj = os.path.expanduser(obj)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, str) and val.startswith('~/'):
                obj[i] = os.path.expanduser(val)
    return obj


def contractuser(string):
    # linux
    if str.startswith(string, '/home'):
        username = string.split(os.path.sep)[2]
        contract_string = '~' + string[len(f'/home/{username}'):]
    elif str.startswith(string, '/root'):
        contract_string = '~' + string[len('/root'):]
    # windows
    elif str.startswith(string, 'C:\\Users\\'):
        username = string.split(os.path.sep)[2]
        contract_string = '~' + string[len(f'C:\\Users\\{username}'):]
    else:
        return string
    return contract_string


def recursive_contractuser(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_contractuser(value)
    elif isinstance(obj, str):
        obj = contractuser(obj)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            obj[i] = recursive_contractuser(val)
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
