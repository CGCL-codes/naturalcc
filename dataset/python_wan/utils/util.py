# -*- coding: utf-8 -*-

import itertools
import os
import re
from collections import Counter
from glob import glob

from ncc.data import constants
from ncc.tokenizers.tokenization import _dpu_sub_tokenizer
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.path_manager import PathManager


def get_child_nodes(node):
    # get tree/token children of node
    token_node, tree_node = [], []
    for name in node['children']:
        if name.startswith(constants.NODE_FIX):
            tree_node.append(name)
        else:
            token_node.append(name)
    return token_node, tree_node,


# is string token
is_string = lambda identifier: \
    len(identifier) > 1 and identifier[0] == identifier[-1] \
    and (identifier[0] == '\'' or identifier[0] == '\"')


def jsonl_gz_writeline(jsonl_gz, data_line):
    line_str = json_io.json_dumps(data_line) + '\n'
    line_bytes = line_str.encode('utf-8')
    jsonl_gz.write(line_bytes)


def load_jsonl_gz(filename):
    with file_io.open(filename, mode='rt') as reader:
        return [json_io.json_dumps(line) for line in reader]


def load_jsonl_gzs(filenames):
    return itertools.chain(*[load_jsonl_gz(fn) for fn in filenames])


def raw_file_index(filename):
    return int(filename.split('/')[-1].split('.')[0].split('_')[-1])


def base_file_index(filename):
    return int(filename.split('/')[-1].split('.')[-3])


def load_raw_filenames(data_dir, sort_func=None, debug=False, ):
    # load raw file
    jsonl_gz_files = glob(data_dir)
    if sort_func is not None:
        jsonl_gz_files = sorted(jsonl_gz_files, key=sort_func)
    if debug:
        return [jsonl_gz_files[0]]
    else:
        return jsonl_gz_files


def load_raw_data(data_dir, load_keys):
    raw_data = {}
    for mode in constants.MODES:
        for key in load_keys:
            mode_data_dir = os.path.join(data_dir, key, '{}.*'.format(mode))
            jsonl_gz_files = PathManager.ls(mode_data_dir)
            raw_data[mode] = list(load_jsonl_gzs(jsonl_gz_files))
    return raw_data


def count_and_filter(token_list, min_freq: int):
    counter = Counter(token_list)
    token_freq = counter.most_common()
    filtered_token_freq = {token: freq for token, freq in token_freq if freq > min_freq}
    return filtered_token_freq


def merge_freqency(multi_dicts):
    all_dict = dict()
    for dct in multi_dicts:
        for key, value in dct.items():
            if key in all_dict:
                all_dict[key] += value
            else:
                all_dict[key] = value
    return all_dict


def is_ascii(identifier) -> bool:
    """
    print(is_ascii('\\xAA'))
    print(is_ascii('\\u02C6-'))
    print(is_ascii('0x10000'))
    print(is_ascii('你好'))
    """
    if ('0x' in identifier) or ('\\x' in identifier) or ('\\u' in identifier):  # hex or unicode
        return False
    else:
        return str.isascii(identifier)


def raw_data_len(src_filename):
    raw_data = list(load_jsonl_gz(src_filename))
    return len(raw_data)


def lower(tokens):
    return list(map(str.lower, tokens))


def stress_tokens(tokens):
    return re.split(r'\s+', ' '.join(tokens).strip())


def filter_tokens(tokens):
    return list(itertools.chain(*[_dpu_sub_tokenizer(token.strip()) for token in tokens if len(token.strip()) > 0]))
