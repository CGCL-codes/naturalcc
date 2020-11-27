# -*- coding: utf-8 -*-
import os
from typing import Dict, Tuple, List, Any, Union
from glob import glob
import gzip, json, jsonlines
import itertools
import math
from ncc.utils import constants
from collections import Counter
from copy import deepcopy
from joblib import Parallel, delayed

# Such methods are widely used. Therefore, we define them ahead of other functions

# get children with NODE_FIX
get_tree_children_func = lambda node: [name for name in node['children'] if name.startswith(constants.NODE_FIX)]

# get children without NODE_FIX
get_token_children_func = lambda node: [name for name in node['children'] if not name.startswith(constants.NODE_FIX)]


def get_child_nodes(node: Dict) -> Tuple[List, List]:
    # get tree/token children of node
    token_node, tree_node = [], []
    for name in node['children']:
        if name.startswith(constants.NODE_FIX):
            tree_node.append(name)
        else:
            token_node.append(name)
    return token_node, tree_node,


# get max node id
last_index = lambda ast_tree: sorted([int(node_ind[len(constants.NODE_FIX):]) for node_ind in ast_tree.keys()])[-1]

# is string token
is_string = lambda identifier: \
    len(identifier) > 1 and identifier[0] == identifier[-1] and (identifier[0] == '\'' or identifier[0] == '\"')


def jsonl_gz_writeline(jsonl_gz: gzip.GzipFile, data_line: Dict) -> None:
    line_str = json.dumps(data_line) + '\n'
    line_bytes = line_str.encode('utf-8')
    jsonl_gz.write(line_bytes)


def load_jsonl_gz(filename: str):
    with gzip.open(filename, mode='rt') as reader:
        lines = reader.readlines()
        return jsonlines.Reader(lines)


def load_jsonl_gzs(filenames):
    return itertools.chain(*[load_jsonl_gz(fn) for fn in filenames])


def raw_file_index(filename: str) -> int:
    return int(filename.split('/')[-1].split('.')[0].split('_')[-1])


def base_file_index(filename: str) -> int:
    return int(filename.split('/')[-1].split('.')[-3])


def load_raw_filenames(data_dir: str, sort_func=None, debug=False, ) -> List[str]:
    # load raw file
    jsonl_gz_files = glob(data_dir)
    if sort_func is not None:
        jsonl_gz_files = sorted(jsonl_gz_files, key=sort_func)
    if debug:
        return [jsonl_gz_files[0]]
    else:
        return jsonl_gz_files


# def load_raw_data(data_dir: str, debug=False, debug_size=None) -> List[Dict]:
#     '''
#     load jsonl.gz files
#     :param data_dir: jsonl.gz file path, like /XX/*.jsonl.gz
#     :param debug: debug mode,
#     :return: code snippets list
#     '''
#     # load raw file
#     jsonl_gz_files = glob(data_dir)
#     if debug:
#         jsonl_gz_files = jsonl_gz_files[:2]
#     raw_dataset = list(load_jsonl_gzs(jsonl_gz_files))
#     if debug:
#         raw_dataset = raw_dataset[:debug_size]
#     return raw_dataset


def load_raw_data(data_dir: str, load_keys: List, debug=False, ) -> Dict:
    raw_data = {}
    for mode in constants.MODES:
        data = {}
        for key in load_keys:
            mode_data_dir = os.path.join(data_dir, key, '{}.*'.format(mode))
            jsonl_gz_files = glob(mode_data_dir)
            if debug:
                jsonl_gz_files = jsonl_gz_files[:1]
            raw_data[mode] = list(load_jsonl_gzs(jsonl_gz_files))
    return raw_data


################################################################


def map_reduce(paralleler: Parallel, func: Any, params: Union[Tuple, List], ) -> Union[Tuple, List]:
    for param in params:
        if type(param) in [list, tuple]:
            data_num = len(param)
            break

    processor_num = paralleler.n_jobs
    batch_len = int(math.ceil(data_num / processor_num))

    # duplicate args
    for ind, param in enumerate(params):
        if type(param) in [list, tuple]:
            continue
        else:
            params[ind] = [deepcopy(param) for _ in range(data_num)]
    params = list(zip(*params))

    result = paralleler(delayed(func)(*param) for param in params)
    result = zip(*result)
    result = [
        list(itertools.chain(*res))
        for res in result
    ]
    return result


################################################################

class CharType():
    null = 0
    upper = 1
    lower = 2
    digit = 3
    operator = 4
    link = 5

    @staticmethod
    def type(char: str) -> int:
        if len(char) == 0:
            return CharType.null
        elif char == '_':
            return CharType.link
        elif str.isdigit(char):
            return CharType.digit
        elif str.isalpha(char):
            if str.isupper(char):
                return CharType.upper
            elif str.lower(char):
                return CharType.lower
        else:
            return CharType.operator


def split_identifier(identifier: str, str_flag=True) -> List:
    '''
    test samples:
         ASTFunc_name23nameNameFF_ -> AST Func name23 name Name FF
         INF -> INF
         &&= -> &&=
         {_Func_name__} -> { Func name }
         __main__ -> main

    :param identifier: variable name
    :param str_flag: true -> return raw string; false return splitted string tokens
    :return: splited subtokens
    '''

    # identifier = ''.join([char for char in identifier \
    #                       if (str.isalpha(char) or str.isdigit(char) or char == '_')])

    if is_string(identifier):
        if str_flag:
            # skip string
            return [identifier]
        else:
            identifier = identifier[1:-1].strip()

    if len(identifier) > 1:
        # skip comment
        if len(identifier) > 1 and (identifier[:2] == '//' or \
                                    (identifier[:2] == '/*' and identifier[-2:] == '*/') \
                ):
            return []
    else:
        return [identifier]

    subtoken_type = CharType.null
    tmp = ''
    subtokens = []

    for char in identifier:
        current_type = CharType.type(char)
        if current_type == CharType.link:  # skip '_'
            if len(tmp) == 0:
                pass
            else:
                subtokens.append(tmp)
                tmp = ''
            subtoken_type = CharType.null
        else:
            if subtoken_type == CharType.null:
                tmp = char
                subtoken_type = CharType.type(char)
            else:
                if subtoken_type == current_type:  # previous char type equals current char type, append it
                    tmp += char
                else:
                    if (subtoken_type == CharType.upper or subtoken_type == CharType.lower) \
                            and current_type == CharType.digit:
                        # previous char type is alpha and current char type is digit, append it,
                        # and change current char type to digit
                        # eg. name 2 -> name2
                        tmp += char
                        subtoken_type = CharType.digit
                    elif subtoken_type == CharType.upper and current_type == CharType.lower:
                        if len(tmp) > 1:
                            # ASTT r -> AST Tr
                            subtokens.append(tmp[:-1])
                            tmp = tmp[-1] + char
                        else:
                            # T r -> Tr
                            tmp += char
                        subtoken_type = current_type
                    elif subtoken_type == CharType.lower and current_type == CharType.upper:
                        # name F -> name F
                        subtokens.append(tmp)
                        tmp = char
                        subtoken_type = current_type
                    elif subtoken_type == CharType.digit and \
                            (current_type == CharType.upper or current_type == CharType.lower):
                        # name23 N/n -> name23 N/n
                        subtokens.append(tmp)
                        tmp = char
                        subtoken_type = current_type
                    elif subtoken_type == CharType.operator and (not current_type == CharType.operator):
                        # { n -> { n
                        subtokens.append(tmp)
                        tmp = char
                        subtoken_type = current_type
                    elif (not subtoken_type == CharType.operator) and current_type == CharType.operator:
                        # name } -> name }
                        subtokens.append(tmp)
                        tmp = char
                        subtoken_type = current_type
                    else:
                        raise Exception
    if len(tmp) > 0:
        subtokens.append(tmp)
    return subtokens


def count_and_filter(token_list: List, min_freq: int) -> Dict:
    counter = Counter(token_list)
    token_freq = counter.most_common()
    filtered_token_freq = {token: freq for token, freq in token_freq if freq > min_freq}
    return filtered_token_freq


def merge_freqency(multi_dicts: List[Dict]) -> Dict:
    all_dict = dict()
    for dct in multi_dicts:
        for key, value in dct.items():
            if key in all_dict:
                all_dict[key] += value
            else:
                all_dict[key] = value
    return all_dict


def filter_func(code_snippet: Dict) -> Dict:
    # define your filter conditions
    if 10 <= len(code_snippet['tree']) <= 100 and \
            3 <= len(code_snippet['tok']) <= 100 and \
            3 <= len(code_snippet['comment']) <= 50:
        return code_snippet
    else:
        return dict()


def is_ascii(identifier: str) -> bool:
    if ('0x' in identifier) or ('\\x' in identifier) or \
            ('\\u' in identifier):  # hex or unicode
        return False
    else:
        return str.isascii(identifier)


if __name__ == '__main__':
    print(is_ascii('\\xAA'))
    print(is_ascii('\\u02C6-'))
    print(is_ascii('0x10000'))
    print(is_ascii('你好'))
