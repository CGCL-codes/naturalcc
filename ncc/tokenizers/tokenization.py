# -*- coding: utf-8 -*-

import re
import itertools
from dpu_utils.codeutils import split_identifier_into_parts

from ncc import LOGGER
from ncc.utils.file_ops import json_io
from ncc.data import constants

NEWLINE_REGEX = re.compile(r"\n")
WHITESPACE_REGEX = re.compile(r"[ \t\n]+")
SPACE_SPLITTER = re.compile(r"\s+")
DPU_IDENTIFIER_SPLITTER = re.compile('[:@_a-zA-Z][_a-zA-Z0-9]*')
URL_REGEX = re.compile(r"https?://\S+\b")

_filter_tokens = lambda tokens: [tok.strip() for tok in tokens if len(tok) > 0]


def json_tokenizer(line, **kwargs):
    tokens = json_io.json_loads(line)
    return tokens


def _space_tokenizer(line, **kwargs):
    """string => space tokenizer => list"""
    tokens = SPACE_SPLITTER.sub(' ', line).strip()
    tokens = tokens.split()
    tokens = _filter_tokens(tokens)
    return tokens


def space_tokenizer(line, **kwargs):
    """json string => list"""
    line = json_io.json_loads(line)
    return _space_tokenizer(line)


def lower_tokenizer(line, **kwargs):
    """json string => list"""
    tokens = json_io.json_loads(line)
    tokens = _filter_tokens(tokens)
    tokens = [str.lower(token) for token in tokens]
    return tokens


def _dpu_sub_tokenizer(tokens, **kwargs):
    """string => list"""
    tokens = [split_identifier_into_parts(tok) if DPU_IDENTIFIER_SPLITTER.match(tok) else [tok] for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    tokens = _filter_tokens(tokens)
    return tokens


def dpu_sub_tokenizer(line, **kwargs):
    tokens = json_io.json_loads(line)
    return _dpu_sub_tokenizer(tokens)


def _space_dpu_sub_tokenizer(line, **kwargs):
    """string => list"""
    tokens = SPACE_SPLITTER.split(line)
    return _dpu_sub_tokenizer(tokens)


def space_dpu_sub_tokenizer(line, **kwargs):
    line = json_io.json_loads(line)
    return _space_dpu_sub_tokenizer(line)


def _bin_ast_tokenizer(line, **kwargs):
    tokens = [node['children'][-1] for node in line.values() if isinstance(node['children'][-1], list)]
    tokens = list(itertools.chain(*tokens))
    return tokens


def bin_ast_tokenizer(line, **kwargs):
    line = json_io.json_loads(line)
    return _bin_ast_tokenizer(line)


def _sbt_tokenizer(line, **kwargs):
    """to collect types & tokens into a list for sbt dictionary generation"""
    type_tokens, tokens = [], []
    for l in line:
        if isinstance(l, str):
            type_tokens.append(l)
        else:  # list
            type_tokens.append(l[0])
            # tokens.extend(_dpu_sub_tokenizer([l[-1]]))
            tokens.append(l[-1])
    return type_tokens, tokens


def sbt_tokenizer(line, **kwargs):
    line = json_io.json_loads(line)
    return _sbt_tokenizer(line)


def normalize_program(fn, **kwargs):
    if not isinstance(fn, (str, bytes)):
        LOGGER.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = NEWLINE_REGEX.sub(rf" {constants.EOL}", fn)
    if kwargs.get('remove_eol', False):
        fn = str.replace(fn, constants.EOL, ' ')
    fn = WHITESPACE_REGEX.sub(" ", fn)
    return fn


def normalize_docstring(docstring, **kwargs):
    # Substitute urls with [URL]
    docstring = NEWLINE_REGEX.sub(rf" {constants.EOL}", docstring)
    if kwargs.get('remove_eol', False):
        docstring = str.replace(docstring, constants.EOL, ' ')
    docstring = URL_REGEX.sub(constants.URL, docstring)
    if kwargs.get('remove_url', False):
        docstring = str.replace(docstring, constants.URL, ' ')
    docstring = WHITESPACE_REGEX.sub(" ", docstring)
    return docstring


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


# is string token
is_string = lambda identifier: \
    len(identifier) > 1 and identifier[0] == identifier[-1] and (identifier[0] == '\'' or identifier[0] == '\"')


def split_identifier(identifier, str_flag=True, **kwargs):
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


if __name__ == '__main__':
    # unit test
    print(split_identifier('@summary'))
    print(_dpu_sub_tokenizer(['@summary']))

    print(split_identifier('@@summary'))
    print(_dpu_sub_tokenizer(['@@summary']))

    print(split_identifier('::=summary'))
    print(_dpu_sub_tokenizer(['::=summary']))
