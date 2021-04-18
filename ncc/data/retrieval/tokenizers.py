import re
import ujson
from collections import Counter
from ncc.data.constants import UNK
from dpu_utils.codeutils import split_identifier_into_parts
import itertools

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def sub_tokenizer(tokens: str):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = ujson.loads(tokens)
    tokens = [split_identifier_into_parts(tok) if IDENTIFIER_TOKEN_REGEX.match(tok) else [tok] for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    tokens = [tok for tok in tokens if len(tok) > 0]
    return tokens


def string_sub_tokenizer(tokens: list):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = [split_identifier_into_parts(tok) if IDENTIFIER_TOKEN_REGEX.match(tok) else tok for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    return tokens


def func_name_tokenizer(tokens, **kwargs):
    tokens = ujson.loads(tokens)
    if len(tokens) >= kwargs['min_func_len']:
        tokens = split_identifier_into_parts(tokens)
    else:
        tokens = []
    return tokens


def lower_tokenizer(tokens: str, **kwargs):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = ujson.loads(tokens)
    return list(map(str.lower, tokens))


def list_tokenizer(line, **kwargs):
    """json string => list"""
    tokens = ujson.loads(line)
    if kwargs.get('func_name', False):
        func_name = ujson.loads(kwargs['func_name'])
        if len(func_name) >= kwargs['min_func_len']:
            tokens = [UNK if token == func_name else token for token in tokens]
        else:
            tokens = []
    return tokens
