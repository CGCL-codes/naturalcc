import re
import ujson
import itertools
from dpu_utils.codeutils import split_identifier_into_parts

SPACE_SPLITTER = re.compile(r"\s+")
DPU_IDENTIFIER_SPLITTER = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def _space_tokenizer(line):
    """string => space tokenizer => list"""
    tokens = SPACE_SPLITTER.sub(' ', line).strip()
    return tokens.split()


def space_tokenizer(line):
    line = ujson.loads(line)
    return _space_tokenizer(line)


def list_tokenizer(line):
    """json string => list"""
    tokens = ujson.loads(line)
    return tokens


def _dpu_sub_tokenizer(line):
    """string => list"""
    tokens = SPACE_SPLITTER.split(line)
    tokens = [split_identifier_into_parts(tok) if DPU_IDENTIFIER_SPLITTER.match(tok) else [tok] for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    tokens = [tok for tok in tokens if len(tok) > 0]
    return tokens


def dpu_sub_tokenizer(line):
    line = ujson.loads(line)
    return _dpu_sub_tokenizer(line)
