import ast
from collections import namedtuple

from .astunparser import Unparser

SrcASTToken = namedtuple("SrcASTToken", "text type")


def separate_list(ast, max_len):
    """
    Handles training / evaluation on long ASTs by splitting
    them into smaller ASTs of length max_len, with a sliding
    window of max_len / 2.

    Example: for an AST ast with length 1700, and max_len = 1000,
    the output will be:
    [[ast[0:1000], 0], [ast[500:1500], 1000], [ast[700:1700], 1500]]

    Input:
        ast : List[Dictionary]
            List of nodes in pre-order traversal.
        max_len : int

    Output:
        aug_asts : List[List[List, int]]
            List of (ast, beginning idx of unseen nodes)
    """
    half_len = int(max_len / 2)
    if len(ast) <= max_len:
        return [[ast, 0]]

    aug_asts = [[ast[:max_len], 0]]
    i = half_len
    while i < len(ast) - max_len:
        aug_asts.append([ast[i: i + max_len], half_len])
        i += half_len
    idx = max_len - (len(ast) - (i + half_len))
    aug_asts.append([ast[-max_len:], idx])
    return aug_asts


class MyListFile(list):
    def write(self, text, type=None):
        text = text.strip()
        if len(text) > 0:
            self.append(SrcASTToken(text, type))

    def flush(self):
        pass

    def tokens(self):
        tokens = [tt.text for tt in self]
        return tokens

    def transpose(self):
        tokens = [tt.text for tt in self]
        types = [tt.type for tt in self]
        return tokens, types


def parse_file(filename):
    with open(filename, 'r') as reader:
        code = reader.read()
        t = ast.parse(code)
        lst = MyListFile()
        Unparser(t, lst)
        tokens, types = lst.transpose()
        del lst
    return tokens, types
