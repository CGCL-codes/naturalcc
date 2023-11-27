# -*- coding: utf-8 -*-


from apted.helpers import Tree
from apted.apted import APTED


def edit_distance(tree1, tree2):
    """
    edit-distance
    tree1 = "{a{b}{c}}"
    tree2 = "{a{b{d}}}"
    """

    tree1, tree2 = map(Tree.from_text, (tree1, tree2))
    apted = APTED(tree1, tree2)
    distance = apted.compute_edit_distance()
    return distance


# tree1 = "{a{b}{c}}"
# tree2 = "{1{2{3}}}"
# print(edit_distance(tree1, tree2))

import itertools
# import sentencepiece as spm
# from dataset.clcdsa.plbart import (
#     SPM_VOCAB_FILE,
# )
# from ncc.data.constants import SPM_SPACE
from ncc.tokenizers.tokenization import SPACE_SPLITTER
from ncc.tokenizers.tokenization import split_identifier


# vocab = spm.SentencePieceProcessor()
# vocab.load(SPM_VOCAB_FILE)


def lexical_distance(code1, code2):
    """
    java = "public void serialize(LittleEndianOutput out) {out.writeShort(field_1_vcenter);}"
    csharp = "public override void Serialize(ILittleEndianOutput out1){out1.WriteShort(field_1_vcenter);}"
    """

    def string2tokens(string):
        string = ''.join([char if str.isalpha(char) else ' ' for char in string])
        string = SPACE_SPLITTER.sub(" ", string)
        tokens = string.split()
        tokens = [split_identifier(tok) for tok in tokens]
        tokens = list(itertools.chain(*tokens))
        tokens = [str.lower(tok) for tok in tokens]
        # tokens = vocab.encode(string, out_type=str)
        # tokens = str.replace(' '.join(tokens), SPM_SPACE, '')
        return tokens

    token_set1 = set(string2tokens(code1))
    token_set2 = set(string2tokens(code2))

    itersection = token_set1 & token_set2
    union = token_set1 | token_set2

    # print(itersection)
    # print(union)

    lexical_distance = round(len(itersection) / len(union) * 100, 2)
    # print(lexical_distance)
    return lexical_distance


def ast2edtree(ast):
    def _dfs(idx):
        if "value" in ast[idx]:
            return "{" + ast[idx]['type'] + "}"
        else:
            tmp = ""
            for child in ast[idx]['children']:
                tmp += _dfs(str(child))
            return "{" + ast[idx]['type'] + tmp + "}"

    return _dfs("0")
