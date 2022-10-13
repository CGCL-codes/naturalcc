# -*- coding: utf-8 -*-

import os
from tree_sitter import Language, Parser
from dataset.clcdsa.dfg.DFG import (
    DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_cpp, DFG_csharp,
)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'csharp': DFG_csharp,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'cpp': DFG_cpp,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(os.path.join(os.path.dirname(__file__), 'my-languages.so'),
                        'c_sharp' if lang == 'csharp' else lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

from dataset.clcdsa.dfg.utils import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
    tree_to_variable_index,
)


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg
