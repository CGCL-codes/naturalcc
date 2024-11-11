import re
from typing import List, Dict, Any, Set, Optional
import logging
from tree_sitter import Parser, Language
import os
import platform
from pathlib import Path
DOCSTRING_REGEX_TOKENIZER = re.compile(r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")


def tokenize_docstring(docstring: str) -> List[str]:
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


def tokenize_code(node, blob: str, nodes_to_exclude: Optional[Set]=None) -> List:
    tokens = []
    traverse(node, tokens)
    return [match_from_span(token, blob) for token in tokens if nodes_to_exclude is None or token not in nodes_to_exclude]


def traverse(node, results: List) -> None:
    if node.type == 'string':
        results.append(node)
        return
    for n in node.children:
        traverse(n, results)
    if not node.children:
        results.append(node)

def nodes_are_equal(n1, n2):
    return n1.type == n2.type and n1.start_point == n2.start_point and n1.end_point == n2.end_point

def previous_sibling(tree, node):
    """
    Search for the previous sibling of the node.

    TODO: C TreeSitter should support this natively, but not its Python bindings yet. Replace later.
    """
    to_visit = [tree.root_node]
    while len(to_visit) > 0:
        next_node = to_visit.pop()
        for i, node_at_i in enumerate(next_node.children):
            if nodes_are_equal(node, node_at_i):
                if i > 0:
                    return next_node.children[i-1]
                return None
        else:
            to_visit.extend(next_node.children)
    return ValueError("Could not find node in tree.")


def node_parent(tree, node):
    to_visit = [tree.root_node]
    while len(to_visit) > 0:
        next_node = to_visit.pop()
        for child in next_node.children:
            if nodes_are_equal(child, node):
                return next_node
        else:
            to_visit.extend(next_node.children)
    raise ValueError("Could not find node in tree.")


def match_from_span(node, blob: str) -> str:
    lines = blob.split('\n')
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]


def traverse_type(node, results: List, kinds) -> None:
    if node.type in kinds:
        results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_type(n, results, kinds)


def remove_redundant_token(content):
    content_splits = content.split()
    content_splits_new = []
    for s in content_splits:
        if s not in ["{", "}"]:
            content_splits_new.append(s)
    return " ".join(content_splits_new)

# def traverse_type(cursor, results: List, kinds) -> None:
#     if cursor.node.type in kinds:
#         results.append(node)
#     if not node.children:
#         return
#     for n in node.children:
#         traverse_type(n, results, kinds)

def print_all_nodes(tree):
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


def get_tree_node_with_kinds(tree, kinds):
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        if cursor.node.type in kinds:
            yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    def __init__(self, language):
        self.PARSER = Parser()
        self.language = language
        if self.language == None:
            self.LOGGER.info("Cannot find prebuilts file for language {}".format(language))

        language_build = self.get_language(self.language)
        self.PARSER.set_language(language_build)

    def get_language(self, language=None):
        root = Path(__file__).parent.parent.parent
        cd = os.getcwd()
        plat = platform.system()     
        p = os.path.join(root, "codetf", "tree-sitter-prebuilts", plat)
        print(p)
        file = f'{language}.so'
        
        return Language(os.path.join(p, file), language)

    def parse(self, code_snippet):
        return self.PARSER.parse(code_snippet)
 
