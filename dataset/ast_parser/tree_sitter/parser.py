# -*- coding: utf-8 -*-


import os

from tree_sitter import Language, Parser

from ncc import LOGGER
from ncc.utils.file_ops import json_io
from ncc.utils.path_manager import PathManager
from .utils import (
    util_ast,
)


class TreeSitterASTParser(object):
    '''parse code data into ast'''
    __slots__ = ('parser', 'to_lower', 'LANGUAGE', 'operators',)

    def __init__(self, SO_FILE, LANGUAGE, to_lower=False, operators_file=None):
        self.parser = Parser()
        try:
            assert PathManager.exists(SO_FILE), FileExistsError(
                f"{SO_FILE} does not exist, automatically download TreeSitter parse file {LANGUAGE}.so."
            )
        except FileExistsError as err:
            LOGGER.warning(err)
            from ncc.utils.hub.tree_sitter.download import download
            download(LANGUAGE)

        if LANGUAGE == 'csharp':
            LANGUAGE = 'c_sharp'
        self.parser.set_language(Language(SO_FILE, LANGUAGE))
        self.LANGUAGE = LANGUAGE
        self.to_lower = to_lower

        if operators_file is None:
            operators_file = os.path.join(os.path.dirname(__file__), 'operators.json')
        with open(operators_file, 'r') as reader:
            self.operators = json_io.json_load(reader)

    def subcode(self, start, end, code_lines):
        '''
        extract substring from code lines
        :param start: start point
        :param end: end point
        :param code_lines: codes.split('\n')
        :return: substring of code
        '''
        if start[0] == end[0]:
            return code_lines[start[0]][start[1]:end[1]]
        elif start[0] < end[0]:
            sub_code_lines = [code_lines[start[0]][start[1]:]]
            for line_num in range(start[0] + 1, end[0]):
                sub_code_lines.append(code_lines[line_num])
            sub_code_lines.append(code_lines[end[0]][:end[1]])
            return b'\n'.join(sub_code_lines)
        else:
            raise NotImplemented

    def define_node_type(self, token):
        '''
        in tree_sitter library, operator and keyword nodes are no pre-define node type, like:
        [type: 'def'/'&&', value: 'def'/'&&']
        :param token: node value
        :return: if token is operator, its type will be EN name
                  if token is keyword, its type will be {}Kw
        '''
        is_keyword = True
        for chr in token:
            if str.isalpha(chr):
                continue
            else:
                is_keyword = False
                break
        if is_keyword:
            return '{}_keyword'.format(str.lower(token))
        else:
            if self.operators and (token in self.operators):
                return self.operators[token]
            else:
                return token

    def build_tree(self, root, code_lines, append_index=False):
        '''
        build ast with tree_sitter, operator and keyword has no pre-defined type
        :param root: ast tree root node
        :param code_lines: [...], ...
        :return:
            [
                {'type': "node_type", 'children': "node_ids(List)", ‘children’: “node_ids(List)” }, # non-leaf node
                {'type': "node_type", 'children': "node_ids(List)", ‘value’: “node_value(str)” }, # leaf node
                ...
            ]
        '''
        ast_tree = {}

        def dfs(cur_node, parent_node_idx):
            children = [child for child in cur_node.children if child.start_point != child.end_point]
            if len(children) == 0:
                # current node has no child node, it's leaf node, build a leaf node
                new_node_idx = len(ast_tree)
                if cur_node.is_named:
                    # leaf node's value is None. we have to extract its value from source code
                    value = self.subcode(cur_node.start_point, cur_node.end_point, code_lines).decode()
                    if not value:  # value='', delete current node
                        return
                    ast_tree[new_node_idx] = {
                        'type': cur_node.type, 'parent': parent_node_idx,
                        'value': value,
                    }
                    if append_index:
                        ast_tree[new_node_idx]["index"] = [cur_node.start_point, cur_node.end_point]
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
                else:
                    # leaf node is operator or keyword
                    ast_tree[new_node_idx] = {
                        'type': self.define_node_type(cur_node.type),
                        'parent': parent_node_idx,
                        'value': cur_node.type,
                    }
                    if append_index:
                        ast_tree[new_node_idx]["index"] = [cur_node.start_point, cur_node.end_point]
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
            else:
                # current node has many child nodes
                cur_node_idx = len(ast_tree)
                ast_tree[cur_node_idx] = {'type': cur_node.type, 'parent': parent_node_idx, 'children': []}
                # update parent node's children
                if parent_node_idx is not None:
                    ast_tree[parent_node_idx]['children'].append(cur_node_idx)
                # update current node's children
                for child_node in children:
                    dfs(child_node, parent_node_idx=cur_node_idx)

        dfs(root, parent_node_idx=None)
        return ast_tree

    def parse_raw_ast(self, code, MIN_AST_SIZE=10, MAX_AST_SIZE=999, append_index=False):
        # must add this head for php code
        if self.LANGUAGE == 'php':
            code = '<?php ' + code

        ast_tree = self.parser.parse(code.encode())

        code_lines = [line.encode() for line in code.split('\n')]

        # 1) build ast tree in Dict type
        try:
            code_tree = self.build_tree(ast_tree.root_node, code_lines, append_index)
            if not (MIN_AST_SIZE < len(code_tree) < MAX_AST_SIZE):
                raise AssertionError(
                    f"Code\'s AST(node num: {len(code_tree)}) ({MIN_AST_SIZE}, {MAX_AST_SIZE}) is too small/large!")
            # if str.lower(code_tree[0]['type']) == 'error':
            #     raise RuntimeError
            # if self.LANGUAGE in {'java'}:
            #     # rename Root's children whose type is ERROR into ‘local_variable_declaration’
            #     roots_children = code_tree[0]['children']
            #     for child in roots_children:
            #         if child == ['ERROR']:
            #             ast_tree[child]['type'] = 'local_variable_declaration'
            #             break

            if self.LANGUAGE == 'php':
                """
                first 3 nodes would be follow:
                0: {'type': 'program', 'parent': None, 'children': [1, 2, 6]}
                1: {'type': 'php_tag', 'parent': 0, 'value': '<?php'}
                2: {'type': 'ERROR', 'parent': 0, 'children': [3, 5]}
                solution: remove 2nd, connect 3rd to 1st, rename 3rd node's type to ‘local_variable_declaration’
                """
                php_tag_node = code_tree.pop(1)
                del code_tree[0]['children'][code_tree[0]['children'].index(1)]
                code_tree[2]['type'] = 'local_variable_declaration'
                # node index: from 2-..., should move forward and update children info
                code_tree[0]['children'] = [index - 1 for index in code_tree[0]['children']]
                for idx in sorted(code_tree.keys())[1:]:
                    new_idx = idx - 1
                    new_node = code_tree.pop(idx)
                    if new_node['parent'] > 1:
                        new_node['parent'] = new_node['parent'] - 1
                    if 'children' in new_node:
                        new_node['children'] = [index - 1 for index in new_node['children'] if index > 0]
                    code_tree[new_idx] = new_node

            if self.LANGUAGE == 'cpp':
                def del_node(ast, index):
                    def _del(idx):
                        node = ast.pop(idx)
                        parent_children = ast[node['parent']]['children']
                        del parent_children[parent_children.index(idx)]
                        return node['parent']

                    parent_idx = _del(index)
                    while len(ast[parent_idx]['children']) == 0:
                        parent_idx = _del(parent_idx)
                    return ast

                pop_indices = [node_idx for node_idx, node_info in code_tree.items() \
                               if node_info['type'] == "LineBreakOp"]
                for idx in pop_indices:
                    code_tree = del_node(code_tree, idx)
                code_tree = \
                    util_ast.reset_indices_for_value_format(code_tree, root_idx=util_ast.get_root_idx(code_tree))

            assert len(code_tree) > 1, AssertionError('AST parsed error.')
            # check whether an ast contains nodes with null children
            for node in code_tree.values():
                if 'children' in node:
                    assert len(node['children']) > 0, AssertionError('AST has a node without child and value')
                if 'value' in node:
                    assert len(node['value']) > 0, AssertionError('AST has a node without child and value')
            return code_tree
        except RecursionError as err:
            # RecursionError: maximum recursion depth exceeded while getting the str of an object
            print(err)
            # raw_ast is too large, skip this ast
            return None
        except AssertionError as err:
            print(err)
            return None


if __name__ == '__main__':
    """error parse"""
    from ncc import __LIBS_DIR__

    # code = "public static string StripExtension(string filename){int idx = filename.IndexOf('.');if (idx != -1){filename = filename.Substring(0, idx);}return filename;}"
    code = "import java . util . * ; class Main { public static void main ( String [ ] args ) { Scanner sc = new Scanner ( System . in ) ; int x = sc . nextInt ( ) ; int y = sc . nextInt ( ) ; System . out . println ( Math . max ( x , y ) ) ; } }"
    parser = TreeSitterASTParser(SO_FILE=os.path.join(__LIBS_DIR__, f"java.so"), LANGUAGE="java")
    tree = parser.parse_raw_ast(code)
