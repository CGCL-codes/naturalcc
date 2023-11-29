# -*- coding: utf-8 -*-

import os
import sys
import ujson
from tree_sitter import Language, Parser
from ..constants import RECURSION_DEPTH
from ..ast.child_value import (
    get_root,
    reset_indices,
)

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(RECURSION_DEPTH)  # recursion depth


class CodeParser(object):
    '''
    Parse code into ast. Supported by TreeSitter and its variants.
    '''
    __slots__ = (
        'parser',
        'language',
        'operators',
    )

    def __init__(self, so_file: str, language: str, operators_file: str = None):
        self.parser = Parser()
        self.parser.set_language(Language(so_file, language))
        self.language = language

        if operators_file is None:
            operators_file = os.path.join(os.path.dirname(__file__), 'operators.json')
        with open(operators_file, 'r') as reader:
            self.operators = ujson.load(reader)

    def _subcode(self, start, end, code_lines):
        '''
        extract substring from code lines
        :param start: start point
        :param end: end point
        :param code_lines: codes.split('\n')
        :return: substring of code
        '''
        if start[0] == end[0]:
            if end[1] == -1:
                return code_lines[start[0]][start[1]:]
            else:
                return code_lines[start[0]][start[1]:end[1]]
        elif start[0] < end[0]:
            sub_code_lines = [code_lines[start[0]][start[1]:]]
            for line_num in range(start[0] + 1, end[0]):
                sub_code_lines.append(code_lines[line_num])
            if end[1] == -1:
                sub_code_lines.append(code_lines[end[0]])
            else:
                sub_code_lines.append(code_lines[end[0]][:end[1]])
            return b'\n'.join(sub_code_lines)
        else:
            raise NotImplemented

    def _node_type(self, token: str) -> str:
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
            return 'Keyword'
        else:
            if self.operators and (token in self.operators):
                return self.operators[token]
            else:
                return token

    def _delete_empty_nodes(self, ast):
        '''delete comment node and its children'''

        def _delete_node(idx, node):
            # update its parent's children
            parent_idx = node['parent']
            parent_node = ast[parent_idx]
            del_idx = parent_node['children'].index(idx)
            parent_node['children'].pop(del_idx)
            # delete node
            ast.pop(idx)
            return parent_idx, parent_node

        def _dfs(idx):
            node = ast[idx]
            child_ids = node.get('children', None)

            if child_ids is not None and len(child_ids) == 0:
                idx, node = _delete_node(idx, node)  # pop comment node and return its parent node
                # if node type is comment, its parent node must have "children"
                while len(node['children']) == 0:
                    idx, node = _delete_node(idx, node)

            if child_ids is None:
                return

            for idx in child_ids:
                _dfs(idx)

        _dfs(idx=0)
        return ast

    def _build_ast(self, root, code_lines):
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

        def _dfs(cur_node, parent_node_idx):
            if len(cur_node.children) == 0:
                # current node has no child node, it's leaf node, build a leaf node
                new_node_idx = len(ast_tree)
                if cur_node.is_named:
                    # leaf node's value is None. we have to extract its value from source code
                    value = self._subcode(cur_node.start_point, cur_node.end_point, code_lines).decode()
                    if not value:  # value='', delete current node
                        return
                    ast_tree[new_node_idx] = {
                        'type': cur_node.type, 'parent': parent_node_idx,
                        'value': value,
                    }
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
                else:
                    # leaf node is operator or keyword
                    ast_tree[new_node_idx] = {
                        'type': self._node_type(cur_node.type),
                        'parent': parent_node_idx,
                        'value': cur_node.type,
                    }
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
            else:
                # current node has many child nodes
                cur_node_idx = len(ast_tree)
                ast_tree[cur_node_idx] = {'type': cur_node.type, 'parent': parent_node_idx, 'children': []}
                # update parent node's children
                if parent_node_idx is not None:
                    ast_tree[parent_node_idx]['children'].append(cur_node_idx)
                # update current node's children
                for child_node in cur_node.children:
                    _dfs(child_node, parent_node_idx=cur_node_idx)

        _dfs(root, parent_node_idx=None)
        ast_tree = self._delete_empty_nodes(ast=ast_tree)
        return ast_tree

    def _remove_php_heads(self, ast):
        """
        first 3 nodes would be follow:
        0: {'type': 'program', 'parent': None, 'children': [1, 2, 6]}
        1: {'type': 'php_tag', 'parent': 0, 'value': '<?php'}
        2: {'type': 'ERROR', 'parent': 0, 'children': [3, 5]}
        solution: remove 2nd, connect 3rd to 1st, rename 3rd node's type to ‘local_variable_declaration’
        """
        _ = ast.pop(1)
        del ast[0]['children'][ast[0]['children'].index(1)]
        ast[2]['type'] = 'local_variable_declaration'
        # node index: from 2-..., should move forward and update children info
        ast[0]['children'] = [index - 1 for index in ast[0]['children']]
        for idx in sorted(ast.keys())[1:]:
            new_idx = idx - 1
            new_node = ast.pop(idx)
            if new_node['parent'] > 1:
                new_node['parent'] = new_node['parent'] - 1
            if 'children' in new_node:
                new_node['children'] = [index - 1 for index in new_node['children'] if index > 0]
            ast[new_idx] = new_node
        return ast

    def _delete_comment_nodes(self, ast):
        '''delete comment node and its children'''

        def _delete_node(idx, node):
            # update its parent's children
            parent_idx = node['parent']
            parent_node = ast[parent_idx]
            del_idx = parent_node['children'].index(idx)
            parent_node['children'].pop(del_idx)
            # delete node
            ast.pop(idx)
            return parent_idx, parent_node

        def _dfs(idx):
            node = ast[idx]
            child_ids = node.get('children', None)

            if 'comment' == node['type']:
                idx, node = _delete_node(idx, node)  # pop comment node and return its parent node
                # if node type is comment, its parent node must have "children"
                while len(node['children']) == 0:
                    idx, node = _delete_node(idx, node)

            if child_ids is None:
                return

            for idx in child_ids:
                _dfs(idx)

        _dfs(idx=0)
        return ast

    def parse(self, code):
        # TreeSitter: must add this head for php code
        code = '<?php ' + code if self.language == 'php' else code
        root = self.parser.parse(code.encode())
        code_lines = [line.encode() for line in code.split('\n')]
        try:
            ast = self._build_ast(root.root_node, code_lines)
            assert len(ast) > 1, AssertionError('AST parsed error.')
            if self.language == 'php':
                ast = self._remove_php_heads(ast)
            # delete comments in ast
            ast = self._delete_comment_nodes(ast)
            # reset ast indices if comment nodes has been removed
            ast = reset_indices(ast)
            # check whether an ast contains nodes with null children
            for node in ast.values():
                if 'children' in node:
                    assert len(node['children']) > 0, AssertionError(f'AST has a node({node}) without child and value')
                if 'value' in node:
                    assert len(node['value']) > 0, AssertionError(f'AST has a node({node}) without child and value')
            return ast
        except RecursionError as err:
            # RecursionError: maximum recursion depth exceeded while getting the str of an object
            # raw_ast is too large, skip this ast
            print(err)
            return None
        except AssertionError as err:
            print(err)
            return None
