# -*- coding: utf-8 -*-


import sys

import re
import os
import itertools
from tree_sitter import Language, Parser
from preprocessing.codesearchnet import (
    RECURSION_DEPTH,
    MEANINGLESS_TOKENS,
    COMMENT_END_TOKENS,
    MAX_CODE_TOKEN_LEN,
    MAX_COMMENT_TOKEN_LIST_LEN,
    NO_METHOD,
)

# ignore those ast whose size is too large. Therefore set it as a small number
sys.setrecursionlimit(RECURSION_DEPTH)  # recursion depth

from preprocessing.codesearchnet.utils import (
    util,
    util_ast,
    util_path,
    util_traversal,
)
from ncc.utils.file_ops import json_io
from ncc.tokenizers.tokenization import split_identifier
from ncc.utils.path_manager import PathManager
from ncc import LOGGER


class CodeParser(object):
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
            from ncc.hub.tree_sitter.download import download
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

    def parse_docstring(self, docstring):
        '''parse comment from docstring'''
        docstring = re.sub(r'\{\@\S+', '', docstring)
        docstring = re.sub(r'{.+}', '', docstring)
        docstring = ''.join([char for char in docstring if char not in MEANINGLESS_TOKENS])
        docstring = [split_identifier(token, str_flag=False) for token in docstring.split(' ')]
        docstring = list(itertools.chain(*docstring))
        docstring = util.stress_tokens(docstring)
        if self.to_lower:
            docstring = util.lower(docstring)
        return docstring

    def parse_docstring_tokens(self, docstring_tokens):
        # parse comment from docstring_tokens
        docstring_tokens = [''.join([char for char in token if char not in MEANINGLESS_TOKENS]) \
                            for token in docstring_tokens]
        docstring_tokens = itertools.chain(
            *[split_identifier(token, str_flag=False) for token in docstring_tokens]
        )
        docstring_tokens = util.stress_tokens(docstring_tokens)
        if self.to_lower:
            docstring_tokens = util.lower(docstring_tokens)
        return docstring_tokens

    def parse_comment(self, docstring, docstring_tokens):
        '''
        our refined comment parse function. if you prefer original comment, use docstring_tokens instead
        '''
        if (docstring_tokens[-1] in COMMENT_END_TOKENS) or \
            (len(docstring_tokens) > MAX_COMMENT_TOKEN_LIST_LEN):
            # if docstring_tokens is too long or docstring_tokens is wrong parsed
            ''' exceptions in CodeSearchNet, eg.
            docstring: 'Set {@link ServletRegistrationBean}s that the filter will be registered against.
                        @param servletRegistrationBeans the Servlet registration beans'
            docstring_tokens:  <class 'list'>: ['Set', '{'] ['.']
            '''

            # skip this code snippet, if there are non-ascii tokens
            if not util.is_ascii(docstring):
                return None
            docstring = re.sub(
                '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ',
                docstring)  # delete url
            # remove additional and useless tails info
            docstring = str.split(docstring, '\n\n')[0].replace('\n', ' ')
            docstring = re.split(r'[\.|:]', docstring)
            docstring = docstring[0] + '.'  # add . at the end of sentence
            comment_tokens = self.parse_docstring(docstring)
        else:
            # skip this code snippet, if there are non-ascii tokens
            for comment_token in docstring_tokens:
                if not util.is_ascii(comment_token):
                    return None
            comment_tokens = self.parse_docstring_tokens(docstring_tokens)

        ########################################################################################
        # add . at the end of sentence
        if comment_tokens[-1] == ':':
            comment_tokens[-1] = '.'
        else:
            comment_tokens.append('.')

        ########################################################################################
        comment_tokens = ' '.join(comment_tokens)
        comment_tokens = re.sub(r'[\-|\*|\=|\~]{2,}', ' ', comment_tokens)  # remove ----+/****+/====+,
        comment_tokens = re.sub(r'[!]{2,}', '!', comment_tokens)  # change !!!! -> !
        comment_tokens = re.sub(r'[`]{2,}', ' ` ', comment_tokens)  # change ```, -> ` ,

        ########################################################################################
        # for rouge
        # remove <**> </**>
        comment_tokens = re.sub(r'<.*?>', '', comment_tokens)
        # remove =>
        comment_tokens = comment_tokens.replace('= >', ' ').replace('=>', ' ')
        # remove < >
        comment_tokens = comment_tokens.replace('<', ' ').replace('>', ' ')

        ########################################################################################
        comment_tokens = re.sub(r'\s+', ' ', comment_tokens)
        comment_tokens = comment_tokens.split(' ')

        new_comment_tokens = []
        for token in comment_tokens:
            token = token.strip()
            if len(token) > 0:
                # rule 1: +XX+ -> XX
                if token[0] == '+' or token[-1] == '+':
                    new_comment_tokens.append(token[1:-1].strip())
                else:
                    new_comment_tokens.append(token.strip())
        comment_tokens = new_comment_tokens

        if 3 < len(comment_tokens) <= 60:
            return comment_tokens
        else:
            return None

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

    def build_tree(self, root, code_lines):
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
            if len(cur_node.children) == 0:
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
                    ast_tree[parent_node_idx]['children'].append(new_node_idx)
                else:
                    # leaf node is operator or keyword
                    ast_tree[new_node_idx] = {
                        'type': self.define_node_type(cur_node.type),
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
                    dfs(child_node, parent_node_idx=cur_node_idx)

        dfs(root, parent_node_idx=None)
        return ast_tree

    def parse_raw_ast(self, code):
        # must add this head for php code
        if self.LANGUAGE == 'php':
            code = '<?php ' + code

        ast_tree = self.parser.parse(code.encode())

        code_lines = [line.encode() for line in code.split('\n')]

        # 1) build ast tree in Dict type
        try:
            code_tree = self.build_tree(ast_tree.root_node, code_lines)
            # if str.lower(code_tree[0]['type']) == 'error':
            #     raise RuntimeError
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

    def parse_method(self, func_name):
        # our defined method parse function
        method = ''
        for char in func_name:
            if str.isalpha(char) or str.isdigit(char):
                method += char
            else:
                method += ' '
        method = self.parse_docstring(method)
        method = [token.strip() for token in method if len(token.strip()) > 1]
        if len(method) > 0:
            return method
        else:
            return [NO_METHOD]

    def parse_code_tokens(self, code_tokens):
        '''
        our refined code tokens parse function. if you prefer original code tokens, use code_tokens instead
        '''
        # skip this code snippet, if there are non-ascii tokens or code token is too long
        # filter comment in code_tokens, eg. //***\n /* */\n
        code_tokens = [
            token for token in code_tokens
            if not (str.startswith(token, '//') or str.startswith(token, '#') or \
                    (str.startswith(token, '/*') and str.endswith(token, '*/')))
        ]

        for idx, token in enumerate(code_tokens):
            code_tokens[idx] = token.strip()
            if not util.is_ascii(code_tokens[idx]) or len(code_tokens[idx]) > MAX_CODE_TOKEN_LEN:
                return None

        code_tokens = util.filter_tokens(code_tokens)
        if self.to_lower:
            code_tokens = util.lower(code_tokens)

        if len(code_tokens) > 3:
            return code_tokens
        else:
            return None
