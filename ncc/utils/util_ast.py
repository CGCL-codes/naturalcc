# -*- coding: utf-8 -*-
import re
import itertools
from copy import deepcopy
from tree_sitter import Language, Parser
from ncc import LOGGER
from ncc.utils import util
from ncc.utils import util_path
from typing import Tuple, List, Dict, Union
from ncc.utils import constants
# sys.setrecursionlimit(99999)  # recursion depth


def subcode(start: Tuple, end: Tuple, code_lines: List) -> str:
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
        return '\n'.join(sub_code_lines)
    else:
        raise NotImplemented


def define_default_node_type(token: str) -> str:
    '''
    in tree_sitter library, operator and keyword nodes are no pre-define node type, like:
    [type: 'def'/'&&', value: 'def'/'&&']
    :param token: node value
    :return: if token is operator, its type will be {}_operator
              if token is keyword, its type will be {}_keyword
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
        return '{}_operator'.format(token)


def build_tree(root, code_lines: List[str]) -> Dict:
    '''
    build ast with tree_sitter, operator and keyword has no pre-defined type
    :param root: ast tree root node
    :param code_lines: [...], ...
    :return: {1*NODEFI1: {'node': 'XX', 'parent': 'None', 'children': [XX, ...]}}
    '''
    ast_tree = {}

    def dfs(cur_node, parent_node_ind):
        if len(cur_node.children) == 0:
            # current node has no child node, it's leaf node, build a leaf node
            if cur_node.is_named:
                # leaf node's value is None. we have to extract its value from source code
                token_name = subcode(cur_node.start_point, cur_node.end_point, code_lines)
                new_node = {
                    'node': cur_node.type,
                    'parent': parent_node_ind,
                    'children': [token_name.strip()],
                }
                new_node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)
                ast_tree[new_node_ind] = new_node
                ast_tree[parent_node_ind]['children'].append(new_node_ind)
            else:
                # leaf node is operator or keyword
                new_node = {
                    'node': define_default_node_type(cur_node.type),
                    'parent': parent_node_ind,
                    'children': [cur_node.type],
                }
                new_node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)
                ast_tree[new_node_ind] = new_node
                ast_tree[parent_node_ind]['children'].append(new_node_ind)
        else:
            # current node has many child nodes
            node_ind = constants.NODE_FIX + str(len(ast_tree) + 1)  # root node index
            node = {
                'node': cur_node.type,
                'parent': parent_node_ind,
                'children': [],
            }
            ast_tree[node_ind] = node
            # update parent node's child nodes
            if parent_node_ind is None:
                pass
            else:
                ast_tree[parent_node_ind]['children'].append(node_ind)

            for child_node in cur_node.children:
                dfs(child_node, parent_node_ind=node_ind)

    dfs(root, parent_node_ind=None)
    return ast_tree


def delete_comment_child_node(ast_tree: Dict) -> Dict:
    '''
    delete comment node
    :param ast_tree:
    :return:
    '''

    def delete_cur_node(node_ind, cur_node):
        # update its parent's children
        parent_ind = cur_node['parent']
        parent_node = ast_tree[parent_ind]
        del_ind = parent_node['children'].index(node_ind)
        del parent_node['children'][del_ind]
        # delete node
        ast_tree.pop(node_ind)
        return parent_ind, parent_node

    def dfs(node_ind):
        cur_node = ast_tree[node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        if cur_node['node'] == 'comment':
            node_ind, cur_node = delete_cur_node(node_ind, cur_node)

            while len(cur_node['children']) == 0:
                node_ind, cur_node = delete_cur_node(node_ind, cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def delete_single_child_ndoe(ast_tree: Dict) -> Dict:
    '''
    delete nodes with single child node
    :param ast_tree:
    :return:
    '''

    def dfs(node_ind):
        cur_node = ast_tree[node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        # each ast tree generally is parsed from a method, so it has a "program" root node and a "method" node
        # therefore, if current node is the root node with single child, we do not delete it
        while len(child_node_indices) == 1 and cur_node['parent'] is not None:
            # update its parent's children
            parent_node = ast_tree[cur_node['parent']]
            del_ind = parent_node['children'].index(node_ind)
            del parent_node['children'][del_ind]
            child_ind = child_node_indices[0]
            # update its children's parent to its parent
            ast_tree[child_ind]['parent'] = cur_node['parent']
            # update its parent's children
            parent_node['children'].insert(del_ind, child_ind)
            # elete itself
            ast_tree.pop(node_ind)

            # update current info
            node_ind = child_ind
            cur_node = ast_tree[node_ind]
            child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def reset_indices(ast_tree: Dict) -> Dict:
    '''
    rename ast tree's node indices with consecutive indices
    :param ast_tree:
    :return:
    '''
    new_ind = 1

    root_ind = 1
    while 1:
        root_node_ind = constants.NODE_FIX + str(root_ind)
        if root_node_ind in ast_tree:
            break
        else:
            root_ind += 1

    def new_ndoe_name():
        nonlocal new_ind
        new_name = '_' + constants.NODE_FIX + str(new_ind)
        new_ind += 1
        return new_name

    def dfs(cur_node_ind):
        cur_node = ast_tree[cur_node_ind]
        # change from cur_node_ind to new_cur_node_ind
        # copy a same node with new name
        new_cur_node_ind = new_ndoe_name()
        ast_tree[new_cur_node_ind] = deepcopy(cur_node)

        # update its parent's child nodes
        if cur_node['parent'] is None:
            pass
        else:
            parent_node = ast_tree[cur_node['parent']]
            parent_node['children'][parent_node['children'].index(cur_node_ind)] = new_cur_node_ind

        if cur_node['children'][0].startswith(constants.NODE_FIX):
            # update its children nodes' parent
            for child_name in cur_node['children']:
                ast_tree[child_name]['parent'] = new_cur_node_ind
        else:
            pass

        # 2. delete old node
        ast_tree.pop(cur_node_ind)

        child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in child_node_indices:
            dfs(child_name)

    dfs(root_node_ind)

    # recover name
    node_names = deepcopy(list(ast_tree.keys()))
    for node_name in node_names:
        node = deepcopy(ast_tree[node_name])
        if node['children'][0].startswith('_' + constants.NODE_FIX):
            node['children'] = [child_name[1:] for child_name in node['children']]
        else:
            pass
        if node['parent'] == None:
            pass
        else:
            node['parent'] = node['parent'][1:]
        ast_tree[node_name[1:]] = node
        ast_tree.pop(node_name)

    return ast_tree


def to_binary_tree(ast_tree: Dict) -> Dict:
    '''
    ast tree -> binary ast tree
    :param ast_tree:
    :return:
    '''
    last_node_ind = util.last_index(ast_tree)

    def dfs(cur_node_ind):
        cur_node = ast_tree[cur_node_ind]
        child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) > 2:
            # add new node
            nonlocal last_node_ind
            last_node_ind += 1
            new_node_ind = constants.NODE_FIX + str(last_node_ind)
            new_node = {
                'node': constants.NODE_TMP,
                'parent': cur_node_ind,
                'children': child_node_indices[1:],
            }
            ast_tree[new_node_ind] = new_node
            # update node's children info
            cur_node['children'] = [child_node_indices[0], new_node_ind]
            # update other childen nodes' parent info
            for child_name in child_node_indices[1:]:
                if child_name.startswith(constants.NODE_FIX) and child_name in ast_tree:
                    ast_tree[child_name]['parent'] = new_node_ind
            # update current node's children info
            child_node_indices = util.get_tree_children_func(cur_node)

        if len(child_node_indices) == 0:
            return

        for child_name in cur_node['children']:
            dfs(child_name)

    dfs(constants.ROOT_NODE_NAME)
    return ast_tree


def split_and_pad_token(token: str, MAX_TOKEN_LIST_LEN: int, to_lower=True, PAD_TOKEN=constants.PAD_WORD) -> List:
    '''
    split token and pad it with PAD_TOKEN till reach MAX_TOKEN_LIST_LEN
    e.g. VariableName ->  [VariableName, [Variable, Name, PAD_TOKEN, PAD_TOKEN, ...]]
    :param token: raw token
    :param MAX_TOKEN_LIST_LEN: max pad length
    :param to_lower:
    :return:
    '''
    token_list = util.split_identifier(token)
    if to_lower:
        token_list = [str.lower(token) for token in token_list]
    token_list.extend([PAD_TOKEN for _ in range(MAX_TOKEN_LIST_LEN - len(token_list))])
    return token_list


def pad_leaf_node(ast_tree: Dict, MAX_LEN: int, to_lower=True, PAD_TOKEN=constants.PAD_WORD) -> Dict:
    '''
    pad leaf node's child into [XX, [XX, ...]]
    :param ast_tree:
    :param MAX_LEN: max pad length
    :return:
    '''
    for key, node in ast_tree.items():
        if len(node['children']) == 1 and (not str.startswith(node['children'][0], constants.NODE_FIX)):
            ast_tree[key]['children'].append(
                split_and_pad_token(ast_tree[key]['children'][0], MAX_LEN, to_lower, PAD_TOKEN)
            )
    return ast_tree


def build_sbt_tree(ast_tree: Dict, node_ind: str, to_lower: bool) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    :param ast_tree:
    :param node_ind:
    :param to_lower:
    :return:
    '''
    if len(ast_tree[node_ind]['children']) > 1 and type(ast_tree[node_ind]['children'][1]) == list:
        token = ast_tree[node_ind]['node'] + '_' + ast_tree[node_ind]['children'][0]
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token, constants.SBT_PARENTHESES[1], token]
    else:
        token = ast_tree[node_ind]['node']
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token]
        for child_ind in ast_tree[node_ind]['children']:
            seq += build_sbt_tree(ast_tree, child_ind, to_lower)
        seq += [constants.SBT_PARENTHESES[1], token]
    return seq


def build_sbtao_tree(ast_tree: Dict, node_ind: str, to_lower: bool) -> List:
    '''
    build structure-based traversal SBT tree
    ref: Deep Code Comment Generation
    :param ast_tree:
    :param node_ind:
    :param to_lower:
    :return:
    '''
    if len(ast_tree[node_ind]['children']) > 1 and type(ast_tree[node_ind]['children'][1]) == list:
        token = ast_tree[node_ind]['node'] + '_' + '<other>'
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token, constants.SBT_PARENTHESES[1], token]
    else:
        token = ast_tree[node_ind]['node']
        if to_lower:
            token = token.lower()
        seq = [constants.SBT_PARENTHESES[0], token]
        for child_ind in ast_tree[node_ind]['children']:
            seq += build_sbtao_tree(ast_tree, child_ind, to_lower)
        seq += [constants.SBT_PARENTHESES[1], token]
    return seq


def build_sbt2_tree(ast_tree: Dict, node_ind: str, to_lower: bool) -> List:
    # our SBT tree
    if len(ast_tree[node_ind]['children']) > 1 and type(ast_tree[node_ind]['children'][1]) == list:
        child_token_list = list(filter(lambda token: not token == constants.PAD_WORD, ast_tree[node_ind]['children'][1]))
        token_list = [ast_tree[node_ind]["node"]] + child_token_list
        if to_lower:
            token_list = [token.lower() for token in token_list]
        seq = [constants.SBT_PARENTHESES[0]] + token_list + [constants.SBT_PARENTHESES[0]] + token_list
    else:
        token_list = ast_tree[node_ind]["node"]
        if to_lower:
            token_list = token_list.lower()
        seq = [constants.SBT_PARENTHESES[0], token_list]
        for child_ind in ast_tree[node_ind]['children']:
            seq += build_sbt2_tree(ast_tree, child_ind, to_lower)
        seq += [constants.SBT_PARENTHESES[1], token_list]
    return seq


def parse_raw_comment(comment: str, to_lower: bool) -> List:
    '''
    parse comment from docstring
    :param comment:
    :param to_lower:
    :return:
    '''
    comment = re.sub(r'\{\@\S+', '', comment)
    comment = re.sub(r'{.+}', '', comment)
    comment = ''.join([char for char in comment if char not in constants.MEANINGLESS_TOKENS])
    comment = [util.split_identifier(token, str_flag=False) for token in comment.split(' ')]
    comment = list(itertools.chain(*comment))
    # comment = list(filter(lambda token: token not in MEANINGLESS_TOKENS, comment))
    comment = re.split(r'\s+', ' '.join(comment).strip())
    if to_lower:
        comment = [str.lower(token) if to_lower else token for token in comment]
    return comment


def parse_comment(comment: List, to_lower: bool) -> List:
    # parse comment from docstring_tokens
    comment = [''.join([char for char in token if char not in constants.MEANINGLESS_TOKENS]) for token in comment]
    comment = itertools.chain(*[util.split_identifier(token, str_flag=False) for token in comment])
    comment = re.split(r'\s+', ' '.join(comment).strip())
    if to_lower:
        comment = [str.lower(token) if to_lower else token for token in comment]
    return comment


def filter_and_seperate(token_list: List, to_lower: bool) -> List:
    token_list = list(itertools.chain(*[util.split_identifier(token.strip()) for token in token_list
                                        if len(token.strip()) > 0]))
    if to_lower:
        token_list = [str.lower(token) if to_lower else token for token in token_list]
    return token_list


class CodeParser(object):
    '''parse code data into ast'''
    __slots__ = ('parser', 'to_lower', 'LANGUAGE',)

    def __init__(self, SO_FILE: str, LANGUAGE: str, to_lower=True, ):
        self.parser = Parser()
        self.parser.set_language(Language(SO_FILE, LANGUAGE))
        self.LANGUAGE = LANGUAGE
        self.to_lower = to_lower

    def parse_comment(self, docstring: str, docstring_tokens: List[str], ) -> Tuple:
        # comment
        error = ''
        if (docstring_tokens[-1] in constants.COMMENT_END_TOKENS) or (len(docstring_tokens) > constants.MAX_COMMENT_TOKEN_LIST_LEN):
            # if docstring_tokens is too long or docstring_tokens is wrong parsed
            ''' exceptions in CodeSearchNet, eg.
            docstring: 'Set {@link ServletRegistrationBean}s that the filter will be registered against.
                        @param servletRegistrationBeans the Servlet registration beans'
            docstring_tokens:  <class 'list'>: ['Set', '{'] ['.']
            '''
            ############################
            # save bad cases
            ############################
            error += 'docstring_tokens: last element is wrong, or length too long.\nuse docstring to generate comment\t'

            # skip this code snippet, if there are non-ascii tokens
            if not util.is_ascii(docstring):
                return None, error + 'contain non-ascii'
            raw_comment = re.sub(
                '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?', ' ',
                docstring)  # delete url
            # remove addtional and useless tails info
            raw_comment = str.split(raw_comment, '\n\n')[0].replace('\n', ' ')
            raw_comment = re.split(r'[\.|:]', raw_comment)
            raw_comment = raw_comment[0] + '.'
            raw_comment = parse_raw_comment(raw_comment, to_lower=self.to_lower)
        else:
            # skip this code snippet, if there are non-ascii tokens
            for comment_token in docstring_tokens:
                if not util.is_ascii(comment_token):
                    return None, error + 'contain non-ascii'
            raw_comment = parse_comment(docstring_tokens, to_lower=self.to_lower)

        # add . at the end of sentence
        if raw_comment[-1] == '.':
            pass
        elif raw_comment[-1] == ':':
            raw_comment[-1] = '.'
        else:
            raw_comment.append('.')

        raw_comment = ' '.join(raw_comment)
        raw_comment = re.sub(r'[\-|\*|\=|\~]{2,}', ' ', raw_comment)  # remove ----+/****+/====+,
        raw_comment = re.sub(r'[!]{2,}', '!', raw_comment)  # change !!!! -> !
        raw_comment = re.sub(r'[`]{2,}', ' ` ', raw_comment)  # change ```, -> ` ,

        ########################################################################################
        # for rouge
        # remove <**> </**>
        raw_comment = re.sub(r'<.*?>', '', raw_comment)
        # remove =>
        raw_comment = raw_comment.replace('= >', ' ').replace('=>', ' ')
        # remove < >
        raw_comment = raw_comment.replace('<', ' ').replace('>', ' ')
        ########################################################################################

        raw_comment = re.sub(r'\s+', ' ', raw_comment)
        raw_comment = raw_comment.split(' ')

        new_raw_comment = []
        for token in raw_comment:
            token = token.strip()
            if len(token) > 0:
                # rule 1: +XX+ -> XX
                if token[0] == '+' or token[-1] == '+':
                    new_raw_comment.append(token[1:-1].strip())
                else:
                    new_raw_comment.append(token.strip())
        comment = new_raw_comment

        try:
            assert 3 < len(comment) <= 60, 'comment token too short/long(3<len<60)'
            return comment, error
        except:
            return None, error + 'contain non-ascii'

    def parse_raw_ast(self, code: str, ) -> Union[None, Dict]:
        # ast tree modal
        if self.LANGUAGE == 'php':
            code = '<?php ' + code
        try:
            ast_tree = self.parser.parse(bytes(code.replace('\t', '    ').replace('\n', ' ').strip(), "utf8"))
        except Exception as err:
            LOGGER.info(err)
            return None

        code_lines = code.split('\n')  # raw code
        # 1) build ast tree in Dict type
        code_tree = build_tree(ast_tree.root_node, code_lines)
        # 2) delete comment node
        code_tree = delete_comment_child_node(code_tree)

        # # save for AST path
        # raw_ast = deepcopy(reset_indices(code_tree))
        # leaf_path = ast_to_path(raw_ast)

        # 3) pop useless head node
        if (code_tree[constants.NODE_FIX + '1']['parent'] is None) and \
                len(code_tree[constants.NODE_FIX + '1']['children']) == 1 and \
                code_tree[constants.NODE_FIX + '1']['children'][0].startswith(constants.NODE_FIX):
            child_node = code_tree[constants.NODE_FIX + '1']['children'][0]
            code_tree[child_node]['parent'] = None
            code_tree.pop(constants.NODE_FIX + '1')
        # 4) reset tree indices
        raw_ast = reset_indices(code_tree)  # reset node indices

        # # parse_base(ast)  # binary & reset node indices
        # # pad_leaf_node(ast, max_sub_token_size)  # pad children to same length
        #
        # if sbt_flag:
        #     sbt1 = parse_deepcom(deepcopy(ast), build_sbt_tree, to_lower=True, )
        #     sbt2 = parse_deepcom(deepcopy(ast), build_sbt2_tree, to_lower=True, )

        return raw_ast

    def parse_method(self, func_name: str, ) -> List:
        # method name modal
        method = ''
        for char in func_name:
            if str.isalpha(char) or str.isdigit(char):
                method += char
            else:
                method += ' '
        method = parse_comment(re.split(r'\s+', method), True)
        method = [token.strip() for token in method if len(token.strip()) > 1]
        if len(method) > 0:
            return method
        else:
            return ['<no_func>']

    def parse_tok(self, code_tokens: List[str], ) -> Tuple:
        # tok modal
        # skip this code snippet, if there are non-ascii tokens or code token is too long
        # filter comment in code_tokens, eg. //***\n /* */\n
        raw_len = len(code_tokens)
        code_tokens = [token for token in code_tokens
                       if not (
                    str.startswith(token, '//') or \
                    str.startswith(token, '#') or \
                    (str.startswith(token, '/*') and str.endswith(token, '*/'))
            )]
        if raw_len != len(code_tokens):
            error = 'contain comments(but removed)\t'
        else:
            error = ''

        raw_code = [None] * len(code_tokens)
        for ind, token in enumerate(code_tokens):
            raw_code[ind] = token.strip()
            if not util.is_ascii(raw_code[ind]) or len(raw_code[ind]) > constants.MAX_CODE_TOKEN_LEN:
                return None, error + 'some token are too long/non-ascii ({})'.format(raw_code[ind])

        raw_seq = filter_and_seperate(raw_code, to_lower=self.to_lower)
        tok = [token.strip() for token in raw_seq if len(token.strip()) > 0]
        try:
            assert len(tok) > 3
            return tok, error
        except:
            return None, error + 'code token too short(<=3)'

    def parse(self, code_snippet: Dict, ) -> Union[Dict, None]:
        # code snippets with seq/tree modalities
        code_snippet_with_modalities = dict(tok=None, tree=None, leaf_path=None, comment=None, func_name=None, )
        code_snippet_with_modalities['tok'] = self.parse_tok(code_snippet['code_snippet'])

        tree = self.parse_ast(code_snippet['code'], leaf_path=True, sbt=True)
        if tree is None:
            return None
        else:
            ast, leaf_path, sbt1, sbt2, = tree
            code_snippet_with_modalities['ast'] = ast
            code_snippet_with_modalities['leaf_path'] = leaf_path

        code_snippet_with_modalities['comment'] = \
            self.parse_comment(code_snippet['docstring'], code_snippet['docstring_tokens'])
        code_snippet_with_modalities['func_name'] = self.parse_method(code_snippet['func_name'])
        return code_snippet_with_modalities


def parse_base(ast_tree: Dict) -> Dict:
    # delete nodes with single node,eg. [1*NODEFIX1] ->  [1*NODEFIX2] -> ['void'] => [1*NODEFIX1] -> ['void']
    ast_tree = delete_single_child_ndoe(ast_tree)
    ast_tree = to_binary_tree(ast_tree)  # to binary ast tree
    ast_tree = reset_indices(ast_tree)  # reset node indices
    return ast_tree


def parse_deepcom(ast_tree: dict, sbt_func: None, to_lower: bool):
    sbt_seq = sbt_func(ast_tree, constants.ROOT_NODE_NAME, to_lower)
    return sbt_seq


if __name__ == '__main__':
    parser = Parser()
    parser.set_language(
        Language('/data/wanyao/yang/ghproj_d/GitHub/naturalcodev2/dataset/parser_zips/languages.so', 'ruby'))
    code = '''
def create_router(name, admin_state_up = true)
    data = {
        'router' =>{
            'name' => name,
            'admin_state_up' => admin_state_up,
        }   
    }
    return post_request(address("routers"), data, @token)
end 
    '''.replace('\t', '    ').replace('\n', ' ').strip()
    ast_tree = parser.parse(bytes(code, "utf8"))
    code_lines = code.split('\n')  # raw code
    # 1) build ast tree in Dict type
    print(ast_tree.root_node.sexp())
    code_tree = build_tree(ast_tree.root_node, code_lines)
    # 2) delete nodes whose children are parentheses, eg. [] and ()
    code_tree = delete_comment_child_node(code_tree)
    # # 4) to binary ast tree, optional
    # code_tree = to_binary(code_tree)
    # 5) reset node indices
    code_tree = reset_indices(code_tree)
    # sbt = to_sbt2(code_tree, ROOT_NODE_NAME, True)
    print(code_tree)

    code_tree = parse_base(code_tree)
    path = util_path.ast_to_path(code_tree)
