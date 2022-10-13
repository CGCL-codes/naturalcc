from pprint import pprint

from dataset.utils.parser import CodeParser
from dataset.utils.test import (LANGUAGES, SO_FILES_MAP)

LANG = 'python'

parser = CodeParser(so_file=SO_FILES_MAP[LANG], language=LANG)

# code = \
#     """
#     def sum_of_list(arr_, idx = 0):
#         arr_len = len(arr_)
#         if idx == arr_len:
#             return 0
#         else:
#             return arr_[idx] + sum_of_list(arr_, idx + 1)
#     """

"""
{0: {'children': [1], 'parent': None, 'type': 'module'},
 1: {'children': [2, 3, 4, 13, 14], 'parent': 0, 'type': 'function_definition'},
 2: {'parent': 1, 'type': 'Keyword', 'value': 'def'},
 3: {'parent': 1, 'type': 'identifier', 'value': 'sum_of_list'},
 4: {'children': [5, 6, 7, 8, 12], 'parent': 1, 'type': 'parameters'},
 5: {'parent': 4, 'type': 'LeftParenOp', 'value': '('},
 6: {'parent': 4, 'type': 'identifier', 'value': 'arr_'},
 7: {'parent': 4, 'type': 'CommaOp', 'value': ','},
 8: {'children': [9, 10, 11], 'parent': 4, 'type': 'default_parameter'},
 9: {'parent': 8, 'type': 'identifier', 'value': 'idx'},
 10: {'parent': 8, 'type': 'AsgnOp', 'value': '='},
 11: {'parent': 8, 'type': 'integer', 'value': '0'},
 12: {'parent': 4, 'type': 'LeftParenOp', 'value': ')'},
 13: {'parent': 1, 'type': 'ColonOp', 'value': ':'},
 14: {'children': [15, 27], 'parent': 1, 'type': 'block'},
 15: {'children': [16], 'parent': 14, 'type': 'expression_statement'},
 16: {'children': [17, 19, 20], 'parent': 15, 'type': 'assignment'},
 17: {'children': [18], 'parent': 16, 'type': 'expression_list'},
 18: {'parent': 17, 'type': 'identifier', 'value': 'arr_len'},
 19: {'parent': 16, 'type': 'AsgnOp', 'value': '='},
 20: {'children': [21], 'parent': 16, 'type': 'expression_list'},
 21: {'children': [22, 23], 'parent': 20, 'type': 'call'},
 22: {'parent': 21, 'type': 'identifier', 'value': 'len'},
 23: {'children': [24, 25, 26], 'parent': 21, 'type': 'argument_list'},
 24: {'parent': 23, 'type': 'LeftParenOp', 'value': '('},
 25: {'parent': 23, 'type': 'identifier', 'value': 'arr_'},
 26: {'parent': 23, 'type': 'LeftParenOp', 'value': ')'},
 27: {'children': [28, 29, 33, 34, 39], 'parent': 14, 'type': 'if_statement'},
 28: {'parent': 27, 'type': 'Keyword', 'value': 'if'},
 29: {'children': [30, 31, 32], 'parent': 27, 'type': 'comparison_operator'},
 30: {'parent': 29, 'type': 'identifier', 'value': 'idx'},
 31: {'parent': 29, 'type': 'EqualOp', 'value': '=='},
 32: {'parent': 29, 'type': 'identifier', 'value': 'arr_len'},
 33: {'parent': 27, 'type': 'ColonOp', 'value': ':'},
 34: {'children': [35], 'parent': 27, 'type': 'block'},
 35: {'children': [36, 37], 'parent': 34, 'type': 'return_statement'},
 36: {'parent': 35, 'type': 'Keyword', 'value': 'return'},
 37: {'children': [38], 'parent': 35, 'type': 'expression_list'},
 38: {'parent': 37, 'type': 'integer', 'value': '0'},
 39: {'children': [40, 41, 42], 'parent': 27, 'type': 'else_clause'},
 40: {'parent': 39, 'type': 'Keyword', 'value': 'else'},
 41: {'parent': 39, 'type': 'ColonOp', 'value': ':'},
 42: {'children': [43], 'parent': 39, 'type': 'block'},
 43: {'children': [44, 45], 'parent': 42, 'type': 'return_statement'},
 44: {'parent': 43, 'type': 'Keyword', 'value': 'return'},
 45: {'children': [46], 'parent': 43, 'type': 'expression_list'},
 46: {'children': [47, 52, 53], 'parent': 45, 'type': 'binary_operator'},
 47: {'children': [48, 49, 50, 51], 'parent': 46, 'type': 'subscript'},
 48: {'parent': 47, 'type': 'identifier', 'value': 'arr_'},
 49: {'parent': 47, 'type': 'LeftBracketOp', 'value': '['},
 50: {'parent': 47, 'type': 'identifier', 'value': 'idx'},
 51: {'parent': 47, 'type': 'RightBracketOp', 'value': ']'},
 52: {'parent': 46, 'type': 'AddOp', 'value': '+'},
 53: {'children': [54, 55], 'parent': 46, 'type': 'call'},
 54: {'parent': 53, 'type': 'identifier', 'value': 'sum_of_list'},
 55: {'children': [56, 57, 58, 59, 63], 'parent': 53, 'type': 'argument_list'},
 56: {'parent': 55, 'type': 'LeftParenOp', 'value': '('},
 57: {'parent': 55, 'type': 'identifier', 'value': 'arr_'},
 58: {'parent': 55, 'type': 'CommaOp', 'value': ','},
 59: {'children': [60, 61, 62], 'parent': 55, 'type': 'binary_operator'},
 60: {'parent': 59, 'type': 'identifier', 'value': 'idx'},
 61: {'parent': 59, 'type': 'AddOp', 'value': '+'},
 62: {'parent': 59, 'type': 'integer', 'value': '1'},
 63: {'parent': 55, 'type': 'LeftParenOp', 'value': ')'}}
"""

# code = \
#     """
#     def remove(array, e):
#         return filter(lambda x: x != e, array)
#     """.strip()

"""
{0: {'children': [1], 'parent': None, 'type': 'module'},
 1: {'children': [2, 3, 4, 10, 11], 'parent': 0, 'type': 'function_definition'},
 2: {'parent': 1, 'type': 'Keyword', 'value': 'def'},
 3: {'parent': 1, 'type': 'identifier', 'value': 'remove'},
 4: {'children': [5, 6, 7, 8, 9], 'parent': 1, 'type': 'parameters'},
 5: {'parent': 4, 'type': 'LeftParenOp', 'value': '('},
 6: {'parent': 4, 'type': 'identifier', 'value': 'array'},
 7: {'parent': 4, 'type': 'CommaOp', 'value': ','},
 8: {'parent': 4, 'type': 'identifier', 'value': 'e'},
 9: {'parent': 4, 'type': 'LeftParenOp', 'value': ')'},
 10: {'parent': 1, 'type': 'ColonOp', 'value': ':'},
 11: {'children': [12], 'parent': 1, 'type': 'block'},
 12: {'children': [13, 14], 'parent': 11, 'type': 'return_statement'},
 13: {'parent': 12, 'type': 'Keyword', 'value': 'return'},
 14: {'children': [15], 'parent': 12, 'type': 'expression_list'},
 15: {'children': [16, 17], 'parent': 14, 'type': 'call'},
 16: {'parent': 15, 'type': 'identifier', 'value': 'filter'},
 17: {'children': [18, 19, 28, 29, 30], 'parent': 15, 'type': 'argument_list'},
 18: {'parent': 17, 'type': 'LeftParenOp', 'value': '('},
 19: {'children': [20, 21, 23, 24], 'parent': 17, 'type': 'lambda'},
 20: {'parent': 19, 'type': 'Keyword', 'value': 'lambda'},
 21: {'children': [22], 'parent': 19, 'type': 'lambda_parameters'},
 22: {'parent': 21, 'type': 'identifier', 'value': 'x'},
 23: {'parent': 19, 'type': 'ColonOp', 'value': ':'},
 24: {'children': [25, 26, 27], 'parent': 19, 'type': 'comparison_operator'},
 25: {'parent': 24, 'type': 'identifier', 'value': 'x'},
 26: {'parent': 24, 'type': 'InequalOp', 'value': '!='},
 27: {'parent': 24, 'type': 'identifier', 'value': 'e'},
 28: {'parent': 17, 'type': 'CommaOp', 'value': ','},
 29: {'parent': 17, 'type': 'identifier', 'value': 'array'},
 30: {'parent': 17, 'type': 'LeftParenOp', 'value': ')'}}
"""

# code = \
#     """
#     def fib(n):
#         if n <= 2:
#             return n
#         else:
#             return fib(n - 1) + fib(n - 2)
#     """.strip()
"""
{0: {'children': [1], 'parent': None, 'type': 'module'},
 1: {'children': [2, 3, 4, 8, 9], 'parent': 0, 'type': 'function_definition'},
 2: {'parent': 1, 'type': 'Keyword', 'value': 'def'},
 3: {'parent': 1, 'type': 'identifier', 'value': 'fib'},
 4: {'children': [5, 6, 7], 'parent': 1, 'type': 'parameters'},
 5: {'parent': 4, 'type': 'LeftParenOp', 'value': '('},
 6: {'parent': 4, 'type': 'identifier', 'value': 'n'},
 7: {'parent': 4, 'type': 'LeftParenOp', 'value': ')'},
 8: {'parent': 1, 'type': 'ColonOp', 'value': ':'},
 9: {'children': [10], 'parent': 1, 'type': 'block'},
 10: {'children': [11, 12, 16, 17, 22], 'parent': 9, 'type': 'if_statement'},
 11: {'parent': 10, 'type': 'Keyword', 'value': 'if'},
 12: {'children': [13, 14, 15], 'parent': 10, 'type': 'comparison_operator'},
 13: {'parent': 12, 'type': 'identifier', 'value': 'n'},
 14: {'parent': 12, 'type': 'LEOp', 'value': '<='},
 15: {'parent': 12, 'type': 'integer', 'value': '2'},
 16: {'parent': 10, 'type': 'ColonOp', 'value': ':'},
 17: {'children': [18], 'parent': 10, 'type': 'block'},
 18: {'children': [19, 20], 'parent': 17, 'type': 'return_statement'},
 19: {'parent': 18, 'type': 'Keyword', 'value': 'return'},
 20: {'children': [21], 'parent': 18, 'type': 'expression_list'},
 21: {'parent': 20, 'type': 'identifier', 'value': 'n'},
 22: {'children': [23, 24, 25], 'parent': 10, 'type': 'else_clause'},
 23: {'parent': 22, 'type': 'Keyword', 'value': 'else'},
 24: {'parent': 22, 'type': 'ColonOp', 'value': ':'},
 25: {'children': [26], 'parent': 22, 'type': 'block'},
 26: {'children': [27, 28], 'parent': 25, 'type': 'return_statement'},
 27: {'parent': 26, 'type': 'Keyword', 'value': 'return'},
 28: {'children': [29], 'parent': 26, 'type': 'expression_list'},
 29: {'children': [30, 39, 40], 'parent': 28, 'type': 'binary_operator'},
 30: {'children': [31, 32], 'parent': 29, 'type': 'call'},
 31: {'parent': 30, 'type': 'identifier', 'value': 'fib'},
 32: {'children': [33, 34, 38], 'parent': 30, 'type': 'argument_list'},
 33: {'parent': 32, 'type': 'LeftParenOp', 'value': '('},
 34: {'children': [35, 36, 37], 'parent': 32, 'type': 'binary_operator'},
 35: {'parent': 34, 'type': 'identifier', 'value': 'n'},
 36: {'parent': 34, 'type': 'SubOp', 'value': '-'},
 37: {'parent': 34, 'type': 'integer', 'value': '1'},
 38: {'parent': 32, 'type': 'LeftParenOp', 'value': ')'},
 39: {'parent': 29, 'type': 'AddOp', 'value': '+'},
 40: {'children': [41, 42], 'parent': 29, 'type': 'call'},
 41: {'parent': 40, 'type': 'identifier', 'value': 'fib'},
 42: {'children': [43, 44, 48], 'parent': 40, 'type': 'argument_list'},
 43: {'parent': 42, 'type': 'LeftParenOp', 'value': '('},
 44: {'children': [45, 46, 47], 'parent': 42, 'type': 'binary_operator'},
 45: {'parent': 44, 'type': 'identifier', 'value': 'n'},
 46: {'parent': 44, 'type': 'SubOp', 'value': '-'},
 47: {'parent': 44, 'type': 'integer', 'value': '2'},
 48: {'parent': 42, 'type': 'LeftParenOp', 'value': ')'}}
"""

code = \
    """
    def fib(i, n=1, m=1):
        if i == 0:
            return n
        else:
            return fib(i - 1, m, n + m)
    """.strip()

"""
{0: {'children': [1], 'parent': None, 'type': 'module'},
 1: {'children': [2, 3, 4, 18, 19], 'parent': 0, 'type': 'function_definition'},
 2: {'parent': 1, 'type': 'Keyword', 'value': 'def'},
 3: {'parent': 1, 'type': 'identifier', 'value': 'fib'},
 4: {'children': [5, 6, 7, 8, 12, 13, 17], 'parent': 1, 'type': 'parameters'},
 5: {'parent': 4, 'type': 'LeftParenOp', 'value': '('},
 6: {'parent': 4, 'type': 'identifier', 'value': 'i'},
 7: {'parent': 4, 'type': 'CommaOp', 'value': ','},
 8: {'children': [9, 10, 11], 'parent': 4, 'type': 'default_parameter'},
 9: {'parent': 8, 'type': 'identifier', 'value': 'n'},
 10: {'parent': 8, 'type': 'AsgnOp', 'value': '='},
 11: {'parent': 8, 'type': 'integer', 'value': '1'},
 12: {'parent': 4, 'type': 'CommaOp', 'value': ','},
 13: {'children': [14, 15, 16], 'parent': 4, 'type': 'default_parameter'},
 14: {'parent': 13, 'type': 'identifier', 'value': 'm'},
 15: {'parent': 13, 'type': 'AsgnOp', 'value': '='},
 16: {'parent': 13, 'type': 'integer', 'value': '1'},
 17: {'parent': 4, 'type': 'LeftParenOp', 'value': ')'},
 18: {'parent': 1, 'type': 'ColonOp', 'value': ':'},
 19: {'children': [20], 'parent': 1, 'type': 'block'},
 20: {'children': [21, 22, 26, 27, 32], 'parent': 19, 'type': 'if_statement'},
 21: {'parent': 20, 'type': 'Keyword', 'value': 'if'},
 22: {'children': [23, 24, 25], 'parent': 20, 'type': 'comparison_operator'},
 23: {'parent': 22, 'type': 'identifier', 'value': 'i'},
 24: {'parent': 22, 'type': 'EqualOp', 'value': '=='},
 25: {'parent': 22, 'type': 'integer', 'value': '0'},
 26: {'parent': 20, 'type': 'ColonOp', 'value': ':'},
 27: {'children': [28], 'parent': 20, 'type': 'block'},
 28: {'children': [29, 30], 'parent': 27, 'type': 'return_statement'},
 29: {'parent': 28, 'type': 'Keyword', 'value': 'return'},
 30: {'children': [31], 'parent': 28, 'type': 'expression_list'},
 31: {'parent': 30, 'type': 'identifier', 'value': 'n'},
 32: {'children': [33, 34, 35], 'parent': 20, 'type': 'else_clause'},
 33: {'parent': 32, 'type': 'Keyword', 'value': 'else'},
 34: {'parent': 32, 'type': 'ColonOp', 'value': ':'},
 35: {'children': [36], 'parent': 32, 'type': 'block'},
 36: {'children': [37, 38], 'parent': 35, 'type': 'return_statement'},
 37: {'parent': 36, 'type': 'Keyword', 'value': 'return'},
 38: {'children': [39], 'parent': 36, 'type': 'expression_list'},
 39: {'children': [40, 41], 'parent': 38, 'type': 'call'},
 40: {'parent': 39, 'type': 'identifier', 'value': 'fib'},
 41: {'children': [42, 43, 47, 48, 49, 50, 54],
      'parent': 39,
      'type': 'argument_list'},
 42: {'parent': 41, 'type': 'LeftParenOp', 'value': '('},
 43: {'children': [44, 45, 46], 'parent': 41, 'type': 'binary_operator'},
 44: {'parent': 43, 'type': 'identifier', 'value': 'i'},
 45: {'parent': 43, 'type': 'SubOp', 'value': '-'},
 46: {'parent': 43, 'type': 'integer', 'value': '1'},
 47: {'parent': 41, 'type': 'CommaOp', 'value': ','},
 48: {'parent': 41, 'type': 'identifier', 'value': 'm'},
 49: {'parent': 41, 'type': 'CommaOp', 'value': ','},
 50: {'children': [51, 52, 53], 'parent': 41, 'type': 'binary_operator'},
 51: {'parent': 50, 'type': 'identifier', 'value': 'n'},
 52: {'parent': 50, 'type': 'AddOp', 'value': '+'},
 53: {'parent': 50, 'type': 'identifier', 'value': 'm'},
 54: {'parent': 41, 'type': 'LeftParenOp', 'value': ')'}}
"""

if __name__ == '__main__':
    ast = parser.parse(code)
    pprint(ast)
