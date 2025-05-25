import re
import json
import os
from collections import defaultdict
import ast

def is_user_defined_type(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 存在类定义，说明涉及用户自定义类型
                return True
    except SyntaxError:
        # 代码解析出错，不是有效的Python代码
        return False

    return False



def find_lines_with_keyword(code, keyword):
    lines_with_keyword = []

    # 将代码分成行
    code_lines = code.split('\n')

    # 遍历每一行
    for line in code_lines:
        # 检查关键字是否在该行中
        if keyword in line:
            lines_with_keyword.append(line)

    # 将匹配的行拼接成一个字符串
    result_string = '\n'.join(lines_with_keyword)

    return result_string

def infer_simple_type_from_assignment(assignment_string):
    # 匹配赋值语句的正则表达式
    assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.*)\s*$')

    match = assignment_pattern.match(assignment_string)

    if match:
        variable_name, value_str = match.groups()

        # 匹配整数的正则表达式
        int_pattern = re.compile(r'^[+-]?\d+$')

        # 匹配浮点数的正则表达式
        float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

        # 匹配布尔值的正则表达式
        bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

        # 匹配字符串的正则表达式
        str_pattern = re.compile(r'^\'(.*)\'$')

        # 匹配bytes的正则表达式
        bytes_pattern = re.compile(r'^b\'(.*)\'$')

        # 匹配列表的正则表达式
        list_pattern = re.compile(r'^\s*\[.*\]\s*$')

        # 匹配元组的正则表达式
        tuple_pattern = re.compile(r'^\s*\((.*)\)\s*$')

        # 匹配字典的正则表达式
        dict_pattern = re.compile(r'^\s*\{.*:.*\}\s*$')

        # 匹配集合的正则表达式
        set_pattern = re.compile(r'^\s*\{[^:{}]*\}\s*$')

        # 检查字符串格式并返回对应的类型
        if int_pattern.match(value_str):
            return "int"
        elif float_pattern.match(value_str):
            return "float"
        elif bool_pattern.match(value_str):
            return "bool"
        elif str_pattern.match(value_str):
            return "str"
        elif bytes_pattern.match(value_str):
            return "bytes"
        elif list_pattern.match(value_str):
            return "list"
        elif tuple_pattern.match(value_str):
            return "tuple"
        elif dict_pattern.match(value_str):
            return "dict"
        elif set_pattern.match(value_str):
            return "set"
        else:
            return None
    else:
        return None


def infer_simple_type_from_assignment_arg(assignment_string):
    # 匹配赋值语句的正则表达式
    assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.*)\s*$')

    match = assignment_pattern.match(assignment_string)

    if match:
        variable_name, value_str = match.groups()

        # 匹配整数的正则表达式
        int_pattern = re.compile(r'^[+-]?\d+$')

        # 匹配浮点数的正则表达式
        float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

        # 匹配布尔值的正则表达式
        bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

        # 匹配字符串的正则表达式
        str_pattern = re.compile(r'^\'(.*)\'$')

        # 匹配bytes的正则表达式
        bytes_pattern = re.compile(r'^b\'(.*)\'$')



        # 检查字符串格式并返回对应的类型
        if int_pattern.match(value_str):
            return "int"
        elif float_pattern.match(value_str):
            return "float"
        elif bool_pattern.match(value_str):
            return "bool"
        elif str_pattern.match(value_str):
            return "str"
        elif bytes_pattern.match(value_str):
            return "bytes"

        else:
            return None
    else:
        return None

def extract_elements(input_string):
    result = {"substring_before_bracket": None, "inner_substrings": None}

    # 找到等号并做标记
    equal_sign_index = input_string.find('=')

    if equal_sign_index != -1:
        # 从等号位置开始继续扫描，直到遇到左括号或者左方括号
        for i in range(equal_sign_index, len(input_string)):
            if input_string[i] == '(':
                # 从等号位置到左括号位置的子串
                substring_before_parenthesis = input_string[equal_sign_index + 1:i].strip()
                result["outer_type"] = "tuple"
                result["substring_before_bracket"] = substring_before_parenthesis

                # 倒着扫描，找到右括号位置
                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ')':
                        # 括号中间的子串
                        inner_substring = input_string[i + 1:j].strip()

                        # 遍历括号中间的子串，检查是否含有 '[' 或者 '('
                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                        else:
                            # 使用逗号分割子串
                            inner_substring_list = inner_substring.split(',')
                            result["inner_substrings"] = inner_substring_list
                        break

                # 结束外层循环
                break
            elif input_string[i] == '[':
                # 从等号位置到左方括号位置的子串
                substring_before_bracket = input_string[equal_sign_index + 1:i].strip()
                result["substring_before_bracket"] = substring_before_bracket

                # 倒着扫描，找到右方括号位置
                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ']':
                        # 括号中间的子串
                        inner_substring = input_string[i + 1:j].strip()

                        # 遍历括号中间的子串，检查是否含有 '[' 或者 '('
                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                        else:
                            # 使用逗号分割子串
                            inner_substring_list = inner_substring.split(',')
                            result["inner_substrings"] = inner_substring_list
                        break

                # 结束外层循环
                break
    else:
        print("Equal sign not found.")

    return result

