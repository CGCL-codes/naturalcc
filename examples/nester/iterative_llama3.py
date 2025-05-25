# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import ast

from llama import Llama
import json
import re
import json
import os
from tqdm import tqdm

from collections import Counter


def filter_list(lst, keywords):
    return [item for item in lst if item in keywords]




def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    # print(string)
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')  # .replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')  # .replace("is ", "")
        if (" " in res and "[" not in res) or res.lower() == "unknown":
            res = None
        return res


def extract_outermost_brackets(input_string):
    stack = []
    outer_part = ""
    inner_part = ""

    for char in input_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack:
                 
                    continue

        if stack:
            inner_part += char
        else:
            outer_part += char

    return outer_part.strip(), inner_part.strip()


def extract_outermost_brackets_for_list(input_string):
    stack = []
    outer_part = ""
    inner_part = ""

    for char in input_string:
        if char == '[':
            stack.append(char)
        elif char == ']':
            if stack:
                stack.pop()
                if not stack:
                    continue

        if stack:
            inner_part += char
        else:
            outer_part += char

    return outer_part.strip(), inner_part.strip()


def extract_parameters_from_method(method_declaration):
    pattern = r'\w+\s*\(([^)]*)\)'
    match = re.search(pattern, method_declaration)

    if match:
        parameters = [param.strip() for param in match.group(1).split(',') if param.strip()]
        return parameters
    else:
        return None




    def analyze_data_flow(self, code_lines):
        current_id = 1
        for line in code_lines:
            if 'def ' in line:
                parameters = extract_parameters_from_method(line)
                for item in parameters:
                    if item not in self.data_flow_analysis:
                        self.data_flow_analysis[item] = set()
                    self.data_flow_analysis[item].add(current_id)
            if ' = ' in line:
                variable = parts[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
                self.data_flow_analysis[variable].add(current_id)
                parts = line.split('(')
                variable = parts[1].split(')')[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
                self.data_flow_analysis[variable].add(current_id)
            elif 'return' in line:
                if 'return' not in self.data_flow_analysis:
                    self.data_flow_analysis['return'] = set()
                pattern = r'return\s+(.*)'
                match = re.search(pattern, line)
                # print(match.group(1))
                # print(self.data_flow_analysis[match.group(1)])
                if match:
                    if match.group(1) == "None":
                        pattern = r'.*'
                        man = re.search(pattern, 'None')
                        self.data_flow_analysis['return'].add(man.group(0))
                        current_id += 1
                        continue;
                    if match.group(1) in self.data_flow_analysis:
                        self.data_flow_analysis['return'].add(match.group(1))
                        current_id += 1
                        continue;
                    blanket = 0
                    if '(' in match.group(1):
                        result_outer, result_inner = extract_outermost_brackets(match.group(1))
                        blanket = 1
                        if '.' not in result_outer:
                            # self.data_flow_analysis['return'].add(result_outer)
                            pattern = r'.*'
                            man = re.search(pattern, result_outer)
                            self.data_flow_analysis['return'].add(man.group(0))
                            current_id += 1
                            continue;
                    if '.' in match.group(1):
                        if blanket == 1:
                            parts = result_outer.split('.', 1)
                        else:
                            parts = match.group(1).split('.', 1)
                        if len(parts) > 1:
                            self.data_flow_analysis['return'].add(parts[0])
                            current_id += 1
                            continue;
                        else:
                            self.data_flow_analysis['return'].add(line)
                            current_id += 1
                            continue;
                elif 'yield' in line:
                    if 'yield' not in self.data_flow_analysis:
                        self.data_flow_analysis['return'] = set()
                    pattern = r'yield\s+(.*)'
                    match = re.search(pattern, line)
                    # print(match.group(1))
                    # print(self.data_flow_analysis[match.group(1)])
                    if match:
                        if match.group(1) == "None":
                            pattern = r'.*'
                            man = re.search(pattern, 'None')
                            self.data_flow_analysis['return'].add(man.group(0))
                            current_id += 1
                            continue;
                        if match.group(1) in self.data_flow_analysis:
                            self.data_flow_analysis['return'].add(match.group(1))
                            current_id += 1
                            continue;
                        blanket = 0
                        if '(' in match.group(1):
                            result_outer, result_inner = extract_outermost_brackets(match.group(1))
                            blanket = 1
                            if '.' not in result_outer:
                                self.data_flow_analysis['return'].add(result_outer)
                                current_id += 1
                                continue;
                        if '.' in match.group(1):
                            if blanket == 1:
                                parts = result_outer.split('.', 1)
                            else:
                                parts = match.group(1).split('.', 1)
                            if len(parts) > 1:
                                self.data_flow_analysis['return'].add(parts[0])
                                current_id += 1
                                continue;
                            else:
                                self.data_flow_analysis['return'].add(line)
                                current_id += 1
                                continue;
                else:
                    pattern = r'.*'
                    man = re.search(pattern, 'None')
                    self.data_flow_analysis['return'].add(man.group(0))

            current_id += 1

    def perform_analysis(self, code):
        code_lines = code.split('\n')
        self.analyze_control_flow(code_lines)
        self.analyze_data_flow(code_lines)


def find_lines_with_keyword(code, keyword):
    lines_with_keyword = []
    code_lines = code.split('\n')
    for line in code_lines:
        if keyword in line:
            lines_with_keyword.append(line)
    result_string = '\n'.join(lines_with_keyword)

    return result_string






def generate_ast_and_detect_type(assignment_string):
    try:
        # 使用 ast 模块解析赋值语句为 AST
        parsed_ast = ast.parse(assignment_string, mode='exec')

        # 提取赋值语句的右侧值部分
        value_node = parsed_ast.body[0].value

        # 根据 AST 结构判断赋值语句的类型
        if isinstance(value_node, ast.Dict):
            return "dict"
        elif isinstance(value_node, ast.Set):
            return "set"
        elif isinstance(value_node, ast.List):
            return "list"
        elif isinstance(value_node, ast.Tuple):
            return "tuple"
        else:
            return None
    except SyntaxError as e:
        return f"Syntax Error: {e}"



    equal_sign_index = input_string.find('=')

    if equal_sign_index != -1:
        
    else:
        print(input_string)
        print("Equal sign not found.")

    return result


def asignment_analysis(string_test, variable):
    # print("key:")
    # print(key)
    # print("string_test:")
    # print(string_test)
    result_type = infer_simple_type_from_assignment(string_test)
    # print("result_type:")
    # print(result_type)
    if result_type == None:
        # print(string_test)
        # result = re.sub(r'\([^)]*\)', '()', string_test)

        string_test_split = extract_elements(string_test)
        # print(key)
        # print(string_test_split)
        if string_test_split["inner_substrings"] and string_test_split["len"] != 1 and (
                generate_ast_and_detect_type(string_test) == "list" or generate_ast_and_detect_type(
            string_test) == "tuple"):
            # if string_test_split["inner_substrings"] and string_test_split["len"] != 1:
            # print(string_test_split)
            part_type = {}
            index = 0
            for part in string_test_split["inner_substrings"]:
                # print(part)
                part_type_ir = infer_simple_type_from_value(part)
                if part_type_ir == None:
                    instructions = [
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful, respectful and honest assistant. You can the the type of the variable when i give you source code. Please provide me with an answer in the following format:the type of the variable is str/int/float/bool/byte/list/tuple/dict/set/unknow"
                            },
                            {
                                "role": "user",
                                "content": part,
                            }
                        ],
                    ]
                    try:
                        results = generator.chat_completion(
                            instructions,  # type: ignore
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        part_type[part] = results[0]['generation']['content']
                    except:
                        part_type[part] = []
                        pass;
                    # print(part_type[part])
                    
                    if matches:
                        part_type[part] = ', '.join(matches)
                    else:
                        part_type[part] = None
                else:
                    part_type[part] = part_type_ir
            print(part_type)
            values = list(part_type.values())
        
            # value_counts = Counter(values)

        
            # most_common_value = value_counts.most_common(1)[0][0]
            if string_test_split["outer_type"] == "list":
                if all(value == values[0] for value in values):
                    result_type = "list[" + values[0] + "]"
                else:
                    # result_type = "list[typing.Optional[" +most_common_value +"]]"
                    result_type = "list[typing.Any]"
            elif string_test_split["outer_type"] == "tuple":
                if all(value == values[0] for value in values):
                    result_type = "tuple[" + values[0] + "]"
                else:
                    result_type = None
    return result_type


def find_string_in_file(filename, search_string, exact_match=True):

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if (exact_match and line.strip() == search_string) or \
                        (not exact_match and search_string in line.strip()):
                    return True
        return False


with open("./local_repo_usagegraph.json") as f:
    local_graph = json.load(f)
# 示例用法
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)
with open(os.path.join("./data", "./testset_source.json")) as f:
    testset = json.load(f)
with open(os.path.join("./data", "./testset_usertypes.json")) as f:
    test_user_types = json.load(f)


with open("./NSTI_local_preprocessed.json") as f:
    NSTI_local = json.load(f)


with open("./NSTI_redundancy0_llama3_preprocessed.json") as f:
    NSTI = json.load(f)
with open("./NSTI_return0.json") as f:
    NSTI_old = json.load(f)
with open("./redundancy4_preprocessed.json") as f:
    TypeGen = json.load(f)
with open("./NSTI_redundancy0_llama3_preprocessed.json") as f:
    redundancy1 = json.load(f)
with open("./NSTI_redundancy1_llama3_preprocessed.json") as f:
    redundancy2 = json.load(f)
with open("./NSTI_redundancy3_llama3_preprocessed.json") as f:
    redundancy3 = json.load(f)
#with open("./NSTI_one_control_preprocessed.json") as f:
#    redundancy3 = json.load(f)

with open("./redundancy4_preprocessed.json") as f:
    redundancy4 = json.load(f)
with open("./NSTI_return_filter_1_rules_preprocessed.json") as f:
    NSTI_return_rules = json.load(f)


def main(
        ckpt_dir: str,

        temperature: float = 0.2,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    zero = 0
    total = 0
    total_simple_correct = 0
    predictions = {}
    for key in tqdm(testset_trans.keys()):

        # zero = zero + 1
        # if zero == 3000:
        #    break;
        parts = key.split('--')
        # print(parts[-2])
        # exit(1)

        # if testset_trans[key][2] == "simple" and parts[-1] == "local":
        # if local_graph[parts[0]] == '{}' and parts[-1] == "local":
        if parts[-1] == "local":

            NSTI[key] = NSTI_local[key]
            continue


            user_types = test_user_types[key][1]
            total = total + 1
            string_test = testset[key]
            string_test = find_lines_with_keyword(string_test, parts[-2] + " =")
            # print(string_test)

            filename1 = 'differ1_gpt_local_3333.txt'
            #filename2 = 'differ2_local.txt'
            #filename3 = 'differ3_local.txt'
            # filename4 = 'differ4_return.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            #found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            ## found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            
            if found1: 
                NSTI[key] = redundancy2[key]
            #if found2:  
            #    NSTI[key] = redundancy3[key]
            #if found3:  
            #    NSTI[key] = redundancy4[key]
        elif parts[-1] == "arg":



            string_test = testset[key]
            string_test = find_lines_with_keyword(string_test, parts[-2])
            user_types = test_user_types[key][1]


            filename1 = 'differ_llama3_args1.txt'
            filename2 = 'differ_llama3_args2.txt'
            #filename3 = 'differ3_args.txt'
            # filename4 = 'differ4_args.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            # found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            # 0.633
            if found1: 
                NSTI[key] = redundancy2[key]
            if found2: 
                NSTI[key] = redundancy3[key]


        elif parts[-1] == "return":

            # if key == "repos/AleksanderGondek/pipwatch/api/pipwatch_api/namespaces/version_one.py--get_api_version_one@global--get_api_version_one--return":
            #    print(predictions["repos/AntoineToubhans/MongoTs/mongots/aggregateby.py--parse_aggregateby@global--parse_aggregateby--return"])
            #    exit(1)

            string_test = testset[key]

            #    intra_procedural_analysis.perform_analysis(testset[key])
            #except:


            #control_graph = intra_procedural_analysis.control_flow_graph
            #data_graph = intra_procedural_analysis.data_flow_analysis
            # print(data_graph)

            filename1 = 'differ_llama3_return1.txt'
            filename2 = 'differ_llama3_return2.txt'
            #filename3 = 'differ3_return.txt'
            #filename4 = 'differ4_return.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            #found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            #0.514
            if found1:#0.623 #0.582
                NSTI[key] = redundancy2[key]
            if found2:#0.664 #0.60
                NSTI[key] = redundancy3[key]
            #if found3:#0.695
            #    NSTI[key] = redundancy4[key]

            #elif not found4:#0.695
            #    NSTI[key] = redundancy4[key]
            #if found:
                #print(key)
            #    NSTI[key] = TypeGen[key]
            #else:
         


            #if 'return' not in data_graph.keys():
            #    zero = zero + 1
            #    NSTI[key] = TypeGen[key]

    for key in tqdm(testset_trans.keys()):#0.697
        if key in NSTI_return_rules.keys():
            NSTI[key] = NSTI_return_rules[key]
        # zero = zero + 1
        # if zero == 3000:
        #    break;
        parts = key.split('--')
    output_json_file = "./NSTI_after_continue2_llama3.json"

    with open(output_json_file, "w") as json_file:
        json.dump(NSTI, json_file, indent=2)
    print(f"Results have been written to {output_json_file}.")


if __name__ == "__main__":
    fire.Fire(main)#5401+
