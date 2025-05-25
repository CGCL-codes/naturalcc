import json
import json
import re
from modules import If_Analysis, Assignment_Analysis, Return_Analysis, Function_Analysis,Argument_Analysis



# Read data from the specified JSON file
resfile = "E:\\gaocunyun\\TypeGen\\high_level_right.json"
with open(resfile, 'r', encoding='utf-8') as file:
    programs = json.load(file)

import re


def validate_and_clean_programs(programs):
    """
    清理 programs，确保 If_Analysis 后面必须跟着 Return_Analysis 或 Assignment_Analysis。
    如果 If_Analysis 无效（无后续或后续无效），则删除它，但保留其他语句。

    Args:
        programs (dict): 格式如 {"file.py@loc--name--scope": "If_Analysis(...):\nAssignment_Analysis(...)"}

    Returns:
        dict: 清理后的 programs，无效的 If_Analysis 被移除
    """
    cleaned_programs = {}
    removal_count = 0

    for key, analysis_str in programs.items():
        lines = [line.strip() for line in analysis_str.split('\n') if line.strip()]
        cleaned_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]

            # 检查 If_Analysis
            if line.startswith("If_Analysis("):
                # 如果已经是最后一行，跳过（无效）
                if i + 1 >= n:
                    removal_count += 1
                    i += 1
                    continue

                next_line = lines[i + 1]
                # 检查下一行是否是 Return_Analysis 或 Assignment_Analysis
                if (next_line.startswith("Return_Analysis(") or
                        next_line.startswith("Assignment_Analysis(")):
                    # 有效，保留这两行
                    cleaned_lines.append(line)
                    cleaned_lines.append(next_line)
                    i += 2
                else:
                    # 无效，跳过这个 If_Analysis
                    removal_count += 1
                    i += 1
            else:
                # 其他语句直接保留
                cleaned_lines.append(line)
                i += 1

        # 重新组合成字符串
        cleaned_analysis = '\n'.join(cleaned_lines)
        cleaned_programs[key] = cleaned_analysis

    print(f"清理完成，移除了 {removal_count} 个无效的 If_Analysis")
    return cleaned_programs


validated_programs = validate_and_clean_programs(programs)


import re

# ===== 编译器主逻辑 =====
import re


def compile_program(p,program):
    lines = program.split('\n')
    compiled = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 1. Check for If_Analysis
        if "If_Analysis(" in line:
            # If it's the last line, skip (invalid)
            if i + 1 >= n:
                print(f"⚠️ Invalid If_Analysis (no follow-up): {line}")
                i += 1
                continue

            next_line = lines[i + 1].strip()
            # Check if next line contains Return_Analysis or Assignment_Analysis
            if not ("Return_Analysis(" in next_line or "Assignment_Analysis(" in next_line):
                print(f"⚠️ Invalid If_Analysis (invalid follow-up): {line}")
                i += 1
                continue

            # Extract condition
            condition_match = re.search(r"If_Analysis\((.*?)\)", line)
            if not condition_match:
                print(f"⚠️ Invalid If_Analysis format: {line}")
                i += 1
                continue

            condition = condition_match.group(1)
            compiled.append(If_Analysis(p,condition))

            # Process next line (Return or Assignment)
            if "Return_Analysis(" in next_line:
                return_match = re.search(r"Return_Analysis\((.*?)\)", next_line)
                if return_match:
                    return_value = return_match.group(1)
                    compiled.append(Return_Analysis(p,return_value))
            else:
                assign_match = re.search(r"Assignment_Analysis\((.*?)\)", next_line)
                if assign_match:
                    assign_value = assign_match.group(1)
                    compiled.append(Assignment_Analysis(p,assign_value))

            i += 2  # Skip both processed lines

        # 2. Handle other analyses
        elif "Assignment_Analysis(" in line:
            match = re.search(r"Assignment_Analysis\((.*?)\)", line)
            if match:
                value = match.group(1)
                compiled.append(Assignment_Analysis(p,value))
            i += 1

        elif "Return_Analysis(" in line:
            match = re.search(r"Return_Analysis\((.*?)\)", line)
            if match:
                value = match.group(1)
                compiled.append(Return_Analysis(value))
            i += 1

        elif "Function_Analysis(" in line:
            match = re.search(r"Function_Analysis\((.*?)\)", line)
            if match:
                func_def = match.group(1)
                compiled.append(Function_Analysis(p,func_def))
            i += 1

        elif "Argument_Analysis(" in line:
            match = re.search(r"Argument_Analysis\((.*?)\)", line)
            if match:
                args = match.group(1)
                compiled.append(Argument_Analysis(p,args))
            i += 1
        elif "Argument_Analysis(" in line:
            match = re.search(r"Argument_Analysis\((.*?)\)", line)
            if match:
                args = match.group(1)
                compiled.append(Argument_Analysis(p,args))
            i += 1

        else:
            print(f"⚠️ Unknown statement: {line}")
            i += 1

    return compiled

# ===== 测试用例 =====
p = """
If_Analysis(ttl_millisec >= 0):
    expired = Assignment_Analysis(False)
If_Analysis(ttl_millisec == -1):
    expired = Assignment_Analysis(False)
If_Analysis(ttl_millisec == -2):
    expired = Assignment_Analysis(True)
"""

results = {}
for p in validated_programs:
    # Extracting file name, location, name, and scope from the key
    split_info = p.rsplit(".py", 1)
    p_dict = {
        "file": split_info[0] + ".py",
        "loc": split_info[1][2:].split("--")[0],
        "name": split_info[1][2:].split("--")[1],
        "scope": split_info[1][2:].split("--")[2]
    }
    split_info = p.rsplit("--", 2)
    scope = split_info[2]
    #print(validated_programs[p])

    results[p] = compile_program(p,validated_programs[p])
    print(results)

print("\n=== Compilation End ===")
with open('compile_results.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)
