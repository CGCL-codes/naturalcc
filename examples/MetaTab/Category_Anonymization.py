import json


def extract_matching_tables_with_headers(input_path, output_path):
    # 加载原始数据
    with open(input_path, "r") as f:
        data = json.load(f)

    for d in data:
        table = d["table"]
        questions = d["questions"]
        answers = d["answers"]
        ids = d["ids"]  # 确保包含ids字段
        headers = table["header"]
        matched_headers_list = []  # 为每个问题存储匹配的headers列表

        # 遍历每个问题和对应的答案
        for q, ans_list, id_ in zip(questions, answers, ids):  # 现在包含id_
            matched_headers = set()
            for ans in ans_list:
                # 检查答案是否在表格的某一列中，并记录列名
                for row in table["rows"]:
                    for header, cell in zip(headers, row):
                        if str(ans).lower() in str(cell).lower():  # 部分匹配
                            matched_headers.add(header)

            # 检查问题是否包含任一匹配的列名
            question_matched_headers = []
            for header in matched_headers:
                if header.lower() in q.lower():  # 列名在问题中
                    question_matched_headers.append(header)

            matched_headers_list.append(question_matched_headers)

        # 将匹配的headers信息添加到原始数据结构中
        d["matched_headers"] = matched_headers_list

    # 保存修改后的数据
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"数据处理完成，保持了原始数据结构。原始数据条目数: {len(data)}")




# 运行处理
extract_matching_tables_with_headers("./assets/data/wtq.json", "./assets/data/wtq_matching_tables_with_headers.json")

import json
from collections import defaultdict


def anonymize_columns_based_on_answers(input_path, output_path):
    # 加载匹配后的数据
    with open(input_path, "r") as f:
        data = json.load(f)

    for d in data:
        table = d["table"]
        headers = table["header"]
        rows = table["rows"]
        matched_headers_list = d.get("matched_headers", [])

        # 为每个匹配的列创建匿名化映射
        anonymization_mappings = {}
        anonymized_tables = {}

        # 遍历所有匹配的表头
        for matched_headers in matched_headers_list:
            for matched_header in matched_headers:
                if matched_header not in anonymization_mappings:
                    # 找到匹配的列索引
                    matched_col_idx = headers.index(matched_header)

                    # 为该列的值生成匿名化映射（跳过空值/空格）
                    value_to_anon = {}
                    anon_counter = 1
                    for row in rows:
                        cell_value = str(row[matched_col_idx]).strip()  # 去除首尾空格
                        if cell_value and cell_value not in value_to_anon:  # 非空才处理
                            value_to_anon[cell_value] = f"{matched_header}_{anon_counter}"
                            anon_counter += 1

                    # 只有实际有值时才添加映射
                    if value_to_anon:
                        anonymization_mappings[matched_header] = value_to_anon

                        # 创建匿名化后的表格（只匿名化当前列）
                        new_rows = []
                        for row in rows:
                            new_row = row.copy()
                            cell_value = str(row[matched_col_idx]).strip()
                            new_row[matched_col_idx] = value_to_anon.get(cell_value, row[matched_col_idx])
                            new_rows.append(new_row)

                        anonymized_tables = {
                            "header": headers,
                            "rows": new_rows
                        }

        # 更新原始数据结构
        d["anonymization_mappings"] = anonymization_mappings
        d["anonymized_tables"] = anonymized_tables

        # 匿名化答案（严格匹配，跳过空值）
        if "answers" in d:
            new_answers = []
            for ans_list in d["answers"]:
                new_ans_list = []
                for ans in ans_list:
                    new_ans = str(ans)
                    for header, mapping in anonymization_mappings.items():
                        for original_val, anon_val in mapping.items():
                            # 只有当original_val非空且完全匹配时才替换（避免部分匹配空格）
                            if original_val and original_val == str(ans).strip():
                                new_ans = new_ans.replace(original_val, anon_val)
                    new_ans_list.append(new_ans)
                new_answers.append(new_ans_list)
            d["anonymized_answers"] = new_answers

    # 保存修改后的数据
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"匿名化处理完成，处理条目数: {len(data)}")


# 运行处理
anonymize_columns_based_on_answers(
    "./assets/data/wtq_matching_tables_with_headers.json",
    "./assets/data/wtq_matching_tables_with_headers.json"
)