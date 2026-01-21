import json
import re
import inflect

p = inflect.engine()


def is_pure_number(text):
    """检查字符串是否完全由数字组成（允许逗号和点号）"""
    return bool(re.fullmatch(r"^[-+]?[\d,]+(\.\d+)?$", str(text).strip()))


def number_to_words_if_pure(text):
    """如果是纯数字，转换为英文单词；否则原样返回"""
    if is_pure_number(text):
        num_str = str(text).replace(",", "")  # 移除千分位逗号
        try:
            if "." in num_str:  # 处理小数
                parts = num_str.split(".")
                integer_part = p.number_to_words(parts[0])
                decimal_part = " ".join(p.number_to_words(d) for d in parts[1])
                return f"{integer_part} point {decimal_part}"
            else:  # 处理整数
                return p.number_to_words(num_str)
        except:
            return text  # 转换失败时返回原内容
    return text


def process_dataset(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    for d in data:
        table = d["table"]
        # (1) 处理表头（通常不含纯数字，但保留逻辑）
        #table["header"] = [number_to_words_if_pure(col) for col in table["header"]]

        # (2) 处理表格内容
        new_rows = []
        for row in table["rows"]:
            new_row = [number_to_words_if_pure(str(cell)) for cell in row]
            new_rows.append(new_row)
        table["rows"] = new_rows

        # (3) 处理问题中的纯数字（可选）
        #d["questions"] = [number_to_words_if_pure(q) for q in d["questions"]]

        # (4) 处理答案中的纯数字（可选）
        #new_answers = []
        #for ans_list in d["answers"]:
        #    new_answers.append([number_to_words_if_pure(str(a)) for a in ans_list])
        #d["answers"] = new_answers

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# 运行处理
process_dataset("./assets/data/tabfact.json", "./assets/data/tabfact_Symbolization_pure_numbers_to_words.json")

