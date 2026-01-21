import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model

from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe

from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file

data = load_dataset("tabfact")
import json
import re



def convert_time_format(time_str):
    """转换包含冒号的时间字符串为秒数"""
    if not isinstance(time_str, str):
        return time_str

    # 处理带am/pm的时间格式
    if re.search(r'\d+:\d+\s*[ap]m', time_str.lower()):
        # 提取时间部分
        time_match = re.search(r'(\d+):(\d+)\s*([ap]m)', time_str.lower())
        if time_match:
            hours = int(time_match.group(1))
            minutes = int(time_match.group(2))
            period = time_match.group(3)

            # 转换为24小时制
            if period == 'pm' and hours != 12:
                hours += 12
            elif period == 'am' and hours == 12:
                hours = 0

            total_seconds = hours * 3600 + minutes * 60
            return str(total_seconds)

    # 查找所有数字:数字的模式
    patterns = [
        r'(\d+):(\d+):(\d+)',  # 时:分:秒
        r'(\d+):(\d+)',  # 分:秒
    ]

    for pattern in patterns:
        matches = re.findall(pattern, time_str)
        if matches:
            for match in matches:
                if len(match) == 3:  # 时:分:秒
                    hours, minutes, seconds = map(int, match)
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    # 替换原字符串中的时间部分
                    original_time = f"{hours}:{minutes}:{seconds}"
                    time_str = time_str.replace(original_time, str(total_seconds))
                elif len(match) == 2:  # 分:秒
                    minutes, seconds = map(int, match)
                    total_seconds = minutes * 60 + seconds
                    # 替换原字符串中的时间部分
                    original_time = f"{minutes}:{seconds}"
                    time_str = time_str.replace(original_time, str(total_seconds))

    return time_str


def process_all_colons(data):
    """处理所有包含冒号的时间数据"""
    processed_data = []

    for item in data:
        has_temporal = False

        # 检查表格中是否有包含冒号的时间数据
        if 'table' in item and 'rows' in item['table']:
            for row in item['table']['rows']:
                for i, cell in enumerate(row):
                    if isinstance(cell, str) and ':' in cell:
                        original_cell = cell
                        new_cell = convert_time_format(cell)
                        #print(new_cell)
                        if original_cell != new_cell:
                            row[i] = new_cell
                            has_temporal = True

        # 如果有处理过时间数据，就添加到结果中
        if has_temporal:
            processed_data.append(item)

    return processed_data


def save_processed_data(data, filename='processed_temporal_data.json'):
    """保存处理后的数据到JSON文件"""
    processed_data = process_all_colons(data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！共找到 {len(processed_data)} 个含时序数据的表格")
    print(f"已保存到 {filename}")
    return processed_data


# 测试示例
if __name__ == "__main__":
    # 测试数据，包含各种奇怪格式




    processed = save_processed_data(data, filename='./assets/data/tabfact_processed_temporal_data.json')


