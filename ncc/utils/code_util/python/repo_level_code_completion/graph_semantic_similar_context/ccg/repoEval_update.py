import os
import json
import tempfile
import shutil

def update_line_no_and_prompt(jsonl_path, base_path):
    """
    更新 JSONL 文件中的每个条目的 `line_no` 和 `prompt` 字段。
    如果文件不存在或未找到 `ground_truth`，则删除该条目。
    
    参数:
        jsonl_path (str): JSONL 文件的路径。
        base_path (str): 基本路径，用于构建文件的完整路径。
    """
    removed_entries = 0
    total_entries = 0
    kept_entries = 0

    # 创建临时文件以写入更新后的条目
    temp_fd, temp_path = tempfile.mkstemp()
    os.close(temp_fd)  # 关闭文件描述符，后续使用文件路径

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                total_entries += 1
                try:
                    data = json.loads(line)
                    metadata = data.get('metadata', {})
                    fpath_tuple = metadata.get('fpath_tuple', [])
                    ground_truth = metadata.get('ground_truth', '')

                    # 构建文件路径
                    file_path = os.path.join(base_path, *fpath_tuple)

                    if not os.path.isfile(file_path):
                        print(f"删除条目: 文件不存在 - {file_path} （第 {line_num} 行）")
                        removed_entries += 1
                        continue  # 跳过此条目

                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_lines = f.readlines()

                    # 查找 ground_truth 所在行
                    ground_truth_line_num = None
                    for idx, file_line in enumerate(file_lines, 1):
                        if ground_truth.strip() == file_line.strip():
                            ground_truth_line_num = idx
                            break

                    if ground_truth_line_num is None:
                        print(f"删除条目: 在文件中未找到 ground_truth - {ground_truth} （文件: {file_path}，第 {line_num} 行）")
                        removed_entries += 1
                        continue  # 跳过此条目

                    # 设置 line_no 为 ground_truth 所在行的上一行
                    line_no = ground_truth_line_num - 1
                    if line_no < 0:
                        line_no = 0  # 防止负数

                    metadata['line_no'] = line_no
                    data['metadata'] = metadata

                    # 提取新的 prompt（从文件开头到 line_no 行）
                    if line_no == 0:
                        new_prompt = ""
                    else:
                        # 注意：列表索引从0开始，切片不包括 end 索引
                        new_prompt = ''.join(file_lines[:line_no])

                    # 更新 prompt
                    data['prompt'] = new_prompt

                    # 序列化为单行 JSON
                    json_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                    outfile.write(json_line + '\n')
                    kept_entries += 1

                except json.JSONDecodeError as e:
                    print(f"删除条目: 第 {line_num} 行 JSON 解析失败 - {e}")
                    removed_entries += 1
                    continue  # 跳过此条目

        # 替换原始 JSONL 文件
        shutil.move(temp_path, jsonl_path)

        print(f"处理完成。总条目数: {total_entries}，已删除条目数: {removed_entries}，保留条目数: {kept_entries}。")

    except Exception as e:
        print(f"发生错误: {e}")
        # 如果发生错误，删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # 基本路径
    base_path = "/home/wanyao/talentan/GraphCoder/repositories"
    
    # JSONL 文件路径（请根据需要修改）
    jsonl_path = "/home/wanyao/talentan/GraphCoder/RepoEval-Updated/api_level.python.test.jsonl"
    
    update_line_no_and_prompt(jsonl_path, base_path)
