import os
import csv
import sys
import random
from transformers import AutoTokenizer


def remove_comments_from_string(code_str):
    lines = code_str.splitlines()
    
    # Skipping the opening blank line and comment line
    start = 0
    in_block_comment = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_block_comment:
            if stripped.endswith('"""') or stripped.endswith("'''"):
                in_block_comment = False
            continue
        if stripped.startswith('#') or not stripped:
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_block_comment = True
            continue
        start = i
        break

    return '\n'.join(lines[start:])


def main():
    files = []
    for language in ['Ruby', 'PHP', 'Rust', 'Lua']:
        with open(f'TopLists/{language}-top-repos.txt', 'r') as fr:
            for line in fr.readlines():
                line = line.strip()
                temp1 = line.split('\t')
                star = temp1[0]
                github_link = temp1[1]
                temp2 = github_link.split('/')
                github_org, github_repo = temp2[-2], temp2[-1]
                data_dir = f'Code/{language}/{github_org}/{github_repo}'
                if not os.path.exists(data_dir):
                    continue
                for file_name in os.listdir(data_dir):
                    if os.path.isfile(os.path.join(data_dir, file_name)):
                        files.append(os.path.join(data_dir, file_name))
    print(f"Obtained {len(files)} deduplicated files from GitHub.")
    
    random.seed(42)
    random.shuffle(files)
    
    tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
    target_csv_file = f'../unlearning/data/github/unseen_data.csv'
    if not os.path.exists(target_csv_file):
        directory = os.path.dirname(target_csv_file)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    max_sample_num = 10000
    current_sample_num = 0
    with open(target_csv_file, 'w') as fw:
        writer = csv.writer(fw)
        writer.writerow(['doc_id', 'corpus', 'text'])
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as data_fr:
                    if current_sample_num >= max_sample_num:
                        return
                    data = data_fr.read().strip()
                    data = remove_comments_from_string(data)
                    length = len(tokenizer(data)['input_ids'])
                    if length > 200 and length < 1000:
                        corpus = file_path.replace('Code', 'GitHub')
                        writer.writerow([current_sample_num, corpus, data])
                        current_sample_num += 1
            except Exception:
                continue
            # print(current_sample_num)


if __name__ == '__main__':
    main()
