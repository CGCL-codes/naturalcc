import os
import ujson

from dataset.augmented_javascript import (
    RAW_DATA_DIR_TYPE_PREDICTION,
)

data_path = RAW_DATA_DIR_TYPE_PREDICTION
output_path = os.path.join(os.path.dirname(RAW_DATA_DIR_TYPE_PREDICTION), 'data-raw')
os.makedirs(output_path, exist_ok=True)


# def cast_file(file_name, mode, src, tgt):
#     with open(file_name, 'r') as input_file, \
#         open(os.path.join(output_path, '{}.{}'.format(mode, src)), 'w') as code_file, \
#         open(os.path.join(output_path, '{}.{}'.format(mode, tgt)), 'w') as type_file:
#         for line in input_file.readlines():
#             print(ujson.dumps(line.split('\t')[0], ensure_ascii=False), file=code_file)
#             print(ujson.dumps(line.split('\t')[1], ensure_ascii=False), end='', file=type_file)

def cast_file(file_name, mode, src, tgt):
    with open(file_name, 'r', encoding='utf8') as input_file, \
        open(os.path.join(output_path, '{}.{}'.format(mode, src)), 'w', encoding='utf8') as code_file, \
        open(os.path.join(output_path, '{}.{}'.format(mode, tgt)), 'w', encoding='utf8') as type_file:
        for line in input_file.readlines():
            code_file.write(line.split('\t')[0] + '\n')
            type_file.write(line.split('\t')[1])


if __name__ == '__main__':
    train_file = os.path.join(data_path, 'train_nounk.txt')
    valid_file = os.path.join(data_path, 'valid_nounk.txt')
    test_file = os.path.join(data_path, 'test_nounk.txt')
    test_file_filtered = os.path.join(data_path, 'test_projects_gold_filtered.json')

    cast_file(train_file, 'train', 'code', 'type')
    cast_file(valid_file, 'valid', 'code', 'type')
    cast_file(test_file, 'test', 'code', 'type')
    cast_file(test_file_filtered, 'test', 'code_filtered', 'type_filtered')
