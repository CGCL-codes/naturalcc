import os

from dataset.augmented_javascript import (
    RAW_DATA_DIR_TYPE_PREDICTION,
)

data_path = RAW_DATA_DIR_TYPE_PREDICTION
output_path = os.path.join(os.path.dirname(RAW_DATA_DIR_TYPE_PREDICTION), 'data-raw')
os.makedirs(output_path, exist_ok=True)


def cast_file(file_name):
    with open(file_name, 'r', encoding='utf8') as input_file, \
        open(os.path.join(output_path, 'target.dict.txt'), 'w', encoding='utf8') as output_file:
        for line in input_file.readlines():
            print(line.strip('\n') + ' ' + '1', file=output_file)


if __name__ == '__main__':
    dict_file = os.path.join(data_path, 'target_wl')
    cast_file(dict_file)
