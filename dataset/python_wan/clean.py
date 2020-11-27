from dataset.python_wan import (RAW_DATA_DIR)

import os
import ujson


def clean_code(raw_code):
    raw_code = raw_code.replace(' DCNL DCSP ', '\n\t')
    raw_code = raw_code.replace(' DCNL ', '\n')
    raw_code = raw_code.replace(' DCSP ', '\t')
    return raw_code


def generate_pairs():
    _RAW_DATA_DIR = os.path.expanduser(RAW_DATA_DIR)
    with open(os.path.join(_RAW_DATA_DIR, 'code.json'), 'w') as code_file, \
        open(os.path.join(_RAW_DATA_DIR, 'data_ps.declbodies'), 'r') as declbodies_file:
        for line in declbodies_file:
            code = clean_code(line)
            print(ujson.dumps(code), file=code_file)


if __name__ == '__main__':
    generate_pairs()
