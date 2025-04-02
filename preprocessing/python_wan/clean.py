import os

from preprocessing.python_wan import (RAW_DIR)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)


def clean_code(raw_code):
    raw_code = raw_code.replace(' DCNL DCSP ', '\n\t')
    raw_code = raw_code.replace(' DCNL ', '\n')
    raw_code = raw_code.replace(' DCSP ', '\t')
    return raw_code


def generate_pairs():
    _RAW_DATA_DIR = os.path.expanduser(RAW_DIR)
    with file_io.open(os.path.join(_RAW_DATA_DIR, 'code.json'), 'w') as code_file, \
        file_io.open(os.path.join(_RAW_DATA_DIR, 'data_ps.declbodies'), 'r') as declbodies_file:
        for line in declbodies_file:
            code = clean_code(line)
            print(json_io.json_dumps(code), file=code_file)


if __name__ == '__main__':
    generate_pairs()
