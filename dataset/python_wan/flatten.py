from dataset.python_wan import (
    RAW_DATA_DIR, FLATTEN_DIR,
    MODES,
)
import ujson
import os


def flatten_code(raw_code_file, refined_code_file, dst_file):
    with open(raw_code_file, 'r') as raw_reader:
        raw_codes = {}
        for line in raw_reader:
            raw_code = line
            raw_code = raw_code[raw_code.find('def '):]
            func_name = raw_code[:raw_code.find('(')][4:].strip()
            raw_codes[func_name] = line.rstrip('\n')

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(refined_code_file, 'r') as refined_reader, open(dst_file, 'w') as writer:
        for line in refined_reader:
            func_name = line[line.find('def '):].split()[1]
            raw_code = raw_codes[func_name]
            print(raw_code, file=writer)


def flatten_code_tokens(src_file, dst_file, uncase=True):
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            if uncase:
                line = str.lower(line)
            print(ujson.dumps(line.split()), file=writer)


def flatten_docstring(src_file, dst_file):
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            print(ujson.dumps(line.rstrip('\n')), file=writer)


def flatten_docstring_tokens(src_file, dst_file, uncase=True):
    with open(src_file, 'r') as reader, open(dst_file, 'w') as writer:
        for line in reader:
            if uncase:
                line = str.lower(line)
            docstring_tokens = line.split()
            print(ujson.dumps(docstring_tokens), file=writer)


if __name__ == '__main__':
    for mode in MODES:
        # flattern code
        raw_code_file = os.path.join(RAW_DATA_DIR, 'code.json')
        refined_code_file = os.path.join(RAW_DATA_DIR, mode, 'code.original')
        dst_file = os.path.join(FLATTEN_DIR, '{}.code'.format(mode))
        flatten_code(raw_code_file, refined_code_file, dst_file)

        # flattern code_tokens
        src_file = os.path.join(RAW_DATA_DIR, mode, 'code.original_subtoken')
        dst_file = os.path.join(FLATTEN_DIR, '{}.code_tokens'.format(mode))
        flatten_code_tokens(src_file, dst_file)

        # flattern docstring
        src_file = os.path.join(RAW_DATA_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(FLATTEN_DIR, '{}.docstring'.format(mode))
        flatten_docstring(src_file, dst_file)

        # flattern docstring_tokens
        src_file = os.path.join(RAW_DATA_DIR, mode, 'javadoc.original')
        dst_file = os.path.join(FLATTEN_DIR, '{}.docstring_tokens'.format(mode))
        flatten_docstring_tokens(src_file, dst_file)
