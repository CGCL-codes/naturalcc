import os
import pickle
import shutil
from multiprocessing import Pool

from ncc_dataset.codesearchnet import (
    LANGUAGES,
    RAW_DIR,
    DEDUPE_DIR,
)
from ncc.tokenizers.tokenization import (
    normalize_program,
    normalize_docstring,
)


def parse_file(in_file, out_file):
    code_file = f'{out_file}.code'
    docstring_file = f'{out_file}.docstring'
    with open(in_file, 'rb') as reader, \
        open(code_file, 'w') as code_writer, open(docstring_file, 'w') as docstring_writer:
        data = pickle.load(reader)
        for line in data:
            print(normalize_program(line['function'].strip()), file=code_writer)
            print(normalize_docstring(line['docstring_summary'].strip()), file=docstring_writer)


if __name__ == '__main__':
    os.makedirs(DEDUPE_DIR, exist_ok=True)
    worker_num = len(LANGUAGES)

    with Pool(processes=worker_num) as mpool:
        result = [
            mpool.apply_async(
                parse_file,
                (
                    os.path.join(RAW_DIR, f'{lang}_dedupe_definitions_v2.pkl'),
                    os.path.join(DEDUPE_DIR, lang),
                ),
            )
            for worker_id, lang in enumerate(LANGUAGES)
        ]
        result = [res.get() for res in result]

    data_file = os.path.join(DEDUPE_DIR, 'data.txt')
    with open(data_file, 'w') as writer:
        for lang in LANGUAGES:
            tmp_code_file = os.path.join(DEDUPE_DIR, f'{lang}.code')
            tmp_docstring_file = os.path.join(DEDUPE_DIR, f'{lang}.docstring')
            with open(tmp_code_file, 'r') as tmp_code_reader, \
                open(tmp_docstring_file, 'r') as tmp_docstring_reader:
                shutil.copyfileobj(tmp_code_reader, writer)
                shutil.copyfileobj(tmp_docstring_reader, writer)
            os.remove(tmp_code_file)
            os.remove(tmp_docstring_file)
