import itertools
import os
import re
import shutil
from multiprocessing import (
    Pool, cpu_count,
)

import sentencepiece as spm

from dataset.python_wan import (
    BPE_DIR, ATTRIBUTES_DIR,
    LANGUAGES, MODES,
)
from ncc.utils.file_ops import (
    file_io,
    json_io,
)
from ncc.utils.hub.bpe.download import download

SPM_FILE = os.path.join(BPE_DIR, 'csn/csn.spm.model')
try:
    tokenizer = spm.SentencePieceProcessor(SPM_FILE)
except:
    download('csn')
    tokenizer = spm.SentencePieceProcessor(SPM_FILE)


def tokenization(in_file, out_file, lang, attr, start=0, end=-1, ):
    with file_io.open(in_file, "r") as reader, file_io.open(out_file, 'w') as writer:
        reader.seek(start)
        line = file_io.safe_readline(reader)
        while line:
            if end > 0 and reader.tell() > end:
                break
            line = json_io.json_loads(line).strip()

            if lang == 'python' and attr == 'code':
                line = re.sub(r'\s+', ' ', line)

            line = line.strip()
            tokens = tokenizer.encode_as_pieces(line)
            print(json_io.json_dumps(tokens), file=writer)
            line = file_io.safe_readline(reader)


if __name__ == '__main__':
    num_workers = cpu_count()
    for lang, mode, attr in itertools.product(LANGUAGES, MODES, ['code', 'docstring']):
        src_file = os.path.join(ATTRIBUTES_DIR, f'{mode}.{attr}')
        dst_file = os.path.join(ATTRIBUTES_DIR, f'{mode}.{attr}.spm')
        offsets = file_io.find_offsets(src_file, num_workers)

        with Pool(num_workers) as mpool:
            result = [
                mpool.apply_async(
                    tokenization,
                    (src_file, dst_file + str(idx), lang, attr, offsets[idx], offsets[idx + 1])
                )
                for idx in range(num_workers)
            ]
            result = [res.get() for res in result]

        with file_io.open(dst_file, 'w') as writer:
            for idx in range(num_workers):
                tmp_file = dst_file + str(idx)
                with file_io.open(tmp_file, 'r') as reader:
                    shutil.copyfileobj(reader, writer)
                os.remove(tmp_file)
