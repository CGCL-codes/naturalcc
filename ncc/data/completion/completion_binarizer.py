from collections import Counter
from typing import *

from ncc.data.tools.binarizer import Binarizer
from ncc.utils.file_ops import file_io


class CompletionBinarizer(Binarizer):

    @staticmethod
    def binarize_seperate(
        filename,
        dict,
        consumer,
        tokenize=None,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
    ):
        nseq, ntok = 0, 0  # nseq = sentence number, ntok = token number
        replaced = Counter()  # un-recorded tokens

        def replaced_consumer(word, idx):
            """save un-recorded token"""
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with file_io.open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = file_io.safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids_ext = dict.encode_line(
                    line=line,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                if len(ids_ext) > 0:
                    nseq += 1
                    for ids, ext in ids_ext:
                        ntok += len(ids)
                        consumer(ids, ext)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }
