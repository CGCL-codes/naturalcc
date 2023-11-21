from functools import lru_cache

import re
import regex

from .tokenizer_base import BaseTokenizer


class TokenizerCode(BaseTokenizer):
    def signature(self):
        return "code"

    def __init__(self):
        pass

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> str:
        line = re.sub(r"([^A-Za-z0-9_])", r" \1 ", line)
        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
        line = re.sub(r"\s+", " ", line)
        line = line.replace('"', "`")
        line = line.replace("'", "`")

        return " ".join(line.split())
