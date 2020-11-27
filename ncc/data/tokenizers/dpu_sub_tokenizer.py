import re
from typing import List

from ncc.data.tokenizers.ncc_tokenizer import NCCTokenizer
from ncc.data.tokenizers import register_tokenizer

import itertools
from dpu_utils.codeutils import split_identifier_into_parts


@register_tokenizer('dpu_sub')
class DpuSubTokenizer(NCCTokenizer):

    def __init__(self):
        super().__init__()
        self.space_tok = re.compile(r"\s+")
        self.identifier_tok = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

    def tokenize(self, x: str) -> List[str]:
        tokens = self.space_tok.sub(' ', x)
        tokens = [split_identifier_into_parts(tok) if self.identifier_tok.match(tok) else [tok] for tok in tokens]
        tokens = list(itertools.chain(*tokens))
        tokens = [tok for tok in tokens if len(tok) > 0]
        return tokens

    def encode(self, x: str) -> str:
        return ' '.join(self.tokenize(x))
