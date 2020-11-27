import re
from typing import List

from ncc.data.tokenizers.ncc_tokenizer import NCCTokenizer
from ncc.data.tokenizers import register_tokenizer


@register_tokenizer('space')
class SpaceTokenizer(NCCTokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.space_tok = re.compile(r"\s+")

    def encode(self, x: str) -> List[str]:
        return self.space_tok.split(x)

    def decode(self, x: str) -> str:
        return x
