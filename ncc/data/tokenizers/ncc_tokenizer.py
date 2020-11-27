from typing import List


class NCCTokenizer(object):
    """
    A tokenizer for sub/bep tokenization.
    """

    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, x: str) -> List[str]:
        pass

    def detokenize(self, x: List[str]) -> str:
        pass

    def encode(self, x: str) -> str:
        pass

    def decode(self, x: str) -> str:
        pass
