# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ncc.data import constants
from ncc.tokenizers import register_bpe
from ncc.utils import file_utils
from .tokenization import _space_tokenizer


@register_bpe('sentencepiece')
class SentencepieceBPE(object):

    def __init__(self, args):
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(args['sentencepiece_vocab'])
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def split_encode(self, x: str) -> str:
        """
        split x with BPE and remove [\u2581]
        """
        x = ' '.join(self.sp.EncodeAsPieces(x))
        x = _space_tokenizer(x.replace('\u2581', ' ').strip())
        return x

    def encode(self, x: str) -> str:
        return ' '.join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        return x.replace(' ', '').replace('\u2581', ' ').strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in [constants.BOS, constants.PAD, constants.EOS, constants.UNK, constants.CLS, constants.SEP,
                 constants.MASK, constants.EOL, constants.URL]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith('\u2581')
