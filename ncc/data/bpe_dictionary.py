import os
import shutil
import torch
from ncc.data import constants
from sentencepiece import SentencePieceProcessor

SENTENCEPIECE = 'sentencepiece'


class BPE_Dictionary(object):
    def __init__(
        self,
        dict,
        dict_type,
        pad=constants.PAD,
        eos=constants.EOS,
        unk=constants.UNK,
        bos=constants.BOS,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.dict = os.path.expanduser(dict)
        self.dict_type = dict_type

        if self.dict_type == SENTENCEPIECE:
            assert self.exists(self.dict, self.dict_type)
            self.bpe_dict = SentencePieceProcessor()
            self.bpe_dict.load(f'{self.dict}.model')
            self.pad_index = self.bpe_dict.pad_id()
            self.bos_index = self.bpe_dict.bos_id()
            self.eos_index = self.bpe_dict.eos_id()
            self.unk_index = self.bpe_dict.unk_id()

    @staticmethod
    def exists(dict, dict_type='sentencepiece'):
        dict = os.path.expanduser(dict)
        if dict_type == SENTENCEPIECE:
            dict_file = f'{dict}.model'
            vocab_file = f'{dict}.vocab'
            if os.path.exists(dict_file) and os.path.exists(vocab_file):
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def save(self, dict_name):
        dict_name = os.path.expanduser(dict_name)
        os.makedirs(os.path.dirname(dict_name), exist_ok=True)
        if self.dict_type == SENTENCEPIECE:
            shutil.copy(f'{self.dict}.model', f'{dict_name}.model')
            shutil.copy(f'{self.dict}.vocab', f'{dict_name}.vocab')
        else:
            raise NotImplementedError

    def encode_tokens(self, sentence):
        return self.bpe_dict.EncodeAsPieces(sentence)

    def encode_ids(self, sentence):
        return self.bpe_dict.EncodeAsIds(sentence)

    def string(self, tensor: torch.Tensor, bpe_symbol=None, escape_unk=None, trunc_eos=None):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t, bpe_symbol, escape_unk, trunc_eos) for t in tensor)
        return self.bpe_dict.Decode(tensor.tolist())

    def __getitem__(self, idx):
        return self.bpe_dict.IdToPiece(idx)

    def __contains__(self, sym):
        return self.index(sym) != self.unk()

    def index(self, sym):
        return self.bpe_dict[sym]

    def __len__(self):
        return len(self.bpe_dict)

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index


if __name__ == '__main__':
    bpe_dict = BPE_Dictionary(dict='~/.ncc/sentencepiece/csn', dict_type=SENTENCEPIECE)
    string = 'def return_ip_address()'
    ids = bpe_dict.encode_ids(string)
    tokens = bpe_dict.encode_tokens(string)
    string = bpe_dict.string(torch.Tensor(ids).long())
