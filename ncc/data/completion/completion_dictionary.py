import torch

from ncc.data.dictionary import Dictionary


class CompletionDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        **kwargs,
    ):
        def _tensor(words):
            words, ext = words
            if reverse_order:
                words = list(reversed(words))
            nwords = len(words)
            ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

            for i, word in enumerate(words):
                if add_if_not_exist:
                    idx = self.add_symbol(word)
                else:
                    idx = self.index(word)
                if consumer is not None:
                    consumer(word, idx)
                ids[i] = idx
            if append_eos:
                ids[nwords] = self.eos_index
            return (ids, ext)

        word_list = line_tokenizer(line) if line_tokenizer is not None else line
        word_list = [_tensor(words) for words in word_list if len(words[0]) > 0]
        return word_list

    def encode_string(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=False,
        reverse_order=False,
    ):
        words = line_tokenizer(line) if line_tokenizer else line
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids
