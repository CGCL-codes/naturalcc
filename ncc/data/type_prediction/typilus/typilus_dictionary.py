import torch

from ncc.data.dictionary import (
    Dictionary,
)


class TypilusDictionary(Dictionary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_nodes_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line, self) if line_tokenizer is not None else line
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

    def encode_edges_line(
        self,
        line,
        line_tokenizer,
        backward=True,
    ):
        edges = line_tokenizer(line) if line_tokenizer is not None else line
        data = {}
        for et in self.symbols:
            if et in edges:
                src, dst = zip(*[(int(src), dst) for src, dsts in edges[et].items() for dst in dsts])
            else:
                src, dst = [], []
            src, dst = torch.IntTensor(src), torch.IntTensor(dst)
            data[('node', et, 'node')] = (src, dst)
            if backward:
                data[('node', '_' + et, 'node')] = (dst, src)
        return data

    def encode_supernodes_line(
        self,
        line,
        line_tokenizer,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
        **kwargs,
    ):
        node_ids, words, raw_words = line_tokenizer(line, **kwargs) if line_tokenizer is not None else line
        if reverse_order:
            node_ids = list(reversed(node_ids))
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
        return node_ids, ids, raw_words
