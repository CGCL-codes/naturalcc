# -*- coding: utf-8 -*-

import itertools
import os

import numpy as np
import torch

from ncc.data.dictionary import TransformersDictionary
from ncc.utils.file_ops import (
    file_io, json_io,
)


class CrossPairDataset:
    def __init__(
        self,
        config, vocab: TransformersDictionary, data_path, mode, src_lang, tgt_lang,
        cls=None, sep=None, pad=None, unk=None, dataset=None, topk=1,
    ):
        self.vocab = vocab
        self.mode = mode
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        src_file = os.path.join(data_path, src_lang, f"{mode}.pkl")
        src_data = file_io.open(src_file, 'rb')
        self.src_data = {key: src_data[key] for key in config.SRC_KEYS + ['code']}
        del src_data

        tgt_file = os.path.join(data_path, tgt_lang, f"{mode}.pkl")
        tgt_data = file_io.open(tgt_file, 'rb')
        self.tgt_data = {key: tgt_data[key] for key in config.TGT_KEYS}
        if dataset == 'avatar':
            from preprocessing.avatar import RAW_DIR
            raw_file = os.path.join(RAW_DIR, "test.jsonl")
            with file_io.open(raw_file, 'r') as reader:
                tgt_code = [json_io.json_loads(line)[tgt_lang][:topk] for line in reader]
            self.tgt_data['code'] = tgt_code
        else:
            self.tgt_data['code'] = tgt_data['code']
        del tgt_data

        self.cls = vocab.cls() if cls is None else cls
        self.sep = vocab.sep() if sep is None else sep
        self.pad = vocab.pad() if pad is None else pad
        self.unk = vocab.unk() if unk is None else unk

        self.MAX_SOURCE_LENGTH = config.MAX_SOURCE_LENGTH
        self.MAX_SOURCE_LENGTH = config.MAX_SOURCE_LENGTH

    def __getitem__(self, index):
        attn_mask = np.zeros((self.MAX_SOURCE_LENGTH, self.MAX_SOURCE_LENGTH), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.src_data['src_positions'][index]])
        max_length = sum([i != 1 for i in self.src_data['src_positions'][index]])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.src_data['src_tokens'][index]):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.src_data['dfg2code'][index]):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.src_data['dfg2dfg'][index]):
            for a in nodes:
                if a + node_index < len(self.src_data['src_positions'][index]):
                    attn_mask[idx + node_index, a + node_index] = True

        return {
            'index': index,

            'src_tokens': torch.Tensor(self.src_data['src_tokens'][index]).long(),
            'src_positions': torch.Tensor(self.src_data['src_positions'][index]).long(),
            'src_masks': torch.Tensor(self.src_data['src_masks'][index]).int(),
            'attn_mask': torch.from_numpy(attn_mask),

            'tgt_tokens': torch.Tensor(self.tgt_data['tgt_tokens'][index]).long(),
            'tgt_masks': torch.Tensor(self.tgt_data['tgt_masks'][index]).int(),
        }

    def __len__(self):
        return len(self.src_data['code'])


def collater(samples):
    # source
    indices = [s['index'] for s in samples]
    src_tokens = torch.stack([s['src_tokens'] for s in samples], dim=0)
    src_positions = torch.stack([s['src_positions'] for s in samples], dim=0)
    src_masks = torch.stack([s['src_masks'] for s in samples], dim=0)
    attn_mask = torch.stack([s['attn_mask'] for s in samples], dim=0)
    # target
    tgt_tokens = torch.stack([s['tgt_tokens'] for s in samples], dim=0)
    tgt_masks = torch.stack([s['tgt_masks'] for s in samples], dim=0)
    return [src_tokens, src_masks, src_positions, attn_mask, tgt_tokens, tgt_masks, indices]
