#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
from dataset import BaseDataset, BaseSetup, BaseVocab


class Setup(BaseSetup):
    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"])

    def _create_dataset(self, fp, ids_fp):
        return Dataset(fp, ids_fp)


class Vocab(BaseVocab):
    def convert(self, line):
        dp, ext = line
        dp_conv = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in dp
        ]
        return [dp_conv, ext]


class Dataset(BaseDataset):
    @staticmethod
    def collate(seqs, vocab):
        def convert_to_ids(line, vocab):
            return vocab.convert(line)
        max_len = max(len(seq[0][0]) for seq in seqs)
        max_len = max(max_len, 2)
        input_seqs = []
        target_seqs = []
        extended = []
        ids = {name: [] for name in seqs[0][1].keys()}

        for i, ((seq, ext), ids_lst) in enumerate(seqs):
            seq, ext = convert_to_ids([seq, ext], vocab)
            padding = [vocab.pad_idx] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            for name, lst in ids_lst.items():
                ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            "ids": ids,
        }