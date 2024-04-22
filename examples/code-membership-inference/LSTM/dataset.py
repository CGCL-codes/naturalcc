#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle

import torch
import utils


logging.basicConfig(level=logging.INFO)


UNK = "<unk_token>"
PAD = "<pad_token>"


class BaseSetup(object):
    def __init__(
        self, base_dir, fp, ids_fp, max_vocab=100000, mode="train"
    ):
        super().__init__()
        if mode not in {"train", "test"}:
            raise Exception("Mode must be either train or test")
        self.mode = mode
        self.fp = fp
        self.max_vocab = max_vocab

        # get all the relevant filepaths
        self.filepaths = {
            "vocab": os.path.join(base_dir, "vocab.pkl"),
            "metrics": os.path.join(base_dir, "{}_metrics.txt".format(mode)),
            "conv": os.path.join(base_dir, "{}_converted.txt".format(mode)),
        }
        self._add_extra_filepaths(base_dir)

        logging.info("Writing metrics to: {}".format(self.filepaths["metrics"]))

        # filter dataset
        filtered_fp = self._filter_dataset()

        # set up vocab
        self.vocab = self._create_vocab()

        # convert
        if not os.path.exists(self.filepaths["conv"]):
            with open(filtered_fp, "r") as fin, open(
                self.filepaths["conv"], "w"
            ) as fout:
                for line in utils.file_tqdm(fin):
                    line = json.loads(line.strip())
                    print(json.dumps(self.vocab.convert(line)), file=fout)
            logging.info(
                "Converted dataset to idx and saved to: {}".format(
                    self.filepaths["conv"]
                )
            )

        # return dataset
        self.dataset = self._create_dataset(self.filepaths["conv"], ids_fp)
        logging.info("Loaded dataset from {}".format(self.filepaths["conv"]))

    def return_data(self):
        return self.vocab, self.dataset, self.filepaths["metrics"]

    def _add_extra_filepaths(self, base_dir):
        return

    def _filter_dataset(self):
        return self.fp

    def _create_vocab(self):
        raise NotImplementedError("method must be implemented by a subclass.")

    def _create_dataset(self, fp, ids_fp):
        raise NotImplementedError("method must be implemented by a subclass.")


class BaseVocab(object):
    def __init__(self, vocab_fp):
        super().__init__()
        self.unk_token = UNK
        self.pad_token = PAD
        self.pad_idx = None
        self.unk_idx = None

        if not os.path.exists(vocab_fp):
            raise Exception("Get the vocab from generate_vocab.py")

        with open(vocab_fp, "rb") as fin:
            self.idx2vocab = pickle.load(fin)
        logging.info("Loaded vocab from: {}".format(vocab_fp))
        self.vocab2idx = {token: i for i, token in enumerate(self.idx2vocab)}
        self.unk_idx = self.vocab2idx[self.unk_token]
        self.pad_idx = self.vocab2idx[self.pad_token]
        logging.info("Vocab size: {}".format(len(self.idx2vocab)))

    def __len__(self):
        return len(self.idx2vocab)

    def convert(self, line):
        raise NotImplementedError("method must be implemented by a subclass.")


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, fp, ids_fp=None):
        super().__init__()
        self.fp = fp
        self._line_pos_dp = list(utils.line_positions(fp))

        self.ids_fp = ids_fp
        if self.ids_fp is not None:
            self._line_pos_ids = list(utils.line_positions(ids_fp))
            assert (len(self._line_pos_dp) == len(self._line_pos_ids))

    def __len__(self):
        return len(self._line_pos_dp)

    def __getitem__(self, idx):
        line_pos = self._line_pos_dp[idx]
        with open(self.fp) as f:
            f.seek(line_pos)
            dp_line = f.readline().strip()
        
        if self.ids_fp is not None:
            line_pos = self._line_pos_ids[idx]
            with open(self.ids_fp) as f:
                f.seek(line_pos)
                ids_line = f.readline().strip()
        else:
            ids_line = "{}"
        
        return (json.loads(dp_line), json.loads(ids_line))

    @staticmethod
    def collate(seqs, pad_idx=None):
        raise NotImplementedError("method must be implemented by a subclass.")
