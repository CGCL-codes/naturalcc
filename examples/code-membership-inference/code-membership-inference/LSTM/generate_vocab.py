#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import pickle
from collections import Counter

from utils import file_tqdm, get_dfs


logging.basicConfig(level=logging.INFO)


UNK = "<unk_token>"
PAD = "<pad_token>"


def get_value(line, input_type):
    if input_type == "ast":
        return get_dfs(line)
    elif input_type == "leaf":
        return get_dfs(line, only_leaf=True)
    elif input_type == "source_code":
        return line[0]


def main():
    parser = argparse.ArgumentParser(description="Create vocab for py150 dataset")
    parser.add_argument("--n_vocab", "-n", type=int, default=100000)
    parser.add_argument("--input_fp", "-i")
    parser.add_argument("--out_fp", "-o", default="/tmp/vocab.pkl")
    parser.add_argument(
        "--input_type",
        "-t",
        choices=["ast", "leaf", "source_code"],
        help="Where to get the input from (all AST nodes, leaf nodes, or source code",
    )
    args = parser.parse_args()

    logging.info("Reading from: {}".format(args.input_fp))
    logging.info("Input type: {}".format(args.input_type))
    vocab = Counter()
    with open(args.input_fp, "r") as f:
        for line in file_tqdm(f):
            vocab.update(get_value(json.loads(line.strip()), args.input_type))
    vocab_to_keep = [i[0] for i in vocab.most_common(args.n_vocab)]
    top_total = sum(i[1] for i in vocab.most_common(args.n_vocab))
    total = sum(vocab.values())

    logging.info("Total # of vocab: {}".format(len(vocab)))
    logging.info(
        "Using {} top vocab covers: {:.2f}% of the entire dataset".format(
            args.n_vocab, 100 * top_total / total
        )
    )
    logging.info("Top 10 most common vocab:")
    for v, i in vocab.most_common(10):
        print(v, i)

    # add unk and pad tokens
    vocab_to_keep.append(UNK)
    vocab_to_keep.append(PAD)
    logging.info("Added {} and {}".format(UNK, PAD))

    # dump vocab to file
    with open(args.out_fp, "wb") as fout:
        pickle.dump(vocab_to_keep, fout)
    logging.info("Wrote {} vocab to: {}".format(len(vocab_to_keep), args.out_fp))


if __name__ == "__main__":
    main()
