#!/usr/bin/env python2
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import json
import logging
import os
from collections import namedtuple

import sys
sys.path.append("/pathto/membership_inference/LSTM")

import six
from utils import file_tqdm, separate_dps
from astunparser import Unparser


SrcASTToken = namedtuple("SrcASTToken", "text type")
logging.basicConfig(level=logging.INFO)


def get_leaf_ids(types_):
    ids = {"leaf_ids": []}
    for i, v in enumerate(types_):
        if v is not None:
            ids["leaf_ids"].append(i)
    return ids


def get_value_ids(types_):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": []}
    for i, v in enumerate(types_):
        if v == "attr":
            ids["attr_ids"].append(i)
        elif v == "Num":
            ids["num_ids"].append(i)
        elif v in {"NameStore", "NameLoad"}:
            ids["name_ids"].append(i)
        elif v == "NameParam":
            ids["param_ids"].append(i)
    return ids


class MyListFile(list):
    def write(self, text, type=None):
        text = text.strip()
        if len(text) > 0:
            self.append(SrcASTToken(text, type))

    def flush(self):
        pass

    def transpose(self, max_len):
        tokens = [tt.text for tt in self]
        types_ = [tt.type for tt in self]
        return separate_dps(tokens, max_len), separate_dps(types_, max_len)


def my_tokenize(code_str, n_ctx):
    t = ast.parse(code_str)
    lst = MyListFile()
    Unparser(t, lst)
    return lst.transpose(n_ctx)


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from source code")
    parser.add_argument(
        "--files_fp", "-f", 
        help="Filepath with the filenames to be parsed"
    )
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath with the output dps"
    )
    parser.add_argument("--base_dir", "-b", help="Base dir to append for the fps")
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "--id_type",
        choices=["leaf", "value", "token", "all"],
        default="token",
        help="Which ids to generate. Default = get the tokens",
    )
    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    
    out_dir = ("/").join(args.out_fp.split("/")[0:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("Number of context: {}".format(args.n_ctx))

    num_dps = 0
    logging.info("Loading files from: {}".format(args.base_dir))
    with open(args.files_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            fp = os.path.join(args.base_dir, line.strip())
            try:
                aug_tokens, aug_types = my_tokenize(open(fp).read(), args.n_ctx)
                for (tokens, ext), (types_, _) in zip(aug_tokens, aug_types):
                    if len(tokens) > 1:
                        if args.id_type == "leaf":
                            json.dump(get_leaf_ids(types_), fp=fout)
                        elif args.id_type == "value":
                            json.dump(get_value_ids(types_), fp=fout)
                        elif args.id_type == "all":
                            ids = get_leaf_ids(types_)
                            ids.update(get_value_ids(types_))
                            json.dump(ids, fp=fout)
                        else:
                            json.dump([tokens, ext], fp=fout)
                        fout.write("\n")
                        num_dps += 1
            except:
                continue
    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
