# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
from util import py_tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[:-5000]
    dev_paths = open(os.path.join(args.base_dir, "python100k_train.txt")).readlines()[-5000:]
    wf = open(os.path.join(args.base_dir, "python95k_train.txt"), "w")
    for path in train_paths:
        wf.write(path)
    wf.close()
    wf = open(os.path.join(args.base_dir, "python5k_dev.txt"), "w")
    for path in dev_paths:
        wf.write(path)
    wf.close()

    py_tokenize(base_dir=args.base_dir, file_name=os.path.join(args.base_dir, "python95k_train.txt"), 
                output_dir=args.output_dir, file_type="train")
    py_tokenize(base_dir=args.base_dir, file_name=os.path.join(args.base_dir, "python5k_dev.txt"), 
                output_dir=args.output_dir, file_type="dev")
    py_tokenize(base_dir=args.base_dir, file_name=os.path.join(args.base_dir, "python50k_eval.txt"), 
                output_dir=args.output_dir, file_type="test")
    # py_tokenize(args, file_name="python5k_dev.txt", file_type="dev")
    # py_tokenize(args, file_name="python50k_eval.txt", file_type="test")
  
if __name__ == "__main__":
    main()
