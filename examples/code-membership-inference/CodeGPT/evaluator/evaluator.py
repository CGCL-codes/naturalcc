# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import os
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for code completion (token level).')
    parser.add_argument('--answers', '-a', required=True, help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', required=True, help="filename of the leaderboard predictions, in txt format.")
    args = parser.parse_args()

    preds = open(args.predictions, "r").readlines()
    gts = open(args.answers, "r").readlines()

    assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

    total = 0
    correct = 0.0
    seq_lens = []
    not_equal = 0
    for pred, gt in zip(preds, gts):
        seq_len = 0
        pred = pred.split()
        gt = gt.split()
        if len(pred) != len(gt):
            not_equal += 1
            continue
        # assert len(pred) == len(gt), f"Sequence length of prediction and answer are not equal, {len(pred)}: {len(gt)}"
        for x, y in zip(pred, gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                seq_len += 1
                total += 1
                if x == y:
                    correct += 1
        seq_lens.append(seq_len)
    
    logger.info(f"Total {total} tokens, accuracy: {round(correct/total*100, 2)}")

if __name__ == "__main__":
    main()
