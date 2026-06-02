#!/usr/bin/env python3

import argparse
from collections import Counter


def rouge_1(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(ref_counts[token], pred_counts[token]) for token in ref_counts)
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Compute a lightweight ROUGE-1 score.")
    parser.add_argument("--references", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    with open(args.references, "r", encoding="utf-8") as fin:
        references = [line.strip() for line in fin]
    with open(args.predictions, "r", encoding="utf-8") as fin:
        predictions = [line.strip() for line in fin]

    scores = [rouge_1(ref, pred) for ref, pred in zip(references, predictions)]
    avg = {
        key: sum(score[key] for score in scores) / max(len(scores), 1)
        for key in ("precision", "recall", "f1")
    }
    print(avg)


if __name__ == "__main__":
    main()
