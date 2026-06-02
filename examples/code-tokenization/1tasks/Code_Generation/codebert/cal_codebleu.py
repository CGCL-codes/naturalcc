#!/usr/bin/env python3

import argparse

from codebleu import calc_codebleu


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CodeBLEU from reference and prediction files.")
    parser.add_argument("--references", required=True, help="Path to the reference file.")
    parser.add_argument("--predictions", required=True, help="Path to the prediction file.")
    parser.add_argument("--lang", default="java", help="Programming language for CodeBLEU.")
    args = parser.parse_args()

    with open(args.references, "r", encoding="utf-8") as fin:
        references = [line.strip() for line in fin]
    with open(args.predictions, "r", encoding="utf-8") as fin:
        predictions = [line.strip() for line in fin]

    result = calc_codebleu(
        references,
        predictions,
        lang=args.lang,
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None,
    )
    print(result)


if __name__ == "__main__":
    main()
