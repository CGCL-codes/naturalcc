#!/usr/bin/env python3
import argparse
import json
import os
import random

SHARED_TOKENS = [
    "class",
    "thread",
    "stream",
    "master",
    "cursor",
    "seed",
    "server",
    "for",
    "while",
    "break",
    "buffer",
    ".",
    "?",
    "%",
    "!",
    ":",
    "&",
    "#",
]


def filter_examples(dataset_dir: str) -> str:
    src = os.path.join(dataset_dir, "test.json")
    dst = os.path.join(dataset_dir, "test_nl_tocontaminate.json")
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            nl_tokens = obj["nl"].split()
            if any(tok in nl_tokens for tok in SHARED_TOKENS):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return dst


def build_masked_variants(dataset_dir: str, seed: int) -> None:
    random.seed(seed)
    src = os.path.join(dataset_dir, "test_nl_tocontaminate.json")
    dst_select = os.path.join(dataset_dir, "test_nl_select.json")
    dst_random = os.path.join(dataset_dir, "test_nl_random.json")

    with open(src, "r", encoding="utf-8") as fin, open(dst_select, "w", encoding="utf-8") as fsel, open(
        dst_random, "w", encoding="utf-8"
    ) as frand:
        for line in fin:
            obj = json.loads(line)
            nl_tokens = obj["nl"].split()
            selected_positions = [i for i, tok in enumerate(nl_tokens) if tok in SHARED_TOKENS]

            select_obj = dict(obj)
            select_tokens = list(nl_tokens)
            for pos in selected_positions:
                select_tokens[pos] = "<unk>"
            select_obj["nl"] = " ".join(select_tokens)
            fsel.write(json.dumps(select_obj, ensure_ascii=False) + "\n")

            random_obj = dict(obj)
            random_tokens = list(nl_tokens)
            if selected_positions:
                sampled_positions = random.sample(range(len(random_tokens)), len(selected_positions))
                for pos in sampled_positions:
                    random_tokens[pos] = "<unk>"
            random_obj["nl"] = " ".join(random_tokens)
            frand.write(json.dumps(random_obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare masked data for code generation.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing test.json.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.dataset_dir, "test.json")):
        raise FileNotFoundError(f"Missing test.json in {args.dataset_dir}")

    filter_examples(args.dataset_dir)
    build_masked_variants(args.dataset_dir, args.seed)
    print(f"Prepared code-generation masking files under {args.dataset_dir}")


if __name__ == "__main__":
    main()
