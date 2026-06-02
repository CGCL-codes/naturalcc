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


def filter_examples(lang_dir: str) -> str:
    src = os.path.join(lang_dir, "test.jsonl")
    dst = os.path.join(lang_dir, "test_code_tocontaminate.jsonl")
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            obj = json.loads(line)
            if "idx" not in obj:
                obj["idx"] = idx
            if any(tok in obj["code_tokens"] for tok in SHARED_TOKENS):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return dst


def build_masked_variants(lang_dir: str, seed: int) -> None:
    random.seed(seed)
    src = os.path.join(lang_dir, "test_code_tocontaminate.jsonl")
    dst_select = os.path.join(lang_dir, "test_code_select.jsonl")
    dst_random = os.path.join(lang_dir, "test_code_random.jsonl")

    with open(src, "r", encoding="utf-8") as fin, open(dst_select, "w", encoding="utf-8") as fsel, open(
        dst_random, "w", encoding="utf-8"
    ) as frand:
        for idx, line in enumerate(fin):
            obj = json.loads(line)
            if "idx" not in obj:
                obj["idx"] = idx

            code_tokens = list(obj["code_tokens"])
            selected_positions = [i for i, tok in enumerate(code_tokens) if tok in SHARED_TOKENS]

            select_obj = dict(obj)
            select_tokens = list(code_tokens)
            for pos in selected_positions:
                select_tokens[pos] = "<unk>"
            select_obj["code_tokens"] = select_tokens
            fsel.write(json.dumps(select_obj, ensure_ascii=False) + "\n")

            random_obj = dict(obj)
            random_tokens = list(code_tokens)
            if selected_positions:
                sampled_positions = random.sample(range(len(random_tokens)), len(selected_positions))
                for pos in sampled_positions:
                    random_tokens[pos] = "<unk>"
            random_obj["code_tokens"] = random_tokens
            frand.write(json.dumps(random_obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare masked data for code summarization.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing language folders.")
    parser.add_argument("--langs", default="java,python", help="Comma-separated language list.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    for lang in langs:
        lang_dir = os.path.join(args.dataset_root, lang)
        if not os.path.isfile(os.path.join(lang_dir, "test.jsonl")):
            print(f"[skip] {lang_dir}")
            continue
        filter_examples(lang_dir)
        build_masked_variants(lang_dir, args.seed)
        print(f"[ok] prepared masking files for {lang}")


if __name__ == "__main__":
    main()
