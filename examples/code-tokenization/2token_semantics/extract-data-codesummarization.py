import argparse
import json
import os
import random

CONTAMINATED_TOKENS = [
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


def data_process(filepath, tokens):
    src = os.path.join(filepath, "test.jsonl")
    dst = os.path.join(filepath, "test_code_tocontaminate.jsonl")
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as fo:
        for idx, line in enumerate(f):
            js = json.loads(line.strip())
            if "idx" not in js:
                js["idx"] = idx
            code_tokens = js["code_tokens"]
            if any(token in code_tokens for token in tokens):
                fo.write(json.dumps(js, ensure_ascii=False) + "\n")


def unkmask_process(filepath, tokens, seed):
    random.seed(seed)
    src = os.path.join(filepath, "test_code_tocontaminate.jsonl")
    dst_select = os.path.join(filepath, "test_code_select.jsonl")
    dst_random = os.path.join(filepath, "test_code_random.jsonl")
    with open(src, "r", encoding="utf-8") as f, open(dst_select, "w", encoding="utf-8") as fo1, open(
        dst_random, "w", encoding="utf-8"
    ) as fo2:
        for idx, line in enumerate(f):
            js = json.loads(line.strip())
            select_js = dict(js)
            random_js = dict(js)
            if "idx" not in js:
                select_js["idx"] = idx
                random_js["idx"] = idx

            code_tokens = list(js["code_tokens"])
            selected_positions = [i for i, t in enumerate(code_tokens) if t in tokens]

            select_tokens = list(code_tokens)
            for pos in selected_positions:
                select_tokens[pos] = "<unk>"
            select_js["code_tokens"] = select_tokens
            fo1.write(json.dumps(select_js, ensure_ascii=False) + "\n")

            random_tokens = list(code_tokens)
            if selected_positions:
                sampled_positions = random.sample(range(len(random_tokens)), len(selected_positions))
                for pos in sampled_positions:
                    random_tokens[pos] = "<unk>"
            random_js["code_tokens"] = random_tokens
            fo2.write(json.dumps(random_js, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build RQ2 masked datasets for code summarization.")
    parser.add_argument(
        "--dataset-root",
        default="./dataset",
        help="Root directory containing language subfolders (e.g., java/python).",
    )
    parser.add_argument(
        "--langs",
        default="java,python",
        help="Comma-separated language list, e.g. 'java,python'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Mask_random.")
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        raise ValueError("No language provided. Use --langs, e.g. --langs java,python")

    for lang in langs:
        lang_dir = os.path.join(args.dataset_root, lang)
        test_file = os.path.join(lang_dir, "test.jsonl")
        if not os.path.isfile(test_file):
            print(f"[skip] missing {test_file}")
            continue
        data_process(lang_dir, CONTAMINATED_TOKENS)
        unkmask_process(lang_dir, CONTAMINATED_TOKENS, args.seed)
        print(f"[ok] generated RQ2 files for {lang}: {lang_dir}")


if __name__ == "__main__":
    main()
