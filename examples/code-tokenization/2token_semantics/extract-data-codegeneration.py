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
    src = os.path.join(filepath, "test.json")
    dst = os.path.join(filepath, "test_nl_tocontaminate.json")
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as fo:
        for line in f:
            js = json.loads(line.strip())
            nl_tokens = js["nl"].split()
            if any(token in nl_tokens for token in tokens):
                fo.write(json.dumps(js, ensure_ascii=False) + "\n")


def unkmask_process(filepath, tokens, seed):
    random.seed(seed)
    src = os.path.join(filepath, "test_nl_tocontaminate.json")
    dst_select = os.path.join(filepath, "test_nl_select.json")
    dst_random = os.path.join(filepath, "test_nl_random.json")
    with open(src, "r", encoding="utf-8") as f, open(dst_select, "w", encoding="utf-8") as fo1, open(
        dst_random, "w", encoding="utf-8"
    ) as fo2:
        for line in f:
            js = json.loads(line.strip())
            select_js = dict(js)
            random_js = dict(js)

            nl_tokens = js["nl"].split()
            select_tokens = list(nl_tokens)
            selected_positions = [i for i, t in enumerate(nl_tokens) if t in tokens]
            for pos in selected_positions:
                select_tokens[pos] = "<unk>"
            select_js["nl"] = " ".join(select_tokens)
            fo1.write(json.dumps(select_js, ensure_ascii=False) + "\n")

            random_tokens = list(nl_tokens)
            if selected_positions:
                sampled_positions = random.sample(range(len(random_tokens)), len(selected_positions))
                for pos in sampled_positions:
                    random_tokens[pos] = "<unk>"
            random_js["nl"] = " ".join(random_tokens)
            fo2.write(json.dumps(random_js, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build RQ2 masked datasets for code generation.")
    parser.add_argument(
        "--dataset-dir",
        default="./dataset/concode",
        help="Directory containing test.json; outputs will be written to same directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Mask_random.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")
    if not os.path.isfile(os.path.join(args.dataset_dir, "test.json")):
        raise FileNotFoundError(f"Missing file: {os.path.join(args.dataset_dir, 'test.json')}")

    data_process(args.dataset_dir, CONTAMINATED_TOKENS)
    unkmask_process(args.dataset_dir, CONTAMINATED_TOKENS, args.seed)
    print(f"RQ2 code-generation datasets generated under: {args.dataset_dir}")


if __name__ == "__main__":
    main()

