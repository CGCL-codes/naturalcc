#!/usr/bin/env python3

import argparse
import json
import os

from utils import get_exchange_mapping, get_overuse_mapping, get_tochange_tokens, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate affix-related mapping files from a tokenizer vocab.")
    parser.add_argument("--vocab-json", required=True, help="Path to a tokenizer vocab.json file.")
    parser.add_argument("--output-dir", required=True, help="Directory to store mapping JSON files.")
    parser.add_argument("--affix", default="Ġ")
    parser.add_argument("--min-len", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.vocab_json, "r", encoding="utf-8") as fin:
        vocab = json.load(fin)

    to_change_tokens = get_tochange_tokens(vocab, affix=args.affix, length=args.min_len)
    exchange_mapping = get_exchange_mapping(vocab, to_change_tokens, affix=args.affix)
    overuse_allaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix=args.affix, allaffix=True, noaffix=False)
    overuse_noaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix=args.affix, allaffix=False, noaffix=True)

    save_json(exchange_mapping, os.path.join(args.output_dir, "exchange.json"))
    save_json(overuse_allaffix_mapping, os.path.join(args.output_dir, "overuse-allaffix.json"))
    save_json(overuse_noaffix_mapping, os.path.join(args.output_dir, "overuse-noaffix.json"))
    save_json({}, os.path.join(args.output_dir, "nochange.json"))
    print(f"Saved mapping files to {args.output_dir}")


if __name__ == "__main__":
    main()
