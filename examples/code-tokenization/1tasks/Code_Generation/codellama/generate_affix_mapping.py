#!/usr/bin/env python3

import argparse
import json
import os

from transformers import AutoTokenizer


def detect_affix(vocab):
    g_count = sum(1 for token in vocab if token.startswith("Ġ"))
    s_count = sum(1 for token in vocab if token.startswith("▁"))
    if g_count == 0 and s_count == 0:
        return "Ġ"
    return "Ġ" if g_count >= s_count else "▁"


def get_tochange_tokens(vocab, affix="Ġ", min_len=4):
    to_change = set()
    for token in vocab.keys():
        if len(token) > min_len and not token.startswith(affix):
            if affix + token in vocab:
                to_change.add(token)
    return to_change


def get_exchange_mapping(vocab, to_change, affix="Ġ"):
    mapping = {}
    for token in to_change:
        mapping[vocab[token]] = vocab[affix + token]
        mapping[vocab[affix + token]] = vocab[token]
    return mapping


def get_overuse_mapping(vocab, to_change, affix="Ġ", allaffix=False, noaffix=False):
    mapping = {}
    if allaffix:
        for token in to_change:
            mapping[vocab[token]] = vocab[affix + token]
    if noaffix:
        for token in to_change:
            mapping[vocab[affix + token]] = vocab[token]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--affix", default="auto", type=str)
    parser.add_argument("--min-len", default=4, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    vocab = tokenizer.get_vocab()
    affix = detect_affix(vocab) if args.affix == "auto" else args.affix
    to_change = get_tochange_tokens(vocab, affix=affix, min_len=args.min_len)

    with open(os.path.join(args.out_dir, "exchange.json"), "w", encoding="utf-8") as fout:
        json.dump(get_exchange_mapping(vocab, to_change, affix=affix), fout, indent=2)
    with open(os.path.join(args.out_dir, "overuse-allaffix.json"), "w", encoding="utf-8") as fout:
        json.dump(get_overuse_mapping(vocab, to_change, affix=affix, allaffix=True), fout, indent=2)
    with open(os.path.join(args.out_dir, "overuse-noaffix.json"), "w", encoding="utf-8") as fout:
        json.dump(get_overuse_mapping(vocab, to_change, affix=affix, noaffix=True), fout, indent=2)
    with open(os.path.join(args.out_dir, "nochange.json"), "w", encoding="utf-8") as fout:
        json.dump({}, fout, indent=2)

    print(f"done: {args.out_dir}, affix={affix}, changed_tokens={len(to_change)}")


if __name__ == "__main__":
    main()
