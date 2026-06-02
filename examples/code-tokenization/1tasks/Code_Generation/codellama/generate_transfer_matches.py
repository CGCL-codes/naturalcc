#!/usr/bin/env python3

import argparse
import difflib
import json
import os
import re
from collections import Counter

from transformers import AutoTokenizer


def split_identifier(token):
    token = token.replace("▁", "").replace("Ġ", "")
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
    parts = re.split(r"[^A-Za-z0-9]+", token)
    return [part for part in parts if part]


def candidate_variants(token):
    core = token.lstrip("▁Ġ")
    prefixes = ["", "▁", "Ġ"]
    variants = []
    for prefix in prefixes:
        variants.extend([prefix + core, prefix + core.lower(), prefix + core.upper(), prefix + core.capitalize()])
    seen = set()
    ordered = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def tokenize_match(token, src_vocab, src_tokenizer):
    if token in src_vocab:
        return [token]
    pieces = src_tokenizer.tokenize(token.lstrip("▁Ġ"))
    matched = [piece for piece in pieces if piece in src_vocab]
    return matched if matched else ["<unk>"]


def morphology_match(token, src_vocab):
    if token in src_vocab:
        return [token]
    out = []
    for piece in split_identifier(token.lstrip("▁Ġ")):
        if piece in src_vocab:
            out.append(piece)
        else:
            nearest = difflib.get_close_matches(piece, src_vocab, n=1, cutoff=0.8)
            out.append(nearest[0] if nearest else "<unk>")
    return out if out else ["<unk>"]


def build_frequency(src_tokenizer, dataset_file):
    freq = Counter()
    if not dataset_file or not os.path.isfile(dataset_file):
        return freq
    with open(dataset_file, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            for field in ("code", "nl"):
                text = obj.get(field, "")
                for token in src_tokenizer.tokenize(text):
                    freq[token] += 1
    return freq


def frequency_match(token, src_vocab_set, src_tokenizer, freq):
    base = tokenize_match(token, src_vocab_set, src_tokenizer)
    out = []
    for piece in base:
        if piece == "<unk>":
            out.append(piece)
            continue
        candidates = [cand for cand in candidate_variants(piece) if cand in src_vocab_set]
        if not candidates:
            out.append(piece)
            continue
        out.append(max(candidates, key=lambda item: freq.get(item, 0)))
    return out if out else ["<unk>"]


def write_match(path, target_vocab_items, source_vocab_items, strategy, src_tokenizer, freq):
    source_tokens = [token for token, _ in source_vocab_items]
    source_set = set(source_tokens)
    source_vocab_dict = dict(source_vocab_items)
    unk_id = source_vocab_dict.get("<unk>", 0)

    with open(path, "w", encoding="utf-8") as fout:
        for token, tid in target_vocab_items:
            if strategy == "tokenizer":
                pieces = tokenize_match(token, source_set, src_tokenizer)
            elif strategy == "morphology":
                pieces = morphology_match(token, source_tokens)
            else:
                pieces = frequency_match(token, source_set, src_tokenizer, freq)
            ids = [str(source_vocab_dict.get(piece, unk_id)) for piece in pieces]
            fout.write(f"{tid}\t{token}\t{','.join(ids)}\t{','.join(pieces)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-file", default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, use_fast=True)

    src_vocab_dict = src_tokenizer.get_vocab()
    tgt_vocab_dict = tgt_tokenizer.get_vocab()
    src_vocab_items = sorted(src_vocab_dict.items(), key=lambda x: x[1])
    tgt_vocab_items = sorted(tgt_vocab_dict.items(), key=lambda x: x[1])
    freq = build_frequency(src_tokenizer, args.dataset_file)

    write_match(os.path.join(args.out_dir, "tokenizer_match.tsv"), tgt_vocab_items, src_vocab_items, "tokenizer", src_tokenizer, freq)
    write_match(os.path.join(args.out_dir, "morphology_match.tsv"), tgt_vocab_items, src_vocab_items, "morphology", src_tokenizer, freq)
    write_match(os.path.join(args.out_dir, "frequency_match.tsv"), tgt_vocab_items, src_vocab_items, "frequency", src_tokenizer, freq)

    metadata = {
        "source_tokenizer": args.source_tokenizer,
        "target_tokenizer": args.target_tokenizer,
        "dataset_file": args.dataset_file,
        "source_vocab_size": len(src_vocab_dict),
        "target_vocab_size": len(tgt_vocab_dict),
    }
    with open(os.path.join(args.out_dir, "match_metadata.json"), "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=False)
    print(f"done: {args.out_dir}")


if __name__ == "__main__":
    main()
