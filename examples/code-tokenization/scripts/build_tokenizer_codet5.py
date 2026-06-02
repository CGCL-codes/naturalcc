#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modified CodeT5 tokenizer from code-text data.")
    parser.add_argument("--base-tokenizer", required=True, help="Path to the base CodeT5 tokenizer.")
    parser.add_argument("--data-files", nargs="+", required=True, help="JSON or JSONL files with `nl` and `code` fields.")
    parser.add_argument("--output-dir", required=True, help="Directory used to save the new tokenizer.")
    parser.add_argument("--vocab-size", type=int, default=0, help="0 means reusing the base tokenizer size.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit-examples", type=int, default=-1)
    return parser.parse_args()


def iter_texts(data_files: List[str], limit_examples: int) -> Iterator[str]:
    seen = 0
    for path in data_files:
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                if limit_examples >= 0 and seen >= limit_examples:
                    return
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("nl", "code"):
                    value = obj.get(key, "")
                    if value:
                        yield value
                seen += 1


def batch_iterator(data_files: List[str], batch_size: int, limit_examples: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for text in iter_texts(data_files, limit_examples):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.data_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    base_vocab_size = len(base_tokenizer)
    target_vocab_size = args.vocab_size if args.vocab_size > 0 else base_vocab_size

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        batch_iterator(args.data_files, args.batch_size, args.limit_examples),
        vocab_size=target_vocab_size,
    )
    new_tokenizer.save_pretrained(str(out_dir))

    with open(out_dir / "build_metadata.json", "w", encoding="utf-8") as fout:
        json.dump(
            {
                "base_tokenizer": args.base_tokenizer,
                "data_files": args.data_files,
                "base_vocab_size": base_vocab_size,
                "final_vocab_size": len(new_tokenizer),
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
