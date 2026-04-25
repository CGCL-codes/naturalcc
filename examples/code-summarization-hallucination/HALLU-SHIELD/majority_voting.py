#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(__file__).resolve().parent / "LLM-as-a-Judge-result"
DEFAULT_JUDGES = ("GPT5-1-Chat", "Qwen3-5-Plus", "Deepseek-V3-2")
DEFAULT_DATASETS = ("easy", "hard")
DEFAULT_BASE_MODELS = ("deepseek", "llama7b", "llama8b", "qwen7b")

RAW_TO_LABEL = {
    "Hallu-Shield wins.": "win",
    "Hallu-Shield loses.": "lose",
    "Tie.": "tie",
}

LABEL_ORDER = ("win", "lose", "tie")


@dataclass
class VoteSummary:
    dataset: str
    base_model: str
    total: int
    win: int
    lose: int
    tie: int
    no_majority_cases: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "dataset": self.dataset,
            "base_model": self.base_model,
            "total": self.total,
            "win": self.win,
            "lose": self.lose,
            "tie": self.tie,
            "no_majority_cases": self.no_majority_cases,
        }


def parse_csv_arg(value: str | None, default: Iterable[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("CSV argument cannot be empty.")
    return values


def read_labels(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    labels: list[str] = []
    for index, raw_line in enumerate(lines, start=1):
        if raw_line not in RAW_TO_LABEL:
            raise ValueError(
                f"Invalid label in {path} at line {index}: {raw_line!r}. "
                f"Expected one of: {', '.join(RAW_TO_LABEL)}"
            )
        labels.append(RAW_TO_LABEL[raw_line])
    return labels


def majority_vote(votes: tuple[str, str, str]) -> tuple[str, bool]:
    counts = Counter(votes).most_common()
    label, count = counts[0]
    if count >= 2:
        return label, False
    return "tie", True


def summarize_group(root: Path, dataset: str, base_model: str, judges: tuple[str, ...]) -> VoteSummary:
    paths = [root / f"{judge}-{dataset}-{base_model}.txt" for judge in judges]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required judge file(s):\n" + "\n".join(f"- {path}" for path in missing)
        )

    judge_labels = [read_labels(path) for path in paths]
    lengths = {len(labels) for labels in judge_labels}
    if len(lengths) != 1:
        raise ValueError(
            f"Line count mismatch for dataset={dataset}, base_model={base_model}: "
            f"{', '.join(str(length) for length in sorted(lengths))}"
        )

    total = len(judge_labels[0])
    final_counts = Counter({label: 0 for label in LABEL_ORDER})
    no_majority_cases = 0

    for votes in zip(*judge_labels):
        final_label, no_majority = majority_vote(votes)
        final_counts[final_label] += 1
        if no_majority:
            no_majority_cases += 1

    if final_counts["win"] + final_counts["lose"] + final_counts["tie"] != total:
        raise RuntimeError(
            f"Internal consistency check failed for dataset={dataset}, base_model={base_model}."
        )

    return VoteSummary(
        dataset=dataset,
        base_model=base_model,
        total=total,
        win=final_counts["win"],
        lose=final_counts["lose"],
        tie=final_counts["tie"],
        no_majority_cases=no_majority_cases,
    )


def print_markdown_table(rows: list[VoteSummary]) -> None:
    print("| dataset | base_model | total | win | lose | tie | no_majority_cases |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.dataset} | {row.base_model} | {row.total} | {row.win} | "
            f"{row.lose} | {row.tie} | {row.no_majority_cases} |"
        )


def print_dataset_aggregate(rows: list[VoteSummary], datasets: tuple[str, ...]) -> None:
    print("\nDataset aggregates:")
    for dataset in datasets:
        matched = [row for row in rows if row.dataset == dataset]
        total = sum(row.total for row in matched)
        win = sum(row.win for row in matched)
        lose = sum(row.lose for row in matched)
        tie = sum(row.tie for row in matched)
        print(f"- {dataset}: total={total}, win={win}, lose={lose}, tie={tie}")


def write_csv(rows: list[VoteSummary], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "base_model",
                "total",
                "win",
                "lose",
                "tie",
                "no_majority_cases",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing judge result files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default=None,
        help=f"Comma-separated judge list (default: {','.join(DEFAULT_JUDGES)})",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated dataset types (default: {','.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--base-models",
        type=str,
        default=None,
        help=f"Comma-separated base model list (default: {','.join(DEFAULT_BASE_MODELS)})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to export detailed results as CSV.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    judges = parse_csv_arg(args.judges, DEFAULT_JUDGES)
    datasets = parse_csv_arg(args.datasets, DEFAULT_DATASETS)
    base_models = parse_csv_arg(args.base_models, DEFAULT_BASE_MODELS)

    rows: list[VoteSummary] = []
    for dataset in datasets:
        for base_model in base_models:
            rows.append(summarize_group(args.root, dataset, base_model, judges))

    print_markdown_table(rows)
    print_dataset_aggregate(rows, datasets)

    if args.output_csv is not None:
        write_csv(rows, args.output_csv)
        print(f"\nCSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
