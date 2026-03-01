#!/usr/bin/env python3
"""Scan dataset folders for report.json and compute distributions for
annotation.platform, annotation.content, annotation.complexity.

Usage:
    python3 src/scripts/annotation_stats.py --root data/data_test --output output/annotation_stats.json

The script will print a human-readable summary and write a JSON file with counters.
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def normalize(v):
    if v is None:
        return "<missing>"
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if v == "":
        return "<empty>"
    return v


def extract_annotation(data):
    # data may have annotation as dict or JSON-string
    ann = data.get("annotation")
    if ann is None:
        # try top-level keys
        return {
            "platform": data.get("platform"),
            "content": data.get("content"),
            "complexity": data.get("complexity"),
        }

    if isinstance(ann, dict):
        return {
            "platform": ann.get("platform"),
            "content": ann.get("content"),
            "complexity": ann.get("complexity"),
        }

    # annotation may be a JSON string
    if isinstance(ann, str):
        try:
            parsed = json.loads(ann)
            if isinstance(parsed, dict):
                return {
                    "platform": parsed.get("platform"),
                    "content": parsed.get("content"),
                    "complexity": parsed.get("complexity"),
                }
        except Exception:
            # not JSON, fallthrough
            pass

    # fallback
    return {"platform": None, "content": None, "complexity": None}


def scan(root: Path):
    counters = {
        "platform": Counter(),
        "content": Counter(),
        "complexity": Counter(),
    }
    totals = {"files_seen": 0, "reports_parsed": 0, "no_annotation": 0}
    files = list(root.rglob("report.json"))
    for f in files:
        totals["files_seen"] += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            # skip unreadable
            continue

        ann = extract_annotation(data)
        if not ann or all(v is None for v in ann.values()):
            totals["no_annotation"] += 1
            continue

        totals["reports_parsed"] += 1
        for key in ("platform", "content", "complexity"):
            val = normalize(ann.get(key))
            counters[key][val] += 1

    return counters, totals, files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/data_test", help="Dataset root directory to scan")
    p.add_argument("--output", default="output/annotation_stats.json", help="Path to write summary JSON")
    p.add_argument("--max-show", type=int, default=30, help="Max distinct tokens to show per field")
    args = p.parse_args()

    root = Path(args.root)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    counters, totals, files = scan(root)

    summary = {
        "root": str(root),
        "files_found": len(files),
        "totals": totals,
        "counters": {k: dict(v) for k, v in counters.items()},
    }

    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # print human readable
    print(f"Scanned root: {root}")
    print(f"report.json files found: {len(files)}")
    print(f"reports parsed (had annotation): {totals['reports_parsed']}")
    print()
    for key in ("platform", "content", "complexity"):
        print(f"--- {key} distribution ---")
        items = counters[key].most_common(args.max_show)
        if not items:
            print("  (no data)")
        else:
            for val, cnt in items:
                print(f"  {val}: {cnt}")
        print()

    print(f"Saved JSON summary to: {outp}")


if __name__ == "__main__":
    main()
