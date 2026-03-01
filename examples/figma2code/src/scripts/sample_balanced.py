#!/usr/bin/env python3
"""Balanced sampler across platform and complexity.

Goal: pick `platform_count * complexity_count * n` samples (ideally)
by selecting up to `n` samples per (platform, complexity) combination.
Within each combination, prefer to increase `content` diversity by
round-robin selecting from different content groups.

Outputs:
 - copies selected sample folders (parent of report.json) into output_dir/0,1,...
 - writes summary JSON to output_dir/summary.json with selected items and stats

Usage:
  python3 src/scripts/sample_balanced.py --root data/data_test --n 5 --output output/sampled_5
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import shutil
import random


def normalize(v):
    if v is None:
        return "<missing>"
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    return v if v != "" else "<empty>"


def load_reports(roots):
    reports = []
    for root in roots:
        root = Path(root)
        for rpt in root.rglob("report.json"):
            try:
                data = json.loads(rpt.read_text(encoding="utf-8"))
            except Exception:
                continue
            # extract annotation (may be dict or JSON string)
            ann = data.get("annotation")
            if isinstance(ann, str):
                try:
                    ann = json.loads(ann)
                except Exception:
                    ann = None

            if isinstance(ann, dict):
                platform = normalize(ann.get("platform"))
                complexity = normalize(ann.get("complexity"))
                content = normalize(ann.get("content"))
            else:
                # try top-level
                platform = normalize(data.get("platform"))
                complexity = normalize(data.get("complexity"))
                content = normalize(data.get("content"))

            reports.append({
                "report_path": str(rpt.resolve()),
                "sample_dir": str(rpt.parent.resolve()),
                "platform": platform,
                "complexity": complexity,
                "content": content,
            })
    return reports


def round_robin_pick(items_by_content, k):
    """Given dict content->list(items), pick up to k items by round-robin
    across content groups to maximize content diversity."""
    # copy lists
    groups = {c: list(v)[:] for c, v in items_by_content.items()}
    for v in groups.values():
        random.shuffle(v)
    picked = []
    # iterate until k reached or no items left
    while len(picked) < k and any(groups.values()):
        for c in list(groups.keys()):
            if groups[c] and len(picked) < k:
                picked.append(groups[c].pop())
    return picked


def load_exclude_keys(exclude_dir):
    """Load directory names (keys) from `exclude_dir`.

    Behavior:
    - If `exclude_dir` exists as given, use it.
    - Otherwise try resolving it relative to this script's directory.
    - Returns a list of directory basenames to exclude.
    """
    p = Path(exclude_dir)
    if not p.exists():
        alt = Path(__file__).parent / exclude_dir
        if alt.exists():
            p = alt
    keys = []
    try:
        if p.exists() and p.is_dir():
            for entry in p.iterdir():
                if entry.is_dir():
                    keys.append(entry.name)
    except Exception:
        # if anything goes wrong, return empty list (no exclusions)
        pass
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", default=["data/data_test"], help="Dataset root(s) to scan")
    p.add_argument("--n", type=int, default=5, help="n per (platform, complexity) cell")
    p.add_argument("--output", default="output/sampled_5_2", help="Output directory to copy selected samples")
    p.add_argument("--exclude_dir", default="../output/sampled_rebuttal_1",
                   help="Directory containing sample keys to exclude (folder names).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    random.seed(args.seed)

    reports = load_reports(args.root)
    # load exclude keys and filter reports accordingly
    exclude_keys = load_exclude_keys(args.exclude_dir)
    if exclude_keys:
        before = len(reports)
        reports = [r for r in reports if Path(r["sample_dir"]).name not in exclude_keys]
        removed = before - len(reports)
        print(f"Excluded {removed} reports based on keys from: {args.exclude_dir}")
    if not reports:
        print("No report.json files found under the provided roots.")
        return

    # build combos
    combos = defaultdict(list)  # (platform, complexity) -> list of reports
    platform_set = set()
    complexity_set = set()
    for r in reports:
        platform_set.add(r["platform"]) if r["platform"] else None
        complexity_set.add(r["complexity"]) if r["complexity"] else None
        combos[(r["platform"], r["complexity"])].append(r)

    platforms = sorted(platform_set)
    complexities = sorted(complexity_set)

    desired_total = len(platforms) * len(complexities) * args.n
    print(f"Found {len(reports)} reports. Platforms: {platforms}, Complexities: {complexities}")
    print(f"Desired total samples (platforms*complexities*n): {desired_total}")

    # For each combo, group by content and pick up to n via round-robin
    selected = []
    per_combo_selected = {}
    for combo, items in combos.items():
        # group by content
        items_by_content = defaultdict(list)
        for it in items:
            items_by_content[it["content"]].append(it)
        k = args.n
        picks = round_robin_pick(items_by_content, k)
        per_combo_selected[combo] = picks
        selected.extend(picks)

    # if we overshot (shouldn't), trim; if undershot because some combos had < n, fill from combos with leftovers
    if len(selected) < desired_total:
        needed = desired_total - len(selected)
        # collect remaining candidates
        remaining = []
        for combo, items in combos.items():
            already = set(id(x) for x in per_combo_selected.get(combo, []))
            for it in items:
                if id(it) not in already:
                    remaining.append((combo, it))
        # sort remaining by combo size descending to favor richer combos
        remaining.sort(key=lambda x: len(combos[x[0]]), reverse=True)
        for _, it in remaining:
            if needed <= 0:
                break
            selected.append(it)
            needed -= 1

    # final dedupe (by sample_dir)
    uniq = {}
    for it in selected:
        uniq[it["sample_dir"]] = it
    final_selected = list(uniq.values())

    print(f"Selected {len(final_selected)} unique samples (after filling).")

    # prepare output
    out = Path(args.output)
    if out.exists():
        # don't auto-delete; make a new dir with suffix if exists
        out = Path(str(out) + f"_{args.n}")
    out.mkdir(parents=True, exist_ok=True)

    # copy sample dirs into output
    summary = {"requested_n": args.n, "desired_total": desired_total, "selected_count": len(final_selected), "items": []}
    for i, it in enumerate(final_selected):
        src = Path(it["sample_dir"])
        # preserve original folder name; if conflict, append an index suffix
        base_name = src.name
        dst = out / base_name
        suffix = 1
        while dst.exists():
            dst = out / f"{base_name}_{suffix}"
            suffix += 1
        try:
            shutil.copytree(src, dst)
        except Exception:
            # if copytree fails (e.g., permission), skip copy but still record
            pass
        summary["items"].append({
            "index": i,
            "sample_dir": str(dst),
            "original_sample_dir": it["sample_dir"],
            "report_path": it["report_path"],
            "platform": it["platform"],
            "complexity": it["complexity"],
            "content": it["content"],
        })

    # stats: counts and proportions for each of the three dimensions
    total_selected = len(final_selected)
    platform_counter = Counter(it["platform"] for it in final_selected)
    complexity_counter = Counter(it["complexity"] for it in final_selected)
    content_counter = Counter(it["content"] for it in final_selected)

    def counter_with_props(counter):
        return {
            k: {"count": v, "proportion": round(v / total_selected, 4)}
            for k, v in counter.items()
        }

    combo_counter = Counter((it["platform"], it["complexity"]) for it in final_selected)

    summary["stats"] = {
        "total_selected": total_selected,
        "by_platform": counter_with_props(platform_counter),
        "by_complexity": counter_with_props(complexity_counter),
        "by_content": counter_with_props(content_counter),
        "by_combo": {f"{p}||{c}": cnt for (p, c), cnt in combo_counter.items()},
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Copied samples to: {out}")
    print(f"Wrote summary to: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
