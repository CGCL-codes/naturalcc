"""Score the built-in C static rules against the Contract-5 ground truth, offline.

No model calls and no network: this is the fast loop for tuning detection rules.
Scoring mirrors ``benchmark_contract5_web.py`` exactly — a finding matches a
ground-truth site when the category matches and the finding line falls inside
the site's function line range.

    uv run python scripts/measure_contract5_static.py            # summary table
    uv run python scripts/measure_contract5_static.py --misses   # list every FN/FP
    uv run python scripts/measure_contract5_static.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.security_c_rules import scan_c_source
from code_agent.security_rtos_rules import scan_rtos_source

DATASET = ROOT / "artifacts" / "contract5" / "rtthread" / "dataset"
PROJECT = DATASET / "project"
GROUND_TRUTH = DATASET / "ground_truth.json"
CATEGORY_LABELS = {"buffer_overflow": "缓冲区溢出", "data_race": "多线程竞争",
                   "memory_leak": "内存泄漏", "command_execution": "命令执行漏洞"}
# Contract-5's buffer_overflow spans CWE-787/120/121/122/119, so the product's
# finer-grained string/array write categories count as evidence for it.
ACCEPTED = {
    "buffer_overflow": {"buffer_overflow", "string_overflow", "array_oob"},
    "data_race": {"data_race"},
    "memory_leak": {"memory_leak"},
    "command_execution": {"command_execution"},
}
DIFFICULTIES = ("easy", "medium", "hard")


def scan_file(relative: str) -> list[dict[str, Any]]:
    source = (PROJECT / relative).read_text(encoding="utf-8", errors="replace")
    findings, _ = scan_c_source(source, relative, 0)
    findings += scan_rtos_source(source, relative, 0)[0]
    return findings


def score(ground_truth: dict[str, Any]) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for case in ground_truth["cases"]:
        by_file.setdefault(case["file"], []).append(case)
    rows, unmatched = [], []
    for relative in sorted(f["path"] for f in ground_truth["detection_files"]):
        findings = scan_file(relative)
        claimed: set[int] = set()
        for case in by_file.get(relative, []):
            accepted = ACCEPTED[case["category"]]
            hits = [f for f in findings if f.get("category") in accepted
                    and case["line_start"] <= f["line"] <= case["line_end"]]
            claimed.update(id(f) for f in hits)
            rows.append({**case, "detected": bool(hits),
                         "matched": [{"category": f["category"], "line": f["line"],
                                      "analyzer_id": f.get("analyzer_id")} for f in hits]})
        unmatched.extend({"file": relative, "category": f.get("category"), "line": f["line"],
                          "analyzer_id": f.get("analyzer_id")}
                         for f in findings if id(f) not in claimed
                         and f.get("category") in {c for s in ACCEPTED.values() for c in s})
    return {"rows": rows, "out_of_scope": unmatched}


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["expected"] == "defect" and r["detected"] for r in rows)
    fn = sum(r["expected"] == "defect" and not r["detected"] for r in rows)
    fp = sum(r["expected"] == "control" and r["detected"] for r in rows)
    tn = sum(r["expected"] == "control" and not r["detected"] for r in rows)

    def ratio(n: int, d: int) -> float | None:
        return n / d if d else None

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "detection_rate": ratio(tp, tp + fn), "warning_accuracy": ratio(tp, tp + fp),
            "false_positive_rate": ratio(fp, fp + tn), "accuracy": ratio(tp + tn, len(rows))}


def rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:6.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--misses", action="store_true", help="list every missed defect and每条误报")
    parser.add_argument("--json", type=Path, help="write the full result as JSON")
    args = parser.parse_args()
    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    result = score(ground_truth)
    rows, overall = result["rows"], metric(result["rows"])
    print("内置 C 静态规则（无模型、无 Cppcheck）在合同 5 数据集上的表现\n")
    print(f"{'类别':<14}{'TP':>4}{'FN':>4}{'FP':>4}{'TN':>4}   检出率    预警准确率   误报率     正确率")
    for category, label in CATEGORY_LABELS.items():
        m = metric([r for r in rows if r["category"] == category])
        print(f"{label:<12}{m['tp']:>5}{m['fn']:>4}{m['fp']:>4}{m['tn']:>4}  {rate(m['detection_rate'])}  "
              f"{rate(m['warning_accuracy'])}  {rate(m['false_positive_rate'])}  {rate(m['accuracy'])}")
    print(f"{'总体':<12}{overall['tp']:>5}{overall['fn']:>4}{overall['fp']:>4}{overall['tn']:>4}  "
          f"{rate(overall['detection_rate'])}  {rate(overall['warning_accuracy'])}  "
          f"{rate(overall['false_positive_rate'])}  {rate(overall['accuracy'])}")
    print("\n按难度（仅注入缺陷）：")
    for difficulty in DIFFICULTIES:
        subset = [r for r in rows if r["expected"] == "defect" and r["difficulty"] == difficulty]
        hit = sum(r["detected"] for r in subset)
        print(f"  {difficulty:<8} {hit}/{len(subset)}  {rate(hit / len(subset) if subset else None)}")
    print(f"\n命中但落在非 site 函数上的告警（计为范围外，不进指标）：{len(result['out_of_scope'])}")
    if args.misses:
        print("\n--- 漏报的注入缺陷 ---")
        for r in rows:
            if r["expected"] == "defect" and not r["detected"]:
                print(f"  {r['site_id']:<9} {r['category']:<18} {r['file']}:{r['defect_line']} {r['function']}() [{r['difficulty']}]")
                print(f"            {r['description'][:110]}")
        print("\n--- 被误报的安全对照 ---")
        for r in rows:
            if r["expected"] == "control" and r["detected"]:
                print(f"  {r['site_id']:<9} {r['category']:<18} {r['file']}:{r['defect_line']} {r['function']}()  <- {r['matched']}")
    if args.json:
        args.json.write_text(json.dumps({"overall": overall, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
