#!/usr/bin/env python3
"""Summarize NaturalCC run records and optional official evaluator JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def summarize(rows: list[dict[str, Any]], security_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    security_rows = security_rows or []
    security_index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in security_rows:
        key = (str(item.get("case_id", "")), str(item.get("codegraph", "unknown")))
        loc = item.get("generated_nonblank_loc")
        findings = item.get("self_check_vulnerability_findings")
        if not isinstance(loc, (int, float)) or loc < 0 or not isinstance(findings, int) or findings < 0:
            raise ValueError(
                "Security rows require generated_nonblank_loc >= 0 and "
                "self_check_vulnerability_findings >= 0"
            )
        security_index[key] = item
    for row in rows:
        item = security_index.get((str(row.get("case_id", "")), str(row.get("codegraph", "unknown"))))
        if item:
            loc = item["generated_nonblank_loc"]
            findings = item["self_check_vulnerability_findings"]
            row["generated_nonblank_loc"] = loc
            row["self_check_vulnerability_findings"] = findings
            row["vulnerability_density_per_100_loc"] = (findings * 100 / loc) if loc else None
            row["vulnerability_density_pass_le_2"] = (findings * 100 / loc <= 2) if loc else None
            row["security_result_source"] = item.get("security_result_source")

    by_case_mode: dict[tuple[str, str], dict[str, Any]] = {}
    modes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        mode = str(row.get("codegraph", "unknown"))
        modes[mode].append(row)
        if case_id:
            by_case_mode[(case_id, mode)] = row
    report: dict[str, Any] = {"runs": len(rows), "by_codegraph": {}, "paired_official": None}
    for mode, group in sorted(modes.items()):
        token_rows = [r for r in group if isinstance(r.get("input_tokens"), (int, float)) and isinstance(r.get("output_tokens"), (int, float))]
        scored = [r for r in group if isinstance(r.get("official_pass"), bool)]
        report["by_codegraph"][mode] = {
            "cases": len(group),
            "run_completed": sum(r.get("status") == "completed" for r in group),
            "duration_seconds_total": sum(r.get("duration_seconds", 0) for r in group if isinstance(r.get("duration_seconds"), (int, float))) or None,
            "codegraph_init_seconds_total": sum(r.get("codegraph_init_seconds", 0) for r in group if isinstance(r.get("codegraph_init_seconds"), (int, float))) or None,
            "llm_calls_total": sum(r.get("llm_calls", 0) for r in group if isinstance(r.get("llm_calls"), (int, float))) or None,
            "tool_calls_total": sum(r.get("tool_calls", 0) for r in group if isinstance(r.get("tool_calls"), (int, float))) or None,
            "official_scored": len(scored),
            "official_passed": sum(r.get("official_pass") is True for r in scored),
            "official_pass_rate": (sum(r.get("official_pass") is True for r in scored) / len(scored)) if scored else None,
            "test_tampering_cases": sum(bool(r.get("test_tampering_detected")) for r in group),
            "input_tokens_total": sum(r["input_tokens"] for r in token_rows) if token_rows else None,
            "output_tokens_total": sum(r["output_tokens"] for r in token_rows) if token_rows else None,
            "cost_usd_total": sum(r.get("cost_usd", 0) for r in group if isinstance(r.get("cost_usd"), (int, float))) or None,
            "security_scored": sum(isinstance(r.get("vulnerability_density_pass_le_2"), bool) for r in group),
            "security_cases_le_2_findings_per_100_loc": sum(r.get("vulnerability_density_pass_le_2") is True for r in group),
            "security_findings_total": sum(r.get("self_check_vulnerability_findings", 0) for r in group if isinstance(r.get("self_check_vulnerability_findings"), int)),
            "security_generated_nonblank_loc_total": sum(r.get("generated_nonblank_loc", 0) for r in group if isinstance(r.get("generated_nonblank_loc"), (int, float))),
            "security_density_findings_per_100_loc": (
                sum(r.get("self_check_vulnerability_findings", 0) for r in group if isinstance(r.get("self_check_vulnerability_findings"), int)) * 100 /
                sum(r.get("generated_nonblank_loc", 0) for r in group if isinstance(r.get("generated_nonblank_loc"), (int, float)) and r.get("generated_nonblank_loc", 0) > 0)
                if any(isinstance(r.get("generated_nonblank_loc"), (int, float)) and r.get("generated_nonblank_loc", 0) > 0 for r in group) else None
            ),
        }
    paired = []
    case_ids = sorted({case_id for case_id, _mode in by_case_mode})
    for case_id in case_ids:
        off = by_case_mode.get((case_id, "off"))
        on = by_case_mode.get((case_id, "on"))
        if off is None or on is None or not isinstance(off.get("official_pass"), bool) or not isinstance(on.get("official_pass"), bool):
            continue
        paired.append({"case_id": case_id, "off": off["official_pass"], "on": on["official_pass"]})
    if paired:
        report["paired_official"] = {
            "paired_cases": len(paired),
            "on_only_pass": sum(p["on"] and not p["off"] for p in paired),
            "off_only_pass": sum(p["off"] and not p["on"] for p in paired),
            "both_pass": sum(p["off"] and p["on"] for p in paired),
            "neither_pass": sum(not p["off"] and not p["on"] for p in paired),
            "cases": paired,
        }
    report["warning"] = "Do not call an agent run 'correct' unless official_pass came from the benchmark's official evaluator."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="NaturalCC results.jsonl files")
    parser.add_argument("--official-results", type=Path, help="Optional JSONL: case_id, codegraph, official_pass, official_metric, official_value")
    parser.add_argument("--security-results", type=Path, help="Optional normalized scanner JSONL: case_id, codegraph, generated_nonblank_loc, self_check_vulnerability_findings")
    parser.add_argument("--output", type=Path, help="Output JSON; default next to first results file")
    args = parser.parse_args()
    rows = read_jsonl(args.results)
    if args.official_results:
        official = read_jsonl([args.official_results])
        index = {(str(r.get("case_id")), str(r.get("codegraph"))): r for r in rows}
        for item in official:
            key = (str(item.get("case_id", "")), str(item.get("codegraph", "")))
            if key not in index:
                raise SystemExit(f"Official result has no matching NaturalCC run: {key}")
            if not isinstance(item.get("official_pass"), bool):
                raise SystemExit(f"official_pass must be true/false for {key}")
            index[key].update({k: item.get(k) for k in ("official_pass", "official_metric", "official_value", "official_result_source")})
    security = read_jsonl([args.security_results]) if args.security_results else []
    report = summarize(rows, security)
    output = args.output or args.results[0].parent / "summary.metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
