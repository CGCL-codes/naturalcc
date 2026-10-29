"""Static-only Contract-3 score: product analyzers, no model, no network.

Runs the comprehensive engines (built-in C rules, Cppcheck, Clang) on the frozen
test sets and scores them with the product classifier ``classify_findings``.

- RTOS suites (balanced, independent, balanced-hard, independent-hard): each
  category fixture is one generated C file; a case is detected when a finding of
  its category falls inside the case function's line range.
- Juliet sets (juliet, juliet-holdout): file-level, like the Web benchmark; the
  Juliet support file io.c is analysed with the targets (as a project-scope scan
  would), but only findings in the target src_*.c files count.

Cppcheck and Clang must be on PATH, e.g.
PATH=/path/to/ncc-static/bin:$PATH uv run python scripts/measure_contract3_static.py --suite balanced,juliet
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import benchmark_contract3_cppcheck as base
from code_agent.scripts import benchmark_contract3_independent as generator
from code_agent.security_analysis import clang_scan, cppcheck_scan
from code_agent.security_c_rules import scan_c_source
from code_agent.security_contracts import classify_findings

RTOS = ("balanced", "independent", "balanced-hard", "independent-hard")
JULIET = ("juliet", "juliet-holdout")
CATEGORIES = ("array_oob", "string_overflow", "null_pointer")


def analyse(project: Path, files: list[Path]) -> list[dict]:
    findings = []
    for scan in (cppcheck_scan, clang_scan):
        findings += scan([str(path) for path in files], str(project), 0)[0]
    for path in files:
        findings += scan_c_source(path.read_text(encoding="utf-8"), path.relative_to(project).as_posix(), 0)[0]
    return classify_findings(findings, str(project))


def rtos(profile: str) -> list[tuple[str, bool, bool]]:
    rows = []
    for category in base.load_manifest()["categories"]:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            fixture, cases = generator.inject_fixture(project, "fixture", [category], fixture_name="fixture.c", profile=profile)
            findings = analyse(project, [fixture])
            for case in cases:
                detected = any(f.get("category") == category["id"] and case["line_start"] <= f["line"] <= case["line_end"] for f in findings)
                rows.append((category["id"], case["expected"] == "defect", detected))
    return rows


def juliet(name: str) -> list[tuple[str, bool, bool]]:
    dataset = json.loads((ROOT / "artifacts" / "contract3" / name / "dataset.json").read_text(encoding="utf-8"))
    rows = []
    for sample in dataset["samples"]:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for file, content in {**dataset["support"], **sample["files"]}.items():
                (project / file).write_text(content, encoding="utf-8")
            findings = analyse(project, [project / file for file in sample["files"]] + [project / "io.c"])
            detected = any(f.get("category") == sample["category"] and f["file"] in sample["files"] for f in findings)
            rows.append((sample["category"], sample["vulnerable"], detected))
    return rows


def summary(name: str, rows: list[tuple[str, bool, bool]]) -> str:
    def rates(subset):
        tp = sum(v and d for _, v, d in subset)
        fn = sum(v and not d for _, v, d in subset)
        fp = sum(not v and d for _, v, d in subset)
        tn = sum(not v and not d for _, v, d in subset)
        return tp, fn, fp, tn
    tp, fn, fp, tn = rates(rows)
    parts = [f"{name:17} 检出率 {tp / (tp + fn):6.1%}  误报率 {fp / (fp + tn):6.1%}  TP/FN/FP/TN={tp}/{fn}/{fp}/{tn}"]
    for category in CATEGORIES:
        ctp, cfn, cfp, ctn = rates([row for row in rows if row[0] == category])
        parts.append(f"{category} 检出{ctp}/{ctp + cfn} 误报{cfp}/{cfp + ctn}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=",".join(RTOS),
                        help=f"Comma-separated: {', '.join(RTOS + JULIET)} (default: all RTOS suites)")
    args = parser.parse_args()
    missing = [tool for tool in ("cppcheck", "clang") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} not on PATH; static-only scores would be incomplete")
    for name in args.suite.split(","):
        if name in RTOS:
            print(summary(name, rtos(name)), flush=True)
        elif name in JULIET:
            print(summary(name, juliet(name)), flush=True)
        else:
            raise SystemExit(f"Unknown suite: {name}")


if __name__ == "__main__":
    main()
