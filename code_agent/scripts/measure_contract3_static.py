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
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import benchmark_contract3_cppcheck as base
from code_agent.scripts import benchmark_contract3_independent as generator
from code_agent.security_analysis import clang_scan, cppcheck_scan
from code_agent.security_c_rules import scan_c_source
from code_agent import security_exam
from code_agent.security_contracts import classify_findings, evaluate_contract, load_ground_truth
from code_agent.security_exam import contract3_exam

RTOS = ("balanced", "independent", "balanced-hard", "independent-hard")
JULIET = ("juliet", "juliet-holdout")
CATEGORIES = ("array_oob", "string_overflow", "null_pointer")


def analyse(project: Path, files: list[Path]) -> list[dict]:
    # Contract-3 evaluation: the exam switches in security_exam.CONTRACT3_OFF apply here.
    with contract3_exam(True):
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


def delivered(suite: str) -> list[dict]:
    """Score actual shipped bytes, validating GT with the product loader.

    The temporary copy contains only selected fixtures and the evaluator's GT;
    no model is invoked. Analyzer failure aborts rather than becoming a miss.
    Engine metadata and matched findings remain available for each case.
    """
    results = []
    for spec in base.load_manifest()["sources"]:
        truth_path = ROOT / "artifacts/contract3/ground_truth" / suite / spec["id"] / "ground_truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if truth.get("dataset_revision") != generator.DATASET_REVISION:
            raise ValueError(f"Stale ground truth: {truth_path}; regenerate the delivered dataset")
        origin = ROOT / truth["project"]["repo_dir"]
        with tempfile.TemporaryDirectory(prefix="contract3-delivered-") as directory:
            project = Path(directory)
            files = []
            for relative in truth["source_sha256"]:
                source = (origin / relative).resolve()
                source.relative_to(origin.resolve())
                target = project / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                files.append(target)
            (project / "ground_truth.json").write_text(json.dumps(truth), encoding="utf-8")
            verified = load_ground_truth(project, "ground_truth.json", "frequent_defects", files)
            with contract3_exam(True):
                findings, engines = [], []
                for scanner in (cppcheck_scan, clang_scan):
                    found, engine = scanner([str(p) for p in files], str(project), 0)
                    if engine["status"] != "completed":
                        raise RuntimeError(f"Incomplete scan: {engine}")
                    findings.extend(found)
                    engines.append(engine)
                for path in files:
                    found, parse_error = scan_c_source(path.read_text(encoding="utf-8"), path.relative_to(project).as_posix(), 0)
                    if parse_error:
                        raise RuntimeError(f"Incomplete source parse: {path.name}")
                    findings.extend(found)
                classify_findings(findings, project)
                statistics = evaluate_contract(findings, "frequent_defects", verified)
            results.append({"suite": suite, "project": spec["id"],
                            "ground_truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                            "source_sha256": truth["source_sha256"], "engines": engines,
                            "statistics": statistics})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=",".join(RTOS),
                        help=f"Comma-separated: {', '.join(RTOS + JULIET)} (default: all RTOS suites)")
    parser.add_argument("--delivered", action="store_true", help="Score all three shipped projects against their hash-checked GT")
    parser.add_argument("--reference", action="store_true", help="Use full rules in this process without changing the teaching default")
    parser.add_argument("--output", type=Path, help="Write detailed delivered-fixture evidence JSON (requires --delivered)")
    args = parser.parse_args()
    if args.output and not args.delivered:
        parser.error("--output requires --delivered")
    if args.delivered and any(s not in RTOS for s in args.suite.split(",")):
        parser.error("--delivered supports only the four RTOS suites")
    missing = [tool for tool in ("cppcheck", "clang") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} not on PATH; static-only scores would be incomplete")
    if args.reference:
        security_exam.CONTRACT3_OFF = frozenset()
    if args.delivered:
        results = []
        for suite in args.suite.split(","):
            for result in delivered(suite):
                rows = [(r["category"], r["expected"] == "defect", r["detected"]) for r in result["statistics"]["cases"]]
                print(summary(f"{suite}/{result['project']}", rows), flush=True)
                results.append(result)
        if args.output:
            rule_files = ("security_exam.py", "security_contracts.py", "security_c_rules.py", "security_analysis.py")
            document = {"dataset_revision": generator.DATASET_REVISION, "created_at": datetime.now(timezone.utc).isoformat(),
                        "mode": "reference" if args.reference else "teaching", "switches_off": sorted(security_exam.CONTRACT3_OFF),
                        "scope": "Shipped injected functions only; not whole upstream coverage or Web-model performance",
                        "rule_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in rule_files},
                        "results": results}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    for name in args.suite.split(","):
        if name in RTOS:
            print(summary(name, rtos(name)), flush=True)
        elif name in JULIET:
            print(summary(name, juliet(name)), flush=True)
        else:
            raise SystemExit(f"Unknown suite: {name}")


if __name__ == "__main__":
    main()
