"""Reproducible Contract-3 frequent-defect benchmark through the product Cppcheck adapter.

The script downloads three pinned RTOS source trees into a caller-chosen temporary
directory, writes clearly named *test-only* injected and control C files, then
calls ``code_agent.security_analysis.cppcheck_scan``.  Upstream source files are
never modified and are not committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Allow both ``python scripts/...`` (the documented command) and module execution.
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
DATA = ROOT / "artifacts" / "contract3"
MANIFEST_PATH = DATA / "manifest.json"
OUT = DATA / "cppcheck"
REPORT = ROOT / "合同-3-result.md"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
IGNORED_DIRS = {".git", ".github", "build", "out", "docs", "doc", "tests", "test", "samples", "sample", "examples", "example", "boards"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(manifest.get("sources", [])) != 3 or len(manifest.get("categories", [])) != 3:
        raise ValueError("Contract-3 manifest must declare exactly three sources and three categories")
    return manifest


def clone_or_verify(spec: dict[str, Any], root: Path) -> Path:
    checkout = root / spec["id"]
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", spec["ref"], spec["url"], str(checkout)], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["commit"]:
        raise ValueError(f"{spec['id']} revision mismatch: expected {spec['commit']}, got {actual}")
    return checkout


def source_files(root: Path, scan_roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative_root in scan_roots:
        subtree = (root / relative_root).resolve()
        try:
            subtree.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"scan root escapes source tree: {relative_root}") from exc
        if not subtree.is_dir():
            raise ValueError(f"missing scan root: {relative_root}")
        for path in subtree.rglob("*"):
            if path.is_dir() or path.name.startswith("contract3_") or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if any(part in IGNORED_DIRS or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
                continue
            files.append(path)
    return sorted(files)


def loc(files: list[Path]) -> int:
    return sum(len(path.read_text(encoding="utf-8", errors="replace").splitlines()) for path in files)


def append_case(lines: list[str], cases: list[dict[str, Any]], source_id: str, category: dict[str, Any], index: int, vulnerable: bool) -> None:
    # The source fixture deliberately does not reveal the positive/negative label.
    # Labels live only in the ground-truth JSON outside the agent's workspace.
    ordinal = index if vulnerable else index + 20
    prefix = f"contract3_{category['id']}_case_{ordinal:02d}"
    lines.extend([f"static int {prefix}(void)", "{"])
    if category["id"] == "array_oob":
        lines.append("    int slots[2] = {0, 1};")
        defect = "    return slots[2];" if vulnerable else "    return slots[1];"
    elif category["id"] == "string_overflow":
        lines.append("    char target[4] = {0};")
        defect = '    strcpy(target, "ABCDE");' if vulnerable else '    strcpy(target, "ABC");'
        lines.append(defect)
        lines.append("    return target[0];")
        defect = None
    elif category["id"] == "null_pointer":
        lines.append("    int *pointer = 0;")
        defect = "    return *pointer;" if vulnerable else "    return 0;"
    else:
        raise ValueError(f"Unknown category: {category['id']}")
    if defect is not None:
        lines.append(defect)
    lines.append("}")
    cases.append({
        "case_id": f"{source_id}-{category['id']}-{'bad' if vulnerable else 'control'}-{index:02d}",
        "project": source_id,
        "category": category["id"],
        "expected": "defect" if vulnerable else "control",
        "function": prefix,
        "line_start": len(lines) - (4 if category["id"] == "string_overflow" else 3),
        "line_end": len(lines),
        "defect_line": len(lines) - 2 if category["id"] == "string_overflow" else len(lines) - 1,
    })
    lines.append("")


def inject_fixture(project: Path, source_id: str, categories: list[dict[str, Any]], *, fixture_name: str = "contract3_injected_frequent_defects.c") -> tuple[Path, list[dict[str, Any]]]:
    """Write a deterministic test-only file. It is excluded from upstream builds."""
    fixture = project / fixture_name
    lines = ["/* Test-only Contract-3 fixture. Do not include in production builds. */", "#include <string.h>", ""]
    cases: list[dict[str, Any]] = []
    for category in categories:
        for index in range(1, 21):
            append_case(lines, cases, source_id, category, index, vulnerable=True)
        for index in range(1, 21):
            append_case(lines, cases, source_id, category, index, vulnerable=False)
    fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for case in cases:
        case["file"] = fixture.name
    return fixture, cases


def finding_category(finding: dict[str, Any], category: dict[str, Any]) -> bool:
    identifier = finding.get("analyzer_id", "")
    cwe = finding.get("rule_id", "").removeprefix("cwe-")
    return identifier in category["cppcheck_ids"] or (cwe.isdigit() and int(cwe) in category["cwes"])


def classify(cases: list[dict[str, Any]], findings: list[dict[str, Any]], categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    category_by_id = {c["id"]: c for c in categories}
    observed = defaultdict(list)
    for finding in findings:
        observed[(finding.get("file"), finding.get("line"))].append(finding)
    rows = []
    for case in cases:
        category = category_by_id[case["category"]]
        matched = []
        for line in range(case["line_start"], case["line_end"] + 1):
            matched.extend(f for f in observed[(case["file"], line)] if finding_category(f, category))
        rows.append({**case, "detected": bool(matched), "matched_findings": matched})
    return rows


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["expected"] == "defect" and r["detected"] for r in rows)
    fn = sum(r["expected"] == "defect" and not r["detected"] for r in rows)
    fp = sum(r["expected"] == "control" and r["detected"] for r in rows)
    tn = sum(r["expected"] == "control" and not r["detected"] for r in rows)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "detection_rate": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def render_report(result: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = ["# 合同指标 3：高频缺陷检测实测结果", "", f"更新时间（UTC）：{result['finished_utc']}", "",
             "本次由 `vulnerability_detection` 的 Cppcheck 适配层执行静态扫描；测试源树来自固定上游提交，"
             "每套工程均加入 60 个缺陷点与 60 个同类安全对照；正负标签仅保存在工作区外的真值工件。响应时延不属于本轮验收判定。", "",
             "## 范围与可复现性", ""]
    for project in result["projects"]:
        diagnostics = len(project["coverage"].get("diagnostics", []))
        lines.append(f"- {project['name']}：`{project['commit']}`，C/C++ 源码 {project['source_lines']:,} 行，扫描耗时 {project['elapsed_seconds']:.2f} 秒，"
                     f"覆盖状态 `{project['coverage']['status']}`（分析器诊断 {diagnostics} 条）。")
    lines.append(f"- 三套工程的平均风险检测时耗：**{result['average_elapsed_seconds']:.2f} 秒**（仅记录，不参与本轮阈值判定）。")
    lines += ["", "每类每工程 20 个注入点 + 20 个安全对照；合计每类 60 个正例与 60 个负例，"
              "共 180 个注入点、180 个对照。统计单位是单个函数范围内的目标 Cppcheck 诊断，"
              "无关诊断不参与本表。检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。", "", "## 指标", "", "| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category in manifest["categories"]:
        m = result["metrics"][category["id"]]
        lines.append(f"| {category['name']} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | {percent(m['detection_rate'])} | {percent(m['false_positive_rate'])} |")
    total = result["metrics"]["overall"]
    lines += [f"| 总体 | {total['tp']} | {total['fn']} | {total['fp']} | {total['tn']} | {percent(total['detection_rate'])} | {percent(total['false_positive_rate'])} |", "",
              "结论：本轮只依据检出率 >85%、误报率 <15% 判定。",
              f"**{'达到' if result['passes_quality_thresholds'] else '未达到'}注入点的质量阈值**；"
              "若工程覆盖状态为 `partial`，表示 Cppcheck 在默认编译配置下有未解析的工程诊断；"
              "它不影响已成功解析的注入文件统计，但不能据此声称已完整分析全部上游代码。", "",
              "## 工件", "", "- `artifacts/contract3/manifest.json`：上游版本、类别映射与协议。",
              "- `artifacts/contract3/cppcheck/ground_truth.json`：逐项文件、函数、行范围、类别与预期。",
              "- `artifacts/contract3/cppcheck/result.json`：逐工程原始计时、统计和匹配的检测结果。",
              "- 脚本每次校验上游 Git 提交；下载的上游源码与注入文件仅保留于脚本指定的工作目录。"]
    return "\n".join(lines) + "\n"


def run(manifest: dict[str, Any], workspace: Path, timeout: int, scan_scope: str) -> dict[str, Any]:
    from code_agent.security_analysis import cppcheck_scan

    started = utc_now()
    all_rows: list[dict[str, Any]] = []
    projects = []
    for spec in manifest["sources"]:
        project = clone_or_verify(spec, workspace)
        baseline = source_files(project, spec["scan_roots"])
        baseline_lines = loc(baseline)
        if baseline_lines < 10_000:
            raise ValueError(f"{spec['id']} has only {baseline_lines} C/C++ source lines after exclusions")
        fixture, cases = inject_fixture(project, spec["id"], manifest["categories"])
        targets = [fixture] if scan_scope == "fixtures" else [*baseline, fixture]
        before = time.perf_counter()
        findings, coverage = cppcheck_scan([str(p) for p in targets], str(project), 2, timeout_seconds=timeout)
        elapsed = time.perf_counter() - before
        if coverage["status"] in {"failed", "unavailable"}:
            raise RuntimeError(f"{spec['id']} analysis did not complete: {coverage}")
        normalized = [{**finding, "file": Path(finding["file"]).name} for finding in findings]
        rows = classify(cases, normalized, manifest["categories"])
        all_rows.extend(rows)
        projects.append({"id": spec["id"], "name": spec["name"], "commit": spec["commit"], "source_lines": baseline_lines,
                         "fixture_sha256": digest(fixture.read_bytes()), "elapsed_seconds": elapsed, "coverage": coverage,
                         "scan_scope": scan_scope, "scanned_files": len(targets)})
    by_category = {category["id"]: metric([r for r in all_rows if r["category"] == category["id"]]) for category in manifest["categories"]}
    overall = metric(all_rows)
    result = {"started_utc": started, "finished_utc": utc_now(), "cppcheck_version": subprocess.check_output([shutil.which("cppcheck") or "cppcheck", "--version"], text=True).strip(),
              "projects": projects, "metrics": {**by_category, "overall": overall}, "cases": all_rows,
              "average_elapsed_seconds": sum(project["elapsed_seconds"] for project in projects) / len(projects),
              "passes_quality_thresholds": all(m["detection_rate"] > .85 and m["false_positive_rate"] < .15 for m in by_category.values())}
    return result


def main() -> None:
    raise SystemExit(
        "The simple Contract-3 baseline has been retired. "
        "Run scripts/benchmark_contract3_independent.py --profile balanced instead."
    )


if __name__ == "__main__":
    main()
