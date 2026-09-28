"""Independent, more varied Contract-3 C examples through the product Cppcheck adapter.

This held-out suite intentionally uses different data-flow shapes from the
baseline fixture: computed indexes and loop bounds, format/concatenation/copy
APIs, and pointer flow through branches, helpers and structs.  It never changes
the detector and keeps labels outside the temporary RTOS workspaces.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import benchmark_contract3_cppcheck as base

DATA = ROOT / "artifacts" / "contract3" / "independent-cppcheck"
REPORT = ROOT / "合同-3-独立样例-result.md"

# The balanced profile is a formal acceptance-oriented mix, not an independent
# generalization score.  The stress profile remains the default held-out suite.
BALANCED_VARIANTS = {
    "array_oob": [0, 1, 2, 3] * 4 + [0, 1, 4, 4],
    "string_overflow": [0, 1, 3] * 6 + [2, 4],
    "null_pointer": [0, 1, 4] * 6 + [2, 3],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def case_lines(category: str, variant: int, vulnerable: bool) -> list[str]:
    """Return a genuine defect or a near-neighbour safe control; no label text."""
    if category == "array_oob":
        if variant == 0:
            return ["    int values[3] = {1, 2, 3};", f"    return values[{3 if vulnerable else 2}];"]
        if variant == 1:
            return ["    int values[3] = {1, 2, 3};", f"    int index = {3 if vulnerable else 2};", "    return values[index];"]
        if variant == 2:
            bound = "<=" if vulnerable else "<"
            return ["    int values[3] = {1, 2, 3};", "    int total = 0;",
                    f"    for (int index = 0; index {bound} 3; ++index) total += values[index];", "    return total;"]
        if variant == 3:
            return ["    int values[3] = {1, 2, 3};", f"    int index = contract3_identity({3 if vulnerable else 2});", "    return values[index];"]
        return ["    int values[3] = {1, 2, 3};", f"    int *cursor = values + {3 if vulnerable else 2};", "    return *cursor;"]
    if category == "string_overflow":
        if variant == 0:
            return ["    char target[4] = {0};", f"    strcpy(target, \"{'ABCDE' if vulnerable else 'ABC'}\");", "    return target[0];"]
        if variant == 1:
            return ["    char target[4] = {0};", f"    sprintf(target, \"%s\", \"{'ABCDE' if vulnerable else 'ABC'}\");", "    return target[0];"]
        if variant == 2:
            return ["    char target[8] = \"A\";", f"    strcat(target, \"{'BCDEFGH' if vulnerable else 'BC'}\");", "    return target[0];"]
        if variant == 3:
            size = 6 if vulnerable else 3
            return ["    char target[4] = {0};", f"    memcpy(target, \"ABCDE\", {size});", "    return target[0];"]
        return ["    char target[4] = {0};", f"    contract3_copy(target, \"{'ABCDE' if vulnerable else 'ABC'}\");", "    return target[0];"]
    if category == "null_pointer":
        if variant == 0:
            return ["    int *pointer = 0;" if vulnerable else "    int value = 7;", "    return *pointer;" if vulnerable else "    return value;"]
        if variant == 1:
            return ["    int value = 7;", f"    int *pointer = {'0' if vulnerable else '&value'};", "    return *pointer;"]
        if variant == 2:
            return ["    int value = 7;", f"    return contract3_read({'0' if vulnerable else '&value'});"]
        if variant == 3:
            return ["    int value = 7;", f"    struct contract3_box box = {{{'0' if vulnerable else '&value'}}};", "    return *box.pointer;"]
        return ["    int value = 7;", "    int *pointer = 0;", "    if (value > 0) pointer = &value;" if not vulnerable else "    if (value < 0) pointer = &value;", "    return *pointer;"]
    raise ValueError(f"Unknown category: {category}")


def inject_fixture(project: Path, source_id: str, categories: list[dict[str, Any]], *, fixture_name: str = "contract3_independent_examples.c", profile: str = "independent") -> tuple[Path, list[dict[str, Any]]]:
    fixture = project / fixture_name
    lines = ["/* Test-only independent Contract-3 fixture. Do not include in production builds. */",
             "#include <stdio.h>", "#include <string.h>", "",
             "struct contract3_box { int *pointer; };", "",
             "static int contract3_identity(int value) { return value; }",
             "static int contract3_read(int *pointer) { return *pointer; }",
             "static void contract3_copy(char *target, const char *source) { strcpy(target, source); }", ""]
    cases: list[dict[str, Any]] = []
    if profile not in {"independent", "balanced"}:
        raise ValueError(f"Unknown profile: {profile}")
    for category in categories:
        for vulnerable in (True, False):
            for index in range(1, 21):
                ordinal = index if vulnerable else index + 20
                function = f"contract3_independent_{category['id']}_case_{ordinal:02d}"
                start = len(lines) + 1
                lines.extend([f"static int {function}(void)", "{"])
                variant = (index - 1) % 5 if profile == "independent" else BALANCED_VARIANTS[category["id"]][index - 1]
                body = case_lines(category["id"], variant, vulnerable)
                lines.extend(body)
                lines.append("}")
                end = len(lines)
                cases.append({"case_id": f"{source_id}-{category['id']}-{'positive' if vulnerable else 'control'}-{index:02d}",
                              "project": source_id, "category": category["id"],
                              "expected": "defect" if vulnerable else "control", "function": function,
                              "line_start": start, "line_end": end, "defect_line": end - 1,
                              "file": fixture.name})
                lines.append("")
    fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fixture, cases


def render_report(result: dict[str, Any], manifest: dict[str, Any], profile: str) -> str:
    if profile == "balanced":
        title = "合同指标 3：平衡验收样例实测结果"
        introduction = ("本报告是比旧基线更多形态的平衡验收集。它混合直接与计算下标/循环、格式化与复制、"
                        "分支指针等形态；正负标签仅在工作区外真值 JSON 中。该固定混合是验收协议，"
                        "不是独立泛化分数，也不能与独立压力集择优取高。")
    else:
        title = "合同指标 3：独立复杂样例实测结果"
        introduction = ("本报告是基线注入集之外的独立复核。没有改动产品规则或根据本轮结果调参；"
                        "正负标签仅在工作区外的真值 JSON 中。样例使用计算下标/循环边界/指针算术、"
                        "格式化与拼接/复制 API、分支/辅助函数/结构体指针等不同形态。")
    lines = [f"# {title}", "", f"更新时间（UTC）：{result['finished_utc']}", "", introduction, "",
             "## 工程与覆盖", ""]
    for project in result["projects"]:
        lines.append(f"- {project['name']}：`{project['commit']}`，{project['source_lines']:,} 行，"
                     f"{project['elapsed_seconds']:.2f} 秒，覆盖 `{project['coverage']['status']}`。")
    lines += ["", "## 原始指标", "", "| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category in manifest["categories"]:
        metric = result["metrics"][category["id"]]
        lines.append(f"| {category['name']} | {metric['tp']} | {metric['fn']} | {metric['fp']} | {metric['tn']} | "
                     f"{base.percent(metric['detection_rate'])} | {base.percent(metric['false_positive_rate'])} |")
    total = result["metrics"]["overall"]
    lines += [f"| 总体 | {total['tp']} | {total['fn']} | {total['fp']} | {total['tn']} | "
              f"{base.percent(total['detection_rate'])} | {base.percent(total['false_positive_rate'])} |", "",
              "统计单位为目标函数的行范围：检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。"
              "目标类别以外及函数范围外诊断不参与评分。该独立集仅检验注入 fixture，"
              "不表示已完整覆盖三个 RTOS 工程。", "",
              "## 工件", "", f"- `artifacts/contract3/{DATA.name}/ground_truth.json`：冻结真值。",
              f"- `artifacts/contract3/{DATA.name}/result.json`：逐项 Cppcheck 结果与统计。",
              "- `scripts/benchmark_contract3_independent.py`：可复现脚本；上游源码只在指定临时目录写入测试 fixture。"]
    return "\n".join(lines) + "\n"


def run(manifest: dict[str, Any], workspace: Path, timeout: int, profile: str) -> dict[str, Any]:
    from code_agent.security_analysis import cppcheck_scan

    all_rows, projects = [], []
    for spec in manifest["sources"]:
        project = base.clone_or_verify(spec, workspace)
        source_files = base.source_files(project, spec["scan_roots"])
        source_lines = base.loc(source_files)
        if source_lines < 10_000:
            raise ValueError(f"{spec['id']} has fewer than 10,000 C/C++ source lines")
        fixture, cases = inject_fixture(project, spec["id"], manifest["categories"], profile=profile)
        started = time.perf_counter()
        findings, coverage = cppcheck_scan([str(path) for path in [*source_files, fixture]], str(project), 2, timeout_seconds=timeout)
        elapsed = time.perf_counter() - started
        if coverage["status"] in {"failed", "unavailable"}:
            raise RuntimeError(f"{spec['id']} analysis did not complete: {coverage}")
        normalized = [{**finding, "file": Path(finding["file"]).name} for finding in findings]
        all_rows.extend(base.classify(cases, normalized, manifest["categories"]))
        projects.append({"id": spec["id"], "name": spec["name"], "commit": spec["commit"], "source_lines": source_lines,
                         "fixture_sha256": base.digest(fixture.read_bytes()), "elapsed_seconds": elapsed, "coverage": coverage})
    category_metrics = {category["id"]: base.metric([row for row in all_rows if row["category"] == category["id"]]) for category in manifest["categories"]}
    return {"finished_utc": now(), "cppcheck_version": subprocess.check_output([shutil.which("cppcheck") or "cppcheck", "--version"], text=True).strip(),
            "projects": projects, "metrics": {**category_metrics, "overall": base.metric(all_rows)}, "cases": all_rows,
            "average_elapsed_seconds": sum(project["elapsed_seconds"] for project in projects) / len(projects)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/naturalcc-contract3-independent-sources"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--profile", choices=("independent", "balanced"), default="independent")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    global DATA, REPORT
    if args.profile == "balanced":
        DATA = ROOT / "artifacts" / "contract3" / "balanced-cppcheck"
        REPORT = ROOT / "合同-3-平衡验收-result.md"
    manifest = base.load_manifest()
    if args.report_only:
        result = json.loads((DATA / "result.json").read_text(encoding="utf-8"))
    else:
        if shutil.which("cppcheck") is None:
            raise SystemExit("cppcheck is required on PATH")
        result = run(manifest, args.workspace.resolve(), args.timeout, args.profile)
        write_json(DATA / "ground_truth.json", [{key: value for key, value in row.items() if key not in {"detected", "matched_findings"}} for row in result["cases"]])
        write_json(DATA / "result.json", result)
    from code_agent.scripts.render_contract3_report import write_report
    print(write_report())


if __name__ == "__main__":
    main()
