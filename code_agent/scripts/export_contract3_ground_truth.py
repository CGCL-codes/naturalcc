"""Export the Contract-3 RTOS ground truth with the files each project is checked over.

For every suite and every pinned RTOS project this writes the upstream detection
files (path and line count), the injected fixture files and the per-function
labels into ``artifacts/contract3/ground_truth/<suite>.json``, and a readable
listing of every injected defect into ``合同-3-测试集与GT清单.md``. Fixtures are
regenerated deterministically with the same names the Web runner uses, so the
labels match ``benchmark_contract3_web.py`` line for line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import benchmark_contract3_cppcheck as base
from code_agent.scripts import benchmark_contract3_independent as independent

DATA = ROOT / "artifacts" / "contract3"
OUT = DATA / "ground_truth"
LISTING = ROOT / "合同-3-测试集与GT清单.md"
SUITES = ["balanced", "independent", "balanced-hard", "independent-hard"]
MIN_LINES = 10_000
SUITE_NAMES = {"balanced": "平衡验收集", "independent": "独立压力集",
               "balanced-hard": "平衡验收集（加难版）", "independent-hard": "独立压力集（加难版）"}
CATEGORY_NAMES = {"array_oob": "数组越界", "string_overflow": "字符串溢出", "null_pointer": "空指针调用"}
# What each injected form tests; the code itself comes from the generator below.
FORMS = {
    ("array_oob", "base", 0): "常量下标越界：`values[3]` 访问 3 元素数组",
    ("array_oob", "base", 1): "下标经局部变量传入后越界",
    ("array_oob", "base", 2): "循环边界写成 `<=`，最后一轮越界",
    ("array_oob", "base", 3): "下标经辅助函数 `contract3_identity` 返回后越界（跨函数）",
    ("array_oob", "base", 4): "指针算术 `values + 3` 后解引用越界",
    ("array_oob", "hard", 0): "守卫写成 `> 3`（应为 `>= 3`），差一越界",
    ("array_oob", "hard", 1): "倒序循环从下标 3 开始，差一越界",
    ("array_oob", "hard", 2): "用 `sizeof` 算出的元素个数直接当下标",
    ("array_oob", "hard", 3): "尾指针 `end[0]` 越界",
    ("array_oob", "hard", 4): "4 元素数组访问 `values[4]`",
    ("string_overflow", "base", 0): "`strcpy` 写入超过目标容量的字面量",
    ("string_overflow", "base", 1): "`sprintf` 写入超过目标容量的字符串",
    ("string_overflow", "base", 2): "`strcat` 拼接后超过目标容量",
    ("string_overflow", "base", 3): "`memcpy` 拷贝长度超过目标容量",
    ("string_overflow", "base", 4): "经辅助函数 `contract3_copy`（内部 `strcpy`）写入超长字符串（跨函数）",
    ("string_overflow", "hard", 0): "目标数组恰好少 1 字节，放不下结尾 `\\0`",
    ("string_overflow", "hard", 1): "`snprintf` 长度参数大于目标容量",
    ("string_overflow", "hard", 2): "`strncpy` 长度参数大于目标容量",
    ("string_overflow", "hard", 3): "`strcat` 拼接后恰好多 1 字节",
    ("string_overflow", "hard", 4): "`memcpy` 长度取源串 `sizeof`，大于目标容量",
    ("null_pointer", "base", 0): "指针初始化为 0 后直接解引用",
    ("null_pointer", "base", 1): "指针赋值为 0 后解引用",
    ("null_pointer", "base", 2): "把 0 作为实参传给会解引用的辅助函数 `contract3_read`（跨函数）",
    ("null_pointer", "base", 3): "结构体成员指针为 0 后解引用",
    ("null_pointer", "base", 4): "分支条件不成立，指针保持 0 后解引用",
    ("null_pointer", "hard", 0): "判空后未返回，继续解引用",
    ("null_pointer", "hard", 1): "条件表达式选出空指针后直接解引用",
    ("null_pointer", "hard", 2): "指针先指向有效对象，又被重新赋值为 0 后解引用",
    ("null_pointer", "hard", 3): "辅助函数 `contract3_select` 按参数返回空指针后解引用（跨函数）",
    ("null_pointer", "hard", 4): "判断条件写反，指针保持 0 后解引用",
}


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def case_form(suite: str, case: dict[str, Any]) -> tuple[str, int]:
    if "variant" in case:
        kind, variant = case["variant"].split("-")
        return kind, int(variant)
    index = int(case["case_id"].rsplit("-", 1)[1])
    variant = (index - 1) % 5 if suite == "independent" else independent.BALANCED_VARIANTS[case["category"]][index - 1]
    return "base", variant


def form_code(category: str, kind: str, variant: int, vulnerable: bool) -> str:
    generate = independent.hard_case_lines if kind == "hard" else independent.case_lines
    return "<br>".join(f"`{line.strip()}`" for line in generate(category, variant, vulnerable))


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_listing(results: list[dict[str, Any]]) -> str:
    first = results[0]
    lines = [
        "# 合同 3：RTOS 测试集与 Ground Truth 清单", "",
        "> 本文由 [scripts/export_contract3_ground_truth.py](scripts/export_contract3_ground_truth.py) 自动生成，请勿手改；"
        "重新生成：`uv run python scripts/export_contract3_ground_truth.py`。", "",
        "## 1. 概览", "",
        "- **工程**：3 套固定版本的公开 RTOS 源码。每套工程的“检测文件”合计不少于 10,000 行，见第 3 节。",
        "- **缺陷**：每套工程按类别（数组越界、字符串溢出、空指针调用）各注入一个测试文件，"
        "每个文件 40 个函数：20 个正例（真实缺陷）+ 20 个同类安全对照。上游源码本身不改动。",
        "- **测试集**：" + "、".join(f"`{r['suite']}`（{SUITE_NAMES[r['suite']]}）" for r in results) +
        "；每套 3 工程 × 3 类 × 40 = 360 个函数。各套的区别只在注入的写法（第 2 节）和函数顺序（加难版打乱）。",
        "- **计分**：只有注入函数参与计分。正例的告警落在函数行范围内（加难版按函数名）记 TP，否则 FN；对照被告警记 FP，否则 TN。"
        "上游源码文件里的告警不计分。",
        "- **机器可读 GT**：`artifacts/contract3/ground_truth/<测试集>.json`，字段见第 5 节。", "",
        "| 测试集 | GT 文件 |", "|---|---|",
        *[f"| {SUITE_NAMES[r['suite']]} | [ground_truth/{r['suite']}.json](artifacts/contract3/ground_truth/{r['suite']}.json) |" for r in results], "",
        "## 2. 缺陷写法（正例与对照）", "",
        "“正例代码”与“对照代码”是注入函数的函数体；对照与正例写法相近，但实际安全。"
        "`base-N` 用于平衡集、压力集，加难版另加 `hard-N`（差一错误与看似危险的安全写法）。", "",
    ]
    for category, name in CATEGORY_NAMES.items():
        lines += [f"### {name}（`{category}`）", "", "| 写法 | 测试点 | 正例代码 | 对照代码 |", "|---|---|---|---|"]
        for kind in ("base", "hard"):
            for variant in range(5):
                lines.append(f"| `{kind}-{variant}` | {FORMS[(category, kind, variant)]} | "
                             f"{cell(form_code(category, kind, variant, True))} | {cell(form_code(category, kind, variant, False))} |")
        lines.append("")
    lines += ["## 3. 工程与检测文件", "",
              "检测文件 = `scan_roots` 下的 C/C++ 源码与头文件（不含注入的测试文件）。完整文件列表与逐文件行数见下方折叠项及 GT JSON 的 `detection_files`。", "",
              "| 工程 | 版本 | 提交 | 扫描范围 | 检测文件数 | 检测文件合计行数 |", "|---|---|---|---|---:|---:|"]
    for project in first["projects"]:
        lines.append(f"| {project['name']} | {project['ref']} | `{project['commit'][:12]}` | "
                     f"{', '.join(f'`{root}`' for root in project['scan_roots'])} | {len(project['detection_files'])} | "
                     f"{project['line_totals']['upstream_detection_files']:,} |")
    lines.append("")
    for project in first["projects"]:
        lines += [f"<details><summary>{project['name']} 检测文件（{len(project['detection_files'])} 个）</summary>", "",
                  "| 文件 | 行数 |", "|---|---:|", *[f"| `{item['path']}` | {item['lines']:,} |" for item in project["detection_files"]],
                  "", "</details>", ""]
    lines += ["## 4. 逐例 Ground Truth", "",
              "测试文件放在各工程根目录。“缺陷行”是正例中触发缺陷的语句；对照列出的是同一位置的安全语句。", ""]
    for result in results:
        suite = result["suite"]
        lines += [f"### {SUITE_NAMES[suite]}（`{suite}`）", ""]
        for project in result["projects"]:
            for fixture in project["fixtures"]:
                rows = [case for case in project["cases"] if case["file"] == fixture["path"]]
                defects = sum(case["expected"] == "defect" for case in rows)
                lines += [f"<details><summary>{project['name']} · {CATEGORY_NAMES[fixture['category']]} · "
                          f"<code>{fixture['path']}</code>（{fixture['lines']} 行，正例 {defects} / 对照 {len(rows) - defects}）</summary>", "",
                          "| 用例 | 标签 | 函数 | 行范围 | 缺陷行 | 写法 | 缺陷行代码 |", "|---|---|---|---|---:|---|---|"]
                for case in rows:
                    kind, variant = case_form(suite, case)
                    lines.append(f"| `{case['case_id']}` | {'**正例**' if case['expected'] == 'defect' else '对照'} | "
                                 f"`{case['function']}` | {case['line_start']}–{case['line_end']} | {case['defect_line']} | "
                                 f"`{kind}-{variant}` | `{cell(case['defect_code'])}` |")
                lines += ["", "</details>", ""]
    lines += ["## 5. GT JSON 字段", "",
              "| 字段 | 含义 |", "|---|---|",
              "| `projects[].detection_files[]` | 要检测的上游源码文件：`path`（相对工程根目录）、`lines` |",
              "| `projects[].line_totals` | `upstream_detection_files` 检测文件合计行数；`fixtures` 测试文件行数；`all_detection_files` 两者之和 |",
              "| `projects[].fixtures[]` | 注入的测试文件：`path`、`category`、`lines`、`sha256` |",
              "| `projects[].cases[]` | 逐函数标签：`case_id`、`category`、`expected`（`defect`/`control`）、`function`、`file`、"
              "`line_start`、`line_end`、`defect_line`、`defect_code`；加难版另有 `variant` |", ""]
    return "\n".join(lines)


def export_suite(manifest: dict[str, Any], workspace: Path, suite: str) -> dict[str, Any]:
    projects = []
    for spec in manifest["sources"]:
        project = base.clone_or_verify(spec, workspace)
        upstream = sorted(base.source_files(project, spec["scan_roots"]))
        detect_files = [{"path": path.relative_to(project).as_posix(), "lines": line_count(path)} for path in upstream]
        upstream_lines = sum(item["lines"] for item in detect_files)
        if upstream_lines < MIN_LINES:
            raise ValueError(f"{spec['id']} detection files total {upstream_lines} lines, fewer than {MIN_LINES}")
        fixtures, cases = [], []
        for category in manifest["categories"]:
            fixture, category_cases = independent.inject_fixture(
                project, spec["id"], [category], fixture_name=f"contract3_{suite.replace('-', '_')}_web_{category['id']}.c", profile=suite)
            fixtures.append({"path": fixture.name, "category": category["id"], "lines": line_count(fixture),
                             "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()})
            source = fixture.read_text(encoding="utf-8").splitlines()
            cases.extend({**case, "defect_code": source[case["defect_line"] - 1].strip()} for case in category_cases)
        fixture_lines = sum(item["lines"] for item in fixtures)
        projects.append({
            "id": spec["id"], "name": spec["name"], "url": spec["url"], "ref": spec["ref"], "commit": spec["commit"],
            "scan_roots": spec["scan_roots"],
            "line_totals": {"upstream_detection_files": upstream_lines, "fixtures": fixture_lines,
                            "all_detection_files": upstream_lines + fixture_lines},
            "fixtures": fixtures,
            "detection_files": detect_files,
            "cases": cases,
        })
    return {
        "suite": suite,
        "title": "合同指标 3：RTOS 注入测试集 ground truth",
        "line_rule": f"每个工程的上游检测文件（detection_files）合计不少于 {MIN_LINES} 行；行数按文件实际行数统计。",
        "scoring_rule": "只有 fixtures 中的注入函数参与计分：expected=defect 为正例、expected=control 为同类安全对照，"
                        "告警须落在 [line_start, line_end] 内或给出对应函数名。上游文件的告警不计分。",
        "projects": projects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/naturalcc-contract3-web-sources"))
    parser.add_argument("--suite", default=",".join(SUITES))
    args = parser.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    suites = args.suite.split(",")
    results = []
    for suite in suites:
        if suite not in SUITES:
            raise SystemExit(f"unknown suite: {suite}")
        result = export_suite(manifest, args.workspace.resolve(), suite)
        results.append(result)
        target = OUT / f"{suite}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        totals = ", ".join(f"{p['id']} {p['line_totals']['upstream_detection_files']} 行/{len(p['detection_files'])} 文件"
                           for p in result["projects"])
        print(f"{suite}: {target.relative_to(ROOT)} ({totals})")
    if suites == SUITES:
        LISTING.write_text(render_listing(results) + "\n", encoding="utf-8")
        print(f"listing: {LISTING.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
