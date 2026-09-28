"""Render the single canonical Contract-3 report from result artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "contract3"
REPORT = ROOT / "合同-3-result.md"
NAMES = {"array_oob": "数组越界", "string_overflow": "字符串溢出", "null_pointer": "空指针调用"}


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def direct_table(result: dict[str, Any]) -> list[str]:
    lines = ["| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, name in NAMES.items():
        value = result["metrics"][category]
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(value['detection_rate'])} | {percent(value['false_positive_rate'])} |")
    value = result["metrics"]["overall"]
    lines.append(f"| 总体 | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(value['detection_rate'])} | {percent(value['false_positive_rate'])} |")
    return lines


def direct_scope(result: dict[str, Any]) -> list[str]:
    return [f"- {item['name']}：`{item['commit']}`，{item['source_lines']:,} 行，{item['elapsed_seconds']:.2f} 秒，覆盖 `{item['coverage']['status']}`。" for item in result["projects"]]


def project_metrics(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    tp = sum(row["expected"] == "defect" and row["detected"] for row in rows)
    fn = sum(row["expected"] == "defect" and not row["detected"] for row in rows)
    fp = sum(row["expected"] == "control" and row["detected"] for row in rows)
    tn = sum(row["expected"] == "control" and not row["detected"] for row in rows)
    return tp, fn, fp, tn


def direct_project_table(result: dict[str, Any]) -> list[str]:
    lines = ["| 工程 | TP | FN | FP | TN | 检出率 | 误报率 | 覆盖 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for project in result["projects"]:
        rows = [row for row in result["cases"] if row["project"] == project["id"]]
        tp, fn, fp, tn = project_metrics(rows)
        lines.append(f"| {project['name']} | {tp} | {fn} | {fp} | {tn} | {percent(tp / (tp + fn))} | {percent(fp / (fp + tn))} | {project['coverage']['status']} |")
    return lines


def web_metrics(checkpoint: dict[str, Any]) -> dict[str, int]:
    rows = [row for record in checkpoint.get("samples", {}).values() if record.get("status") == "scored" for row in (record.get("rows") or [])]
    return {"n": len(rows), "tp": sum(row["expected"] == "defect" and row["detected"] for row in rows),
            "fn": sum(row["expected"] == "defect" and not row["detected"] for row in rows),
            "fp": sum(row["expected"] == "control" and row["detected"] for row in rows),
            "tn": sum(row["expected"] == "control" and not row["detected"] for row in rows)}


def web_section(title: str, checkpoint: dict[str, Any] | None, artifact: str) -> list[str]:
    lines = [f"### {title}", ""]
    if not checkpoint:
        return lines + ["尚未运行；启动 Web 服务后执行对应 `--suite`。"]
    value = web_metrics(checkpoint)
    scored = sum(record.get("status") == "scored" for record in checkpoint.get("samples", {}).values())
    return lines + ["Web 模式与合同 5 一样，固定 OpenRouter/Sonnet 配置，经 `/api/agent/threads` 与 Run API 执行；超时、模型失败和无效 JSON 不计为安全或正确。",
                    f"- 状态：{checkpoint.get('stop_reason', '未知')}。",
                    f"- 已有效评分：{scored}/9 组，{value['n']} 个函数；TP/FN/FP/TN：{value['tp']}/{value['fn']}/{value['fp']}/{value['tn']}。",
                    f"- 工件：`artifacts/contract3/{artifact}/`。"]


def web_summary(checkpoint: dict[str, Any] | None) -> tuple[str, str]:
    if not checkpoint:
        return "尚未运行", "—"
    value = web_metrics(checkpoint)
    scored = sum(record.get("status") == "scored" for record in checkpoint.get("samples", {}).values())
    detection = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else None
    fpr = value["fp"] / (value["fp"] + value["tn"]) if value["fp"] + value["tn"] else None
    return f"{scored}/9 组有效评分；{checkpoint.get('stop_reason', '未知')}", f"{percent(detection)} / {percent(fpr)}"


def web_progress(checkpoint: dict[str, Any] | None) -> list[str]:
    if not checkpoint:
        return []
    lines = ["", "| 工程×类别 | 状态 | 已评分函数 | TP/FN/FP/TN | 最近 Run |", "|---|---|---:|---|---|"]
    for sample_id, record in sorted(checkpoint.get("samples", {}).items()):
        rows = (record.get("rows") or []) if record.get("status") == "scored" else []
        tp, fn, fp, tn = project_metrics(rows)
        attempt = (record.get("attempts") or [{}])[-1]
        lines.append(f"| {sample_id} | {record.get('status', '未开始')} | {len(rows)} | {tp}/{fn}/{fp}/{tn} | `{attempt.get('run_id', '—')}` |")
    return lines


def web_rows(checkpoint: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not checkpoint:
        return []
    return [row for record in checkpoint.get("samples", {}).values() if record.get("status") == "scored"
            for row in (record.get("rows") or [])]


def counted_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp, fn, fp, tn = project_metrics(rows)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def meets_requirement(value: dict[str, int]) -> bool:
    detection = rate(value["tp"], value["tp"] + value["fn"])
    false_positive = rate(value["fp"], value["fp"] + value["tn"])
    return detection is not None and false_positive is not None and detection > .85 and false_positive < .15


def web_category_table(checkpoint: dict[str, Any] | None) -> list[str]:
    rows = web_rows(checkpoint)
    lines = ["| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category, name in NAMES.items():
        value = counted_metrics([row for row in rows if row["category"] == category])
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    value = counted_metrics(rows)
    lines.append(f"| 总体 | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    return lines


def web_project_table(checkpoint: dict[str, Any] | None) -> list[str]:
    rows = web_rows(checkpoint)
    labels = {"freertos_kernel": "FreeRTOS Kernel", "zephyr": "Zephyr", "rt_thread": "RT-Thread"}
    lines = ["| 工程 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for project, name in labels.items():
        value = counted_metrics([row for row in rows if row["project"] == project])
        lines.append(f"| {name} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {percent(rate(value['tp'], value['tp'] + value['fn']))} | {percent(rate(value['fp'], value['fp'] + value['tn']))} |")
    return lines


def render() -> str:
    manifest = load(DATA / "manifest.json") or {}
    balanced = load(DATA / "balanced-cppcheck" / "result.json")
    independent = load(DATA / "independent-cppcheck" / "result.json")
    web_balanced = load(DATA / "web-sonnet45-balanced" / "checkpoint.json")
    web_independent = load(DATA / "web-sonnet45-independent" / "checkpoint.json")
    lines = ["# 合同指标 3：高频缺陷检测总报告", "", f"更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
             "本报告是唯一的指标 3 Markdown 入口。各测试集的真值、原始结果、模型 Run 和覆盖状态保留在各自工件目录；不得择优挑选或平均不同测试集的分数。", "",
             "## 判定口径", "", "每套工程针对数组越界、字符串溢出、空指针调用各有 20 个正例和 20 个安全对照。检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。只计目标类别且行号落在冻结函数范围内的发现。", ""]
    lines += ["## 当前结果总览", "", "| 测试集 / 运行方式 | 完成状态 | 检出率 / 误报率 | 定位 |", "|---|---|---|---|"]
    balanced_local = "—" if not balanced else f"{percent(balanced['metrics']['overall']['detection_rate'])} / {percent(balanced['metrics']['overall']['false_positive_rate'])}"
    independent_local = "—" if not independent else f"{percent(independent['metrics']['overall']['detection_rate'])} / {percent(independent['metrics']['overall']['false_positive_rate'])}"
    balanced_web_status, balanced_web_rate = web_summary(web_balanced)
    independent_web_status, independent_web_rate = web_summary(web_independent)
    lines += [f"| 平衡验收集 Web | {balanced_web_status} | {balanced_web_rate} | Web API 验收链路 |",
              f"| 平衡验收集本地 | {'已完成' if balanced else '未保留'} | {balanced_local} | 推荐本地验收口径 |",
              f"| 独立压力集 Web | {independent_web_status} | {independent_web_rate} | 能力边界 Web 链路 |",
              f"| 独立压力集本地 | {'已完成' if independent else '未保留'} | {independent_local} | 能力边界对照 |", ""]
    if manifest:
        lines += ["## 冻结测试来源", ""]
        for source in manifest.get("sources", []):
            lines.append(f"- {source['name']}：`{source['url']}`，固定提交 `{source['commit']}`，扫描根目录 `{', '.join(source['scan_roots'])}`。")
        lines += ["", "测试 fixture 只写入临时上游 checkout，不修改或提交上游源码；真值 JSON 不进入 Web Agent 工作区。", ""]
    if balanced or web_balanced:
        lines += ["## 平衡验收集（推荐验收口径）", "", "该集混合计算下标、循环、格式化/复制和分支指针形态；每类每工程仍固定 20 正例 + 20 对照。它是推荐的本地检测指标证据，但不是独立泛化分数。", ""]
        if web_balanced:
            lines += web_section("平衡验收集 Web", web_balanced, "web-sonnet45-balanced") + web_progress(web_balanced) + [""]
            lines += web_category_table(web_balanced) + [""] + web_project_table(web_balanced) + [""]
            web_balanced_passed = all(
                meets_requirement(counted_metrics([row for row in web_rows(web_balanced) if row["category"] == category]))
                for category in NAMES)
            lines += [f"平衡验收集 Web：**{'符合' if web_balanced_passed else '不符合'}**各类别“检出率 >85%、误报率 <15%”的要求。", ""]
        if balanced:
            values = balanced["metrics"]
            passed = all(values[category]["detection_rate"] > .85 and values[category]["false_positive_rate"] < .15 for category in NAMES)
            lines += ["### 平衡验收集本地", ""] + direct_scope(balanced) + [""] + direct_table(balanced) + [""] + direct_project_table(balanced) + ["", f"结论：**{'符合' if passed else '不符合'}**本轮“检出率 >85%、误报率 <15%”的检测统计要求。", "FreeRTOS 与 RT-Thread 的 `partial` 表示默认 Cppcheck 配置有工程解析诊断；注入文件统计有效，但不能据此声称完整覆盖全部上游工程。", ""]
    if independent or web_independent:
        lines += ["## 独立复杂压力集（能力边界）", "", "该集用于暴露跨辅助函数、结构体指针、指针算术和复杂字符串流的边界；不能替换、平均或择优合并平衡验收集。", ""]
        if web_independent:
            lines += web_section("独立压力集 Web", web_independent, "web-sonnet45-independent") + web_progress(web_independent) + [""]
            lines += web_category_table(web_independent) + [""] + web_project_table(web_independent) + [""]
            web_values = web_metrics(web_independent)
            category_passed = all(
                meets_requirement(counted_metrics([row for row in web_rows(web_independent) if row["category"] == category]))
                for category in NAMES)
            overall_rate = percent(rate(web_values["tp"], web_values["tp"] + web_values["fn"]))
            lines += [f"独立压力集 Web 的总体检出率为 {overall_rate}，{'符合' if category_passed else '未达到'}各类别均 >85%、误报率 <15% 的验收口径；该集仅用于记录复杂形态下的能力边界。", ""]
        if independent:
            lines += ["### 独立压力集本地", ""] + direct_table(independent) + [""] + direct_project_table(independent) + [""]
    lines += ["## 测试链路与适用边界", "", "- 本地：通过产品 `security_analysis.cppcheck_scan` 适配层调用 Cppcheck；不执行目标程序。", "- Web：与合同 5 共用本机 Web Agent API、线程、Run、events 和持久化引擎；固定 OpenRouter/Sonnet 配置，无模型 fallback。", "- Web 运行按固定顺序串行执行，避免并发的限流、超时重试及账单归因干扰；未完成、超时、API 故障或无效 JSON 不计为正确。", "- 平衡验收集证明规定注入形态下的检测统计；独立压力集展示未覆盖形态的边界。两者均不能单独证明完整上游工程覆盖、所有路径安全或增量分析性能。", "", "## 运行与续跑", "", "```bash", "# 推荐本地验收", "uv run python scripts/benchmark_contract3_independent.py --profile balanced --workspace /tmp/naturalcc-contract3-independent-sources", "", "# Web 模式：先按 README 启动 agent_web_api.py，再在另一终端执行", "uv run python scripts/benchmark_contract3_web.py --suite balanced", "uv run python scripts/benchmark_contract3_web.py --suite independent", "", "# 不发模型请求，仅从已有原始工件刷新本报告", "uv run python scripts/render_contract3_report.py", "```", "",
              "## 工件索引", "", "- `artifacts/contract3/manifest.json`：固定上游版本与类别映射。", "- `artifacts/contract3/balanced-cppcheck/`：推荐验收集真值及原始结果。", "- `artifacts/contract3/independent-cppcheck/`：独立压力集真值及原始结果。", "- `artifacts/contract3/web-sonnet45-balanced/`、`web-sonnet45-independent/`：Web checkpoint、结构化预测与 Run 证据。"]
    return "\n".join(lines) + "\n"


def write_report() -> Path:
    REPORT.write_text(render(), encoding="utf-8")
    return REPORT


if __name__ == "__main__":
    print(write_report())
