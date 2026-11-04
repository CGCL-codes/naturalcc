"""Contract-5 whole-project evaluation through the local Web Agent HTTP API.

Dataset: ``artifacts/contract5/rtthread/dataset`` (RT-Thread v5.2.1 subset with
10 injected defects per category and safe controls; see
``scripts/build_contract5_dataset.py``).  One Thread + Run per ground-truth
target file; the Agent workspace is the whole injected project and ground truth
stays outside it.  Scoring is per injected/control site (function match or line
range) plus a per-file verdict.

Start ``agent_web_api.py`` with the OpenRouter runtime default matching
``--model``, export ``OPENROUTER_API_KEY``, then run this script.  Without
credits use ``--print-prompts`` (no network) or ``--report-only``.
No direct inference, no model fallback, no source execution or edits.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

try:
    from .check_openrouter_credits import check_credits
except ImportError:  # executed as a script
    from check_openrouter_credits import check_credits

DATA = ROOT / "artifacts" / "contract5" / "rtthread"
DATASET = DATA / "dataset"
PROJECT = DATASET / "project"
GROUND_TRUTH = DATASET / "ground_truth.json"
MANIFEST = DATASET / "manifest.json"
PROMPT_FILE = DATA / "prompt.md"
REPORT = ROOT / "合同-5-result.md"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
CATEGORIES = {"buffer_overflow": "缓冲区溢出", "data_race": "多线程竞争",
              "memory_leak": "内存泄漏", "command_execution": "命令执行漏洞"}
CWE_TO_CATEGORY = {787: "buffer_overflow", 120: "buffer_overflow", 121: "buffer_overflow", 122: "buffer_overflow", 119: "buffer_overflow",
                   362: "data_race", 366: "data_race", 367: "data_race",
                   401: "memory_leak", 772: "memory_leak",
                   78: "command_execution", 77: "command_execution"}
DIFFICULTIES = ("easy", "medium", "hard")
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
# Target files reach ~138k characters, so a full read is ~40k tokens and the
# cumulative input across a run needs far more headroom than a single call.
BUDGET = {"max_llm_calls": 14, "max_tool_calls": 36, "max_input_tokens": 600000,
          "max_output_tokens": 24000, "max_seconds": 1800, "max_compaction_calls": 8}
PRODUCT_FILES = ("security_review.py", "security_c_rules.py", "security_rtos_rules.py",
                 "security_analysis.py", "security_contracts.py",
                 "plugins/vulnerability_detection.py", "agent_core/tools/pipeline.py")
OUT = DATA / "web-sonnet45"  # replaced per --model in main()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_text(text: str) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return text.replace(key, "[REDACTED]") if key else text


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(safe_text(text), encoding="utf-8")
    temp.replace(path)


def save_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def prompt_template() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    match = re.search(r"^<!-- prompt -->[ \t]*\n(.*?)^<!-- /prompt -->", text, re.S | re.M)
    if not match or "{file}" not in match.group(1):
        raise ValueError("prompt.md must contain a <!-- prompt --> block with a {file} placeholder")
    return match.group(1).strip()


def goal(file: str) -> str:
    return prompt_template().replace("{file}", file)


def load_ground_truth() -> dict[str, Any]:
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(GROUND_TRUTH.read_bytes()) != manifest["ground_truth_sha256"]:
        raise ValueError("ground_truth.json does not match manifest.json")
    return gt


def verify_project(gt: dict[str, Any]) -> None:
    """Refuse to run on a project whose files differ from the frozen manifest."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {row["file"]: row["sha256"] for row in manifest["project_files"]}
    for relative, sha in expected.items():
        path = PROJECT / relative
        if not path.exists() or digest(path.read_bytes()) != sha:
            raise ValueError(f"project file changed or missing: {relative}; rebuild with build_contract5_dataset.py")
    for row in gt["target_files"]:
        if row["sha256"] != digest((PROJECT / row["file"]).read_bytes()):
            raise ValueError(f"target file hash mismatch: {row['file']}")


def ordered_targets(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed order: alternate vulnerable and clean files, otherwise by path."""
    vulnerable = sorted((f for f in gt["target_files"] if f["verdict"] == "vulnerable"), key=lambda f: f["file"])
    clean = sorted((f for f in gt["target_files"] if f["verdict"] == "clean"), key=lambda f: f["file"])
    ordered: list[dict[str, Any]] = []
    while vulnerable or clean:
        if vulnerable:
            ordered.append(vulnerable.pop(0))
        if clean:
            ordered.append(clean.pop(0))
    return ordered


def _balanced_object(text: str) -> str:
    """Return the first brace-balanced JSON object in text, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object in answer")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("Unterminated JSON object in answer")


def extract_json(answer: str) -> str:
    """Tolerate fences, a stray ``json`` language tag and surrounding prose; never guess content."""
    text = answer.strip()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    if len(fenced) > 1:
        raise ValueError("Ambiguous multiple JSON answers")
    if fenced:
        text = fenced[0].strip()
    # A stripped fence can leave the bare language tag on its own first line.
    text = re.sub(r"\A(?:json|JSON)\s*\n", "", text).strip()
    return text if text.startswith("{") else _balanced_object(text)


def parse_answer(answer: str, line_count: int) -> dict[str, Any]:
    value = json.loads(extract_json(answer))
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        raise ValueError("Missing findings array")
    if not isinstance(value.get("summary"), str):
        raise ValueError("Missing summary")
    for finding in value["findings"]:
        if not isinstance(finding, dict):
            raise ValueError("Invalid finding")
        category = finding.get("category")
        if category not in CATEGORIES:
            cwe = finding.get("cwe")
            if type(cwe) is int and cwe in CWE_TO_CATEGORY and category is None:
                finding["category"] = CWE_TO_CATEGORY[cwe]
            else:
                raise ValueError("Out-of-scope category")
        if type(finding.get("line")) is not int or not 1 <= finding["line"] <= line_count:
            raise ValueError("Invalid source line")
        if "function" in finding and not isinstance(finding["function"], str):
            raise ValueError("Invalid function name")
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            raise ValueError("Missing evidence")
    return value


def classify(sites: list[dict[str, Any]], prediction: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match findings to ground-truth sites by category + (function name or line range)."""
    findings = prediction["findings"]
    matched_ids: set[int] = set()
    rows = []
    for site in sites:
        hits = [f for f in findings if f["category"] == site["category"] and (
            f.get("function") == site["function"] or site["line_start"] <= f["line"] <= site["line_end"])]
        matched_ids.update(id(f) for f in hits)
        rows.append({**site, "detected": bool(hits), "matched_findings": hits})
    unmatched = [f for f in findings if id(f) not in matched_ids]
    return rows, unmatched


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["expected"] == "defect" and r["detected"] for r in rows)
    fn = sum(r["expected"] == "defect" and not r["detected"] for r in rows)
    fp = sum(r["expected"] == "control" and r["detected"] for r in rows)
    tn = sum(r["expected"] == "control" and not r["detected"] for r in rows)

    def ratio(n: int, d: int) -> float | None:
        return n / d if d else None

    return {"n": len(rows), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "detection_rate": ratio(tp, tp + fn), "warning_accuracy": ratio(tp, tp + fp),
            "false_positive_rate": ratio(fp, fp + tn), "accuracy": ratio(tp + tn, tp + tn + fp + fn)}


def rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def file_metric(records: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    for target in targets:
        prediction = records.get(target["file"], {}).get("prediction")
        if prediction is None:
            continue
        predicted = bool(prediction["findings"])
        actual = target["verdict"] == "vulnerable"
        tp += predicted and actual
        tn += (not predicted) and (not actual)
        fp += predicted and not actual
        fn += (not predicted) and actual
    n = tp + tn + fp + fn
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / n if n else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def scored_rows(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for record in checkpoint.get("samples", {}).values() if record.get("status") == "scored"
            for row in record.get("rows", [])]


def credit_failure(events: list[dict[str, Any]]) -> bool:
    errors = [e.get("payload", {}).get("error", {}) for e in events if e.get("type") == "run.failed"]
    return any(re.search(r"\b402\b|insufficient.*credit|not enough credit|credit.*insufficient",
                         json.dumps(e), re.I) for e in errors)


def credit_stop_reason(events: list[dict[str, Any]]) -> str:
    errors = [e.get("payload", {}).get("error", {}) for e in events if e.get("type") == "run.failed"]
    if any("in_flight_budget_exhausted" in json.dumps(e) for e in errors):
        return "OpenRouter 402：在途请求额度占用，余额未必耗尽；已停止，待在途请求结算后可重试"
    return "OpenRouter 返回额度不足，已停止；充值后从此文件继续"


def refresh_scores(checkpoint: dict[str, Any], gt: dict[str, Any]) -> None:
    """Re-parse saved completed answers offline; never spend tokens on format repair."""
    sites_by_file = sites_index(gt)
    lines_by_file = {f["file"]: f["lines"] for f in gt["target_files"]}
    for file, rec in checkpoint.get("samples", {}).items():
        if rec.get("status") != "unscored" or not rec.get("attempts"):
            continue
        evidence = OUT / "runs" / f"{rec['attempts'][-1]['run_id']}.json"
        if not evidence.exists():
            continue
        state = json.loads(evidence.read_text(encoding="utf-8"))["state"]
        if state.get("status") != "completed":
            continue
        try:
            prediction = parse_answer(state["final_answer"], lines_by_file[file])
        except (ValueError, TypeError, KeyError):
            continue
        rows, unmatched = classify(sites_by_file.get(file, []), prediction)
        rec.update({"status": "scored", "prediction": prediction, "rows": rows, "out_of_scope_findings": unmatched})
        rec.pop("error", None)


def sites_index(gt: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for site in gt["sites"]:
        index.setdefault(site["file"], []).append(site)
    return index


def report(checkpoint: dict[str, Any], gt: dict[str, Any], model: str) -> str:
    records = checkpoint.get("samples", {})
    targets = ordered_targets(gt)
    rows = scored_rows(checkpoint)
    overall = metric(rows)
    files = file_metric(records, targets)
    scored_files = sum(r.get("status") == "scored" for r in records.values())
    attempted = sum(bool(r.get("attempts")) for r in records.values())
    project = gt["project"]
    lines = ["# 合同第 5 项：操作系统底层程序高危代码检测实测结果", "",
             f"更新时间（UTC）：{now()}", ""]
    if model != DEFAULT_MODEL:
        lines += [f"> **本轮不是合同成绩。** 使用的是 `{model}`，并非合同要求的 Claude/GPT 级模型，"
                  f"本轮仅用于验证数据集与测试链路（提示词、读取覆盖、JSON 解析、site 计分、报告生成）是否正确。"
                  f"合同成绩须用 `{DEFAULT_MODEL}` 或更强的 Claude/GPT 模型重跑，结果写入 `web-<model>/` 独立目录。", ""]
    lines += [
             f"数据集：{project['name']} {project['ref']}（提交 `{project['commit'][:12]}`）子集，"
             f"{project['file_count']} 个文件、源码 {project['source_lines']} 行；"
             f"待检文件 {len(gt['target_files'])} 个、合计 {gt['target_file_lines']} 行；"
             f"每类注入缺陷 10 处，共 {sum(c['defects'] for c in gt['categories'].values())} 处缺陷 + "
             f"{sum(c['controls'] for c in gt['categories'].values())} 处安全对照。"
             "数据集与真值见 `artifacts/contract5/rtthread/dataset/`，构建方法见 `scripts/build_contract5_dataset.py`。", "",
             f"模型：OpenRouter / `{model}`；状态：{checkpoint.get('stop_reason', '尚未开始')}；"
             f"已尝试 {attempted}/{len(targets)} 个文件，已有效评分 {scored_files}/{len(targets)} 个。", ""]
    if not rows:
        lines += ["**尚未产生任何有效评分结果。** 下列指标表在运行 `uv run python scripts/benchmark_contract5_web.py` 后自动填充；"
                  "API/额度错误、未完成 Run、输出格式错误不会被当作无漏洞或正确结果。", ""]
    else:
        lines += [("全部待检文件已有效完成，以下为完整结果。" if scored_files == len(targets)
                   else "**这是部分文件的阶段性结果，不是完整数据集的得分。**") +
                  "API/额度错误、未完成 Run、输出格式错误不计入 TP/TN/FP/FN。", ""]
    lines += ["检测系统的静态层可单独评估（不调模型、不联网）："
              "`uv run python scripts/measure_contract5_static.py`；"
              "其成绩与本轮端到端成绩的差异分析见 `合同-5-数据集说明.md` 第 5.2、5.3 节。", "",
              "## 指标（site 级：每处注入缺陷 / 安全对照函数为一个样本）", "",
              "| 指标 | 公式 | 当前结果 |", "|---|---|---|",
              f"| 预警准确率（合同口径） | TP/(TP+FP) | {rate(overall['warning_accuracy'])} ({overall['tp']}/{overall['tp'] + overall['fp']}) |",
              f"| 误报率（合同口径） | FP/(FP+TN) | {rate(overall['false_positive_rate'])} ({overall['fp']}/{overall['fp'] + overall['tn']}) |",
              f"| 检出率 | TP/(TP+FN) | {rate(overall['detection_rate'])} ({overall['tp']}/{overall['tp'] + overall['fn']}) |",
              f"| 正确率 | (TP+TN)/N | {rate(overall['accuracy'])} ({overall['tp'] + overall['tn']}/{overall['n']}) |",
              f"| TP / FN / FP / TN | — | {overall['tp']} / {overall['fn']} / {overall['fp']} / {overall['tn']} |", "",
              "合同阈值：预警准确率 > 90%，误报率 < 8%；甲方期望数据集难度使正确率落在 85%–90%。"
              "本脚本只输出原始数值，不输出达标结论。", "",
              "| 类别 | TP | FN | FP | TN | 检出率 | 预警准确率 | 误报率 | 正确率 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for category, name in CATEGORIES.items():
        m = metric([r for r in rows if r["category"] == category])
        lines.append(f"| {name} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | {rate(m['detection_rate'])} | "
                     f"{rate(m['warning_accuracy'])} | {rate(m['false_positive_rate'])} | {rate(m['accuracy'])} |")
    lines += [f"| 总体 | {overall['tp']} | {overall['fn']} | {overall['fp']} | {overall['tn']} | {rate(overall['detection_rate'])} | "
              f"{rate(overall['warning_accuracy'])} | {rate(overall['false_positive_rate'])} | {rate(overall['accuracy'])} |", "",
              "| 缺陷难度 | 缺陷数 | 已评分 | 检出 | 检出率 |", "|---|---:|---:|---:|---:|"]
    for difficulty in DIFFICULTIES:
        total = sum(s["expected"] == "defect" and s["difficulty"] == difficulty for s in gt["sites"])
        scored = [r for r in rows if r["expected"] == "defect" and r["difficulty"] == difficulty]
        hit = sum(r["detected"] for r in scored)
        lines.append(f"| {difficulty} | {total} | {len(scored)} | {hit} | {rate(hit / len(scored) if scored else None)} |")
    lines += ["", "## 文件级判定", "",
              "文件级：findings 非空视为预测有漏洞；真值 verdict=vulnerable 的文件至少含一处注入缺陷，clean 文件不含缺陷（可含安全对照）。", "",
              "| 指标 | 当前结果 |", "|---|---|",
              f"| 文件级正确率 | {rate(files['accuracy'])} ({files['tp'] + files['tn']}/{files['n']}) |",
              f"| 文件级误报率（clean 文件被报有漏洞） | {rate(files['false_positive_rate'])} ({files['fp']}/{files['fp'] + files['tn']}) |",
              f"| 文件级 TP / FN / FP / TN | {files['tp']} / {files['fn']} / {files['fp']} / {files['tn']} |", ""]
    errors = [r for r in rows if r["detected"] != (r["expected"] == "defect")]
    if errors:
        lines += ["## 与真值不一致的 site", "",
                  "以下均保留在上述计分中；人工复核意见不修改标签，也不产生另一套更高的成绩。", "",
                  "| site | 文件 | 函数 | 类别 | 难度 | 判定 | 真值说明 |", "|---|---|---|---|---|---|---|"]
        for r in errors:
            verdict = "漏报 FN" if r["expected"] == "defect" else "误报 FP"
            reason = (r["matched_findings"][0]["reason"] if r["matched_findings"] else "")[:160]
            lines.append(f"| `{r['id']}` | `{r['file']}` | `{r['function']}` | {CATEGORIES[r['category']]} | {r['difficulty']} | {verdict} | "
                         f"{r['description'][:160]}{'；模型理由：' + reason if reason else ''} |")
        lines.append("")
    out_of_scope = sum(len(rec.get("out_of_scope_findings", [])) for rec in records.values())
    lines += [f"范围外告警（不落在任何注入/对照函数上、不计入 site 级指标）：{out_of_scope} 条；明细见 checkpoint。", "",
              "## 逐文件进度", "", "| 文件 | 行数 | 真值 | 缺陷/对照 | 状态 | 预测告警数 | site TP/FN/FP/TN | 文件判定 | Run |",
              "|---|---:|---|---|---|---:|---|---|---|"]
    for target in targets:
        rec = records.get(target["file"], {})
        prediction = rec.get("prediction")
        m = metric(rec.get("rows", [])) if prediction else None
        file_ok = "—" if prediction is None else ("正确" if bool(prediction["findings"]) == (target["verdict"] == "vulnerable") else "错误")
        run_id = rec["attempts"][-1].get("run_id", "") if rec.get("attempts") else ""
        counts = f"{m['tp']}/{m['fn']}/{m['fp']}/{m['tn']}" if m else "—"
        lines.append(f"| `{target['file']}` | {target['lines']} | {'有缺陷' if target['verdict'] == 'vulnerable' else '无缺陷'} | "
                     f"{target['defect_count']}/{target['control_count']} | {rec.get('status', '未测试')} | "
                     f"{len(prediction['findings']) if prediction else '—'} | {counts} | {file_ok} | `{run_id[:8]}` |")
    lines += ["", "## 测试链路、额度与证据", "",
              f"- 模型：OpenRouter / `{model}`；无模型 fallback。可选的 GPT/Claude 更强模型用 `--model` 指定，"
              "并在 Web 后端启动时把运行时默认模型设为同一值；不同模型的结果写入不同的 `web-<model>/` 目录，不混合。",
              "- 链路：Web `/api/agent/threads` → 新消息 → Run `/step` → 读取状态和 events；与页面共用持久化 Agent 引擎，不是脚本直接询问模型。",
              "- 每个待检文件一个独立对话；工作区为整个注入后的工程目录，target_files 只给该文件；"
              "不向 Agent 提供 ground truth、注入位置或正负标签，真值文件不在工作区内。",
              "- 静态只读审计：允许读取源码与调用只读扫描工具；不编译、不执行、不授权 write/execute 工具（等待审批的工具调用一律拒绝）。",
              f"- 固定次序：有缺陷文件与无缺陷文件交替，各自按路径排序；不按模型表现筛文件。每个 Run 预算：`{json.dumps(BUDGET)}`。",
              "- 提示词模板：`artifacts/contract5/rtthread/prompt.md`；计分：category 相同且 function 同名或 line 落在函数范围内即命中。",
              f"- 完整预测、断点与配置：`artifacts/contract5/rtthread/{OUT.name}/checkpoint.json`；Run 事件日志保存在同目录 `runs/`（不入库）。",
              "- 脚本启动时核对工程全部文件与 ground truth 的 SHA256，防止在被改动的工程上计分。"]
    balance = checkpoint.get("credits", [])
    if balance:
        lines += ["", "## 额度记录", "",
                  f"- 首次余额：${balance[0]['effective_remaining_usd']:.6f}；最近余额：${balance[-1]['effective_remaining_usd']:.6f}"
                  f"（查询时间 {balance[-1]['checked_utc']}）。",
                  "- 两次查询之间的差值可能包含期间其他请求，不等同于本实验精确账单；余额入账可能延迟，在途占用也可能触发 402。"]
    lines += ["", "## 运行与续跑", "", "启动 Web 服务（运行时默认模型与 `--model` 一致）并导出 OPENROUTER_API_KEY 后：", "",
              "```bash", "uv run python scripts/benchmark_contract5_web.py                 # 全部待检文件，默认续跑",
              "uv run python scripts/benchmark_contract5_web.py --limit 2       # 低成本冒烟：前 2 个文件",
              "uv run python scripts/benchmark_contract5_web.py --print-prompts # 不联网，打印每个文件的提示词",
              "uv run python scripts/benchmark_contract5_web.py --report-only   # 不联网，重新生成本报告", "```", "",
              "脚本保留已评分结果（包括错误），不会反复测到答对；仅额度失败的文件会在充值后重开 Run，非额度失败需加 `--retry-failed`。", ""]
    return "\n".join(lines)


class WebClient:
    def __init__(self, base: str):
        url = urlparse(base)
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local HTTP Web backend is allowed")
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # Local API must not forward the key through a proxy.

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, self.base + path, timeout=660, allow_redirects=False, **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"Web {path}: HTTP {response.status_code}: {safe_text(response.text[:800])}")
        return response.json()


def clear_workspace_scratch() -> None:
    """Drop tool scratch output so each target file is audited independently."""
    scratch = PROJECT / ".code-agent"
    if scratch.exists():
        shutil.rmtree(scratch)


def is_skippable(record: dict[str, Any], retry_failed: bool) -> bool:
    if record.get("prediction") is not None or record.get("status") == "unscored":
        return True
    return record.get("status") == "failed" and not retry_failed


def run(checkpoint: dict[str, Any], gt: dict[str, Any], targets: list[dict[str, Any]], base: str, model: str, retry_failed: bool = False) -> None:
    client = WebClient(base)
    bootstrap = client.call("GET", "/api/bootstrap")
    config = bootstrap["runtime_default_model_config"]
    if (config.get("provider"), config.get("model"), config.get("base_url", "").rstrip("/")) != (
            "openrouter", model, "https://openrouter.ai/api/v1") or config.get("fallback_models"):
        raise ValueError(f"Web runtime defaults do not match OpenRouter / {model} / no fallback")
    signature = {"model_config": config, "budget": BUDGET, "prompt_sha256": digest(prompt_template().encode("utf-8")),
                 "ground_truth_sha256": digest(GROUND_TRUTH.read_bytes()), "manifest_sha256": digest(MANIFEST.read_bytes()),
                 "product_sha256": {name: digest((ROOT / name).read_bytes()) for name in PRODUCT_FILES if (ROOT / name).exists()},
                 "order": [t["file"] for t in ordered_targets(gt)], "base_url": base, "scoring": "site:category+function_or_line_range"}
    if checkpoint.get("protocol", signature) != signature:
        raise ValueError("Evaluation protocol changed (model/prompt/dataset/product rules); use a new output directory instead of mixing results")
    checkpoint["protocol"] = signature
    sites_by_file = sites_index(gt)

    def persist() -> None:
        save_json(OUT / "checkpoint.json", checkpoint)
        atomic_write(REPORT, report(checkpoint, gt, model))

    def credits() -> float:
        receipt = check_credits()
        checkpoint.setdefault("credits", []).append(receipt)
        persist()
        print(f"Available credit: ${receipt['effective_remaining_usd']:.6f}", flush=True)
        return receipt["effective_remaining_usd"]

    total = len(targets)
    for index, target in enumerate(targets, 1):
        file = target["file"]
        rec = checkpoint["samples"].setdefault(file, {"attempts": []})
        if is_skippable(rec, retry_failed):
            print(f"[{index}/{total}] {file} {rec.get('status')}：跳过（已有结果）", flush=True)
            continue
        if credits() <= 0:
            checkpoint["stop_reason"] = "额度不足，等待充值后续跑"
            persist()
            return
        clear_workspace_scratch()
        attempt = rec["attempts"][-1] if rec["attempts"] else None
        if attempt is None or rec.get("status") in {"credit_exhausted", "failed"}:
            thread = client.call("POST", "/api/agent/threads", json={
                "title": f"合同5 · {file}", "workspace": str(PROJECT), "runtime_mode": "agent",
                "runtime_model_config": config, "budget": BUDGET})
            attempt = {"thread_id": thread["thread_id"], "created_utc": now()}
            rec["attempts"].append(attempt)
            rec["status"] = "created"
            persist()
        if "run_id" not in attempt:
            if attempt.get("message_pending"):
                raise RuntimeError("Message response lost; inspect saved Thread before retrying to avoid duplicate billing")
            attempt["message_pending"] = True
            persist()
            created = client.call("POST", f"/api/agent/threads/{attempt['thread_id']}/messages", json={
                "content": goal(file), "target_files": [file],
                "context_items": [{"path": file, "absolute_path": str(PROJECT / file), "external": False, "type": "file"}],
                "budget": BUDGET, "api_key": os.environ["OPENROUTER_API_KEY"]})
            attempt["run_id"] = created["run_id"]
            attempt.pop("message_pending")
            persist()
        run_path = f"/api/agent/runs/{attempt['run_id']}"
        state = client.call("GET", run_path)
        if state["runtime_model_config"] != config:
            raise ValueError("Run model snapshot does not match the frozen configuration")
        while state["status"] not in TERMINAL:
            print(f"[{index}/{total}] {file} {state['status']} LLM={state.get('llm_calls', 0)}", flush=True)
            if state["status"] == "waiting_approval":
                state = client.call("POST", run_path + "/reject")  # never approve execute/write tools
            elif state["status"] == "paused":
                raise RuntimeError(f"Run paused: {attempt['run_id']}; review it in the Web UI")
            else:
                state = client.call("POST", run_path + "/step")
            events = client.call("GET", run_path + "/events")["events"]
            save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
            rec["status"] = state["status"]
            persist()
        events = client.call("GET", run_path + "/events")["events"]
        save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
        attempt.update({"status": state["status"], "finished_utc": now()})
        if credit_failure(events):
            rec["status"] = "credit_exhausted"
            checkpoint["stop_reason"] = credit_stop_reason(events)
            persist()
            credits()
            return
        if state["status"] == "completed":
            try:
                prediction = parse_answer(state["final_answer"], target["lines"])
                rows, unmatched = classify(sites_by_file.get(file, []), prediction)
                rec.update({"status": "scored", "prediction": prediction, "rows": rows, "out_of_scope_findings": unmatched})
                rec.pop("error", None)
                m = metric(rows)
                print(f"[{index}/{total}] {file} 已评分：告警 {len(prediction['findings'])} 条，site TP/FN/FP/TN={m['tp']}/{m['fn']}/{m['fp']}/{m['tn']}", flush=True)
            except (ValueError, TypeError) as exc:
                rec.update({"status": "unscored", "error": str(exc)})
                print(f"[{index}/{total}] {file} 输出无法解析：{exc}", flush=True)
        else:
            rec["status"] = "failed"
            checkpoint["stop_reason"] = f"非额度故障：{file} / {state['status']}；查看 Run 证据"
            persist()
            if retry_failed:
                continue
            credits()
            return
        persist()
    checkpoint["stop_reason"] = "待检文件队列已处理完毕（未评分项见进度表）"
    credits()
    persist()


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id; must equal the Web runtime default")
    parser.add_argument("--limit", type=int, help="only the first N target files in the fixed order (smoke test)")
    parser.add_argument("--files", help="comma-separated project-relative target files to run (subset)")
    parser.add_argument("--retry-failed", action="store_true", help="re-run files whose last Run failed for non-credit reasons")
    parser.add_argument("--report-only", action="store_true", help="regenerate the report without network access")
    parser.add_argument("--print-prompts", action="store_true", help="print the exact prompt per target file; no network")
    args = parser.parse_args()
    OUT = DATA / f"web-{model_slug(args.model)}"
    gt = load_ground_truth()
    targets = ordered_targets(gt)
    if args.files:
        wanted = {f.strip() for f in args.files.split(",") if f.strip()}
        unknown = wanted - {t["file"] for t in targets}
        if unknown:
            raise SystemExit(f"unknown target files: {sorted(unknown)}")
        targets = [t for t in targets if t["file"] in wanted]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        targets = targets[:args.limit]
    if args.print_prompts:
        for target in targets:
            print(f"===== {target['file']} ({target['lines']} 行) =====")
            print(goal(target["file"]))
            print()
        print(f"共 {len(targets)} 个文件；每个 Run 预算 {json.dumps(BUDGET)}；模型 {args.model}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUT / "checkpoint.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "created_utc": now(), "model": args.model, "samples": {}, "credits": [], "stop_reason": "尚未开始"}
        refresh_scores(checkpoint, gt)
        if args.report_only:
            save_json(path, checkpoint)
            atomic_write(REPORT, report(checkpoint, gt, args.model))
            print(f"Report: {REPORT}", flush=True)
            return
        try:
            if not os.environ.get("OPENROUTER_API_KEY", "").strip():
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            verify_project(gt)
            checkpoint["stop_reason"] = "测试进行中"
            run(checkpoint, gt, targets, args.base_url, args.model, args.retry_failed)
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint["stop_reason"] = safe_text(f"已停止：{type(exc).__name__}: {exc}")
            print(checkpoint["stop_reason"], flush=True)
            raise SystemExit(1) from None
        finally:
            save_json(path, checkpoint)
            atomic_write(REPORT, report(checkpoint, gt, args.model))
            print(f"Report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
