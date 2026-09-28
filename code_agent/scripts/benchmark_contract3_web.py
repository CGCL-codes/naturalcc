"""Run Contract-3 through the same local HTTP Web Agent API as Contract-5.

This is deliberately distinct from the deterministic local Cppcheck run.  It
uses the configured Web Agent/model and permits only read-level tools, so it
does not compile, execute, modify source, or auto-approve an execute tool.
Start ``agent_web_api.py`` first and export ``OPENROUTER_API_KEY``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts import benchmark_contract3_cppcheck as local
from code_agent.scripts import benchmark_contract3_independent as independent

DATA = ROOT / "artifacts" / "contract3"
OUT = DATA / "web-sonnet45"
REPORT = ROOT / "合同-3-web-result.md"
MODEL = "anthropic/claude-sonnet-4.5"
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
BUDGET = {"max_llm_calls": 6, "max_tool_calls": 12, "max_input_tokens": 120000,
          "max_output_tokens": 24000, "max_seconds": 600, "max_compaction_calls": 8}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_text(value: str) -> str:
    secret = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return value.replace(secret, "[REDACTED]") if secret else value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(safe_text(json.dumps(value, ensure_ascii=False, indent=2)) + "\n", encoding="utf-8")
    temporary.replace(path)


def goal(category: dict[str, Any], fixture: str) -> str:
    return f'''请审计当前工作区的 `{fixture}`，仅判断“{category['name']}”这一类风险。
先读取完整文件，然后必须调用只读工具 `vulnerability_detection` 扫描该文件，参数应包含
`target_files:["{fixture}"]`、`scan_scope:"targets"`、`severity_threshold:"low"`、`max_findings:1000`。
结合源码和工具证据判断；扫描无告警不等于安全。只做静态审计：不修改、编译、运行任何文件，
不执行 shell，不调用需要 execute/write 授权的工具，不搜索外部资料。
不要根据函数名猜测正负样本；每个函数都独立按代码证据判断。
最终回答只输出 JSON（可使用唯一 json 代码块）：
{{"findings":[{{"category":"{category['id']}","line":7,"reason":"具体代码证据与触发条件"}}],"summary":"简要结论"}}。
findings 中 category 只能为 `{category['id']}`，line 必须是该文件真实的 1-based 行号；
确认没有该类风险时 findings 为 []。以上 JSON 仅为格式示例，不表示文件实际有漏洞。'''


def parse_answer(answer: str, category: str, line_count: int) -> dict[str, Any]:
    text = answer.strip()
    fenced = re.findall(r"```json\s*([\s\S]*?)\s*```", text)
    if len(fenced) > 1:
        raise ValueError("Ambiguous multiple JSON answers")
    value = json.loads(fenced[0] if fenced else text)
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        raise ValueError("Missing findings array")
    if not isinstance(value.get("summary"), str):
        raise ValueError("Missing summary")
    for finding in value["findings"]:
        if not isinstance(finding, dict) or finding.get("category") != category:
            raise ValueError("Out-of-scope category")
        if type(finding.get("line")) is not int or not 1 <= finding["line"] <= line_count:
            raise ValueError("Invalid source line")
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            raise ValueError("Missing evidence")
    return value


def classify(cases: list[dict[str, Any]], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    findings = prediction["findings"]
    rows = []
    for case in cases:
        matched = [f for f in findings if case["line_start"] <= f["line"] <= case["line_end"]]
        rows.append({**case, "detected": bool(matched), "matched_findings": matched})
    return rows


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return local.metric(rows)


def rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def credit_failure(events: list[dict[str, Any]]) -> bool:
    errors = [event.get("payload", {}).get("error", {}) for event in events if event.get("type") == "run.failed"]
    return any(re.search(r"\b402\b|insufficient.*credit|not enough credit|credit.*insufficient",
                         json.dumps(error), re.I) for error in errors)


def refresh_credit_status(checkpoint: dict[str, Any]) -> None:
    """Keep a provider-credit stop distinct from a product/model verdict."""
    found = False
    for record in checkpoint.get("samples", {}).values():
        if record.get("status") != "failed" or not record.get("attempts"):
            continue
        run_id = record["attempts"][-1].get("run_id")
        evidence = OUT / "runs" / f"{run_id}.json"
        if evidence.exists() and credit_failure(json.loads(evidence.read_text(encoding="utf-8")).get("events", [])):
            record["status"] = "credit_exhausted"
            found = True
    if found and str(checkpoint.get("stop_reason", "")).startswith("失败："):
        checkpoint["stop_reason"] = "OpenRouter 额度不足或请求输出预留超过可用额度；未把该组计分"


def prepare_samples(manifest: dict[str, Any], workspace: Path, suite: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare 3×3 target fixtures; labels are never placed in those workspaces."""
    samples, projects = [], []
    categories = {category["id"]: category for category in manifest["categories"]}
    for spec in manifest["sources"]:
        project = local.clone_or_verify(spec, workspace)
        source_lines = local.loc(local.source_files(project, spec["scan_roots"]))
        if source_lines < 10_000:
            raise ValueError(f"{spec['id']} has fewer than 10,000 source lines")
        projects.append({"id": spec["id"], "name": spec["name"], "commit": spec["commit"], "source_lines": source_lines})
        for category_id, category in categories.items():
            fixture, cases = independent.inject_fixture(
                project, spec["id"], [category], fixture_name=f"contract3_{suite}_web_{category_id}.c", profile=suite)
            samples.append({"id": f"{spec['id']}-{category_id}", "project": spec["id"],
                            "project_path": str(project), "category": category, "fixture": fixture.name,
                            "fixture_sha256": digest(fixture.read_bytes()), "cases": cases,
                            "line_count": len(fixture.read_text(encoding="utf-8").splitlines())})
    return samples, projects


class WebClient:
    def __init__(self, base: str):
        parsed = urlparse(base)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local HTTP Web backend is allowed")
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, self.base + path, timeout=660, allow_redirects=False, **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"Web {path}: HTTP {response.status_code}: {safe_text(response.text[:800])}")
        return response.json()


def render_report(checkpoint: dict[str, Any], manifest: dict[str, Any], suite: str = "baseline") -> str:
    records = checkpoint["samples"]
    completed = [record for record in records.values() if record.get("rows") is not None]
    rows = [row for record in completed for row in record["rows"]]
    by_category = {category["id"]: metric([row for row in rows if row["category"] == category["id"]])
                   for category in manifest["categories"]}
    overall = metric(rows)
    attempted = sum(bool(record.get("attempts")) for record in records.values())
    title = "合同指标 3：Web Agent 独立复杂样例实测结果" if suite == "independent" else "合同指标 3：Web Agent 可选实测结果"
    lines = [f"# {title}", "", f"更新时间（UTC）：{now()}", "",
             "此报告是与合同指标 5 相同的 Web Agent HTTP 链路的可选模式，不是本地 Cppcheck 直测的替代。"
             "每个‘工程×类别’为一个独立 Run；模型可读代码并调用只读漏洞扫描工具，"
             "脚本不自动批准 execute/write 工具。", "",
             f"状态：{checkpoint['stop_reason']}；已尝试 {attempted}/9 组，已有效评分 {len(completed)}/9 组。",
             "未完成、模型输出格式无效、API 故障均不计为安全或正确。", "",
             "## 原始统计", "", "| 类别 | TP | FN | FP | TN | 检出率 | 误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for category in manifest["categories"]:
        value = by_category[category["id"]]
        lines.append(f"| {category['name']} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} | {rate(value['detection_rate'])} | {rate(value['false_positive_rate'])} |")
    lines += [f"| 总体 | {overall['tp']} | {overall['fn']} | {overall['fp']} | {overall['tn']} | {rate(overall['detection_rate'])} | {rate(overall['false_positive_rate'])} |", "",
              "计分：一个目标函数范围中出现至少一条目标类别告警，正例计 TP、对照计 FP；"
              "未出现则分别计 FN、TN。检出率 = TP/(TP+FN)，误报率 = FP/(FP+TN)。"
              "函数范围外的模型告警单列保留为范围外发现，不进入此两项指标。", "",
              "## 协议与证据", "",
              "- 上游工程、固定提交、每类每工程 20 个注入点 + 20 个安全对照，见 `artifacts/contract3/manifest.json`。",
              f"- 真值文件与预测/Run 证据保存在 `artifacts/contract3/{OUT.name}/`；真值不放入 Agent 工作区。",
              f"- 模型：OpenRouter / `{MODEL}`，无 fallback；每个 Run 预算：`{json.dumps(BUDGET)}`。",
              "- 调用链：Web `/api/agent/threads` → message → Run `/step`/状态/events，与指标 5 一致；不是脚本直接调用模型。",
              "- Web 模式只扫描注入 fixture，源工程用于满足规模、真实上下文和可复现版本要求；不能把该模式表述为完整上游工程覆盖。",
              "- 可用 `--limit 1` 做一个 40 函数的低成本冒烟样本；只有完整 9 组才是 180 正例 + 180 对照的完整 Web 结果。"]
    return "\n".join(lines) + "\n"


def run(checkpoint: dict[str, Any], manifest: dict[str, Any], samples: list[dict[str, Any]], projects: list[dict[str, Any]], base: str, suite: str, retry_failed: bool = False) -> None:
    client = WebClient(base)
    bootstrap = client.call("GET", "/api/bootstrap")
    config = bootstrap["runtime_default_model_config"]
    if (config.get("provider"), config.get("model"), config.get("base_url", "").rstrip("/")) != ("openrouter", MODEL, "https://openrouter.ai/api/v1") or config.get("fallback_models"):
        raise ValueError("Web runtime defaults must be OpenRouter / Sonnet 4.5 / no fallback")
    protocol = {"model_config": config, "budget": BUDGET, "manifest_sha256": digest((DATA / "manifest.json").read_bytes()),
                "projects": projects, "fixture_protocol": "one deterministic category fixture per project; labels outside workspace",
                "base_url": base}
    # Keep the pre-suite baseline checkpoint resumable; independent output has its
    # own directory and an explicit protocol discriminator.
    if suite != "baseline":
        protocol["suite"] = suite
    if checkpoint.get("protocol", protocol) != protocol:
        raise ValueError("Evaluation protocol changed; do not mix results in the existing checkpoint")
    checkpoint["protocol"] = protocol

    def persist() -> None:
        save_json(OUT / "checkpoint.json", checkpoint)
        from code_agent.scripts.render_contract3_report import write_report
        write_report()

    for index, sample in enumerate(samples, 1):
        record = checkpoint["samples"].setdefault(sample["id"], {"attempts": []})
        if record.get("rows") is not None or record.get("status") == "unscored" or (record.get("status") == "failed" and not retry_failed):
            continue
        thread = client.call("POST", "/api/agent/threads", json={
            "title": f"合同3 · {sample['id']}", "workspace": sample["project_path"], "runtime_mode": "agent",
            "runtime_model_config": config, "budget": BUDGET})
        attempt = {"thread_id": thread["thread_id"], "created_utc": now()}
        record["attempts"].append(attempt)
        record["status"] = "created"
        persist()
        created = client.call("POST", f"/api/agent/threads/{attempt['thread_id']}/messages", json={
            "content": goal(sample["category"], sample["fixture"]), "target_files": [sample["fixture"]],
            "context_items": [{"path": sample["fixture"], "absolute_path": str(Path(sample["project_path"]) / sample["fixture"]), "external": False, "type": "file"}],
            "budget": BUDGET, "api_key": os.environ["OPENROUTER_API_KEY"]})
        attempt["run_id"] = created["run_id"]
        run_path = f"/api/agent/runs/{attempt['run_id']}"
        state = client.call("GET", run_path)
        if state["runtime_model_config"] != config:
            raise ValueError("Run model snapshot changed")
        while state["status"] not in TERMINAL:
            print(f"[{index}/9] {sample['id']} {state['status']} LLM={state.get('llm_calls', 0)}", flush=True)
            if state["status"] == "waiting_approval":
                # Contract-3 Web mode intentionally has no automatic execute/write approval.
                state = client.call("POST", run_path + "/reject")
            elif state["status"] == "paused":
                raise RuntimeError(f"Run paused: {attempt['run_id']}")
            else:
                state = client.call("POST", run_path + "/step")
            events = client.call("GET", run_path + "/events")["events"]
            save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
            record["status"] = state["status"]
            persist()
        events = client.call("GET", run_path + "/events")["events"]
        save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
        attempt.update({"status": state["status"], "finished_utc": now()})
        if state["status"] != "completed":
            if credit_failure(events):
                record["status"] = "credit_exhausted"
                checkpoint["stop_reason"] = "OpenRouter 额度不足或请求输出预留超过可用额度；未把该组计分"
                persist()
                return
            record["status"] = "failed"
            checkpoint["stop_reason"] = f"失败：{sample['id']} / {state['status']}；请查看 Run 证据"
            persist()
            if retry_failed:
                continue
            return
        try:
            prediction = parse_answer(state["final_answer"], sample["category"]["id"], sample["line_count"])
            rows = classify(sample["cases"], prediction)
            record.update({"status": "scored", "prediction": prediction, "rows": rows,
                           "out_of_scope_findings": [f for f in prediction["findings"] if not any(c["line_start"] <= f["line"] <= c["line_end"] for c in sample["cases"])]})
        except (TypeError, ValueError) as exc:
            record.update({"status": "unscored", "error": str(exc)})
        persist()
    checkpoint["stop_reason"] = "样本队列已处理完毕（未评分项保留在 checkpoint）"
    persist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/naturalcc-contract3-web-sources"))
    parser.add_argument("--suite", choices=("balanced", "independent"), default="balanced",
                        help="balanced is the recommended acceptance mix; independent is the held-out stress mix")
    parser.add_argument("--limit", type=int, help="Only run the first N project/category groups (for a low-cost smoke test)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry prior transient failed groups and continue the queue; attempts remain in checkpoint")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.suite == "independent":
        OUT = DATA / "web-sonnet45-independent"
    else:
        OUT = DATA / "web-sonnet45-balanced"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = local.load_manifest()
        checkpoint_path = OUT / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {"created_utc": now(), "samples": {}, "stop_reason": "尚未开始"}
        refresh_credit_status(checkpoint)
        if args.report_only:
            save_json(checkpoint_path, checkpoint)
            from code_agent.scripts.render_contract3_report import write_report
            write_report()
            return
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise SystemExit("OPENROUTER_API_KEY is not set")
        samples, projects = prepare_samples(manifest, args.workspace.resolve(), args.suite)
        if args.limit is not None:
            if args.limit < 1:
                raise SystemExit("--limit must be at least 1")
            samples = samples[:args.limit]
        try:
            checkpoint["stop_reason"] = "测试进行中"
            run(checkpoint, manifest, samples, projects, args.base_url, args.suite, args.retry_failed)
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint["stop_reason"] = safe_text(f"已停止：{type(exc).__name__}: {exc}")
            raise SystemExit(1) from None
        finally:
            save_json(checkpoint_path, checkpoint)
            from code_agent.scripts.render_contract3_report import write_report
            print(f"Report: {write_report()}", flush=True)


if __name__ == "__main__":
    main()
