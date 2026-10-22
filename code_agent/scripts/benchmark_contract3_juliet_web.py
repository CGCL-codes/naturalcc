"""Run the frozen Juliet Contract-3 set through the same Web Agent API as Contract-5.

Build the set first with ``build_contract3_juliet.py``.  Each sample gets its own
Thread and workspace (renamed ``src_*.c`` plus Juliet support files); labels,
Juliet names and paths never enter the workspace.  Scoring is file-level like
Contract-5: a sample is predicted defective when the final JSON lists at least
one finding of the target category.  Results are resumable; scored samples
(including wrong ones) are never re-asked.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from code_agent.scripts.benchmark_contract5_web import WebClient, credit_failure, credit_stop_reason, digest, safe_text, save_json
from code_agent.scripts.check_openrouter_credits import check_credits
from code_agent.security_review import REVIEW_VERSION

DATA = ROOT / "artifacts" / "contract3" / "juliet"
OUT = DATA / f"web-sonnet45-{REVIEW_VERSION}"
PRODUCT_FILES = ("security_review.py", "plugins/vulnerability_detection.py", "agent_core/tools/pipeline.py")
MODEL = "anthropic/claude-sonnet-4.5"
NAMES = {"array_oob": "数组越界", "string_overflow": "字符串溢出", "null_pointer": "空指针调用"}
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
BUDGET = {"max_llm_calls": 8, "max_tool_calls": 20, "max_input_tokens": 120000,
          "max_output_tokens": 24000, "max_seconds": 600, "max_compaction_calls": 8}


# Statuses that are harness/provider problems, not model outcomes: start a fresh Thread.
RETRYABLE = {"credit_exhausted", "infra_failed"}


def infrastructure_failure(events: list[dict]) -> bool:
    errors = [event.get("payload", {}).get("error", {}) for event in events if event.get("type") == "run.failed"]
    markers = ("database is locked", "APIConnectionError", "APITimeoutError")
    return any(marker in json.dumps(error) for error in errors for marker in markers)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def goal(category: str, files: list[str]) -> str:
    targets = json.dumps(files)
    return f'''请审计当前工作区的 {", ".join(f"`{name}`" for name in files)}，仅判断“{NAMES[category]}”这一类风险。
工作区中的 `std_testcase.h`、`std_testcase_io.h`、`io.c` 是公共支持代码，可读取作为上下文，但不在报告范围内。
先读取全部目标文件（必要时读取支持代码），然后必须调用只读工具 `vulnerability_detection` 扫描目标文件，参数应包含
`target_files:{targets}`、`scan_scope:"targets"`、`scan_type:"frequent_defects"`、
`severity_threshold:"low"`、`max_findings:1000`。
结合源码、跨函数/跨文件数据流、宏与全局常量取值和工具证据判断；扫描无告警不等于安全，扫描有告警也需核实。
只做静态审计：不修改、编译、运行任何文件，不执行 shell，不调用需要 execute/write 授权的工具，不搜索外部资料或数据集。
最终回答只输出 JSON（可使用唯一 json 代码块）：
{{"findings":[{{"category":"{category}","file":"{files[0]}","line":7,"reason":"具体代码证据与触发条件"}}],"summary":"简要结论"}}。
findings 中 category 只能为 `{category}`，file 只能是上述目标文件之一，line 必须是该文件真实的 1-based 行号；
确认没有该类风险时 findings 为 []。以上 JSON 仅为格式示例，不表示文件实际有漏洞。'''


def load() -> tuple[dict, dict, list[dict]]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    raw = (DATA / "dataset.json").read_bytes()
    if digest(raw) != manifest["dataset_sha256"]:
        raise ValueError("dataset.json SHA256 mismatch")
    dataset = json.loads(raw)
    by_id = {sample["id"]: sample for sample in dataset["samples"]}
    # Frozen order: case number, then category, then vulnerable/safe pair.
    order = manifest.get("order") or [f"{category}-{number:02d}-{kind}" for number in range(1, manifest["per_category"] + 1)
                                      for category in NAMES for kind in ("v", "s")]
    return manifest, dataset, [by_id[identifier] for identifier in order]


def parse_answer(answer: str, category: str, files: dict[str, str]) -> dict:
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
        if finding.get("file") not in files:
            raise ValueError("Invalid file")
        if type(finding.get("line")) is not int or not 1 <= finding["line"] <= len(files[finding["file"]].splitlines()):
            raise ValueError("Invalid source line")
        if not isinstance(finding.get("reason"), str) or not finding["reason"].strip():
            raise ValueError("Missing evidence")
    return value


def metric(rows: list[tuple[dict, bool]]) -> dict:
    tp = sum(sample["vulnerable"] and detected for sample, detected in rows)
    fn = sum(sample["vulnerable"] and not detected for sample, detected in rows)
    fp = sum(not sample["vulnerable"] and detected for sample, detected in rows)
    tn = sum(not sample["vulnerable"] and not detected for sample, detected in rows)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "detection_rate": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def scored(checkpoint: dict, samples: list[dict]) -> list[tuple[dict, bool]]:
    records = checkpoint.get("samples", {})
    return [(sample, bool(records[sample["id"]]["prediction"]["findings"])) for sample in samples
            if records.get(sample["id"], {}).get("status") == "scored"]


def summary(checkpoint: dict, samples: list[dict]) -> str:
    rows = scored(checkpoint, samples)
    lines = [f"=== 指标 3 Web Juliet 第三方测试集 ===", f"已评分 {len(rows)}/{len(samples)}；状态：{checkpoint.get('stop_reason')}",
             "类别              TP  FN  FP  TN   检出率    误报率"]
    for category, name in [*NAMES.items(), (None, "总体")]:
        value = metric([row for row in rows if category is None or row[0]["category"] == category])
        rate = lambda x: "N/A" if x is None else f"{x:.2%}"
        lines.append(f"{name:<16} {value['tp']:>2}  {value['fn']:>2}  {value['fp']:>2}  {value['tn']:>2}  "
                     f"{rate(value['detection_rate']):>8}  {rate(value['false_positive_rate']):>8}")
    return "\n".join(lines)


def run(checkpoint: dict, manifest: dict, dataset: dict, samples: list[dict], base: str, limit: int | None = None, workers: int = 1) -> None:
    client = WebClient(base)
    config = client.call("GET", "/api/bootstrap")["runtime_default_model_config"]
    if (config.get("provider"), config.get("model"), config.get("base_url", "").rstrip("/")) != ("openrouter", MODEL, "https://openrouter.ai/api/v1") or config.get("fallback_models"):
        raise ValueError("Web runtime defaults must be OpenRouter / Sonnet 4.5 / no fallback")
    protocol = {"model_config": config, "budget": BUDGET, "goal_sha256": digest(goal("array_oob", ["src_a.c"]).encode()),
                "manifest_sha256": digest((DATA / "manifest.json").read_bytes()),
                "product_sha256": {name: digest((ROOT / name).read_bytes()) for name in PRODUCT_FILES}, "order": [s["id"] for s in samples], "base_url": base}
    if checkpoint.get("protocol", protocol) != protocol:
        raise ValueError("Evaluation protocol changed; do not mix results in the existing checkpoint")
    checkpoint["protocol"] = protocol

    def persist() -> None:
        save_json(OUT / "checkpoint.json", checkpoint)
        from code_agent.scripts.render_contract3_report import write_report
        write_report()

    def credits() -> float:
        # The balance check is advisory; a real shortfall still surfaces as a 402 on the Run.
        for attempt in range(3):
            try:
                receipt = check_credits()
                break
            except RuntimeError as exc:
                error = str(exc)
                time.sleep(10 * (attempt + 1))
        else:
            checkpoint["credits"].append({"checked_utc": now(), "error": error})
            persist()
            return float("inf")
        checkpoint["credits"].append(receipt)
        persist()
        return receipt["effective_remaining_usd"]

    lock = threading.RLock()
    stop = threading.Event()
    parallel = workers > 1
    local = threading.local()
    total = len(samples)

    def halt(reason: str) -> None:
        with lock:
            checkpoint["stop_reason"] = reason
            persist()
        stop.set()

    def one(index: int, sample: dict) -> None:
        if stop.is_set():
            return
        if not parallel and credits() <= 0:
            return halt("额度不足，等待充值后续跑")
        # requests.Session is not thread-safe; each worker keeps its own client.
        if not hasattr(local, "client"):
            local.client = WebClient(base)
        web = local.client
        with lock:
            record = checkpoint["samples"].setdefault(sample["id"], {"attempts": []})
        workspace = OUT / "workspaces" / f"sample_{index:03d}"
        workspace.mkdir(parents=True, exist_ok=True)
        for name, text in {**dataset["support"], **sample["files"]}.items():
            path = workspace / name
            if path.exists() and path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Workspace source changed: {sample['id']} {name}")
            path.write_text(text, encoding="utf-8")
        files = list(sample["files"])
        attempt = record["attempts"][-1] if record["attempts"] else None
        if attempt is None or record.get("status") in RETRYABLE:
            thread = web.call("POST", "/api/agent/threads", json={
                "title": f"合同3 Juliet · sample_{index:03d}", "workspace": str(workspace),
                "runtime_mode": "agent", "runtime_model_config": config, "budget": BUDGET})
            attempt = {"thread_id": thread["thread_id"], "created_utc": now()}
            with lock:
                record["attempts"].append(attempt)
                record["status"] = "created"
                persist()
        if "run_id" not in attempt:
            if attempt.get("message_pending"):
                raise RuntimeError("Message response lost; inspect saved Thread before retrying to avoid duplicate billing")
            with lock:
                attempt["message_pending"] = True
                persist()
            created = web.call("POST", f"/api/agent/threads/{attempt['thread_id']}/messages", json={
                "content": goal(sample["category"], files), "target_files": files,
                "context_items": [{"path": name, "absolute_path": str(workspace / name), "external": False, "type": "file"} for name in files],
                "budget": BUDGET, "api_key": os.environ["OPENROUTER_API_KEY"]})
            with lock:
                attempt["run_id"] = created["run_id"]
                attempt.pop("message_pending")
                persist()
        run_path = f"/api/agent/runs/{attempt['run_id']}"
        state = web.call("GET", run_path)
        if state["runtime_model_config"] != config:
            raise ValueError("Run model snapshot does not match the frozen configuration")
        while state["status"] not in TERMINAL:
            if not parallel:
                print(f"[{index}/{total}] {sample['id']} {state['status']} LLM={state.get('llm_calls', 0)}", flush=True)
            if state["status"] == "waiting_approval":
                state = web.call("POST", run_path + "/reject")
            elif state["status"] == "paused":
                raise RuntimeError(f"Run paused: {attempt['run_id']}")
            else:
                for retry in range(5):
                    try:
                        state = web.call("POST", run_path + "/step")
                        break
                    except RuntimeError as exc:
                        # SQLite contention under concurrency surfaces as HTTP 500; the step is retried, not scored.
                        if "HTTP 500" not in str(exc) or retry == 4:
                            raise
                        time.sleep(5 * (retry + 1))
            record["status"] = state["status"]
        events = web.call("GET", run_path + "/events")["events"]
        save_json(OUT / "runs" / f"{attempt['run_id']}.json", {"state": state, "events": events})
        used = set(re.findall(r"c-security-review-v[\w-]*", json.dumps(events)))
        if used - {REVIEW_VERSION}:
            raise RuntimeError(f"Backend review version {sorted(used)} != {REVIEW_VERSION}; restart both with the same CODE_AGENT_REVIEW_PROFILE")
        with lock:
            attempt.update({"status": state["status"], "finished_utc": now(),
                            "usage": {key: state.get(key) for key in ("input_tokens", "output_tokens", "llm_calls", "tool_calls")}})
            if credit_failure(events):
                record["status"] = "credit_exhausted"
                return halt(credit_stop_reason(events))
            if state["status"] == "completed":
                try:
                    record["prediction"] = parse_answer(state["final_answer"], sample["category"], sample["files"])
                    record["status"] = "scored"
                except (ValueError, TypeError) as exc:
                    record.update({"status": "unscored", "error": str(exc)})
            elif infrastructure_failure(events):
                record["status"] = "infra_failed"
                print(f"[{index}/{total}] {sample['id']}: infra_failed（后端数据库锁，未计分，可重跑）", flush=True)
                persist()
                return None
            else:
                record["status"] = "failed"
                return halt(f"非额度故障：{sample['id']} / {state['status']}；查看 Run 证据")
            outcome = record["status"]
            if outcome == "scored":
                detected = bool(record["prediction"]["findings"])
                outcome = ("TP" if detected else "FN") if sample["vulnerable"] else ("FP" if detected else "TN")
            print(f"[{index}/{total}] {sample['id']} ({sample['testcase'].split('__', 1)[1]}): {outcome}", flush=True)
            persist()

    pending = [(index, sample) for index, sample in enumerate(samples, 1)
               if checkpoint["samples"].get(sample["id"], {}).get("status") not in {"scored", "unscored", "failed"}]
    if limit is not None:
        pending = pending[:limit]
    if parallel and credits() <= 0:
        return halt("额度不足，等待充值后续跑")
    if parallel:
        print(f"并行运行 {len(pending)} 条，workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(one, index, sample) for index, sample in pending]:
                future.result()
    else:
        for index, sample in pending:
            one(index, sample)
    if stop.is_set():
        return
    finished = all(checkpoint["samples"].get(s["id"], {}).get("status") in {"scored", "unscored", "failed"} for s in samples)
    checkpoint["stop_reason"] = ("样本队列已处理完毕（未评分项保留在 checkpoint）" if finished
                                 else f"按 --limit {limit} 暂停；再次运行将从下一条续跑")
    credits()
    persist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--limit", type=int, help="Only run the first N samples of the frozen order (smoke test)")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent samples; each sample still has its own Thread and workspace")
    parser.add_argument("--set", choices=("juliet", "juliet-holdout", "juliet-probe", "juliet-hard"), default="juliet",
                        help="juliet is the tuning set; juliet-holdout is frozen separately and never used for tuning")
    args = parser.parse_args()
    global DATA, OUT
    DATA = ROOT / "artifacts" / "contract3" / args.set
    OUT = DATA / f"web-sonnet45-{REVIEW_VERSION}"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "benchmark.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, dataset, samples = load()
        path = OUT / "checkpoint.json"
        checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "created_utc": now(), "samples": {}, "credits": [], "stop_reason": "尚未开始"}
        try:
            if not args.report_only:
                if not os.environ.get("OPENROUTER_API_KEY", "").strip():
                    raise SystemExit("OPENROUTER_API_KEY is not set")
                checkpoint["stop_reason"] = "测试进行中"
                run(checkpoint, manifest, dataset, samples, args.base_url, args.limit, args.workers)
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint["stop_reason"] = safe_text(f"已停止：{type(exc).__name__}: {exc}")
            raise SystemExit(1) from None
        finally:
            save_json(path, checkpoint)
            from code_agent.scripts.render_contract3_report import write_report
            print(summary(checkpoint, samples), flush=True)
            print(f"总报告：{write_report()}", flush=True)


if __name__ == "__main__":
    main()
