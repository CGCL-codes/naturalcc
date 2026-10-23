#!/usr/bin/env python3
"""Run NaturalCC on prepared benchmark task cards; official scoring is imported separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".java", ".py",
    ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
}
IGNORED_DIRS = {".codegraph", ".code-agent", ".venv", "venv", "node_modules", "build", "dist", "__pycache__"}
TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
AUTO_APPROVABLE_RISKS = {"write", "execute", "network", "git_write"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover_cases(dataset_dir: Path) -> list[Path]:
    if (dataset_dir / "TASK.md").is_file() and (dataset_dir / "workspace").is_dir():
        return [dataset_dir]
    return sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and (p / "TASK.md").is_file() and (p / "workspace").is_dir()),
        key=lambda p: p.name.casefold(),
    )


def source_files(workspace: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(workspace.rglob("*"), key=lambda p: p.as_posix().casefold()):
        rel = path.relative_to(workspace)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Workspace contains a symlink; refusing unsafe copy: {path}")
        if path.is_file() and path.suffix.casefold() in SOURCE_SUFFIXES:
            found.append(rel.as_posix())
    return found


def select_targets(case: Path, workspace: Path, explicit: list[str] | None = None) -> list[str]:
    requested = explicit or []
    list_file = case / "target_files.txt"
    if not requested and list_file.is_file():
        requested = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines()]
        requested = [line for line in requested if line and not line.startswith("#")]
    if requested:
        selected: list[str] = []
        for raw in requested:
            normalized = raw.replace("\\", "/")
            rel = Path(normalized)
            if rel.is_absolute() or ".." in rel.parts or not (workspace / rel).is_file():
                raise ValueError(f"Target must be an existing relative file under workspace: {raw}")
            if normalized not in selected:
                selected.append(normalized)
        return selected
    files = source_files(workspace)
    if len(files) == 1:
        return files
    if len(files) > 1:
        raise ValueError(
            f"{case.name} has {len(files)} source files. Add target_files.txt next to TASK.md "
            "or pass --target-file; do not make the whole repository an edit target."
        )
    raise ValueError(f"No recognized source files in {workspace}")


def file_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix().casefold()):
        rel = path.relative_to(root)
        if path.is_file() and not path.is_symlink() and not any(part in IGNORED_DIRS for part in rel.parts):
            result[rel.as_posix()] = digest(path)
    return result


def looks_like_test(path: str) -> bool:
    parts = [part.casefold() for part in Path(path).parts]
    name = parts[-1] if parts else ""
    test_dirs = {"test", "tests", "testing", "spec", "specs", "__tests__"}
    test_names = (
        name.startswith("test_"), name.endswith("_test.py"), name.endswith("_test.c"),
        name.endswith("_test.cpp"), name.endswith(".test.js"), name.endswith(".test.ts"),
        name.endswith(".spec.js"), name.endswith(".spec.ts"),
    )
    return any(part in test_dirs for part in parts[:-1]) or any(test_names)


def added_nonblank_lines(before_root: Path, after_root: Path, changed: list[str]) -> int:
    """Count added nonblank lines without depending on Git metadata."""
    import difflib

    total = 0
    for name in changed:
        old, new = before_root / name, after_root / name
        old_lines = old.read_text(encoding="utf-8", errors="replace").splitlines() if old.is_file() else []
        new_lines = new.read_text(encoding="utf-8", errors="replace").splitlines() if new.is_file() else []
        total += sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("+ ") and line[2:].strip())
    return total


def request_json(base: str, method: str, endpoint: str, payload: dict[str, Any] | None = None, timeout: int = 60):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    try:
        parsed = urllib.parse.urlparse(request.full_url)
        if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
            # Local NaturalCC API traffic must not be sent through a WSL/system proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            response_context = opener.open(request, timeout=timeout)
        else:
            response_context = urllib.request.urlopen(request, timeout=timeout)
        with response_context as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {endpoint}: {message}") from exc


def windows_to_wsl_path(value: str) -> Path:
    raw = value.strip().strip("\"").strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        if os.name != "nt":
            converter = shutil.which("wslpath") or shutil.which("cygpath")
            if not converter:
                raise ValueError("Windows drive paths require WSL or Git Bash; pass a native path for this shell")
            args = [converter, "-u", raw] if Path(converter).name.casefold() == "wslpath" else [converter, "-u", raw]
            raw = subprocess.check_output(args, text=True).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        # Treat CLI-relative dataset/output paths as repository-relative so the
        # documented commands work even when launched outside the repo root.
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def safe_write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def approval_request(state: dict[str, Any]) -> dict[str, Any]:
    pending = state.get("pending_approval") or {}
    risk = pending.get("risk")
    if risk not in AUTO_APPROVABLE_RISKS:
        raise ValueError(f"Refusing unknown or non-approvable risk level: {risk!r}")
    tool_call = pending.get("tool_call") or {}
    return {"risk": risk, "tool_call_id": tool_call.get("id")}


def run_case(args, case: Path, run_dir: Path, targets: list[str], budget: dict[str, int]) -> dict[str, Any]:
    source_workspace = case / "workspace"
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source_workspace,
        workspace,
        ignore=shutil.ignore_patterns(".codegraph", ".code-agent"),
        symlinks=False,
    )
    task_text = (case / "TASK.md").read_text(encoding="utf-8")
    before = file_hashes(workspace)
    case_meta = case / "case.json"
    metadata = {}
    if case_meta.is_file():
        try:
            metadata = json.loads(case_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON: {case_meta}")

    run_record: dict[str, Any] = {
        "case_id": str(metadata.get("instance_id") or metadata.get("instance_key") or metadata.get("instance") or case.name),
        "case_dir": str(case),
        "workspace": str(workspace),
        "target_files": targets,
        "task_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
        "case_json_sha256": digest(case_meta) if case_meta.is_file() else None,
        "codegraph": args.codegraph,
        "approval_policy": "auto_all" if args.auto_approve_all else "manual",
        "auto_approvals": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "status": "preparing",
        "official_pass": None,
        "vulnerability_density_per_100_loc": None,
        "codegraph_init_seconds": None,
    }
    safe_write(run_dir / "manifest.json", run_record)
    (run_dir / "TASK.md").write_text(task_text, encoding="utf-8")

    caps = {"codegraph": {"enabled": args.codegraph == "on", "auto_sync": True, "hide_workspace_search": True}}
    config: dict[str, Any] = {}
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model
    if args.base_url:
        config["base_url"] = args.base_url
    thread_payload: dict[str, Any] = {
        "title": f"Benchmark {args.dataset_dir.name} {case.name} ({args.codegraph})",
        "workspace": str(workspace),
        "runtime_mode": "agent",
        "budget": budget,
        "capabilities": caps,
    }
    if config:
        thread_payload["runtime_model_config"] = config

    thread = request_json(args.api, "POST", "/api/agent/threads", thread_payload)
    thread_id = thread["thread_id"]
    run_record["thread_id"] = thread_id
    run_record["runtime_model_config"] = (thread.get("thread") or {}).get("runtime_model_config")
    safe_write(run_dir / "manifest.json", run_record)

    if args.codegraph == "on":
        init_started = time.monotonic()
        init = request_json(args.api, "POST", f"/api/agent/threads/{urllib.parse.quote(thread_id)}/codegraph/init", {}, timeout=600)
        run_record["codegraph_init_seconds"] = round(time.monotonic() - init_started, 3)
        safe_write(run_dir / "codegraph_init.json", init)
        status = (init.get("status") or {})
        if status and not status.get("ready", status.get("initialized", False)):
            raise RuntimeError(f"CodeGraph init returned but index is not ready: {status}")

    instruction = (
        "Solve this benchmark task in the configured workspace. Modify product code only; do not edit, remove, "
        "weaken, or replace benchmark-provided tests or evaluator files. Do not use the case metadata as task context. "
        "At completion, summarize changed files and verification performed.\n\n" + task_text
    )
    key = os.environ.get("NATURALCC_API_KEY", "").strip()
    if not key and args.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and args.provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    msg_payload: dict[str, Any] = {
        "content": instruction,
        "target_files": targets,
        "budget": budget,
        "capabilities": caps,
    }
    if key:
        msg_payload["api_key"] = key
    created = request_json(args.api, "POST", f"/api/agent/threads/{urllib.parse.quote(thread_id)}/messages", msg_payload)
    run_id = created["run_id"]
    run_record["run_id"] = run_id
    safe_write(run_dir / "manifest.json", run_record)

    errors: list[str] = []
    worker: threading.Thread | None = None

    def start_run():
        nonlocal worker
        def call():
            try:
                request_json(args.api, "POST", f"/api/agent/runs/{urllib.parse.quote(run_id)}/run", {}, timeout=budget["max_seconds"] + 120)
            except Exception as exc:  # state endpoint remains the source of truth
                errors.append(str(exc))
        worker = threading.Thread(target=call, daemon=True)
        worker.start()

    start_run()
    approval_started = None
    last_status = None
    state: dict[str, Any] = {}
    deadline = time.monotonic() + budget["max_seconds"] + args.run_grace_seconds
    while time.monotonic() < deadline:
        time.sleep(args.poll_seconds)
        state = request_json(args.api, "GET", f"/api/agent/runs/{urllib.parse.quote(run_id)}", timeout=15)
        status = state.get("status", "unknown")
        if status != last_status:
            print(f"{case.name}: {status}", flush=True)
            last_status = status
        if status in TERMINAL:
            break
        if status in {"waiting_approval", "paused"}:
            approval_started = approval_started or time.monotonic()
            if status == "waiting_approval" and args.auto_approve_all:
                approval = approval_request(state)
                pending_tool = (state.get("pending_approval") or {}).get("tool_call") or {}
                request_json(
                    args.api,
                    "POST",
                    f"/api/agent/runs/{urllib.parse.quote(run_id)}/approve",
                    approval,
                )
                record = {
                    "risk": approval["risk"],
                    "tool": pending_tool.get("name"),
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                }
                run_record["auto_approvals"].append(record)
                print(f"{case.name}: auto-approved {record['risk']} tool {record['tool']}", flush=True)
                safe_write(run_dir / "manifest.json", run_record)
                if approval_started is not None:
                    deadline += time.monotonic() - approval_started
                    approval_started = None
                last_status = None
                continue
            if not sys.stdin.isatty():
                run_record["status"] = status
                run_record["pending_approval"] = state.get("pending_approval")
                safe_write(run_dir / "run_state.json", state)
                raise RuntimeError(f"Run {run_id} needs UI approval; rerun with an interactive terminal after preparing resume support")
            pending = state.get("pending_approval") or {}
            print(f"\nRun {run_id} needs {pending.get('risk', 'user')} approval in NaturalCC UI.")
            print("Approve/reject it in the UI, then press Enter here. The script never auto-approves tools.")
            input()
            if approval_started is not None:
                deadline += time.monotonic() - approval_started
                approval_started = None
            state = request_json(args.api, "GET", f"/api/agent/runs/{urllib.parse.quote(run_id)}", timeout=15)
            if state.get("status") in {"running", "queued"} and (worker is None or not worker.is_alive()):
                start_run()
            last_status = None
            continue
        if worker is not None and not worker.is_alive() and status in {"queued", "running"}:
            start_run()

    if state.get("status") not in TERMINAL:
        run_record["timeout"] = True
    events = request_json(args.api, "GET", f"/api/agent/runs/{urllib.parse.quote(run_id)}/events", timeout=30)
    after = file_hashes(workspace)
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    test_changes = [path for path in changed if looks_like_test(path)]
    audit = {
        "changed_files": changed,
        "added_files": sorted(set(after) - set(before)),
        "deleted_files": sorted(set(before) - set(after)),
        "test_path_changes": test_changes,
        "test_tampering_detected": bool(test_changes),
        "before_sha256": before,
        "after_sha256": after,
        "generated_nonblank_added_loc": added_nonblank_lines(source_workspace, workspace, changed),
        "self_check_vulnerability_findings": None,
        "vulnerability_density_per_100_loc": None,
        "official_pass": None,
    }
    run_record.update({
        "status": state.get("status", "unknown"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round((datetime.now(timezone.utc) - datetime.fromisoformat(run_record["started_at"])).total_seconds(), 3),
        "llm_calls": state.get("llm_calls"),
        "tool_calls": state.get("tool_calls"),
        "input_tokens": state.get("input_tokens"),
        "output_tokens": state.get("output_tokens"),
        "cost_usd": state.get("cost_usd"),
        "model": (state.get("runtime_model_config") or {}).get("model"),
        "provider": (state.get("runtime_model_config") or {}).get("provider"),
        "run_errors": errors,
        "test_tampering_detected": bool(test_changes),
    })
    safe_write(run_dir / "run_state.json", state)
    safe_write(run_dir / "events.json", events)
    safe_write(run_dir / "audit.json", audit)
    safe_write(run_dir / "result.json", run_record)
    return run_record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Dataset folder containing task cards")
    parser.add_argument("--codegraph", choices=("on", "off"), default="off")
    parser.add_argument("--api", default=os.environ.get("NATURALCC_API_BASE", "http://127.0.0.1:7860"))
    parser.add_argument("--output-dir", help="Output root; defaults to test/benchmark_runs")
    parser.add_argument("--limit", type=int, default=0, help="Maximum cases; 0 means all")
    parser.add_argument("--target-file", action="append", help="Relative target file; repeat as needed")
    parser.add_argument("--provider", choices=("openrouter", "deepseek"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-compaction-calls", type=int, default=16,
                        help="Maximum context-compaction maintenance model calls per case")
    parser.add_argument("--max-tool-calls", type=int, default=60)
    parser.add_argument("--max-input-tokens", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=24000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--run-grace-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="Validate tasks and print plan; no API calls")
    parser.add_argument(
        "--auto-approve-all",
        action="store_true",
        help=(
            "Automatically approve every pending write/execute/network/git_write action for benchmark Runs. "
            "Dangerous: commands and network tools run with backend host privileges; use only trusted cases."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.dataset_dir = windows_to_wsl_path(args.dataset_dir)
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")
    if args.limit < 0:
        raise SystemExit("--limit must be zero or greater")
    cases = discover_cases(args.dataset_dir)
    if not cases:
        raise SystemExit("No cases found. Each runnable task needs TASK.md and workspace/; prepare full-repository workspaces first.")
    if args.limit:
        cases = cases[:args.limit]
    prepared = [(case, select_targets(case, case / "workspace", args.target_file)) for case in cases]
    print(f"Dataset: {args.dataset_dir}\nCases: {len(prepared)}\nCodeGraph: {args.codegraph}\nMode: {'dry-run' if args.dry_run else 'execute'}")
    for case, targets in prepared:
        print(f"READY {case.name}: workspace={case / 'workspace'} targets={','.join(targets)}")
    if args.dry_run:
        print("Dry-run complete. No API request, model call, or file mutation was made.")
        return 0

    output_root = windows_to_wsl_path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "benchmark_runs"
    if output_root == args.dataset_dir or args.dataset_dir in output_root.parents:
        raise SystemExit("Output directory must not be inside the source dataset directory")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        request_json(args.api, "GET", "/api/health", timeout=10)
    except Exception as exc:
        raise SystemExit(f"NaturalCC API not reachable ({args.api}): {exc}")

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    dataset_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.dataset_dir.name).strip("_") or "dataset"
    run_root = output_root / f"{dataset_name}-{stamp}-{args.codegraph}"
    run_root.mkdir(parents=False, exist_ok=False)
    budget = {
        "max_llm_calls": args.max_llm_calls,
        "max_compaction_calls": args.max_compaction_calls,
        "max_tool_calls": args.max_tool_calls,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "max_seconds": args.max_seconds,
    }
    results = []
    for case, targets in prepared:
        try:
            result = run_case(args, case, run_root / case.name, targets, budget)
        except KeyboardInterrupt:
            print("Interrupted by user; current workspace/run artifacts were retained.", file=sys.stderr)
            return 130
        except Exception as exc:
            result = {"case_id": case.name, "status": "runner_error", "error": str(exc), "codegraph": args.codegraph, "official_pass": None}
            safe_write(run_root / case.name / "runner_error.json", result)
            results.append(result)
            print(f"ERROR {case.name}: {exc}", file=sys.stderr)
            break
        results.append(result)
    with (run_root / "results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "dataset": dataset_name,
        "codegraph": args.codegraph,
        "cases_attempted": len(results),
        "completed_runs": sum(1 for item in results if item.get("status") == "completed"),
        "official_scored": sum(1 for item in results if item.get("official_pass") is not None),
        "test_tampering_cases": sum(1 for item in results if item.get("test_tampering_detected")),
        "official_score": None,
        "note": "Agent run status is not benchmark correctness. Attach official evaluator results before reporting benchmark accuracy.",
    }
    safe_write(run_root / "summary.json", summary)
    print(f"\nRun artifacts: {run_root}\nOfficial score: not run (use the dataset's official evaluator on clean validation workspaces).")
    return 0 if all(item.get("status") == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
