from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .workspace import resolve_workspace_path


ALLOWED_EXECUTABLES = {
    "python",
    "python3",
    "pytest",
    "node",
    "npm",
    "make",
    "cmake",
    "ctest",
    "g++",
    "gcc",
    "c++",
    "clang++",
    "mvn",
    "mvnw",
    "gradle",
    "gradlew",
    "cargo",
    "git",
}
DENIED_GIT_SUBCOMMANDS = {"push", "commit", "reset", "clean", "checkout", "switch"}
_ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
_CANCELLED_RUNS: set[str] = set()
_PROCESS_LOCK = threading.Lock()


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        process.terminate()


def cancel_run_commands(run_id: str) -> None:
    with _PROCESS_LOCK:
        _CANCELLED_RUNS.add(run_id)
        process = _ACTIVE_PROCESSES.get(run_id)
    if process is not None:
        _terminate_process(process)


def _run(context: ToolContext, args: dict) -> ToolResult:
    argv = args["argv"]
    if not argv:
        raise ValueError("argv must not be empty")
    executable = Path(argv[0]).name
    if argv[0] != executable:
        raise ValueError("command executable must be a bare executable name")
    if executable not in ALLOWED_EXECUTABLES:
        raise ValueError(f"executable is not allowed: {executable}")
    if executable == "git" and len(argv) > 1 and argv[1] in DENIED_GIT_SUBCOMMANDS:
        raise ValueError(f"git subcommand is not allowed: {argv[1]}")
    cwd = resolve_workspace_path(context, args.get("cwd", "."), must_exist=True)
    if not cwd.is_dir():
        raise ValueError("command cwd must be a directory")
    timeout_seconds = min(args.get("timeout_seconds", 60), 900)
    max_output_chars = min(args.get("max_output_chars", 20_000), 200_000)
    runtime_bin = str(Path(sys.executable).parent)
    inherited_path = os.environ.get("PATH", "")
    env = {
        "PATH": os.pathsep.join(part for part in (runtime_bin, inherited_path) if part),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
        "NO_COLOR": "1",
    }
    if os.name == "nt":
        for key in ("COMSPEC", "SystemRoot", "WINDIR", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    if executable in {"python", "python3"}:
        launch_argv = [sys.executable, *argv[1:]]
    else:
        resolved_executable = shutil.which(executable, path=env["PATH"])
        launch_argv = [resolved_executable or argv[0], *argv[1:]]
    process = subprocess.Popen(
            launch_argv,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name != "nt",
        )
    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES[context.run_id] = process
        _CANCELLED_RUNS.discard(context.run_id)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        stdout, stderr = process.communicate()
        return ToolResult(
            "timeout",
            f"command timed out after {timeout_seconds}s",
            data={"stdout": (stdout or "")[:max_output_chars], "stderr": (stderr or "")[:max_output_chars]},
        )
    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.pop(context.run_id, None)
    with _PROCESS_LOCK:
        cancelled = context.run_id in _CANCELLED_RUNS
        _CANCELLED_RUNS.discard(context.run_id)
    stdout = stdout or ""
    stderr = stderr or ""
    if cancelled:
        return ToolResult(
            "cancelled",
            "command cancelled with run",
            data={"stdout": stdout[:max_output_chars], "stderr": stderr[:max_output_chars]},
            exit_code=process.returncode,
        )
    combined = (stdout + ("\n" if stdout and stderr else "") + stderr)[:max_output_chars]
    status = "success" if process.returncode == 0 else "error"
    return ToolResult(
        status,
        combined or f"command exited with {process.returncode}",
        data={
            "argv": argv,
            "launch_argv": launch_argv,
            "cwd": cwd.relative_to(context.workspace).as_posix() or ".",
            "stdout": stdout[:max_output_chars],
            "stderr": stderr[:max_output_chars],
        },
        exit_code=process.returncode,
        truncated=len(stdout) + len(stderr) > max_output_chars,
        error=None if status == "success" else {"type": "CommandFailed", "message": f"exit code {process.returncode}"},
    )


def command_tool_spec() -> ToolSpec:
    return ToolSpec(
        "command.run",
        "Run an approved argv-only build or test command in the locked workspace.",
        {
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
                "max_output_chars": {"type": "integer"},
            },
            "required": ["argv"],
            "additionalProperties": False,
        },
        RiskLevel.EXECUTE,
        _run,
        idempotent=False,
        parallel_safe=False,
        default_timeout_seconds=60,
        max_output_chars=20_000,
    )
