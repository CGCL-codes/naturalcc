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


# [maint-note:b5d051ae53af]
# 维护导航：_terminate_process；下列为本函数源码中的接口与路径索引。
# 输入 process：subprocess.Popen。
# 返回契约：None。
# 分支条件：process.poll() is not None。
# 分支条件：os.name != 'nt'。
# 返回路径：无返回值。
# 捕获异常：(OSError, ProcessLookupError)。
# 直接调用：process.poll。
# 直接调用：os.killpg。
# 直接调用：process.terminate。
# [/maint-note:b5d051ae53af]
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


# [maint-note:5e2c1d434660]
# 维护导航：cancel_run_commands；下列为本函数源码中的接口与路径索引。
# 输入 run_id：str。
# 返回契约：None。
# 分支条件：process is not None。
# 直接调用：_CANCELLED_RUNS.add。
# 直接调用：_ACTIVE_PROCESSES.get。
# 直接调用：_terminate_process。
# 资源作用域：_PROCESS_LOCK；退出该 with 时调用对应上下文清理。
# [/maint-note:5e2c1d434660]
def cancel_run_commands(run_id: str) -> None:
    with _PROCESS_LOCK:
        _CANCELLED_RUNS.add(run_id)
        process = _ACTIVE_PROCESSES.get(run_id)
    if process is not None:
        _terminate_process(process)


# [maint-note:cf5a92c95b4c]
# 维护导航：_run；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not argv。
# 分支条件：argv[0] != executable。
# 分支条件：executable not in ALLOWED_EXECUTABLES。
# 分支条件：executable == 'git' and len(argv) > 1 and (argv[1] in DENIED_GIT_SUBCOMMANDS)。
# 分支条件：not cwd.is_dir()。
# 分支条件：os.name == 'nt'。
# 分支条件：value。
# 分支条件：executable in {'python', 'python3'}。
# 分支条件：cancelled。
# 遍历边界：key ← ('COMSPEC', 'SystemRoot', 'WINDIR', 'PATHEXT', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA')。
# 返回路径：ToolResult('timeout', f'command timed out after {timeout_seconds}s', data={'stdout': (stdout or '')[:max_output_chars], 'stderr': (stderr or '')[:max_output_chars]})。
# 返回路径：ToolResult('cancelled', 'command cancelled with run', data={'stdout': stdout[:max_output_chars], 'stderr': stderr[:max_output_chars]}, exit_code=process.returncode)。
# 返回路径：ToolResult(status, combined or f'command exited with {process.returncode}', data={'argv': argv, 'launch_argv': launch_argv, 'cwd': cwd.relative_to(context.workspace).as_posix() or '.', 'stdout': stdout[:max_output_chars], 'stderr': stderr[:。
# 显式抛出：ValueError('argv must not be empty')。
# 显式抛出：ValueError('command executable must be a bare executable name')。
# 显式抛出：ValueError(f'executable is not allowed: {executable}')。
# 显式抛出：ValueError(f'git subcommand is not allowed: {argv[1]}')。
# 显式抛出：ValueError('command cwd must be a directory')。
# 捕获异常：subprocess.TimeoutExpired。
# 写入目标：env[key]。
# 写入目标：_ACTIVE_PROCESSES[context.run_id]。
# 直接调用：ValueError。
# 直接调用：Path。
# 直接调用：len。
# 直接调用：resolve_workspace_path。
# 直接调用：args.get。
# 直接调用：cwd.is_dir。
# 直接调用：min。
# 直接调用：str。
# 直接调用：os.environ.get。
# 直接调用：os.pathsep.join。
# 直接调用：shutil.which。
# 直接调用：subprocess.Popen。
# 直接调用：_CANCELLED_RUNS.discard。
# 直接调用：process.communicate。
# 直接调用：_terminate_process。
# 直接调用：ToolResult。
# 直接调用：_ACTIVE_PROCESSES.pop。
# 直接调用：cwd.relative_to(context.workspace).as_posix。
# 直接调用：cwd.relative_to。
# 集合派生：part 来自 (runtime_bin, inherited_path)；过滤 part。
# 资源作用域：_PROCESS_LOCK；退出该 with 时调用对应上下文清理。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# 表达式分流：stdout and stderr；真值与假值分支分别计算，不同时执行。
# 表达式分流：process.returncode == 0；真值与假值分支分别计算，不同时执行。
# 表达式分流：status == 'success'；真值与假值分支分别计算，不同时执行。
# [/maint-note:cf5a92c95b4c]
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


# [maint-note:87f19fe60d2a]
# 维护导航：command_tool_spec；下列为本函数源码中的接口与路径索引。
# 返回契约：ToolSpec。
# 返回路径：ToolSpec('command.run', 'Run an approved argv-only build or test command in the locked workspace.', {'type': 'object', 'properties': {'argv': {'type': 'array', 'items': {'type': 'string'}}, 'cwd': {'type': 'string'}, 'timeout_seconds': {'ty。
# 直接调用：ToolSpec。
# [/maint-note:87f19fe60d2a]
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
