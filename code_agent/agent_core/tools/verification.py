from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .command import _run as _run_command


def discover_test_commands(workspace: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    if (workspace / "pyproject.toml").is_file() or any(workspace.glob("test*.py")) or (workspace / "tests").is_dir():
        commands.append(["python", "-m", "pytest"])
    if (workspace / "package.json").is_file():
        try:
            package = json.loads((workspace / "package.json").read_text(encoding="utf-8"))
        except Exception:
            package = {}
        if "test" in package.get("scripts", {}):
            commands.append(["npm", "test"])
    if (workspace / "pom.xml").is_file():
        commands.append(["mvn", "test"])
    if (workspace / "build.gradle").is_file() or (workspace / "build.gradle.kts").is_file():
        commands.append(["gradle", "test"])
    if (workspace / "Cargo.toml").is_file():
        commands.append(["cargo", "test"])
    if (workspace / "CMakeLists.txt").is_file():
        commands.append(["ctest", "--test-dir", "build", "--output-on-failure"])
    if (workspace / "Makefile").is_file():
        commands.append(["make", "test"])
    return commands


def _git_status(context: ToolContext, _args: dict) -> ToolResult:
    if shutil.which("git") is None:
        return ToolResult.success("Git is not installed", data={"is_repo": False, "status": ""})
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=context.workspace,
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        return ToolResult.success("Workspace is not a Git repository", data={"is_repo": False, "status": ""})
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=context.workspace,
        text=True,
        capture_output=True,
    )
    return ToolResult.success(status.stdout, data={"is_repo": True, "status": status.stdout})


def _git_diff(context: ToolContext, _args: dict) -> ToolResult:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--"],
        cwd=context.workspace,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return ToolResult.failure(result.stderr or "git diff failed", "GitError")
    return ToolResult.success(result.stdout, data={"diff": result.stdout})


def _discover(context: ToolContext, _args: dict) -> ToolResult:
    commands = discover_test_commands(context.workspace)
    return ToolResult.success(f"Discovered {len(commands)} verification commands", data={"commands": commands})


def _run_tests(context: ToolContext, args: dict) -> ToolResult:
    discovered = discover_test_commands(context.workspace)
    requested = args.get("argv")
    if requested is None:
        if not discovered:
            raise ValueError("no supported test command was discovered")
        requested = discovered[0]
    if requested not in discovered:
        raise ValueError("tests.run only accepts a command returned by tests.discover")
    result = _run_command(
        context,
        {
            "argv": requested,
            "timeout_seconds": args.get("timeout_seconds", 300),
            "max_output_chars": args.get("max_output_chars", 40_000),
        },
    )
    result.data["verification"] = True
    return result


def verification_tool_specs() -> list[ToolSpec]:
    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        ToolSpec("git.status", "Read Git workspace status.", empty_schema, RiskLevel.READ, _git_status),
        ToolSpec("git.diff", "Read the current Git diff.", empty_schema, RiskLevel.READ, _git_diff),
        ToolSpec("tests.discover", "Discover likely project test commands without running them.", empty_schema, RiskLevel.READ, _discover),
        ToolSpec(
            "tests.run",
            "Run one approved test command that was discovered from project files.",
            {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "timeout_seconds": {"type": "integer"},
                    "max_output_chars": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            RiskLevel.EXECUTE,
            _run_tests,
            idempotent=False,
            parallel_safe=False,
            default_timeout_seconds=300,
        ),
    ]
