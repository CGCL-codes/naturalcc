from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .command import _run as _run_command


# [maint-note:9e55fa088b92]
# 维护导航：discover_test_commands；下列为本函数源码中的接口与路径索引。
# 输入 workspace：Path。
# 返回契约：list[list[str]]。
# 分支条件：(workspace / 'pyproject.toml').is_file() or any(workspace.glob('test*.py')) or (workspace / 'tests').is_dir()。
# 分支条件：(workspace / 'package.json').is_file()。
# 分支条件：'test' in package.get('scripts', {})。
# 分支条件：(workspace / 'pom.xml').is_file()。
# 分支条件：(workspace / 'build.gradle').is_file() or (workspace / 'build.gradle.kts').is_file()。
# 分支条件：(workspace / 'Cargo.toml').is_file()。
# 分支条件：(workspace / 'CMakeLists.txt').is_file() and (workspace / 'build').is_dir()。
# 分支条件：(workspace / 'Makefile').is_file()。
# 返回路径：commands。
# 捕获异常：Exception。
# 直接调用：(workspace / 'pyproject.toml').is_file。
# 直接调用：any。
# 直接调用：workspace.glob。
# 直接调用：(workspace / 'tests').is_dir。
# 直接调用：commands.append。
# 直接调用：(workspace / 'package.json').is_file。
# 直接调用：json.loads。
# 直接调用：(workspace / 'package.json').read_text。
# 直接调用：package.get。
# 直接调用：(workspace / 'pom.xml').is_file。
# 直接调用：(workspace / 'build.gradle').is_file。
# 直接调用：(workspace / 'build.gradle.kts').is_file。
# 直接调用：(workspace / 'Cargo.toml').is_file。
# 直接调用：(workspace / 'CMakeLists.txt').is_file。
# 直接调用：(workspace / 'build').is_dir。
# 直接调用：(workspace / 'Makefile').is_file。
# [/maint-note:9e55fa088b92]
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
    # ctest requires an existing configured build tree. A source-level
    # CMakeLists.txt alone is not enough to run it and would produce a
    # platform-specific FileNotFoundError on Windows.
    if (workspace / "CMakeLists.txt").is_file() and (workspace / "build").is_dir():
        commands.append(["ctest", "--test-dir", "build", "--output-on-failure"])
    if (workspace / "Makefile").is_file():
        commands.append(["make", "test"])
    return commands


# [maint-note:20eed4178e36]
# 维护导航：matches_test_command；下列为本函数源码中的接口与路径索引。
# 输入 argv：list[str]。
# 输入 discovered：list[list[str]]。
# 返回契约：bool。
# 分支条件：not isinstance(argv, list) or not argv or (not all((isinstance(arg, str) for arg in argv)))。
# 分支条件：argv in discovered。
# 分支条件：['python', '-m', 'pytest'] not in discovered。
# 分支条件：argv[:1] == ['pytest']。
# 分支条件：argv[0] in {'python', 'python3'} and argv[1:3] == ['-m', 'pytest']。
# 返回路径：False。
# 返回路径：True。
# 返回路径：all((flag in {'-q', '-qq', '--quiet', '-v', '-vv', '--verbose'} for flag in flags))。
# 直接调用：isinstance。
# 直接调用：all。
# 集合派生：arg 来自 argv。
# 集合派生：flag 来自 flags。
# [/maint-note:20eed4178e36]
def matches_test_command(argv: list[str], discovered: list[list[str]]) -> bool:
    """Recognize full discovered suites, allowing only pytest reporting flags.

    Do not accept selectors, paths, collection-only, config overrides or plugin
    options: a successful partial/dry run is not full-suite verification.
    """
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        return False
    if argv in discovered:
        return True
    if ["python", "-m", "pytest"] not in discovered:
        return False
    if argv[:1] == ["pytest"]:
        flags = argv[1:]
    elif argv[0] in {"python", "python3"} and argv[1:3] == ["-m", "pytest"]:
        flags = argv[3:]
    else:
        return False
    return all(flag in {"-q", "-qq", "--quiet", "-v", "-vv", "--verbose"} for flag in flags)


# [maint-note:08c481f41782]
# 维护导航：_git_status；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict。
# 返回契约：ToolResult。
# 分支条件：shutil.which('git') is None。
# 分支条件：probe.returncode != 0。
# 返回路径：ToolResult.success('Git is not installed', data={'is_repo': False, 'status': ''})。
# 返回路径：ToolResult.success('Workspace is not a Git repository', data={'is_repo': False, 'status': ''})。
# 返回路径：ToolResult.success(stdout, data={'is_repo': True, 'status': stdout})。
# 直接调用：shutil.which。
# 直接调用：ToolResult.success。
# 直接调用：subprocess.run。
# [/maint-note:08c481f41782]
def _git_status(context: ToolContext, _args: dict) -> ToolResult:
    if shutil.which("git") is None:
        return ToolResult.success("Git is not installed", data={"is_repo": False, "status": ""})
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=context.workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if probe.returncode != 0:
        return ToolResult.success("Workspace is not a Git repository", data={"is_repo": False, "status": ""})
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=context.workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    stdout = status.stdout or ""
    return ToolResult.success(stdout, data={"is_repo": True, "status": stdout})


# [maint-note:dbe0660fb6b9]
# 维护导航：_git_diff；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict。
# 返回契约：ToolResult。
# 分支条件：result.returncode != 0。
# 返回路径：ToolResult.failure(result.stderr or 'git diff failed', 'GitError')。
# 返回路径：ToolResult.success(stdout, data={'diff': stdout})。
# 直接调用：subprocess.run。
# 直接调用：ToolResult.failure。
# 直接调用：ToolResult.success。
# [/maint-note:dbe0660fb6b9]
def _git_diff(context: ToolContext, _args: dict) -> ToolResult:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--"],
        cwd=context.workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        return ToolResult.failure(result.stderr or "git diff failed", "GitError")
    stdout = result.stdout or ""
    return ToolResult.success(stdout, data={"diff": stdout})


# [maint-note:accba6c384fa]
# 维护导航：_discover；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(f'Discovered {len(commands)} verification commands. Run one with tests.run using its command_index.', data={'commands': commands, 'command_index': list(range(len(commands)))})。
# 直接调用：discover_test_commands。
# 直接调用：ToolResult.success。
# 直接调用：len。
# 直接调用：list。
# 直接调用：range。
# [/maint-note:accba6c384fa]
def _discover(context: ToolContext, _args: dict) -> ToolResult:
    commands = discover_test_commands(context.workspace)
    return ToolResult.success(
        f"Discovered {len(commands)} verification commands. Run one with tests.run using its command_index.",
        data={
            "commands": commands,
            "command_index": list(range(len(commands))),
        },
    )


# [maint-note:2dd978f8565a]
# 维护导航：_run_tests；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：command_index is not None。
# 分支条件：not isinstance(command_index, int) or not 0 <= command_index < len(discovered)。
# 分支条件：requested is None。
# 分支条件：not discovered。
# 分支条件：not matches_test_command(requested, discovered)。
# 返回路径：result。
# 显式抛出：ValueError(f'invalid command_index; choose an index from 0 to {len(discovered) - 1} returned by tests.discover')。
# 显式抛出：ValueError('no supported test command was discovered')。
# 显式抛出：ValueError('tests.run only accepts a command returned by tests.discover')。
# 写入目标：result.data['verification']。
# 直接调用：discover_test_commands。
# 直接调用：args.get。
# 直接调用：isinstance。
# 直接调用：len。
# 直接调用：ValueError。
# 直接调用：matches_test_command。
# 直接调用：_run_command。
# [/maint-note:2dd978f8565a]
def _run_tests(context: ToolContext, args: dict) -> ToolResult:
    discovered = discover_test_commands(context.workspace)
    command_index = args.get("command_index")
    requested = args.get("argv")
    if command_index is not None:
        if not isinstance(command_index, int) or not 0 <= command_index < len(discovered):
            raise ValueError(
                f"invalid command_index; choose an index from 0 to {len(discovered) - 1} returned by tests.discover"
            )
        requested = discovered[command_index]
    elif requested is None:
        if not discovered:
            raise ValueError("no supported test command was discovered")
        requested = discovered[0]
    if not matches_test_command(requested, discovered):
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


# [maint-note:5dc630868ad8]
# 维护导航：verification_tool_specs；下列为本函数源码中的接口与路径索引。
# 返回契约：list[ToolSpec]。
# 返回路径：[ToolSpec('git.status', 'Read Git workspace status.', empty_schema, RiskLevel.READ, _git_status), ToolSpec('git.diff', 'Read the current Git diff.', empty_schema, RiskLevel.READ, _git_diff), ToolSpec('tests.discover', 'Discover likely proje。
# 直接调用：ToolSpec。
# [/maint-note:5dc630868ad8]
def verification_tool_specs() -> list[ToolSpec]:
    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        ToolSpec("git.status", "Read Git workspace status.", empty_schema, RiskLevel.READ, _git_status),
        ToolSpec("git.diff", "Read the current Git diff.", empty_schema, RiskLevel.READ, _git_diff),
        ToolSpec("tests.discover", "Discover likely project test commands without running them.", empty_schema, RiskLevel.READ, _discover),
        ToolSpec(
            "tests.run",
            "Run a discovered test command. Prefer command_index from tests.discover; omit arguments to run the first discovered command. Pytest quiet/verbose flags are accepted; test selection and collection-only options are not.",
            {
                "type": "object",
                "properties": {
                    "command_index": {"type": "integer", "minimum": 0},
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
