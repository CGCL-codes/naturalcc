from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from code_agent.agent_core.contracts import RiskLevel, ToolContext
from code_agent.agent_core.policy import PolicyDecision, PolicyEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.agent_core.tools.command import ALLOWED_EXECUTABLES, cancel_run_commands


def test_command_allows_cpp_compiler_executables():
    assert {"g++", "gcc", "c++", "clang++"} <= ALLOWED_EXECUTABLES


def test_policy_requires_approval_for_mutation_and_execution():
    policy = PolicyEngine()
    assert policy.decide("run", "read", RiskLevel.READ) == PolicyDecision.ALLOW
    assert policy.decide("run", "edit", RiskLevel.WRITE) == PolicyDecision.REQUIRE_APPROVAL
    policy.grant("run", RiskLevel.WRITE)
    assert policy.decide("run", "edit", RiskLevel.WRITE) == PolicyDecision.ALLOW


def test_exact_patch_and_command_are_workspace_scoped(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    registry = build_default_registry(include_mutating=True)
    context = ToolContext(
        "run",
        tmp_path,
        tmp_path / ".artifacts",
        approved_risks={RiskLevel.WRITE, RiskLevel.EXECUTE},
    )
    changed = registry.execute(
        "workspace.apply_patch",
        {"path": "app.py", "old_text": "value = 1", "new_text": "value = 2"},
        context,
    )
    restored = registry.execute("workspace.restore_snapshot", {"path": "app.py"}, context)
    command = registry.execute(
        "command.run",
        {"argv": ["python3", "-c", "print('verified')"], "timeout_seconds": 5},
        context,
    )
    denied = registry.execute(
        "command.run",
        {"argv": ["rm", "-rf", "."], "timeout_seconds": 5},
        context,
    )
    assert changed.changed_files == ["app.py"]
    assert restored.changed_files == ["app.py"]
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert command.data["stdout"].strip() == "verified"
    assert denied.status == "error"


def test_snapshot_restore_returns_to_run_initial_content(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    registry = build_default_registry(include_mutating=True)
    context = ToolContext("run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE})

    registry.execute(
        "workspace.apply_patch",
        {"path": "app.py", "old_text": "value = 1", "new_text": "value = 2"},
        context,
    )
    registry.execute(
        "workspace.apply_patch",
        {"path": "app.py", "old_text": "value = 2", "new_text": "value = 3"},
        context,
    )
    restored = registry.execute("workspace.restore_snapshot", {"path": "app.py"}, context)

    assert restored.status == "success"
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_create_file_requires_approval_and_never_overwrites(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    unapproved = ToolContext("run", tmp_path, tmp_path / ".artifacts")
    denied = registry.execute(
        "workspace.create_file",
        {"path": "graph.cpp", "content": "int main() {}\n"},
        unapproved,
    )
    assert denied.status == "error"
    assert denied.error["type"] == "ApprovalRequired"
    assert not (tmp_path / "graph.cpp").exists()

    approved = ToolContext(
        "run",
        tmp_path,
        tmp_path / ".artifacts-approved",
        approved_risks={RiskLevel.WRITE},
    )
    created = registry.execute(
        "workspace.create_file",
        {"path": "graph.cpp", "content": "int main() {}\n"},
        approved,
    )
    duplicate = registry.execute(
        "workspace.create_file",
        {"path": "graph.cpp", "content": "overwritten\n"},
        approved,
    )

    assert created.status == "success"
    assert created.changed_files == ["graph.cpp"]
    assert created.data["created"] is True
    assert "+++ b/graph.cpp" in created.data["diff"]
    assert duplicate.status == "error"
    assert duplicate.error["type"] == "FileExistsError"
    assert (tmp_path / "graph.cpp").read_text(encoding="utf-8") == "int main() {}\n"


def test_create_file_requires_existing_parent_and_stays_in_workspace(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    context = ToolContext(
        "run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE}
    )

    missing_parent = registry.execute(
        "workspace.create_file",
        {"path": "missing/graph.cpp", "content": "content"},
        context,
    )
    escaped = registry.execute(
        "workspace.create_file",
        {"path": "../escaped.cpp", "content": "content"},
        context,
    )
    sensitive = registry.execute(
        "workspace.create_file",
        {"path": ".env", "content": "API_KEY=must-not-be-written"},
        context,
    )

    assert missing_parent.status == "error"
    assert missing_parent.error["type"] == "FileNotFoundError"
    assert escaped.status == "error"
    assert "escapes workspace" in escaped.summary
    assert not (tmp_path.parent / "escaped.cpp").exists()
    assert sensitive.status == "error"
    assert "sensitive path" in sensitive.summary
    assert not (tmp_path / ".env").exists()


def test_restore_snapshot_removes_file_created_during_run(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    context = ToolContext(
        "run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE}
    )
    registry.execute(
        "workspace.create_file",
        {"path": "new.py", "content": "value = 1\n"},
        context,
    )
    registry.execute(
        "workspace.apply_patch",
        {"path": "new.py", "old_text": "value = 1", "new_text": "value = 2"},
        context,
    )

    restored = registry.execute(
        "workspace.restore_snapshot", {"path": "new.py"}, context
    )

    assert restored.status == "success"
    assert restored.data["restored_absence"] is True
    assert not (tmp_path / "new.py").exists()


def test_create_directory_supports_parent_chain_and_is_idempotent(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    unapproved = ToolContext("run", tmp_path, tmp_path / ".unapproved")
    denied = registry.execute(
        "workspace.create_directory",
        {"path": "zidong/library", "parents": True},
        unapproved,
    )
    assert denied.status == "error"
    assert denied.error["type"] == "ApprovalRequired"

    context = ToolContext(
        "run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE}
    )
    missing_parent = registry.execute(
        "workspace.create_directory",
        {"path": "zidong/library"},
        context,
    )
    created = registry.execute(
        "workspace.create_directory",
        {"path": "zidong/library", "parents": True},
        context,
    )
    repeated = registry.execute(
        "workspace.create_directory",
        {"path": "zidong/library"},
        context,
    )

    assert missing_parent.status == "error"
    assert missing_parent.error["type"] == "FileNotFoundError"
    assert created.status == "success"
    assert created.data["created_directories"] == ["zidong", "zidong/library"]
    assert (tmp_path / "zidong" / "library").is_dir()
    assert repeated.status == "success"
    assert repeated.data["created"] is False


def test_create_directory_rejects_files_escape_and_sensitive_paths(tmp_path: Path):
    (tmp_path / "occupied").write_text("file", encoding="utf-8")
    registry = build_default_registry(include_mutating=True)
    context = ToolContext(
        "run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE}
    )

    occupied = registry.execute(
        "workspace.create_directory", {"path": "occupied"}, context
    )
    escaped = registry.execute(
        "workspace.create_directory",
        {"path": "../escaped-directory"},
        context,
    )
    sensitive = registry.execute(
        "workspace.create_directory",
        {"path": ".ssh/cache", "parents": True},
        context,
    )

    assert occupied.error["type"] == "FileExistsError"
    assert "escapes workspace" in escaped.summary
    assert "sensitive path" in sensitive.summary
    assert not (tmp_path.parent / "escaped-directory").exists()
    assert not (tmp_path / ".ssh").exists()


def test_restore_snapshot_removes_only_empty_created_directories(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    context = ToolContext(
        "run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.WRITE}
    )
    registry.execute(
        "workspace.create_directory",
        {"path": "parent/child", "parents": True},
        context,
    )

    child = registry.execute(
        "workspace.restore_snapshot", {"path": "parent/child"}, context
    )
    parent = registry.execute(
        "workspace.restore_snapshot", {"path": "parent"}, context
    )

    assert child.status == "success"
    assert parent.status == "success"
    assert not (tmp_path / "parent").exists()


def test_running_command_can_be_cancelled_with_its_run(tmp_path: Path):
    registry = build_default_registry(include_mutating=True)
    context = ToolContext("cancel-run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.EXECUTE})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            registry.execute,
            "command.run",
            {"argv": ["python3", "-c", "import time; time.sleep(10)"], "timeout_seconds": 20},
            context,
        )
        time.sleep(0.25)
        cancel_run_commands("cancel-run")
        result = future.result(timeout=5)
    assert result.status == "cancelled"


def test_command_rejects_absolute_executable_spoofing(tmp_path: Path):
    fake = tmp_path / "python3"
    fake.write_text("#!/bin/sh\necho spoofed\n", encoding="utf-8")
    fake.chmod(0o755)
    registry = build_default_registry(include_mutating=True)
    context = ToolContext("run", tmp_path, tmp_path / ".artifacts", approved_risks={RiskLevel.EXECUTE})
    result = registry.execute("command.run", {"argv": [str(fake)]}, context)
    assert result.status == "error"
    assert "bare executable name" in result.summary
