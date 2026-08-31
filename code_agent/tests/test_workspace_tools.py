from pathlib import Path

from code_agent.agent_core.contracts import ToolContext
from code_agent.agent_core.tool_registry import build_default_registry


def test_workspace_tools_read_search_and_reject_traversal(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    artifacts = tmp_path / ".agent-artifacts"
    registry = build_default_registry(include_mutating=False)
    context = ToolContext("run", tmp_path, artifacts)

    listed = registry.execute("workspace.list", {"path": "src"}, context)
    read = registry.execute("workspace.read", {"path": "src/main.py"}, context)
    search = registry.execute("workspace.search", {"query": "hello", "path": "src"}, context)
    escaped = registry.execute("workspace.read", {"path": "../secret.txt"}, context)

    assert listed.data["entries"][0]["path"] == "src/main.py"
    assert "return 'world'" in read.data["content"]
    assert search.data["matches"][0]["line"] == 1
    assert escaped.status == "error"


def test_workspace_tools_hide_common_secret_files(tmp_path: Path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("safe", encoding="utf-8")
    registry = build_default_registry(include_mutating=False)
    context = ToolContext("run", tmp_path, tmp_path / "artifacts")

    listed = registry.execute("workspace.list", {}, context)
    denied = registry.execute("workspace.read", {"path": ".env"}, context)
    searched = registry.execute("workspace.search", {"query": "secret"}, context)

    assert ".env" not in [entry["path"] for entry in listed.data["entries"]]
    assert denied.status == "error"
    assert searched.data["matches"] == []


def test_default_registry_exposes_naturalcc_as_read_only_tools():
    registry = build_default_registry(include_mutating=False)
    assert registry.get("naturalcc.parse").risk_level.value == "read"
    assert registry.get("naturalcc.symbol_search").risk_level.value == "read"
    assert build_default_registry(include_mutating=True).get("tests.run").risk_level.value == "execute"


def test_workspace_read_allows_only_explicitly_authorized_external_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "shared.py"
    sibling = tmp_path / "secret.py"
    external.write_text("shared = True\n", encoding="utf-8")
    sibling.write_text("secret = True\n", encoding="utf-8")
    registry = build_default_registry(include_mutating=False)
    context = ToolContext(
        "run",
        workspace,
        workspace / ".agent-artifacts",
        authorized_paths={external.resolve()},
    )

    allowed = registry.execute("workspace.read", {"path": str(external)}, context)
    denied = registry.execute("workspace.read", {"path": str(sibling)}, context)

    assert allowed.status == "success"
    assert allowed.data["path"] == external.resolve().as_posix()
    assert "shared = True" in allowed.data["content"]
    assert denied.status == "error"
