from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.api.agent_routes import create_agent_router
from code_agent.agent_core.contracts import ModelResponse, RiskLevel, ToolCall, ToolContext
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.agent_core.tools.codegraph import CodeGraphClient


GRAPH_CAPABILITIES = {
    "codegraph": {
        "enabled": True,
        "auto_sync": True,
        "hide_workspace_search": True,
    }
}


def _fake_codegraph(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    script = tmp_path / "fake_codegraph.py"
    log_path = tmp_path / "codegraph.log"
    script.write_text(
        """from pathlib import Path
import os
import sys

command = sys.argv[1] if len(sys.argv) > 1 else ""
log = Path(os.environ["FAKE_CODEGRAPH_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(command + "\\n")
database = Path.cwd() / ".codegraph" / "codegraph.db"
if command == "version":
    print("codegraph 1.5.0")
elif command == "init":
    database.parent.mkdir(parents=True, exist_ok=True)
    database.touch()
    print("initialized")
elif command == "status":
    if not database.is_file():
        print("CodeGraph not initialized", file=sys.stderr)
        raise SystemExit(1)
    print("CodeGraph ready")
elif command == "sync":
    if not database.is_file():
        print("CodeGraph not initialized", file=sys.stderr)
        raise SystemExit(1)
    print("synchronized")
elif command == "explore":
    print("src/main.py:1: def main():")
    print("src/main.py:2:     return helper()")
else:
    print("unsupported command", file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        executable = tmp_path / "codegraph.cmd"
        executable.write_text(
            '@python "%~dp0fake_codegraph.py" %*\r\n',
            encoding="utf-8",
        )
    else:
        executable = tmp_path / "codegraph"
        executable.write_text(
            f"#!{sys.executable}\n" + script.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        executable.chmod(0o755)
    monkeypatch.setenv("CODE_AGENT_CODEGRAPH_BIN", str(executable))
    monkeypatch.setenv("FAKE_CODEGRAPH_LOG", str(log_path))
    return executable, log_path


def _initialize_fake_index(workspace: Path) -> None:
    index = workspace / ".codegraph"
    index.mkdir()
    (index / "codegraph.db").touch()


def test_codegraph_client_detects_path_binary_and_explores(tmp_path: Path, monkeypatch):
    _fake_codegraph(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _initialize_fake_index(workspace)

    client = CodeGraphClient(workspace)
    status = client.status()
    result = client.explore("main callers")

    assert status["installed"] is True
    assert status["available"] is True
    assert status["initialized"] is True
    assert status["ready"] is True
    assert status["index_path"] == ".codegraph/codegraph.db"
    assert result.returncode == 0
    assert "def main" in result.stdout


def test_graph_mode_replaces_workspace_search_in_model_tools(tmp_path: Path, monkeypatch):
    _fake_codegraph(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _initialize_fake_index(workspace)
    model = ScriptedModelGateway([ModelResponse(content="done")])
    engine = RunEngine(
        EventStore(tmp_path / "agent.db"),
        build_default_registry(include_mutating=False),
        model,
    )

    run_id = engine.create_run(
        workspace,
        "inspect the call graph",
        capabilities=GRAPH_CAPABILITIES,
    )
    completed = engine.run(run_id)
    names = {item["function"]["name"] for item in model.requests[0].tools}

    assert completed["status"] == "completed"
    assert "codegraph_explore" in names
    assert "workspace_search" not in names
    assert "use codegraph_explore" in model.requests[0].messages[0]["content"]


def test_graph_mode_rejects_workspace_search_even_if_model_requests_it(
    tmp_path: Path,
    monkeypatch,
):
    _fake_codegraph(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _initialize_fake_index(workspace)
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("search-1", "workspace_search", {"query": "needle"})
                ]
            ),
            ModelResponse(content="used graph policy"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(
        workspace,
        "inspect safely",
        capabilities=GRAPH_CAPABILITIES,
    )

    completed = engine.run(run_id)
    finished = [event for event in store.list_events(run_id) if event.type == "tool.finished"]

    assert completed["status"] == "completed"
    assert finished[0].payload["result"]["error"]["type"] == "CodeGraphPreferred"


def test_changed_source_is_synced_before_next_graph_explore(tmp_path: Path, monkeypatch):
    _executable, log_path = _fake_codegraph(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _initialize_fake_index(workspace)
    source = workspace / "main.py"
    source.write_text("value = 1\n", encoding="utf-8")
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit-1",
                        "workspace_apply_patch",
                        {"path": "main.py", "old_text": "value = 1", "new_text": "value = 2"},
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("graph-1", "codegraph_explore", {"query": "main"})
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    run_id = engine.create_run(
        workspace,
        "edit and inspect",
        capabilities=GRAPH_CAPABILITIES,
    )

    assert engine.step(run_id)["status"] == "waiting_approval"
    engine.approve(run_id, RiskLevel.WRITE)
    completed = engine.run(run_id)

    assert completed["status"] == "completed"
    assert completed["codegraph_dirty_files"] == []
    assert log_path.read_text(encoding="utf-8").splitlines().count("sync") >= 2
    event_types = [event.type for event in store.list_events(run_id)]
    assert "codegraph.sync_started" in event_types
    assert "codegraph.sync_finished" in event_types


def test_codegraph_visualization_reads_workspace_relative_database(tmp_path: Path):
    workspace = tmp_path / "workspace"
    index = workspace / ".codegraph"
    index.mkdir(parents=True)
    database = index / "codegraph.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            name TEXT,
            qualified_name TEXT,
            kind TEXT,
            file_path TEXT
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes VALUES ('1', 'main', 'app.main', 'function', 'src/app.py');
        INSERT INTO nodes VALUES ('2', 'helper', 'app.helper', 'function', 'src/app.py');
        INSERT INTO edges VALUES ('1', '2', 'calls');
        """
    )
    connection.commit()
    connection.close()
    output = workspace / ".code-agent" / "runs" / "test" / "graph.html"

    data = CodeGraphClient(workspace).visualize(output, keyword="main")

    rendered = output.read_text(encoding="utf-8")
    assert data["node_count"] == 2
    assert data["edge_count"] == 1
    assert "<canvas id=\"graph\">" in rendered
    assert "src/app.py" in rendered


def test_thread_capabilities_round_trip_through_event_store(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread(
        "Graph task",
        workspace=tmp_path,
        capabilities=GRAPH_CAPABILITIES,
    )

    thread = store.get_thread(thread_id)

    assert thread["capabilities"] == GRAPH_CAPABILITIES


def test_codegraph_thread_api_persists_switch_and_reports_status(
    tmp_path: Path,
    monkeypatch,
):
    _fake_codegraph(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _initialize_fake_index(workspace)
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    created = client.post(
        "/api/agent/threads",
        json={
            "title": "Graph task",
            "workspace": str(workspace),
            "capabilities": GRAPH_CAPABILITIES,
        },
    ).json()
    thread_id = created["thread_id"]
    status = client.get(f"/api/agent/threads/{thread_id}/codegraph/status")

    assert created["thread"]["capabilities"] == GRAPH_CAPABILITIES
    assert status.status_code == 200
    assert status.json()["status"]["ready"] is True

    disabled = client.patch(
        f"/api/agent/threads/{thread_id}",
        json={
            "capabilities": {
                "codegraph": {
                    "enabled": False,
                    "auto_sync": True,
                    "hide_workspace_search": True,
                }
            },
            "expected_version": created["thread"]["version"],
        },
    )
    assert disabled.status_code == 200
    assert disabled.json()["thread"]["capabilities"]["codegraph"]["enabled"] is False
