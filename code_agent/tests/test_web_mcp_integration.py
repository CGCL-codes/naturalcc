import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse, ToolCall
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.api import agent_routes


_SERVER = Path(__file__).parent / "fixtures" / "mcp_notes_server.py"
_SECRET = "test-mcp-credential-not-for-model"


def _client(tmp_path, monkeypatch, model, *, mode="normal", two_servers=False):
    servers = {}
    for name in (["notes", "docs"] if two_servers else ["notes"]):
        servers[name] = {
            "command": sys.executable,
            "args": [str(_SERVER), name, str(tmp_path / f"{name}.jsonl"), mode],
            "env_allowlist": ["NCC_MCP_TEST_TOKEN"],
            "read_only_tools": ["lookup"],
            "resources": True,
        }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    monkeypatch.setenv("CODE_AGENT_MCP_CONFIG", str(path))
    monkeypatch.setenv("CODE_AGENT_DB", str(tmp_path / "agent.db"))
    monkeypatch.setenv("CODE_AGENT_AUTO_MEMORY_MODE", "off")
    monkeypatch.setenv("NCC_MCP_TEST_TOKEN", _SECRET)
    monkeypatch.setattr(agent_routes, "RoutedModelGateway", lambda **kwargs: model)
    app = FastAPI()
    app.include_router(agent_routes.build_default_agent_router())
    return TestClient(app)


def _create(client, tmp_path):
    created = client.post("/api/agent/runs", json={"workspace": str(tmp_path), "goal": "Read project notes."})
    assert created.status_code == 200, created.text
    return f"/api/agent/runs/{created.json()['run_id']}"


def _result(state, call_id):
    return json.loads(next(msg["content"] for msg in state["messages"] if msg.get("tool_call_id") == call_id))


def _ledger(tmp_path, name):
    path = tmp_path / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_real_web_router_lists_calls_and_reads_mcp_with_exact_approval(tmp_path, monkeypatch):
    model = ScriptedModelGateway([ModelResponse(tool_calls=[
        ToolCall("notes-read", "mcp__notes__lookup"),
        ToolCall("docs-read", "mcp__docs__lookup"),
        ToolCall("resources", "mcp__notes__resources_list"),
        ToolCall("reject", "mcp__docs__write_note", {"note": "don't send"}),
    ])])
    with _client(tmp_path, monkeypatch, model, two_servers=True) as client:
        path = "/api/agent/mcp/servers"
        assert {server["status"] for server in client.get(path).json()["servers"]} == {"disconnected"}
        assert client.post(f"{path}/notes/connect", headers={"Origin": "https://example.org"}).status_code == 403
        for server in ("notes", "docs"):
            connected = client.post(f"{path}/{server}/connect").json()["server"]
            assert connected["status"] == "connected", connected
            assert f"mcp__{server}__lookup" in connected["tools"]
            assert _SECRET not in json.dumps(connected)

        run = _create(client, tmp_path)
        state = client.post(f"{run}/run").json()
        assert state["status"] == "waiting_approval"
        assert state["pending_approval"]["risk"] == "execute"
        assert state["pending_approval"]["tool_call"]["id"] == "reject"
        assert "Note from notes" in json.dumps(_result(state, "notes-read"))
        assert "Note from docs" in json.dumps(_result(state, "docs-read"))
        resource = _result(state, "resources")["data"]["result"]["resources"][0]
        model._responses.extend([
            ModelResponse(tool_calls=[ToolCall("page", "mcp__notes__resources_read", {
                "resource_id": resource["resource_id"],
            })]),
            ModelResponse(content="Read the authorized resource."),
        ])

        assert client.post(f"{run}/reject").status_code == 200
        state = client.post(f"{run}/run").json()
        assert state["pending_approval"]["risk"] == "network"
        assert client.post(f"{run}/approve", json={"risk": "network", "tool_call_id": "wrong"}).status_code == 409
        assert client.post(f"{run}/approve", json={"risk": "network"}).status_code == 409
        assert all(entry["method"] != "resources/read" for entry in _ledger(tmp_path, "notes"))
        assert client.post(f"{run}/approve", json={"risk": "network", "tool_call_id": "page"}).status_code == 200
        state = client.post(f"{run}/run").json()
        assert state["status"] == "completed"
        assert "Resource from notes" in json.dumps(_result(state, "page"))
        assert not any(entry["params"].get("name") == "write_note" for entry in _ledger(tmp_path, "docs"))
        assert _SECRET not in json.dumps(state)
        assert _SECRET not in client.get(f"{run}/events").text
        assert _SECRET not in json.dumps([request.tools for request in model.requests])


def test_mcp_disconnect_invalidates_old_approval_and_does_not_replay(tmp_path, monkeypatch):
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("old", "mcp__notes__write_note", {"note": "stale"})]),
        ModelResponse(content="Stale call rejected."),
        ModelResponse(tool_calls=[ToolCall("drop", "mcp__notes__write_note", {"note": "once"})]),
        ModelResponse(content="Outcome requires external confirmation."),
    ])
    with _client(tmp_path, monkeypatch, model, mode="drop") as client:
        server = "/api/agent/mcp/servers/notes"
        assert client.post(f"{server}/connect").json()["server"]["status"] == "connected"
        old_run = _create(client, tmp_path)
        assert client.post(f"{old_run}/run").json()["status"] == "waiting_approval"
        assert client.post(f"{server}/disconnect").json()["server"]["status"] == "disconnected"
        assert client.post(f"{server}/connect").json()["server"]["status"] == "connected"
        assert client.post(f"{old_run}/approve", json={"risk": "execute", "tool_call_id": "old"}).status_code == 409
        assert client.post(f"{old_run}/reject").status_code == 200
        assert client.post(f"{old_run}/run").json()["status"] == "completed"
        assert not _ledger(tmp_path, "notes")

        run = _create(client, tmp_path)
        assert client.post(f"{run}/run").json()["status"] == "waiting_approval"
        assert client.post(f"{run}/approve", json={"risk": "execute", "tool_call_id": "drop"}).status_code == 200
        state = client.post(f"{run}/run").json()
        assert state["status"] == "paused"
        assert _result(state, "drop")["status"] == "uncertain"
        assert client.get("/api/agent/mcp/servers").json()["servers"][0]["tools"] == []
        assert client.post(f"{run}/resume").status_code == 200
        assert client.post(f"{run}/run").json()["status"] == "completed"
        assert [entry["params"]["arguments"]["note"] for entry in _ledger(tmp_path, "notes")] == ["once"]


def test_mcp_calls_require_separate_approvals(tmp_path, monkeypatch):
    first = ToolCall("first", "mcp__notes__write_note", {"note": "first"})
    second = ToolCall("second", "mcp__notes__write_note", {"note": "second"})
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[first, second]),
        ModelResponse(content="First approved, second rejected."),
    ])
    with _client(tmp_path, monkeypatch, model) as client:
        client.post("/api/agent/mcp/servers/notes/connect").raise_for_status()
        run = _create(client, tmp_path)
        assert client.post(f"{run}/run").json()["pending_approval"]["tool_call"]["id"] == "first"
        client.post(f"{run}/approve", json={"risk": "execute", "tool_call_id": "first"}).raise_for_status()
        assert client.post(f"{run}/run").json()["pending_approval"]["tool_call"]["id"] == "second"
        assert len(_ledger(tmp_path, "notes")) == 1
        client.post(f"{run}/reject").raise_for_status()
        assert client.post(f"{run}/run").json()["status"] == "completed"


def test_mcp_catalog_notification_revokes_old_definition_and_new_run_rediscovers(tmp_path, monkeypatch):
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("old", "mcp__notes__write_note", {"note": "old"})]),
        ModelResponse(tool_calls=[ToolCall("lookup", "mcp__notes__lookup")]),
        ModelResponse(content="Catalog change triggered."),
        ModelResponse(tool_calls=[ToolCall("new", "mcp__notes__write_note", {"note": "new"})]),
        ModelResponse(content="Record written."),
    ])
    with _client(tmp_path, monkeypatch, model, mode="change") as client:
        client.post("/api/agent/mcp/servers/notes/connect").raise_for_status()
        old_run = _create(client, tmp_path)
        old_approval = client.post(f"{old_run}/run").json()["pending_approval"]
        trigger = _create(client, tmp_path)
        assert client.post(f"{trigger}/run").json()["status"] == "completed"
        assert client.post(f"{old_run}/approve", json={"risk": "execute", "tool_call_id": "old"}).status_code == 409

        fresh = _create(client, tmp_path)
        approval = client.post(f"{fresh}/run").json()["pending_approval"]
        assert approval["mcp_tool_version"] != old_approval["mcp_tool_version"]
        assert any("Updated." in tool["function"]["description"] for tool in model.requests[-1].tools)
        client.post(f"{fresh}/approve", json={"risk": "execute", "tool_call_id": "new"}).raise_for_status()
        assert client.post(f"{fresh}/run").json()["status"] == "completed"
        executed = [entry["params"]["arguments"]["note"] for entry in _ledger(tmp_path, "notes")
                    if entry["params"]["name"] == "write_note"]
        assert executed == ["new"]


def test_cancel_signals_an_in_flight_mcp_call_and_records_uncertainty(tmp_path, monkeypatch):
    model = ScriptedModelGateway([ModelResponse(tool_calls=[
        ToolCall("wait", "mcp__notes__write_note", {"note": "dispatched"}),
    ])])
    with _client(tmp_path, monkeypatch, model, mode="wait") as client:
        client.post("/api/agent/mcp/servers/notes/connect").raise_for_status()
        run = _create(client, tmp_path)
        assert client.post(f"{run}/run").json()["status"] == "waiting_approval"
        client.post(f"{run}/approve", json={"risk": "execute", "tool_call_id": "wait"}).raise_for_status()
        with ThreadPoolExecutor(max_workers=1) as worker:
            running = worker.submit(client.post, f"{run}/run")
            deadline = time.monotonic() + 5
            while not _ledger(tmp_path, "notes") and time.monotonic() < deadline:
                time.sleep(0.02)
            assert len(_ledger(tmp_path, "notes")) == 1
            cancelled_at = time.monotonic()
            state = client.post(f"{run}/cancel").json()
            assert time.monotonic() - cancelled_at < 7
            assert state["status"] == "cancelled"
            assert _result(state, "wait")["status"] == "uncertain"
            assert running.result(timeout=2).json()["status"] == "cancelled"
