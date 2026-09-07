from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse, ToolCall
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.api.agent_routes import create_agent_router


def make_client(db_path: Path, responses: list[ModelResponse]):
    store = EventStore(db_path)
    engine = RunEngine(store, build_default_registry(include_mutating=True), ScriptedModelGateway(responses))
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    return TestClient(app), store


def test_directory_approval_survives_restart_and_executes_once(tmp_path: Path):
    client, store = make_client(tmp_path / "agent.db", [
        ModelResponse(tool_calls=[ToolCall("mkdir-1", "workspace.create_directory", {
            "path": "\u65b0\u5efa\u6587\u4ef6\u5939/child", "parents": True,
        })]),
    ])
    run_id = client.post("/api/agent/runs", json={
        "workspace": str(tmp_path), "goal": "create a nested directory",
    }).json()["run_id"]
    url = f"/api/agent/runs/{run_id}"
    waiting = client.post(f"{url}/run").json()
    assert waiting["status"] == "waiting_approval"
    assert not (tmp_path / "\u65b0\u5efa\u6587\u4ef6\u5939").exists()

    client, store = make_client(store.db_path, [ModelResponse(content="Directory created")])
    assert client.get(url).json()["pending_approval"]["tool_call"]["id"] == "mkdir-1"
    for payload in [{"risk": "execute"}, {"risk": "write", "tool_call_id": "stale-call"}]:
        assert client.post(f"{url}/approve", json=payload).status_code == 409
        assert not store.approvals_for(run_id)
    for _ in range(2):
        response = client.post(f"{url}/approve", json={"risk": "write", "tool_call_id": "mkdir-1"})
        assert response.status_code == 200
    for _ in range(2):
        assert client.post(f"{url}/run").json()["status"] == "completed"
    assert (tmp_path / "\u65b0\u5efa\u6587\u4ef6\u5939" / "child").is_dir()
    events = store.list_events(run_id)
    assert sum(event.type == "approval.resolved" for event in events) == 1
    assert sum(event.type == "tool.started" for event in events) == 1


def test_missing_run_returns_404_without_granting_approval(tmp_path: Path):
    client, store = make_client(tmp_path / "agent.db", [])
    url = "/api/agent/runs/missing-run"
    assert client.get(url).status_code == 404
    response = client.post(f"{url}/approve", json={"risk": "write"})
    assert response.status_code == 404
    assert "unknown run" in response.json()["detail"]
    assert not store.approvals_for("missing-run")


def test_missing_snapshot_is_a_state_conflict_not_a_missing_run(tmp_path: Path):
    client, store = make_client(tmp_path / "agent.db", [])
    store.create_run("incomplete", str(tmp_path), "interrupted initialization", {})
    url = "/api/agent/runs/incomplete"
    assert client.get(url).status_code == 409
    assert client.post(f"{url}/approve", json={"risk": "write"}).status_code == 409
    assert not store.approvals_for("incomplete")


def test_directory_creation_does_not_loop_on_project_test_commands(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(sample)\nenable_testing()\n",
        encoding="utf-8",
    )
    client, store = make_client(tmp_path / "agent.db", [
        ModelResponse(tool_calls=[ToolCall("mkdir-1", "workspace.create_directory", {"path": "zidong"})]),
        ModelResponse(content="The directory zidong was created."),
    ])
    run_id = client.post("/api/agent/runs", json={
        "workspace": str(tmp_path), "goal": "create the zidong directory",
    }).json()["run_id"]

    waiting = client.post(f"/api/agent/runs/{run_id}/run").json()
    assert waiting["status"] == "waiting_approval"
    client.post(
        f"/api/agent/runs/{run_id}/approve",
        json={"risk": "write", "tool_call_id": "mkdir-1"},
    )
    completed = client.post(f"/api/agent/runs/{run_id}/run").json()

    assert completed["status"] == "completed"
    assert completed["llm_calls"] == 2
    assert completed["input_tokens"] < 10000
    assert (tmp_path / "zidong").is_dir()
    events = store.list_events(run_id)
    assert not any(event.type == "verification.required" for event in events)
