from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse, ToolCall
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.memory_store import MemoryStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.api.agent_routes import create_agent_router


def test_agent_api_create_run_step_events_and_cancel(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    thread_id = client.post("/api/agent/threads", json={"title": "Repository work"}).json()["thread_id"]
    created = client.post(
        "/api/agent/runs",
        json={"workspace": str(tmp_path), "goal": "inspect", "thread_id": thread_id},
    ).json()
    run_id = created["run_id"]
    stepped = client.post(f"/api/agent/runs/{run_id}/step").json()
    events = client.get(f"/api/agent/runs/{run_id}/events").json()

    assert stepped["status"] == "completed"
    assert events["events"][-1]["type"] == "run.completed"
    assert client.get(f"/api/agent/runs/{run_id}").status_code == 200
    assert client.get("/api/agent/runs").json()["runs"][0]["id"] == run_id
    assert client.get("/api/agent/threads").json()["threads"][0]["id"] == thread_id


def test_agent_api_rejects_invalid_run_control_transitions(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app, raise_server_exceptions=False)
    run_id = client.post(
        "/api/agent/runs",
        json={"workspace": str(tmp_path), "goal": "control safely"},
    ).json()["run_id"]

    assert client.post(f"/api/agent/runs/{run_id}/resume").status_code == 409
    assert client.post(f"/api/agent/runs/{run_id}/cancel").status_code == 200
    assert client.post(f"/api/agent/runs/{run_id}/pause").status_code == 409


def test_agent_api_create_run_accepts_target_files(tmp_path: Path):
    (tmp_path / "calculator.c").write_text("// English comment\n", encoding="utf-8")
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    created = client.post(
        "/api/agent/runs",
        json={
            "workspace": str(tmp_path),
            "goal": "translate this file",
            "target_files": ["calculator.c"],
        },
    ).json()

    assert created["state"]["target_files"] == ["calculator.c"]


def test_agent_api_uses_request_scoped_api_key_without_persisting_it(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    created = client.post(
        "/api/agent/runs",
        json={
            "workspace": str(tmp_path),
            "goal": "inspect",
            "api_key": "request-only-key",
        },
    ).json()
    state = client.post(f"/api/agent/runs/{created['run_id']}/step").json()

    assert state["status"] == "completed"
    assert model.requests[0].metadata["api_key"] == "request-only-key"
    assert "request-only-key" not in str(created["state"])
    assert "request-only-key" not in str(state)


def test_agent_thread_message_uses_request_scoped_api_key(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    thread_id = client.post(
        "/api/agent/threads",
        json={"title": "Repository work", "workspace": str(tmp_path)},
    ).json()["thread_id"]
    created = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={"content": "inspect", "api_key": "request-only-key"},
    ).json()
    state = client.post(f"/api/agent/runs/{created['run_id']}/step").json()

    assert state["status"] == "completed"
    assert model.requests[0].metadata["api_key"] == "request-only-key"
    assert "request-only-key" not in str(created["state"])
    assert "request-only-key" not in str(state)


def test_agent_api_approval_is_persistent_and_idempotent(tmp_path: Path):
    target = tmp_path / "x.py"
    target.write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"node -e \\"console.log(123)\\""}}',
        encoding="utf-8",
    )
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit-1",
                        "workspace.apply_patch",
                        {"path": "x.py", "old_text": "x = 1", "new_text": "x = 2"},
                    )
                ]
            ),
            ModelResponse(tool_calls=[ToolCall("test-1", "tests.run", {})]),
            ModelResponse(content="edited"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    run_id = client.post("/api/agent/runs", json={"workspace": str(tmp_path), "goal": "edit x"}).json()["run_id"]

    assert client.post(f"/api/agent/runs/{run_id}/step").json()["status"] == "waiting_approval"
    first = client.post(f"/api/agent/runs/{run_id}/approve", json={"risk": "write"}).json()
    second = client.post(f"/api/agent/runs/{run_id}/approve", json={"risk": "write"}).json()
    assert first["status"] == "running"
    assert second["status"] == "running"
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "waiting_approval"
    client.post(f"/api/agent/runs/{run_id}/approve", json={"risk": "execute"})
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"
    assert target.read_text(encoding="utf-8") == "x = 2\n"
    events = client.get(f"/api/agent/runs/{run_id}/events").json()["events"]
    assert any(event["type"] == "verification.finished" for event in events)


def test_agent_api_can_reject_pending_action_and_manage_memory(tmp_path: Path):
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit-1",
                        "workspace.apply_patch",
                        {"path": "missing.py", "old_text": "a", "new_text": "b"},
                    )
                ]
            ),
            ModelResponse(content="action was rejected"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    memories = MemoryStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model, memory_store=memories)
    app = FastAPI()
    app.include_router(create_agent_router(engine, memories))
    client = TestClient(app)
    run_id = client.post("/api/agent/runs", json={"workspace": str(tmp_path), "goal": "edit"}).json()["run_id"]

    assert client.post(f"/api/agent/runs/{run_id}/step").json()["status"] == "waiting_approval"
    rejected = client.post(f"/api/agent/runs/{run_id}/reject").json()
    assert rejected["status"] == "running"
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"
    events = client.get(f"/api/agent/runs/{run_id}/events").json()["events"]
    assert any(
        event["type"] == "approval.resolved" and event["payload"]["decision"] == "rejected"
        for event in events
    )

    created = client.post(
        "/api/agent/memories",
        json={
            "scope": "project",
            "kind": "constraint",
            "project_id": str(tmp_path),
            "subject": "runtime",
            "content": "Keep Python 3.12",
            "confidence": 0.9,
        },
    ).json()
    memory_id = created["memory_id"]
    assert client.post(f"/api/agent/memories/{memory_id}/activate").status_code == 200
    assert client.get("/api/agent/memories", params={"project_id": str(tmp_path)}).json()["memories"]
    assert client.delete(f"/api/agent/memories/{memory_id}").json()["deleted"] == memory_id


def test_thread_api_restores_settings_conversation_and_last_run(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="Thread reply")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)

    created_thread = client.post(
        "/api/agent/threads",
        json={
            "title": "Repository work",
            "workspace": str(tmp_path),
            "model": "deepseek-chat",
            "runtime_mode": "agent",
            "budget": {"max_llm_calls": 12, "max_tool_calls": 20},
        },
    ).json()
    thread_id = created_thread["thread_id"]
    created_message = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={"content": "Inspect this project", "target_files": []},
    ).json()
    run_id = created_message["run_id"]
    assert client.post(f"/api/agent/runs/{run_id}/run").json()["status"] == "completed"

    detail = client.get(f"/api/agent/threads/{thread_id}").json()

    assert detail["thread"]["workspace"] == str(tmp_path.resolve())
    assert detail["thread"]["model"] == "deepseek-chat"
    assert detail["thread"]["budget"]["max_llm_calls"] == 12
    assert [message["content"] for message in detail["messages"]] == [
        "Inspect this project",
        "Thread reply",
    ]
    assert detail["last_run"]["run_id"] == run_id
    assert detail["last_run"]["status"] == "completed"


def test_thread_delete_api_removes_terminal_conversation_and_reports_missing(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    thread_id = client.post(
        "/api/agent/threads",
        json={"title": "Delete me", "workspace": str(tmp_path)},
    ).json()["thread_id"]

    deleted = client.delete(f"/api/agent/threads/{thread_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": thread_id}
    assert client.get(f"/api/agent/threads/{thread_id}").status_code == 404
    assert client.delete("/api/agent/threads/missing-thread").status_code == 404


def test_thread_delete_api_rejects_conversation_with_active_run(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    thread_id = client.post(
        "/api/agent/threads",
        json={"title": "Active", "workspace": str(tmp_path)},
    ).json()["thread_id"]
    client.post(
        "/api/agent/runs",
        json={"workspace": str(tmp_path), "goal": "queued work", "thread_id": thread_id},
    )

    response = client.delete(f"/api/agent/threads/{thread_id}")

    assert response.status_code == 409
    assert "Cancel" in response.json()["detail"]
    assert client.get(f"/api/agent/threads/{thread_id}").status_code == 200


def test_context_resolve_searches_workspace_and_authorizes_absolute_path(tmp_path: Path):
    (tmp_path / "StudentManager.java").write_text("class StudentManager {}", encoding="utf-8")
    external = tmp_path.parent / f"{tmp_path.name}-external.py"
    external.write_text("enabled = True\n", encoding="utf-8")
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    thread_id = client.post(
        "/api/agent/threads",
        json={"title": "Context", "workspace": str(tmp_path)},
    ).json()["thread_id"]

    matches = client.post(
        "/api/agent/context/resolve",
        json={"thread_id": thread_id, "value": "@student"},
    ).json()
    resolved = client.post(
        "/api/agent/context/resolve",
        json={"thread_id": thread_id, "value": str(external)},
    ).json()

    assert matches["matches"][0]["path"] == "StudentManager.java"
    assert resolved["matches"][0]["path"] == str(external.resolve())
    assert resolved["matches"][0]["external"] is True
    assert str(external.resolve()) in client.get(f"/api/agent/threads/{thread_id}").json()["thread"]["authorized_paths"]


def test_run_budget_api_increases_limit_and_updates_thread_default(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    thread_id = client.post(
        "/api/agent/threads",
        json={
            "title": "Budget",
            "workspace": str(tmp_path),
            "budget": {"max_llm_calls": 4, "max_tool_calls": 5},
        },
    ).json()["thread_id"]
    run_id = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={"content": "Use a larger budget"},
    ).json()["run_id"]

    updated = client.patch(
        f"/api/agent/runs/{run_id}/budget",
        json={"budget": {"max_llm_calls": 9, "max_tool_calls": 12}},
    ).json()

    assert updated["budget"]["max_llm_calls"] == 9
    thread = client.get(f"/api/agent/threads/{thread_id}").json()["thread"]
    assert thread["budget"]["max_tool_calls"] == 12


def test_thread_message_persists_selected_context_items(tmp_path: Path):
    target = tmp_path / "src" / "main.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    app = FastAPI()
    app.include_router(create_agent_router(engine))
    client = TestClient(app)
    thread_id = client.post(
        "/api/agent/threads",
        json={"title": "Context", "workspace": str(tmp_path)},
    ).json()["thread_id"]
    context_item = {
        "path": "src/main.py",
        "absolute_path": str(target.resolve()),
        "external": False,
        "type": "file",
    }

    response = client.post(
        f"/api/agent/threads/{thread_id}/messages",
        json={
            "content": "Inspect the selected file",
            "target_files": ["src/main.py"],
            "context_items": [context_item],
        },
    )

    assert response.status_code == 200
    thread = client.get(f"/api/agent/threads/{thread_id}").json()["thread"]
    assert thread["context_items"] == [context_item]
