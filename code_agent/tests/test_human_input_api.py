import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_agent.agent_core.contracts import ModelResponse, ToolCall
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.api import agent_routes


def _client(tmp_path: Path, monkeypatch, gateway: ScriptedModelGateway) -> TestClient:
    monkeypatch.setenv("CODE_AGENT_DB", str(tmp_path / "agent.db"))
    monkeypatch.setenv("CODE_AGENT_AUTO_MEMORY_MODE", "off")
    monkeypatch.delenv("CODE_AGENT_MCP_CONFIG", raising=False)
    monkeypatch.setattr(agent_routes, "RoutedModelGateway", lambda **kwargs: gateway)
    app = FastAPI()
    app.include_router(agent_routes.build_default_agent_router())
    return TestClient(app)


def _run_url(client: TestClient, workspace: Path, *, thread_id: str | None = None) -> str:
    created = client.post("/api/agent/runs", json={
        "workspace": str(workspace), "goal": "Ask for the desired verification scope.", "thread_id": thread_id,
    })
    assert created.status_code == 200, created.text
    return f"/api/agent/runs/{created.json()['run_id']}"


def test_waiting_input_survives_restart_and_answer_is_typed_and_idempotent(tmp_path: Path, monkeypatch):
    gateway = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("ask-verification", "user_ask", {
            "question": "Which verification scope should I use?",
            "answer_type": "choice",
            "options": [{"id": "unit", "label": "Unit tests"}, {"id": "all", "label": "Full suite"}],
        })]),
        ModelResponse(content="I will use unit tests."),
    ])
    with _client(tmp_path, monkeypatch, gateway) as client:
        run_url = _run_url(client, tmp_path)
        response = client.post(f"{run_url}/run")
        assert response.status_code == 200, response.text
        waiting = response.json()
        assert waiting["status"] == "waiting_input"
        pending = waiting["pending_user_input"]
        assert pending["tool_call_id"] == "ask-verification"
        assert pending["answer_type"] == "choice"
        ndjson = client.get(f"{run_url}/events.ndjson")
        assert ndjson.status_code == 200, ndjson.text
        assert "user_input.requested" in {json.loads(line)["type"] for line in ndjson.text.splitlines() if line}

    with _client(tmp_path, monkeypatch, gateway) as client:
        assert client.get(f"{run_url}/input").json() == {"request": pending, "status": "waiting_input"}
        answer = {"request_id": pending["request_id"], "expected_version": pending["version"], "answer": "unit"}
        assert client.post(f"{run_url}/input", json={**answer, "request_id": "other-request"}).status_code == 409
        assert client.post(f"{run_url}/input", json={**answer, "expected_version": pending["version"] + 1}).status_code == 409
        assert client.post(f"{run_url}/input", json={**answer, "expected_version": True}).status_code == 422
        assert client.post(f"{run_url}/input", json={**answer, "answer": 1}).status_code == 422
        assert client.post(f"{run_url}/input", json={**answer, "answer": True}).status_code == 422
        assert client.post(f"{run_url}/input", json={**answer, "answer": "different-choice"}).status_code == 422
        assert client.get(f"{run_url}/input").json()["request"] == pending

        answer["api_key"] = "synthetic-input-only-credential"
        resumed = client.post(f"{run_url}/input", json=answer)
        assert resumed.status_code == 200, resumed.text
        assert not resumed.json().get("pending_user_input")
        assert answer["api_key"] not in resumed.text
        assert client.post(f"{run_url}/input", json=answer).status_code == 200
        assert client.post(f"{run_url}/input", json={**answer, "answer": "all"}).status_code == 409

    with _client(tmp_path, monkeypatch, gateway) as client:
        credential = "synthetic-continuation-only-credential"
        response = client.post(f"{run_url}/run", json={"api_key": credential})
        assert response.status_code == 200, response.text
        completed = response.json()
        assert completed["status"] == "completed"
        assert client.get(f"{run_url}/input").json() == {"request": None, "status": "completed"}
        assert client.post(f"{run_url}/input", json=answer).status_code == 200
        assert gateway.requests[-1].metadata["api_key"] == credential
        events = client.get(f"{run_url}/events")
        assert len([event for event in events.json()["events"] if event["type"] == "user_input.resolved"]) == 1
        assert credential not in events.text and credential not in response.text
        assert answer["api_key"] not in events.text
        messages = [message for message in completed["messages"] if message.get("tool_call_id") == "ask-verification"]
        assert len(messages) == 1
        assert "unit" in messages[0]["content"]


def test_waiting_confirmation_can_be_cancelled_and_blocks_thread_deletion(tmp_path: Path, monkeypatch):
    gateway = ScriptedModelGateway([ModelResponse(tool_calls=[ToolCall("ask-confirm", "user_ask", {
        "question": "Run all checks?", "answer_type": "confirm",
    })])])
    with _client(tmp_path, monkeypatch, gateway) as client:
        thread = client.post("/api/agent/threads", json={"title": "Waiting thread", "workspace": str(tmp_path)})
        assert thread.status_code == 200, thread.text
        thread_id = thread.json()["thread_id"]
        thread_url = f"/api/agent/threads/{thread_id}"
        run_url = _run_url(client, tmp_path, thread_id=thread_id)
        waiting = client.post(f"{run_url}/run").json()
        assert waiting["status"] == "waiting_input"
        pending = waiting["pending_user_input"]
        answer = {"request_id": pending["request_id"], "expected_version": pending["version"], "answer": "false"}
        assert client.post(f"{run_url}/input", json=answer).status_code == 422
        assert client.delete(thread_url).status_code == 409
        assert client.post(f"{run_url}/cancel").json()["status"] == "cancelled"
        assert client.get(f"{run_url}/input").json() == {"request": None, "status": "cancelled"}
        assert client.post(f"{run_url}/input", json={**answer, "answer": False}).status_code == 409
        assert client.get("/api/agent/runs/missing/input").status_code == 404
