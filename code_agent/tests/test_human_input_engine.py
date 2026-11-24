import json
import time

import pytest

from code_agent.agent_core.contracts import ModelResponse, RiskLevel, RunBudget, ToolCall
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry


def test_clarification_freezes_time_and_replans_other_tools_in_same_batch(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr("code_agent.agent_core.run_engine.time.time", lambda: now)
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[
            ToolCall("edit", "workspace.create_file", {"path": "planned.txt", "content": "old plan"}),
            ToolCall("ask", "user_ask", {"question": "What should the file contain?", "answer_type": "text"}),
        ]),
        ModelResponse(content="I will use the clarified requirement."),
    ])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(), model, enable_user_input=True)
    run_id = engine.create_run(tmp_path, "Create a file", RunBudget(max_seconds=10))
    waiting = engine.run(run_id)
    pending = waiting["pending_user_input"]
    assert pending["question"] == "What should the file contain?"
    assert any(tool["function"]["name"] == "user_ask" for tool in model.requests[0].tools)
    assert waiting["status"] == "waiting_input"
    assert not store.list_events(run_id, event_type="approval.requested")
    assert not store.list_events(run_id, event_type="tool.started")

    now += 300
    assert engine.run(run_id)["version"] == waiting["version"]
    assert len(model.requests) == 1
    engine.update_budget(run_id, {"max_seconds": 20})
    answered = engine.answer_user_input(
        run_id, pending["request_id"], pending["version"], "Use the release notes."
    )
    assert answered["started_at_epoch"] == now
    assert engine.run(run_id)["status"] == "completed"
    results = [message for message in model.requests[-1].messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == ["edit", "ask"]
    assert json.loads(results[0]["content"])["error"]["type"] == "UserInputRequired"
    assert json.loads(results[1]["content"])["data"]["answer"] == "Use the release notes."
    assert not (tmp_path / "planned.txt").exists()


def test_unfinished_local_call_requires_reconciliation_after_restart(tmp_path, monkeypatch):
    class WorkerStopped(BaseException):
        pass

    (tmp_path / "README.md").write_text("fixture", encoding="utf-8")
    db = tmp_path / "agent.db"
    registry = build_default_registry()
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("read", "workspace.read", {"path": "README.md"})]),
        ModelResponse(content="The user checked the result."),
    ])
    engine = RunEngine(EventStore(db), registry, model, enable_user_input=True)
    run_id = engine.create_run(tmp_path, "Read the README")

    def stop_after_start(*args):
        raise WorkerStopped()

    monkeypatch.setattr(registry, "execute", stop_after_start)
    with pytest.raises(WorkerStopped):
        engine.run(run_id)
    restarted = RunEngine(EventStore(db), registry, model, enable_user_input=True)
    waiting = restarted.run(run_id)
    pending = waiting["pending_uncertain_tool"]
    assert waiting["status"] == "paused"
    assert pending["tool_call"]["id"] == "read"
    with pytest.raises(ValueError, match="reconcile"):
        restarted.resume(run_id)
    with pytest.raises(ValueError, match="stale"):
        restarted.resolve_uncertain_tool(run_id, "read", pending["version"] + 1, "completed")
    restarted.resolve_uncertain_tool(run_id, "read", pending["version"], "completed", "Read manually.")
    assert restarted.run(run_id)["status"] == "completed"
    assert len(restarted.store.list_events(run_id, event_type="tool.started")) == 1
    result = next(message for message in model.requests[-1].messages if message.get("tool_call_id") == "read")
    assert json.loads(result["content"])["status"] == "uncertain"
    assert json.loads(result["content"])["data"]["outcome"] == "completed"


def test_local_approval_shows_scope_and_uses_run_deadline(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr("code_agent.agent_core.run_engine.time.time", lambda: now)
    model = ScriptedModelGateway([ModelResponse(tool_calls=[
        ToolCall("edit", "workspace.create_file", {"path": "planned.txt", "content": "new"})
    ])])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(), model, enable_user_input=True)
    run_id = engine.create_run(tmp_path, "Create a file", RunBudget(max_seconds=10))
    approval = engine.run(run_id)["pending_approval"]
    assert approval["scope"] == "run_risk"
    assert approval["status"] == "pending"
    assert approval["expires_at"]

    now += 11
    assert engine.get_state(run_id)["pending_approval"]["status"] == "expired"
    with pytest.raises(ValueError, match="expired"):
        engine.approve(run_id, RiskLevel.WRITE, "edit")
    with pytest.raises(ValueError, match="changed"):
        engine.reject(run_id, "older-edit")
    engine.reject(run_id, "edit")
    assert not (tmp_path / "planned.txt").exists()
