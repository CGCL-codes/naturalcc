import json
from pathlib import Path

import pytest

from code_agent.agent_core.context_builder import ContextPlanner
from code_agent.agent_core.contracts import ModelResponse, ModelStreamEvent, RunBudget, ToolCall
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import (
    DeepSeekRequestSerializer,
    ModelGateway,
    ProviderContextOverflow,
    ScriptedModelGateway,
    _provider_context_overflow,
)
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.token_budget import DeepSeekModelProfile, DeepSeekTokenCounter
from code_agent.agent_core.tool_registry import build_default_registry


@pytest.fixture(scope="module")
def planner():
    tokenizer = Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
    return ContextPlanner(
        DeepSeekTokenCounter.from_directory(tokenizer),
        DeepSeekRequestSerializer(),
        DeepSeekModelProfile(),
    )


def _engine(store, model, planner):
    return RunEngine(
        store, build_default_registry(include_mutating=False), model, context_planner=planner
    )


def test_saved_tool_result_keeps_ids_and_is_readable_after_restart(tmp_path, planner):
    raw = "green amber cyan violet gray\n" * 600
    for name in ("first.txt", "second.txt"):
        (tmp_path / name).write_text(raw, encoding="utf-8")
    store = EventStore(tmp_path / "agent.db")
    calls = [ToolCall(name, "workspace_read", {"path": f"{name}.txt"}) for name in ("first", "second")]
    model = ScriptedModelGateway([ModelResponse(tool_calls=calls), ModelResponse(content="one")])
    engine = _engine(store, model, planner)
    run_id = engine.create_run(tmp_path, "Read both files.")
    engine.step(run_id)

    restarted_model = ScriptedModelGateway([ModelResponse(content="unused")])
    restarted = _engine(store, restarted_model, planner)
    state = restarted.get_state(run_id)
    request = restarted._build_token_planned_context(state, RunBudget(), [], [])
    tool_messages = [message for message in request.messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["first", "second"]
    current = [message for message in state["messages"] if message["role"] == "tool"]
    assert all(raw in json.loads(message["content"])["data"]["content"] for message in current)
    reference = json.loads(tool_messages[0]["content"])["result_ref"]

    reader_model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("page", "tool_result_read", {"ref": reference, "limit": 80})]),
        ModelResponse(content="done"),
    ])
    reader = _engine(store, reader_model, planner)
    result = reader.run(run_id)
    assert result["status"] == "completed"
    assert result["tool_calls"] == 3
    replayed = [message for message in reader_model.requests[0].messages if message["role"] == "tool"]
    assert replayed == tool_messages
    page = next(message for message in result["messages"] if message.get("tool_call_id") == "page")
    assert json.loads(page["content"])["data"]["next_start"] == 80

    other_model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("foreign", "tool_result_read", {"ref": reference})]),
        ModelResponse(content="done"),
    ])
    other = _engine(store, other_model, planner)
    foreign = other.run(other.create_run(tmp_path, "Can another Run read it?"))
    denied = next(message for message in foreign["messages"] if message.get("tool_call_id") == "foreign")
    assert json.loads(denied["content"])["error"]["type"] == "ToolResultUnavailable"


class _OverflowGateway(ModelGateway):
    def __init__(self, *, repeat=False, partial=False):
        self.requests = []
        self.repeat = repeat
        self.partial = partial

    def generate(self, request):
        raise AssertionError("Unexpected maintenance call")

    def stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            response = ModelResponse(tool_calls=[ToolCall("read", "workspace_read", {"path": "file.txt"})])
        elif len(self.requests) == 2 or self.repeat:
            if self.partial:
                yield ModelStreamEvent("content", "partial stream")
            raise ProviderContextOverflow()
        else:
            response = ModelResponse(content="done", input_tokens=233)
        yield ModelStreamEvent("completed", response=response)


@pytest.mark.parametrize(("repeat", "partial", "status", "attempts"), [
    (False, False, "completed", 3),
    (True, False, "failed", 3),
    (False, True, "failed", 2),
])
def test_provider_overflow_only_retries_a_smaller_complete_request(tmp_path, planner, repeat, partial, status, attempts):
    (tmp_path / "file.txt").write_text("green amber cyan violet gray\n" * 160, encoding="utf-8")
    model = _OverflowGateway(repeat=repeat, partial=partial)
    store = EventStore(tmp_path / "agent.db")
    engine = _engine(store, model, planner)
    run_id = engine.create_run(tmp_path, "Read the file.")
    assert engine.step(run_id)["status"] == "running"
    state = engine.step(run_id)
    if not partial:
        assert state["context_overflow_recovery"]["dispatched"] is False
    # Reload the pending recovery from SQLite before continuing.
    engine = _engine(store, model, planner)
    state = engine.run(run_id)
    assert state["status"] == status
    assert state["llm_calls"] == len(model.requests) == attempts
    assert state["input_tokens"] == (233 if status == "completed" else 0)
    assert state["tool_calls"] == 1
    if attempts == 3:
        rejected, retry = model.requests[1:]
        assert 0 < retry.metadata["timeout_seconds"] <= RunBudget().max_seconds
        rejected_tool, = [message for message in rejected.messages if message["role"] == "tool"]
        retry_tool, = [message for message in retry.messages if message["role"] == "tool"]
        assert rejected_tool["tool_call_id"] == retry_tool["tool_call_id"] == "read"
        assert len(retry_tool["content"]) < len(rejected_tool["content"])
        assert json.loads(retry_tool["content"])["result_ref"]


def test_provider_error_classification_requires_context_evidence():
    error = RuntimeError("provider rejected")
    error.status_code = 413
    error.body = {"error": {"message": "content-length exceeds upload limit"}}
    assert _provider_context_overflow(error) is None
    error.body = {"error": {"message": "This model's maximum context length is 12,800 tokens"}}
    overflow = _provider_context_overflow(error, partial=True)
    assert overflow.limit_tokens == 12_800
    assert overflow.partial is True
