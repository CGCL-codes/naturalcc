import json
from pathlib import Path

import pytest

from code_agent.agent_core.context_builder import ContextPlanner
from code_agent.agent_core.contracts import ModelResponse, RunBudget
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.memory_store import MemoryStore
from code_agent.agent_core.model_gateway import DeepSeekRequestSerializer, ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.token_budget import DeepSeekModelProfile, DeepSeekTokenCounter
from code_agent.agent_core.tool_registry import build_default_registry


_GOAL = "Will the formatter package signed artifacts correctly?"


@pytest.fixture(scope="module")
def context_planner():
    tokenizer = Path(__file__).resolve().parents[1] / "resources" / "deepseek_v3_tokenizer"
    return ContextPlanner(
        DeepSeekTokenCounter.from_directory(tokenizer),
        DeepSeekRequestSerializer(),
        DeepSeekModelProfile(context_window_tokens=16_384),
    )


@pytest.fixture
def recall_case(tmp_path, context_planner):
    store = EventStore(tmp_path / "agent.db")
    memory = MemoryStore(store.db_path)
    workspace = str(tmp_path.resolve())

    def note(subject, content, project=workspace):
        memory_id = memory.create_candidate(
            scope="project", kind="workflow", project_id=project,
            subject=subject, content=content, source_run_id=None, source_revision=None, confidence=0.9,
        )
        memory.activate(memory_id)
        return memory_id

    lexical = note("formatter", "Apply formatting to Python code.")
    relevant = note("Release prerequisite", "Check fixture compatibility before publishing.")
    foreign = note("Private memo", "Confidential note from an unrelated project.", "other-project")
    search = memory.search_with_quality(_GOAL, project_id=workspace)
    assert [item["id"] for item in search.matches] == [lexical]
    assert not search.strong_matches

    def run(replies, *, max_llm_calls=3, max_input_tokens=120_000, max_output_tokens=24_000, max_cost_usd=None):
        model = ScriptedModelGateway(replies)
        engine = RunEngine(
            store, build_default_registry(include_mutating=False), model,
            memory_store=memory, context_planner=context_planner,
        )
        run_id = engine.create_run(tmp_path, _GOAL, RunBudget(
            max_llm_calls=max_llm_calls, max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens, max_cost_usd=max_cost_usd,
        ))
        state = engine.run(run_id)
        events = store.list_events(run_id)
        trace, = [event.payload for event in events if event.type == "memory.recall_evaluated"]
        return state, model.requests, trace

    return run, lexical, relevant, foreign


@pytest.mark.parametrize("invalid", ["empty_text", "malformed_json", "bad_schema", "foreign_id"])
def test_selector_repairs_invalid_output_once_and_keeps_budget_for_foreground(recall_case, invalid):
    run, lexical, relevant, foreign = recall_case
    invalid_response = {
        "empty_text": "",
        "malformed_json": '{"ids":',
        "bad_schema": json.dumps({"ids": relevant}),
        "foreign_id": json.dumps({"ids": [foreign]}),
    }[invalid]
    state, requests, trace = run([
        ModelResponse(content=invalid_response, input_tokens=31, output_tokens=13),
        ModelResponse(content=json.dumps({"ids": [relevant]}), input_tokens=37, output_tokens=7),
        ModelResponse(content="foreground answer", input_tokens=101, output_tokens=5),
    ])

    first, repair, foreground = requests
    assert first.purpose == "memory_recall"
    assert repair.purpose == "memory_recall" and foreground.purpose == "agent"
    assert not first.tools and not repair.tools
    assert 0 < first.max_output_tokens <= 512
    assert first.max_output_tokens <= repair.max_output_tokens <= 768
    assert state["status"] == "completed" and state["final_answer"] == "foreground answer"
    assert state["llm_calls"] == 3
    assert state["input_tokens"] == 169 and state["output_tokens"] == 25
    assert trace["selection_source"] == "model"
    assert trace["attempt_count"] == 2
    assert trace["selected_ids"] == [relevant]
    assert relevant in foreground.messages[-1]["content"]
    assert lexical not in foreground.messages[-1]["content"]
    assert foreign not in foreground.messages[-1]["content"]


def test_valid_empty_selection_is_not_retried_or_replaced_by_lexical_hits(recall_case):
    run, lexical, relevant, _ = recall_case
    state, requests, trace = run([
        ModelResponse(content='{"ids":[]}'),
        ModelResponse(content="foreground answer"),
    ])

    assert [request.purpose for request in requests] == ["memory_recall", "agent"]
    assert state["status"] == "completed" and state["llm_calls"] == 2
    assert trace["selection_source"] == "model"
    assert trace["attempt_count"] == 1
    assert trace["selected_ids"] == []
    assert lexical not in requests[-1].messages[-1]["content"]
    assert relevant not in requests[-1].messages[-1]["content"]


@pytest.mark.parametrize("max_llm_calls", [2, 3])
def test_invalid_selector_falls_back_to_lexical_and_reserves_foreground_call(recall_case, max_llm_calls):
    run, lexical, relevant, foreign = recall_case
    replies = [ModelResponse(content=json.dumps({"ids": [foreign]}))]
    if max_llm_calls == 3:
        replies.append(ModelResponse(content="still invalid"))
    replies.append(ModelResponse(content="foreground answer"))

    state, requests, trace = run(replies, max_llm_calls=max_llm_calls)

    assert len(requests) == state["llm_calls"] == max_llm_calls
    assert requests[-1].purpose == "agent"
    assert state["status"] == "completed" and state["final_answer"] == "foreground answer"
    assert trace["selection_source"] == "lexical_fallback"
    assert trace["selected_ids"] == [lexical]
    assert lexical in requests[-1].messages[-1]["content"]
    assert relevant not in requests[-1].messages[-1]["content"]
    assert foreign not in requests[-1].messages[-1]["content"]


def test_repair_preflight_reserves_input_for_foreground(recall_case, context_planner):
    run, lexical, _, _ = recall_case
    _, baseline_requests, _ = run([
        ModelResponse(content='{"ids":[]}'),
        ModelResponse(content="foreground answer"),
    ])
    foreground = baseline_requests[-1]
    estimated_foreground = context_planner.counter.count_request(
        foreground.messages, foreground.tools, context_planner.serializer, context_planner.profile,
    ).input_tokens

    state, requests, trace = run([
        ModelResponse(content="malformed output", input_tokens=1_820),
        ModelResponse(content="foreground answer", input_tokens=100),
    ], max_input_tokens=estimated_foreground + 2_000)

    assert [request.purpose for request in requests] == ["memory_recall", "agent"]
    assert state["status"] == "completed" and state["input_tokens"] == 1_920
    assert trace["selection_source"] == "lexical_fallback"
    assert trace["selected_ids"] == [lexical]


def test_repair_preflight_reserves_output_for_foreground(recall_case):
    run, lexical, _, _ = recall_case
    state, requests, trace = run([
        ModelResponse(content='{"', output_tokens=276),
        ModelResponse(content="foreground answer", output_tokens=8),
    ], max_output_tokens=1_300)

    assert [request.purpose for request in requests] == ["memory_recall", "agent"]
    assert [request.max_output_tokens for request in requests] == [276, 1024]
    assert state["status"] == "completed" and state["output_tokens"] == 284
    assert trace["selection_source"] == "lexical_fallback"
    assert trace["selected_ids"] == [lexical]


def test_repair_stops_when_first_attempt_reaches_run_cost_limit(recall_case):
    run, _, _, _ = recall_case
    state, requests, trace = run([
        ModelResponse(content="", cost_usd=0.01),
    ], max_cost_usd=0.01)

    assert [request.purpose for request in requests] == ["memory_recall"]
    assert state["status"] == "budget_exhausted" and state["llm_calls"] == 1
    assert trace["selection_source"] == "lexical_fallback" and trace["attempt_count"] == 1
