from pathlib import Path
import json

import pytest

from code_agent.agent_core.contracts import ModelResponse, RiskLevel, RunBudget, ToolCall
from code_agent.agent_core.compaction import (
    CompactionService,
    FrozenCompactionSource,
    FrozenThreadCompactionSource,
    canonical_message_hash,
)
from code_agent.agent_core.compaction_prompts import (
    ANALYZER_PROMPT_VERSION,
    SUMMARIZER_PROMPT_VERSION,
)
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.context_builder import ContextBuilder, ContextPlanner
from code_agent.agent_core.memory_store import MemoryStore
from code_agent.agent_core.model_gateway import DeepSeekRequestSerializer, ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry
from code_agent.agent_core.token_budget import DeepSeekModelProfile, DeepSeekTokenCounter


def test_read_only_run_completes_and_persists_trace(tmp_path: Path):
    (tmp_path / "README.md").write_text("agent facts", encoding="utf-8")
    model = ScriptedModelGateway(
        [
            ModelResponse(tool_calls=[ToolCall("c1", "workspace.read", {"path": "README.md"})]),
            ModelResponse(content="Found agent facts."),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "read the readme", RunBudget(max_llm_calls=3, max_tool_calls=3))
    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["final_answer"] == "Found agent facts."
    event_types = [event.type for event in store.list_events(run_id)]
    assert event_types.count("tool.proposed") == 1
    assert event_types.count("tool.finished") == 1


def test_default_agent_budget_allows_multi_step_code_tasks():
    budget = RunBudget()

    assert budget.max_llm_calls >= 24
    assert budget.max_tool_calls >= 24


def test_run_engine_executes_model_safe_tool_aliases(tmp_path: Path):
    (tmp_path / "README.md").write_text("agent facts", encoding="utf-8")
    model = ScriptedModelGateway(
        [
            ModelResponse(tool_calls=[ToolCall("c1", "workspace_read", {"path": "README.md"})]),
            ModelResponse(content="alias worked."),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "read via alias", RunBudget(max_llm_calls=3, max_tool_calls=3))

    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["final_answer"] == "alias worked."
    tool_events = [event for event in store.list_events(run_id) if event.type == "tool.finished"]
    assert tool_events[0].payload["tool"] == "workspace.read"


def test_budget_exhaustion_is_durable(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(tool_calls=[ToolCall("c1", "workspace.list", {"path": "."})])])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "loop", RunBudget(max_llm_calls=1, max_tool_calls=0))
    result = engine.run(run_id)
    assert result["status"] == "budget_exhausted"


def test_budget_exhaustion_after_model_response_records_unexecuted_tools(tmp_path: Path):
    model = ScriptedModelGateway([
        ModelResponse(tool_calls=[ToolCall("c1", "workspace.read", {"path": "README.md"})])
    ])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "read", RunBudget(max_llm_calls=1, max_tool_calls=3))

    result = engine.run(run_id)

    assert result["status"] == "budget_exhausted"
    event = store.list_events(run_id)[-1]
    assert event.type == "run.budget_exhausted"
    assert event.payload["reason"] == "max_llm_calls"
    assert event.payload["unexecuted_tool_calls"][0]["name"] == "workspace.read"
    assert all(item.type != "tool.started" for item in store.list_events(run_id))


def test_final_answer_on_last_allowed_model_call_completes(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="final")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "one call", RunBudget(max_llm_calls=1))

    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["final_answer"] == "final"


def test_elapsed_time_budget_is_checked_before_model_call(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="should not run")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "stop immediately", RunBudget(max_seconds=0))

    result = engine.run(run_id)

    assert result["status"] == "budget_exhausted"
    assert model.requests == []
    assert store.list_events(run_id)[-1].payload["reason"] == "max_seconds"


def test_reported_model_cost_stops_run_before_next_tool(tmp_path: Path):
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[ToolCall("c1", "workspace.list", {"path": "."})],
                cost_usd=1.5,
            )
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "respect cost", RunBudget(max_cost_usd=1.0))

    result = engine.run(run_id)

    assert result["status"] == "budget_exhausted"
    assert result["cost_usd"] == 1.5
    assert all(event.type != "tool.started" for event in store.list_events(run_id))


def test_run_accumulates_prompt_cache_usage_and_records_trace(tmp_path: Path):
    model = ScriptedModelGateway(
        [
            ModelResponse(
                content="done",
                input_tokens=120,
                output_tokens=8,
                prompt_cache_hit_tokens=90,
                prompt_cache_miss_tokens=30,
            )
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(tmp_path, "measure prompt cache")

    result = engine.run(run_id)

    assert result["prompt_cache_hit_tokens"] == 90
    assert result["prompt_cache_miss_tokens"] == 30
    usage_events = [
        event for event in store.list_events(run_id)
        if event.type == "model.usage_recorded"
    ]
    assert usage_events[0].payload["prompt_cache_hit_tokens"] == 90
    assert usage_events[0].payload["prompt_cache_miss_tokens"] == 30


def test_run_uses_context_builder_without_creating_ungoverned_memory(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="Use Python 3.12.")])
    event_store = EventStore(tmp_path / "agent.db")
    memory_store = MemoryStore(tmp_path / "agent.db")
    engine = RunEngine(
        event_store,
        build_default_registry(include_mutating=False),
        model,
        context_builder=ContextBuilder(max_chars=2000),
        memory_store=memory_store,
    )
    run_id = engine.create_run(tmp_path, "record project runtime", RunBudget(max_llm_calls=2))
    assert engine.run(run_id)["status"] == "completed"
    assert model.requests[0].messages[0]["role"] == "system"
    assert memory_store.list(status="candidate") == []


def test_run_splits_pinned_and_retrieved_memory_for_cache_friendly_prompt(
    tmp_path: Path,
):
    event_store = EventStore(tmp_path / "agent.db")
    memory_store = MemoryStore(tmp_path / "agent.db")
    project_id = str(tmp_path.resolve())
    pinned_id = memory_store.create_candidate(
        scope="project",
        kind="project_constraint",
        project_id=project_id,
        subject="API constraint",
        content="Keep the public API stable",
        source_run_id=None,
        source_revision=None,
        confidence=1.0,
    )
    retrieved_id = memory_store.create_candidate(
        scope="project",
        kind="workflow",
        project_id=project_id,
        subject="formatter workflow",
        content="Run the compact formatter after edits",
        source_run_id=None,
        source_revision=None,
        confidence=0.9,
    )
    memory_store.activate(pinned_id)
    memory_store.activate(retrieved_id)
    model = ScriptedModelGateway([ModelResponse(content="done")])
    engine = RunEngine(
        event_store,
        build_default_registry(include_mutating=False),
        model,
        memory_store=memory_store,
    )

    run_id = engine.create_run(tmp_path, "apply the formatter workflow")
    assert engine.run(run_id)["status"] == "completed"

    system_prompt = model.requests[0].messages[0]["content"]
    assert system_prompt.count(pinned_id) == 1
    assert system_prompt.count(retrieved_id) == 1
    assert system_prompt.index(pinned_id) < system_prompt.index("[USER_GOAL]")
    assert system_prompt.index(retrieved_id) > system_prompt.index("[USER_GOAL]")


def test_target_files_are_persisted_and_visible_to_model(tmp_path: Path):
    (tmp_path / "calculator.c").write_text("// English comment\n", encoding="utf-8")
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)

    run_id = engine.create_run(
        tmp_path,
        "translate this file's comments",
        RunBudget(max_llm_calls=1),
        target_files=["calculator.c"],
    )
    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["target_files"] == ["calculator.c"]
    created = store.list_events(run_id)[0]
    assert created.payload["target_files"] == ["calculator.c"]
    system_prompt = model.requests[0].messages[0]["content"]
    assert '"target_files": ["calculator.c"]' in system_prompt
    assert "Do not edit files outside target_files" in system_prompt


def test_context_compaction_is_visible_in_event_trace(tmp_path: Path):
    (tmp_path / "README.md").write_text("context data", encoding="utf-8")
    model = ScriptedModelGateway(
        [
            ModelResponse(tool_calls=[ToolCall("c1", "workspace.read", {"path": "README.md"})]),
            ModelResponse(content="done"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_builder=ContextBuilder(max_chars=220),
    )
    run_id = engine.create_run(tmp_path, "read with compact context")

    assert engine.run(run_id)["status"] == "completed"
    assert "context.compacted" in [event.type for event in store.list_events(run_id)]


def test_multi_risk_approval_does_not_replay_completed_tool_calls(tmp_path: Path):
    target = tmp_path / "value.py"
    target.write_text("value = 1\n", encoding="utf-8")
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "write-1",
                        "workspace.apply_patch",
                        {"path": "value.py", "old_text": "value = 1", "new_text": "value = 2"},
                    ),
                    ToolCall(
                        "execute-1",
                        "command.run",
                        {"argv": ["python3", "-c", "print('ok')"]},
                    ),
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    run_id = engine.create_run(tmp_path, "edit and execute")

    assert engine.step(run_id)["status"] == "waiting_approval"
    engine.approve(run_id, RiskLevel.WRITE)
    waiting = engine.run(run_id)
    assert waiting["status"] == "waiting_approval"
    assert [call["name"] for call in waiting["pending_tool_calls"]] == ["command.run"]
    engine.approve(run_id, RiskLevel.EXECUTE)
    assert engine.run(run_id)["status"] == "completed"

    started_tools = [
        event.payload["tool_call"]["name"]
        for event in store.list_events(run_id)
        if event.type == "tool.started"
    ]
    assert started_tools.count("workspace.apply_patch") == 1
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_agent_can_create_a_new_file_after_write_approval(tmp_path: Path):
    graph_source = """#include <vector>

int main() {
    std::vector<std::vector<int>> adjacency(10);
    adjacency[0].push_back(1);
    return 0;
}
"""
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "create-graph",
                        "workspace_create_file",
                        {"path": "graph.cpp", "content": graph_source},
                    )
                ]
            ),
            ModelResponse(content="Created graph.cpp."),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    run_id = engine.create_run(
        tmp_path,
        "Create a 10-node directed graph using an adjacency list.",
    )

    waiting = engine.step(run_id)
    assert waiting["status"] == "waiting_approval"
    assert waiting["pending_tool_calls"][0]["name"] == "workspace.create_file"
    engine.approve(run_id, RiskLevel.WRITE)
    completed = engine.run(run_id)

    assert completed["status"] == "completed"
    assert (tmp_path / "graph.cpp").read_text(encoding="utf-8") == graph_source
    assert "graph.cpp" in completed["working_state"]["changed_files"]
    assert any(
        definition["function"]["name"] == "workspace_create_file"
        for definition in model.requests[0].tools
    )
    finished = [
        event
        for event in store.list_events(run_id)
        if event.type == "tool.finished"
    ]
    assert finished[0].payload["tool"] == "workspace.create_file"
    assert finished[0].payload["result"]["status"] == "success"


def test_agent_can_create_directory_then_file_with_one_write_grant(tmp_path: Path):
    (tmp_path / "zidong").mkdir()
    source = "#pragma once\nclass Library {};\n"
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "create-directory",
                        "workspace_create_directory",
                        {"path": "zidong/library"},
                    ),
                    ToolCall(
                        "create-header",
                        "workspace_create_file",
                        {"path": "zidong/library/library.hpp", "content": source},
                    ),
                ]
            ),
            ModelResponse(content="Created the library directory and header."),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    run_id = engine.create_run(
        tmp_path,
        "Create zidong/library and add library.hpp inside it.",
    )

    waiting = engine.step(run_id)
    assert waiting["status"] == "waiting_approval"
    assert [item["name"] for item in waiting["pending_tool_calls"]] == [
        "workspace.create_directory",
        "workspace.create_file",
    ]
    engine.approve(run_id, RiskLevel.WRITE)
    completed = engine.run(run_id)

    assert completed["status"] == "completed"
    assert (tmp_path / "zidong" / "library").is_dir()
    assert (tmp_path / "zidong" / "library" / "library.hpp").read_text(
        encoding="utf-8"
    ) == source
    assert "zidong/library/library.hpp" in completed["working_state"]["changed_files"]
    assert any(
        definition["function"]["name"] == "workspace_create_directory"
        for definition in model.requests[0].tools
    )
    finished_tools = [
        event.payload["tool"]
        for event in store.list_events(run_id)
        if event.type == "tool.finished"
    ]
    assert finished_tools == ["workspace.create_directory", "workspace.create_file"]


def test_restart_after_unfinished_tool_pauses_as_uncertain(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="must not be called")]),
    )
    run_id = engine.create_run(tmp_path, "recover")
    state = engine.get_state(run_id)
    engine._record(state, "run.started", {}, "running")
    engine._record(
        state,
        "tool.started",
        {"tool_call": ToolCall("interrupted", "workspace.read", {"path": "x"}).to_dict()},
    )

    recovered = engine.step(run_id)

    assert recovered["status"] == "paused"
    assert store.list_events(run_id)[-1].type == "tool.uncertain"


def test_terminal_failure_releases_workspace_lease(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=True),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    run_id = engine.create_run(tmp_path, "fail after write")
    assert store.acquire_workspace_lease(str(tmp_path), run_id)
    state = engine.get_state(run_id)
    engine._record(state, "run.failed", {"error": "simulated"}, "failed")

    assert engine.step(run_id)["status"] == "failed"
    assert store.acquire_workspace_lease(str(tmp_path), "next-run")


def test_edit_cannot_complete_before_discovered_tests_pass(tmp_path: Path):
    target = tmp_path / "value.py"
    target.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "test_value.py").write_text(
        "from value import value\n\ndef test_value():\n    assert value == 2\n",
        encoding="utf-8",
    )
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "edit-1",
                        "workspace.apply_patch",
                        {"path": "value.py", "old_text": "value = 1", "new_text": "value = 2"},
                    )
                ]
            ),
            ModelResponse(content="premature completion"),
            ModelResponse(tool_calls=[ToolCall("tests-1", "tests.run", {})]),
            ModelResponse(content="verified completion"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(store, build_default_registry(include_mutating=True), model)
    run_id = engine.create_run(tmp_path, "fix and verify")

    assert engine.step(run_id)["status"] == "waiting_approval"
    engine.approve(run_id, RiskLevel.WRITE)
    assert engine.run(run_id)["status"] == "waiting_approval"
    assert "verification.blocked_completion" in [event.type for event in store.list_events(run_id)]
    assert all(event.type != "run.completed" for event in store.list_events(run_id))

    engine.approve(run_id, RiskLevel.EXECUTE)
    result = engine.run(run_id)
    assert result["status"] == "completed"
    assert result["final_answer"] == "verified completion"
    verification_results = result["working_state"]["verification"]["results"]
    assert verification_results[-1]["passed"] is True
    assert verification_results[-1]["status"] == "success"


def test_thread_run_hydrates_recent_conversation_and_persists_reply(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread(
        "Refactor service",
        workspace=str(tmp_path),
        budget={"max_llm_calls": 4, "max_tool_calls": 4},
        summary="Keep the public API stable.",
    )
    store.append_conversation_message(
        thread_id,
        role="user",
        content="Inspect the service.",
    )
    store.append_conversation_message(
        thread_id,
        role="assistant",
        content="The service has one oversized method.",
    )
    model = ScriptedModelGateway([ModelResponse(content="Refactor complete.")])
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)

    run_id = engine.create_run(tmp_path, "Refactor it.", thread_id=thread_id)
    result = engine.run(run_id)

    assert result["status"] == "completed"
    prompt = model.requests[0].messages
    assert "Keep the public API stable." in prompt[0]["content"]
    assert [message["content"] for message in prompt if message["role"] != "system"] == [
        "Inspect the service.",
        "The service has one oversized method.",
        "Refactor it.",
    ]
    conversation = store.list_conversation_messages(thread_id)
    assert [message["content"] for message in conversation][-2:] == [
        "Refactor it.",
        "Refactor complete.",
    ]
    assert conversation[-1]["run_id"] == run_id


def test_active_run_budget_can_only_increase_above_usage(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    run_id = engine.create_run(
        tmp_path,
        "budget",
        RunBudget(max_llm_calls=4, max_tool_calls=5),
    )
    state = engine.get_state(run_id)
    state["llm_calls"] = 2
    engine._record(state, "test.usage", {})

    updated = engine.update_budget(run_id, {"max_llm_calls": 8})

    assert updated["budget"]["max_llm_calls"] == 8
    assert store.get_run(run_id)["budget"]["max_llm_calls"] == 8
    assert store.list_events(run_id)[-1].type == "run.budget_updated"
    with pytest.raises(ValueError, match="below current usage"):
        engine.update_budget(run_id, {"max_llm_calls": 1})


def test_thread_budget_exhaustion_persists_visible_terminal_message(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Limited run", workspace=str(tmp_path))
    model = ScriptedModelGateway(
        [ModelResponse(tool_calls=[ToolCall("read-1", "workspace.list", {"path": "."})])]
    )
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)
    run_id = engine.create_run(
        tmp_path,
        "Inspect once",
        RunBudget(max_llm_calls=1, max_tool_calls=3),
        thread_id=thread_id,
    )

    assert engine.run(run_id)["status"] == "budget_exhausted"
    conversation = store.list_conversation_messages(thread_id)

    assert conversation[-1]["role"] == "assistant"
    assert conversation[-1]["metadata"]["status"] == "budget_exhausted"
    assert "max_llm_calls" in conversation[-1]["content"]


def test_thread_cancel_persists_visible_terminal_message_once(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Cancelled run", workspace=str(tmp_path))
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    run_id = engine.create_run(tmp_path, "Cancel me", thread_id=thread_id)

    engine.cancel(run_id)
    assert engine.step(run_id)["status"] == "cancelled"

    assistant_messages = [
        message
        for message in store.list_conversation_messages(thread_id)
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["metadata"]["status"] == "cancelled"


def test_terminal_run_cannot_be_reactivated_by_pause_or_resume(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([ModelResponse(content="unused")]),
    )
    run_id = engine.create_run(tmp_path, "Cancel permanently")
    assert engine.cancel(run_id)["status"] == "cancelled"
    event_count = len(store.list_events(run_id))

    with pytest.raises(ValueError, match="terminal"):
        engine.pause(run_id)
    with pytest.raises(ValueError, match="paused"):
        engine.resume(run_id)
    assert engine.cancel(run_id)["status"] == "cancelled"

    assert engine.get_state(run_id)["status"] == "cancelled"
    assert len(store.list_events(run_id)) == event_count


def test_explicit_external_target_is_visible_in_model_authorization_state(tmp_path: Path):
    external = tmp_path.parent / f"{tmp_path.name}-shared.py"
    external.write_text("shared = True\n", encoding="utf-8")
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread(
        "External edit",
        workspace=str(tmp_path),
        authorized_paths=[str(external)],
    )
    model = ScriptedModelGateway([ModelResponse(content="ready")])
    engine = RunEngine(store, build_default_registry(include_mutating=False), model)

    run_id = engine.create_run(
        tmp_path,
        "Inspect the external target",
        thread_id=thread_id,
        target_files=[str(external)],
    )
    assert engine.run(run_id)["status"] == "completed"

    system_prompt = model.requests[0].messages[0]["content"]
    assert str(external.resolve()) in system_prompt
    assert "user-authorized external paths" in system_prompt


def test_mid_run_compaction_uses_maintenance_calls_and_keeps_recent_tail(
    tmp_path: Path,
):
    (tmp_path / "README.md").write_text("recent tool evidence", encoding="utf-8")
    analysis = {
        "covered_range": {"from": 0, "to": 0},
        "active_goals": [],
        "user_constraints": [],
        "decisions": [],
        "completed_actions": [],
        "changed_files": [],
        "symbols_touched": [],
        "tool_findings": [],
        "failed_attempts": [],
        "verification_results": [],
        "unresolved_questions": [],
        "next_steps": [],
        "contradictions": [],
        "discarded_noise": [],
    }
    checkpoint = {
        "version": 1,
        "covered_range": {"from": 0, "to": 0},
        "task_objective": "Inspect README and preserve constraints.",
        "must_preserve": ["Keep public API stable"],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": [],
        "important_artifacts": [],
        "source_refs": ["message:0"],
        "repository_revision": None,
    }
    model = ScriptedModelGateway(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("read-1", "workspace.read", {"path": "README.md"})
                ]
            ),
            ModelResponse(content=json.dumps(analysis)),
            ModelResponse(content=json.dumps(checkpoint)),
            ModelResponse(content="done"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    registry = build_default_registry(include_mutating=False)
    engine = RunEngine(store, registry, model)
    run_id = engine.create_run(
        tmp_path,
        "Inspect README",
        RunBudget(max_llm_calls=3, max_tool_calls=3, max_compaction_calls=4),
    )
    assert engine.step(run_id)["status"] == "running"

    state = engine.get_state(run_id)
    state["messages"] = [
        {"role": "user", "content": "old constraint " * 2200},
        {"role": "assistant", "content": "old acknowledged"},
        *state["messages"],
    ]
    store.save_snapshot(run_id, state, expected_version=state["version"])

    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    profile = DeepSeekModelProfile(
        context_window_tokens=8192,
        default_output_reserve_tokens=512,
        safety_margin_tokens=128,
        provider_framing_tokens=64,
        compaction_trigger_ratio=0.60,
        compaction_target_ratio=0.45,
        analyzer_output_tokens=1024,
        summarizer_output_tokens=512,
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    serializer = DeepSeekRequestSerializer()
    engine.context_planner = ContextPlanner(counter, serializer, profile)
    engine.compaction_service = CompactionService(
        store=store,
        gateway=model,
        counter=counter,
        serializer=serializer,
        profile=profile,
    )

    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["llm_calls"] == 2
    assert result["compaction_calls"] == 2
    assert result["active_run_checkpoint_id"]
    event_types = [event.type for event in store.list_events(run_id)]
    assert "context.compaction_committed" in event_types
    committed_event = next(
        event
        for event in store.list_events(run_id)
        if event.type == "context.compaction_committed"
    )
    assert committed_event.payload["fallback_used"] is False, committed_event.payload
    final_request = [
        request for request in model.requests if request.purpose == "agent"
    ][-1]
    assert final_request.max_output_tokens == 512
    assert final_request.tools == engine._model_tool_definitions(result)
    final_text = json.dumps(final_request.messages, ensure_ascii=False)
    assert "CONTEXT_CHECKPOINT" in final_text
    assert "recent tool evidence" in final_text
    assert "old constraint" not in final_text


def test_context_hard_limit_stops_before_main_model_call(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="must not run")])
    store = EventStore(tmp_path / "agent.db")
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    profile = DeepSeekModelProfile(
        context_window_tokens=512,
        default_output_reserve_tokens=128,
        safety_margin_tokens=64,
        provider_framing_tokens=32,
        compaction_trigger_ratio=0.60,
        compaction_target_ratio=0.40,
        analyzer_output_tokens=128,
        summarizer_output_tokens=64,
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=ContextPlanner(
            counter,
            DeepSeekRequestSerializer(),
            profile,
        ),
    )
    run_id = engine.create_run(tmp_path, "x" * 5000)

    result = engine.run(run_id)

    assert result["status"] == "budget_exhausted"
    assert model.requests == []
    events = store.list_events(run_id)
    assert events[-2].type == "context.hard_limit_blocked"
    assert events[-1].type == "run.budget_exhausted"
    assert events[-1].payload["reason"] == "context_hard_limit"


def test_run_input_budget_is_preflighted_before_agent_call(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="must not run")])
    store = EventStore(tmp_path / "agent.db")
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=ContextPlanner(
            counter,
            DeepSeekRequestSerializer(),
            DeepSeekModelProfile(
                context_window_tokens=8192,
                default_output_reserve_tokens=512,
            ),
        ),
    )
    run_id = engine.create_run(
        tmp_path,
        "inspect repository",
        RunBudget(max_input_tokens=1),
    )

    result = engine.run(run_id)

    assert result["status"] == "budget_exhausted"
    assert model.requests == []
    assert store.list_events(run_id)[-1].payload["reason"] == "max_input_tokens"


def test_agent_output_limit_uses_remaining_run_budget(tmp_path: Path):
    model = ScriptedModelGateway([ModelResponse(content="done")])
    store = EventStore(tmp_path / "agent.db")
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=ContextPlanner(
            counter,
            DeepSeekRequestSerializer(),
            DeepSeekModelProfile(
                context_window_tokens=8192,
                default_output_reserve_tokens=512,
            ),
        ),
    )
    run_id = engine.create_run(
        tmp_path,
        "finish briefly",
        RunBudget(max_output_tokens=100),
    )

    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert model.requests[-1].max_output_tokens == 100


def test_compaction_safe_point_requires_closed_tool_exchange(tmp_path: Path):
    engine = RunEngine(
        EventStore(tmp_path / "agent.db"),
        build_default_registry(include_mutating=False),
        ScriptedModelGateway([]),
    )
    state = {
        "status": "running",
        "pending_tool_calls": [],
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "workspace.read", "args": {}},
                    {"id": "c2", "name": "workspace.list", "args": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
    }

    assert engine._compaction_safe(state) is False

    state["messages"].append(
        {"role": "tool", "tool_call_id": "c2", "content": "done"}
    )
    assert engine._compaction_safe(state) is True

    state["status"] = "waiting_approval"
    assert engine._compaction_safe(state) is False


def test_new_run_hydrates_thread_checkpoint_and_only_messages_after_watermark(
    tmp_path: Path,
):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread(
        "Long conversation",
        workspace=tmp_path,
        summary="legacy summary should be superseded",
    )
    for sequence in range(1, 11):
        store.append_conversation_message(
            thread_id,
            role="user" if sequence % 2 else "assistant",
            content=f"message-{sequence}",
        )
    checkpoint = {
        "version": 1,
        "covered_range": {"from": 1, "to": 6},
        "task_objective": "Continue the repository task.",
        "must_preserve": ["Public API is stable"],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": [],
        "important_artifacts": [],
        "source_refs": ["conversation:1"],
        "repository_revision": None,
    }
    thread_source_hash = canonical_message_hash(
        store.list_conversation_messages(
            thread_id, limit=None, through_sequence=6
        )
    )
    compaction = store.create_compaction(
        scope="thread",
        thread_id=thread_id,
        covered_from=1,
        covered_to=6,
        source_version=0,
        source_hash=thread_source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version="analyzer-v1",
        summarizer_prompt_version="summarizer-v1",
    )
    store.update_compaction(
        compaction["id"], status="summarizing", analysis={"covered": True}
    )
    store.update_compaction(
        compaction["id"], status="validating", checkpoint=checkpoint
    )
    store.commit_compaction(
        compaction["id"], expected_source_hash=thread_source_hash
    )

    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    model = ScriptedModelGateway([ModelResponse(content="continued")])
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=ContextPlanner(
            counter,
            DeepSeekRequestSerializer(),
            DeepSeekModelProfile(),
        ),
    )

    run_id = engine.create_run(tmp_path, "new goal", thread_id=thread_id)
    result = engine.run(run_id)

    assert result["status"] == "completed"
    request = model.requests[0]
    request_text = json.dumps(request.messages, ensure_ascii=False)
    assert request_text.count("CONTEXT_CHECKPOINT") == 2  # open and close tags
    non_system_content = [
        message["content"]
        for message in request.messages
        if message["role"] != "system"
    ]
    assert "message-1" not in non_system_content
    assert "message-6" not in non_system_content
    for sequence in range(7, 11):
        assert f"message-{sequence}" in request_text
    assert "new goal" in request_text
    assert store.get_thread(thread_id)["summary"] == (
        "legacy summary should be superseded"
    )


def test_new_run_compacts_oversized_completed_thread_prefix(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Pressure", workspace=tmp_path)
    store.append_conversation_message(
        thread_id,
        role="user",
        content="old thread context " * 2200,
    )
    store.append_conversation_message(
        thread_id, role="assistant", content="old context acknowledged"
    )
    store.append_conversation_message(
        thread_id, role="user", content="recent question"
    )
    store.append_conversation_message(
        thread_id, role="assistant", content="recent answer"
    )
    analysis = {
        "covered_range": {"from": 1, "to": 1},
        "active_goals": [],
        "user_constraints": [],
        "decisions": [],
        "completed_actions": [],
        "changed_files": [],
        "symbols_touched": [],
        "tool_findings": [],
        "failed_attempts": [],
        "verification_results": [],
        "unresolved_questions": [],
        "next_steps": [],
        "contradictions": [],
        "discarded_noise": [],
    }
    checkpoint = {
        "version": 1,
        "covered_range": {"from": 1, "to": 1},
        "task_objective": "Continue the thread.",
        "must_preserve": ["old thread constraint"],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": [],
        "important_artifacts": [],
        "source_refs": ["conversation:1"],
        "repository_revision": None,
    }
    model = ScriptedModelGateway(
        [
            ModelResponse(content=json.dumps(analysis)),
            ModelResponse(content=json.dumps(checkpoint)),
            ModelResponse(content="continued"),
        ]
    )
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    profile = DeepSeekModelProfile(
        context_window_tokens=12288,
        default_output_reserve_tokens=512,
        safety_margin_tokens=128,
        provider_framing_tokens=64,
        compaction_trigger_ratio=0.55,
        compaction_target_ratio=0.40,
        analyzer_output_tokens=1024,
        summarizer_output_tokens=512,
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    serializer = DeepSeekRequestSerializer()
    planner = ContextPlanner(counter, serializer, profile)
    service = CompactionService(
        store=store,
        gateway=model,
        counter=counter,
        serializer=serializer,
        profile=profile,
    )
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=planner,
        compaction_service=service,
    )

    run_id = engine.create_run(tmp_path, "new work", thread_id=thread_id)
    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert [request.purpose for request in model.requests] == [
        "compaction_analysis",
        "compaction_summary",
        "agent",
    ]
    assert result["compaction_calls"] == 2
    assert store.get_thread(thread_id)["checkpoint_covered_sequence"] == 1
    agent_text = json.dumps(model.requests[-1].messages, ensure_ascii=False)
    assert "CONTEXT_CHECKPOINT" in agent_text
    assert "recent question" in agent_text
    assert "old thread context" not in agent_text


def test_run_engine_resumes_persisted_summarizing_compaction_before_agent_call(
    tmp_path: Path,
):
    checkpoint = {
        "version": 1,
        "covered_range": {"from": 0, "to": 0},
        "task_objective": "Resume safely.",
        "must_preserve": [],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": [],
        "important_artifacts": [],
        "source_refs": ["message:0"],
        "repository_revision": None,
    }
    model = ScriptedModelGateway(
        [
            ModelResponse(content=json.dumps(checkpoint)),
            ModelResponse(content="resumed"),
        ]
    )
    store = EventStore(tmp_path / "agent.db")
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
    )
    run_id = engine.create_run(tmp_path, "resume compaction")
    state = engine.get_state(run_id)
    state["messages"] = [
        {"role": "user", "content": "old context " * 2200},
        {"role": "user", "content": "recent tail"},
    ]
    store.save_snapshot(run_id, state, expected_version=state["version"])
    source = FrozenCompactionSource.freeze(
        run_id=run_id,
        thread_id=None,
        source_snapshot_version=state["version"],
        covered_from=0,
        covered_to=0,
        recent_tail_from=1,
        messages=state["messages"],
        working_state=state["working_state"],
        old_checkpoint=None,
    )
    record = store.create_compaction(
        scope="run",
        run_id=run_id,
        covered_from=0,
        covered_to=0,
        source_version=state["version"],
        source_hash=source.source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version=ANALYZER_PROMPT_VERSION,
        summarizer_prompt_version=SUMMARIZER_PROMPT_VERSION,
    )
    analysis = {
        "covered_range": {"from": 0, "to": 0},
        "active_goals": [],
        "user_constraints": [],
        "decisions": [],
        "completed_actions": [],
        "changed_files": [],
        "symbols_touched": [],
        "tool_findings": [],
        "failed_attempts": [],
        "verification_results": [],
        "unresolved_questions": [],
        "next_steps": [],
        "contradictions": [],
        "discarded_noise": [],
    }
    assert store.reserve_compaction_call(
        record["id"], max_maintenance_calls=8
    )
    store.record_compaction_response_usage(
        record["id"], input_tokens=10, output_tokens=5, cost_usd=0.01
    )
    store.update_compaction(
        record["id"], status="summarizing", analysis=analysis
    )
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    serializer = DeepSeekRequestSerializer()
    profile = DeepSeekModelProfile(context_window_tokens=16384)
    engine.context_planner = ContextPlanner(counter, serializer, profile)
    engine.compaction_service = CompactionService(
        store=store,
        gateway=model,
        counter=counter,
        serializer=serializer,
        profile=profile,
    )

    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["active_run_checkpoint_id"] == record["id"]
    assert result["compaction_calls"] == 2
    assert result["compaction_input_tokens"] >= 10
    assert result["compaction_output_tokens"] >= 5
    assert [request.purpose for request in model.requests] == [
        "compaction_summary",
        "agent",
    ]


def test_create_run_resumes_persisted_thread_compaction(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Resume thread", workspace=tmp_path)
    store.append_conversation_message(
        thread_id, role="user", content="preserve this"
    )
    store.append_conversation_message(
        thread_id, role="assistant", content="acknowledged"
    )
    covered = store.list_conversation_messages(thread_id, limit=None)
    source = FrozenThreadCompactionSource.freeze(
        thread_id=thread_id,
        source_thread_version=0,
        covered_messages=covered,
        working_state={
            "current_objective": "continue",
            "next_expected_action": "start the new run",
        },
        old_checkpoint=None,
    )
    record = store.create_compaction(
        scope="thread",
        thread_id=thread_id,
        covered_from=source.covered_from,
        covered_to=source.covered_to,
        source_version=0,
        source_hash=source.source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version=ANALYZER_PROMPT_VERSION,
        summarizer_prompt_version=SUMMARIZER_PROMPT_VERSION,
    )
    analysis = {
        "covered_range": source.covered_range,
        "active_goals": [],
        "user_constraints": [],
        "decisions": [],
        "completed_actions": [],
        "changed_files": [],
        "symbols_touched": [],
        "tool_findings": [],
        "failed_attempts": [],
        "verification_results": [],
        "unresolved_questions": [],
        "next_steps": [],
        "contradictions": [],
        "discarded_noise": [],
    }
    assert store.reserve_compaction_call(record["id"], max_maintenance_calls=8)
    store.record_compaction_response_usage(
        record["id"], input_tokens=11, output_tokens=7, cost_usd=0.01
    )
    store.update_compaction(record["id"], status="summarizing", analysis=analysis)
    checkpoint = {
        "version": 1,
        "covered_range": source.covered_range,
        "task_objective": "continue",
        "must_preserve": ["preserve this"],
        "decisions": [],
        "current_repository_state": [],
        "completed_work": [],
        "failed_approaches": [],
        "verification_state": [],
        "open_work": [],
        "important_artifacts": [],
        "source_refs": ["conversation:1"],
        "repository_revision": None,
    }
    model = ScriptedModelGateway(
        [ModelResponse(content=json.dumps(checkpoint)), ModelResponse(content="done")]
    )
    tokenizer_root = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "deepseek_v3_tokenizer"
    )
    counter = DeepSeekTokenCounter.from_directory(tokenizer_root)
    serializer = DeepSeekRequestSerializer()
    profile = DeepSeekModelProfile(context_window_tokens=16_384)
    engine = RunEngine(
        store,
        build_default_registry(include_mutating=False),
        model,
        context_planner=ContextPlanner(counter, serializer, profile),
        compaction_service=CompactionService(
            store=store,
            gateway=model,
            counter=counter,
            serializer=serializer,
            profile=profile,
        ),
    )

    run_id = engine.create_run(tmp_path, "continue", thread_id=thread_id)
    result = engine.run(run_id)

    assert result["status"] == "completed"
    assert result["compaction_calls"] == 2
    assert result["compaction_input_tokens"] >= 11
    assert store.get_thread(thread_id)["active_thread_checkpoint_id"] == record["id"]
    assert [request.purpose for request in model.requests] == [
        "compaction_summary",
        "agent",
    ]
