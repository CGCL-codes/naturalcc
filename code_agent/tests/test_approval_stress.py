from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from code_agent.agent_core.contracts import ModelResponse, RiskLevel, RunBudget, ToolCall
from code_agent.agent_core.event_store import EventStore
from code_agent.agent_core.model_gateway import ScriptedModelGateway
from code_agent.agent_core.run_engine import RunEngine
from code_agent.agent_core.tool_registry import build_default_registry


def _write_stress_engine(tmp_path: Path, write_count: int) -> tuple[RunEngine, str]:
    calls: list[ToolCall] = []
    for index in range(write_count):
        path = tmp_path / f"part_{index:03d}.py"
        path.write_text("value = 0\n", encoding="utf-8")
        calls.append(
            ToolCall(
                f"write-{index:03d}",
                "workspace.apply_patch",
                {
                    "path": path.name,
                    "old_text": "value = 0",
                    "new_text": "value = 1",
                },
            )
        )
    model = ScriptedModelGateway(
        [ModelResponse(tool_calls=calls), ModelResponse(content="all edits completed")]
    )
    engine = RunEngine(
        EventStore(tmp_path / "agent.db"),
        build_default_registry(include_mutating=True),
        model,
    )
    run_id = engine.create_run(
        tmp_path,
        "apply a large batch of independent edits",
        RunBudget(max_llm_calls=4, max_tool_calls=write_count + 4, max_seconds=60),
    )
    return engine, run_id


def test_sixty_four_write_calls_keep_one_durable_approval_gate(tmp_path: Path):
    engine, run_id = _write_stress_engine(tmp_path, 64)

    waiting = engine.run(run_id)

    assert waiting["status"] == "waiting_approval"
    assert waiting["pending_approval"]["tool_call"]["id"] == "write-000"
    assert len(waiting["pending_tool_calls"]) == 64

    engine.approve(run_id, RiskLevel.WRITE)
    completed = engine.run(run_id)

    assert completed["status"] == "completed"
    assert completed["pending_approval"] is None
    assert all(
        (tmp_path / f"part_{index:03d}.py").read_text(encoding="utf-8") == "value = 1\n"
        for index in range(64)
    )
    events = engine.store.list_events(run_id)
    assert sum(event.type == "approval.requested" for event in events) == 1
    assert sum(event.type == "tool.finished" for event in events) == 64


def test_duplicate_approval_and_run_requests_are_serialized(tmp_path: Path):
    engine, run_id = _write_stress_engine(tmp_path, 1)
    assert engine.run(run_id)["status"] == "waiting_approval"

    with ThreadPoolExecutor(max_workers=16) as pool:
        approval_states = list(
            pool.map(lambda _: engine.approve(run_id, RiskLevel.WRITE), range(32))
        )

    assert all(state["status"] == "running" for state in approval_states)
    events = engine.store.list_events(run_id)
    assert sum(event.type == "approval.resolved" for event in events) == 1

    with ThreadPoolExecutor(max_workers=16) as pool:
        completed_states = list(pool.map(lambda _: engine.run(run_id), range(32)))

    assert all(state["status"] == "completed" for state in completed_states)
    events = engine.store.list_events(run_id)
    assert sum(event.type == "tool.started" for event in events) == 1
    assert sum(event.type == "run.completed" for event in events) == 1


def test_legacy_waiting_snapshot_reconstructs_pending_approval(tmp_path: Path):
    engine, run_id = _write_stress_engine(tmp_path, 1)
    waiting = engine.run(run_id)
    legacy = engine.store.load_snapshot(run_id)
    assert legacy is not None
    legacy.pop("pending_approval", None)
    engine.store.save_snapshot(run_id, legacy, expected_version=waiting["version"])

    restored = engine.get_state(run_id)

    assert restored["status"] == "waiting_approval"
    assert restored["pending_approval"]["risk"] == "write"
    assert restored["pending_approval"]["tool_call"]["id"] == "write-000"


def test_wrong_risk_cannot_resolve_a_pending_write(tmp_path: Path):
    engine, run_id = _write_stress_engine(tmp_path, 1)
    assert engine.run(run_id)["status"] == "waiting_approval"

    try:
        engine.approve(run_id, RiskLevel.EXECUTE)
    except ValueError as exc:
        assert "requires write risk" in str(exc)
    else:
        raise AssertionError("wrong-risk approval unexpectedly succeeded")

    state = engine.get_state(run_id)
    assert state["status"] == "waiting_approval"
    assert state["pending_approval"]["risk"] == "write"
