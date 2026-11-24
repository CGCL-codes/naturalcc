import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from code_agent.agent_core.contracts import RunStatus
from code_agent.agent_core.event_store import EventStore, VersionConflict
from code_agent.agent_core.human_input import (
    USER_INPUT_TOOL_NAME,
    HumanInputValidationError,
    user_input_tool_definition,
    validate_answer,
    validate_question,
)


def test_question_and_answer_shapes_keep_typed_values():
    schema = user_input_tool_definition()["function"]
    assert schema["name"] == USER_INPUT_TOOL_NAME == "user_ask"
    assert schema["parameters"]["properties"]["answer_type"]["enum"] == ["text", "choice", "confirm"]

    text = validate_question({"question": "  补充什么信息？ ", "answer_type": "text"})
    assert text == {"question": "补充什么信息？", "answer_type": "text", "options": []}
    assert validate_answer(text, "  第一行\n第二行 \n") == "第一行\n第二行"

    confirm = validate_question({"question": "Continue?", "answer_type": "confirm", "options": []})
    assert validate_answer(confirm, False) is False
    assert validate_answer(confirm, True) is True

    choice = validate_question({
        "question": "Choose a mode.",
        "answer_type": "choice",
        "options": [{"id": "quick", "label": " Quick "}, {"id": "full", "label": "Full"}],
    })
    assert choice["options"] == [{"id": "quick", "label": "Quick"}, {"id": "full", "label": "Full"}]
    assert validate_answer(choice, "quick") == "quick"
    with pytest.raises(HumanInputValidationError):
        validate_answer(choice, "Quick")


@pytest.mark.parametrize("question", [
    {"question": "  ", "answer_type": "text"},
    {"question": "x" * 1_001, "answer_type": "text"},
    {"question": "q", "answer_type": "boolean"},
    {"question": "q", "answer_type": "confirm", "options": [{"id": "x", "label": "X"}]},
    {"question": "q", "answer_type": "choice", "options": [{"id": "x", "label": "X"}]},
    {"question": "q", "answer_type": "choice", "options": [{"id": "x", "label": "X"}] * 2},
    {"question": "q", "answer_type": "choice", "options": [{"id": "x", "label": "X"}, {"id": "bad id", "label": "Y"}]},
    {"question": "q", "answer_type": "text", "authorization": True},
])
def test_invalid_question_is_rejected(question):
    with pytest.raises(HumanInputValidationError):
        validate_question(question)


@pytest.mark.parametrize("answer_type,answer", [
    ("confirm", 1), ("confirm", "true"), ("confirm", None),
    ("text", "   "), ("text", "x" * 4_001), ("text", "bad\x00value"), ("text", "\ud800"),
])
def test_invalid_answer_is_rejected(answer_type, answer):
    pending = validate_question({"question": "Question?", "answer_type": answer_type})
    with pytest.raises(HumanInputValidationError):
        validate_answer(pending, answer)


@pytest.fixture
def running_store(tmp_path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Pending decision", workspace=str(tmp_path))
    store.create_run("run", str(tmp_path), "goal", {}, thread_id)
    event = store.append_event("run", "run.started", {}, expected_version=0, status=RunStatus.RUNNING.value)
    state = {"version": event.sequence, "status": RunStatus.RUNNING.value, "run_id": "run", "messages": []}
    store.save_snapshot("run", state, expected_version=event.sequence)
    return store, thread_id, state


def test_waiting_input_event_run_status_and_snapshot_are_committed_together(running_store):
    store, thread_id, state = running_store
    question = validate_question({"question": "Continue?", "answer_type": "confirm"})
    state["pending_user_input"] = {**question, "tool_call_id": "question-1"}
    event = store.append_event_and_snapshot(
        "run", state, "input.requested", state["pending_user_input"], status=RunStatus.WAITING_INPUT.value,
    )

    restarted = EventStore(store.db_path)
    assert event == restarted.list_events("run")[-1]
    assert state == restarted.load_snapshot("run")
    assert state["version"] == restarted.get_run("run")["version"] == event.sequence == 2
    assert state["status"] == restarted.get_run("run")["status"] == "waiting_input"
    assert restarted.get_thread(thread_id)["active_run_id"] == "run"


def test_snapshot_failure_rolls_back_event_and_status(running_store):
    store, _, state = running_store
    previous = dict(state)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("""
            CREATE TRIGGER reject_snapshot BEFORE UPDATE ON snapshots
            BEGIN SELECT RAISE(ABORT, 'test snapshot failure'); END
        """)

    with pytest.raises(sqlite3.IntegrityError, match="test snapshot failure"):
        store.append_event_and_snapshot("run", state, "input.requested", {}, status="waiting_input")

    assert store.load_snapshot("run") == state == previous
    assert store.get_run("run")["version"] == 1
    assert store.get_run("run")["status"] == "running"
    assert [event.type for event in store.list_events("run")] == ["run.started"]


def test_racing_transitions_only_commit_one_answer(running_store):
    store, _, state = running_store
    pending = {**state, "pending_user_input": {"tool_call_id": "question-1"}}
    store.append_event_and_snapshot("run", pending, "input.requested", {}, status="waiting_input")

    def answer(value):
        next_state = {**pending, "pending_user_input": None, "answer": value}
        try:
            store.append_event_and_snapshot("run", next_state, "input.resolved", {"answer": value}, status="running")
        except VersionConflict:
            assert next_state["version"] == 2
            assert next_state["status"] == "waiting_input"
            return None
        return value

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(answer, ["fast", "full"]))

    winner, = [result for result in results if result is not None]
    assert store.load_snapshot("run")["answer"] == winner
    assert [event.sequence for event in store.list_events("run")] == [1, 2, 3]
    assert store.get_run("run")["status"] == "running"
