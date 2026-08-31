from pathlib import Path

import pytest

from code_agent.agent_core.event_store import EventStore, ThreadActiveConflict, VersionConflict


def test_event_store_is_monotonic_idempotent_and_resumable(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    store.create_run("run-1", str(tmp_path), "goal", {"max_llm_calls": 3})
    first = store.append_event("run-1", "run.started", {"x": 1}, expected_version=0, idempotency_key="start")
    repeated = store.append_event("run-1", "run.started", {"x": 1}, expected_version=1, idempotency_key="start")
    assert first.sequence == 1
    assert repeated.sequence == 1
    assert len(store.list_events("run-1")) == 1
    with pytest.raises(VersionConflict):
        store.append_event("run-1", "bad", {}, expected_version=0)

    store.save_snapshot("run-1", {"status": "paused", "messages": []}, expected_version=1)
    assert store.load_snapshot("run-1")["status"] == "paused"


def test_workspace_lease_has_single_writer(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    assert store.acquire_workspace_lease(str(tmp_path), "run-a") is True
    assert store.acquire_workspace_lease(str(tmp_path), "run-b") is False
    store.release_workspace_lease(str(tmp_path), "run-a")
    assert store.acquire_workspace_lease(str(tmp_path), "run-b") is True


def test_thread_persists_settings_and_ordered_conversation(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    external = tmp_path.parent / "shared.py"
    thread_id = store.create_thread(
        "Repository work",
        workspace=str(tmp_path),
        model="deepseek-chat",
        runtime_mode="agent",
        budget={"max_llm_calls": 40, "max_tool_calls": 60},
        authorized_paths=[str(external)],
    )
    store.create_run(
        "run-1",
        str(tmp_path),
        "inspect",
        {"max_llm_calls": 40},
        thread_id,
    )
    first = store.append_conversation_message(
        thread_id,
        role="user",
        content="Inspect the repository",
        run_id="run-1",
    )
    second = store.append_conversation_message(
        thread_id,
        role="assistant",
        content="Inspection complete",
        run_id="run-1",
        kind="final",
        metadata={"status": "completed"},
    )
    updated = store.update_thread(
        thread_id,
        summary="The public API must remain stable.",
        budget={"max_llm_calls": 50, "max_tool_calls": 80},
        expected_version=0,
    )

    detail = store.get_thread(thread_id)
    messages = store.list_conversation_messages(thread_id)

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["metadata"]["status"] == "completed"
    assert detail["workspace"] == str(tmp_path.resolve())
    assert detail["model"] == "deepseek-chat"
    assert detail["budget"]["max_llm_calls"] == 50
    assert detail["authorized_paths"] == [str(external.resolve())]
    assert detail["summary"] == "The public API must remain stable."
    assert detail["last_run_id"] == "run-1"
    assert updated["version"] == 1


def test_legacy_run_migration_creates_recoverable_thread(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    store.create_run("legacy-run", str(tmp_path), "Legacy goal", {"max_llm_calls": 3})
    store.save_snapshot(
        "legacy-run",
        {
            "run_id": "legacy-run",
            "workspace": str(tmp_path),
            "goal": "Legacy goal",
            "status": "completed",
            "messages": [
                {"role": "user", "content": "Legacy goal"},
                {"role": "assistant", "content": "Legacy answer", "tool_calls": []},
            ],
            "final_answer": "Legacy answer",
        },
        expected_version=0,
    )
    legacy_updated_at = store.get_run("legacy-run")["updated_at"]

    assert store.migrate_legacy_runs() == 1
    assert store.migrate_legacy_runs() == 0

    run = store.get_run("legacy-run")
    thread = store.get_thread(run["thread_id"])
    messages = store.list_conversation_messages(run["thread_id"])

    assert thread["title"] == "Legacy goal"
    assert thread["last_run_id"] == "legacy-run"
    assert thread["updated_at"] == legacy_updated_at
    assert [message["content"] for message in messages] == ["Legacy goal", "Legacy answer"]


def test_delete_thread_removes_runtime_graph_without_touching_other_threads(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    deleted_thread = store.create_thread("Delete me", workspace=tmp_path)
    kept_thread = store.create_thread("Keep me", workspace=tmp_path)
    store.create_run("run-delete", str(tmp_path), "delete work", {}, deleted_thread)
    store.create_run("run-keep", str(tmp_path), "keep work", {}, kept_thread)
    store.append_event(
        "run-delete",
        "run.completed",
        {},
        expected_version=0,
        status="completed",
    )
    store.save_snapshot(
        "run-delete",
        {"status": "completed", "messages": []},
        expected_version=1,
    )
    store.grant_approval("run-delete", "write")
    assert store.acquire_workspace_lease(str(tmp_path), "run-delete") is True
    store.append_conversation_message(
        deleted_thread,
        role="user",
        content="Remove this conversation",
        run_id="run-delete",
    )

    assert store.delete_thread(deleted_thread) == deleted_thread

    with pytest.raises(KeyError):
        store.get_thread(deleted_thread)
    with pytest.raises(KeyError):
        store.get_run("run-delete")
    with pytest.raises(KeyError):
        store.list_conversation_messages(deleted_thread)
    assert store.list_events("run-delete") == []
    assert store.load_snapshot("run-delete") is None
    assert store.approvals_for("run-delete") == set()
    assert store.acquire_workspace_lease(str(tmp_path), "replacement-run") is True
    assert store.get_thread(kept_thread)["title"] == "Keep me"
    assert store.get_run("run-keep")["thread_id"] == kept_thread


@pytest.mark.parametrize("status", ["queued", "running", "waiting_approval", "paused"])
def test_delete_thread_rejects_active_runs(tmp_path: Path, status: str):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Active", workspace=tmp_path)
    store.create_run("run-active", str(tmp_path), "work", {}, thread_id)
    if status != "queued":
        store.append_event(
            "run-active",
            "run.status",
            {},
            expected_version=0,
            status=status,
        )

    with pytest.raises(ThreadActiveConflict, match="Cancel"):
        store.delete_thread(thread_id)

    assert store.get_thread(thread_id)["id"] == thread_id
    assert store.get_run("run-active")["status"] == status


def test_delete_thread_rejects_unknown_thread(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")

    with pytest.raises(KeyError, match="unknown thread"):
        store.delete_thread("missing-thread")


def test_conversation_query_can_load_complete_tail_without_fixed_limit(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Complete tail", workspace=tmp_path)
    for index in range(1, 7):
        store.append_conversation_message(
            thread_id,
            role="user",
            content=f"message-{index}",
        )

    complete = store.list_conversation_messages(
        thread_id,
        limit=None,
        after_sequence=2,
    )

    assert [message["sequence"] for message in complete] == [3, 4, 5, 6]
