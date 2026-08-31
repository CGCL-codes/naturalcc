from pathlib import Path
import sqlite3

import pytest

from code_agent.agent_core.compaction import canonical_message_hash
from code_agent.agent_core.contracts import RunBudget
from code_agent.agent_core.event_store import EventStore


def test_run_budget_has_separate_compaction_limit():
    budget = RunBudget.from_dict({"max_llm_calls": 3, "max_compaction_calls": 6})

    assert budget.max_llm_calls == 3
    assert budget.max_compaction_calls == 6


def test_event_store_migrates_compaction_schema(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")

    with store._connect() as connection:
        names = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(compactions)")
        }
        thread_names = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(threads)")
        }
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert {"analysis_json", "checkpoint_json", "source_hash"} <= names
    assert {
        "active_thread_checkpoint_id",
        "checkpoint_covered_sequence",
    } <= thread_names
    assert user_version == 6


def _create_run_compaction(
    store: EventStore,
    *,
    thread_id: str,
    run_id: str,
    source_hash: str,
    covered_to: int,
):
    snapshot = store.load_snapshot(run_id) or {}
    return store.create_compaction(
        scope="run",
        thread_id=thread_id,
        run_id=run_id,
        covered_from=int(snapshot.get("compacted_message_to", -1)) + 1,
        covered_to=covered_to,
        source_version=0,
        source_hash=source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version="analyzer-v1",
        summarizer_prompt_version="summarizer-v1",
    )


def _prepare_validating(
    store: EventStore,
    compaction_id: str,
    *,
    source_hash: str,
    covered_to: int,
):
    store.update_compaction(
        compaction_id,
        status="summarizing",
        analysis={"covered_range": {"from": 0, "to": covered_to}},
        analyzer_input_tokens=100,
        analyzer_output_tokens=20,
    )
    return store.update_compaction(
        compaction_id,
        status="validating",
        checkpoint={
            "version": 1,
            "covered_range": {"from": 0, "to": covered_to},
            "source_hash": source_hash,
        },
        summarizer_input_tokens=40,
        summarizer_output_tokens=10,
    )


def test_run_compaction_lifecycle_and_atomic_active_switch(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Compaction", workspace=tmp_path)
    store.create_run("run-1", str(tmp_path), "work", {}, thread_id)
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(9)
    ]
    store.save_snapshot(
        "run-1",
        {
            "status": "running",
            "messages": messages,
            "active_run_checkpoint_id": None,
            "compacted_message_to": -1,
        },
        expected_version=0,
    )

    first = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-1",
        source_hash=canonical_message_hash(messages[:5]),
        covered_to=4,
    )
    assert first["status"] == "analyzing"
    assert first["version"] == 1
    validating = _prepare_validating(
        store,
        first["id"],
        source_hash=canonical_message_hash(messages[:5]),
        covered_to=4,
    )
    assert validating["analysis"]["covered_range"]["to"] == 4
    committed = store.commit_compaction(
        first["id"], expected_source_hash=canonical_message_hash(messages[:5])
    )
    assert committed["status"] == "committed"
    snapshot = store.load_snapshot("run-1")
    assert snapshot["active_run_checkpoint_id"] == first["id"]
    assert snapshot["compacted_message_to"] == 4

    second = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-1",
        source_hash=canonical_message_hash(messages[5:9]),
        covered_to=8,
    )
    assert second["version"] == 2
    _prepare_validating(
        store,
        second["id"],
        source_hash=canonical_message_hash(messages[5:9]),
        covered_to=8,
    )
    store.commit_compaction(
        second["id"], expected_source_hash=canonical_message_hash(messages[5:9])
    )

    assert store.get_compaction(first["id"])["status"] == "superseded"
    assert store.latest_committed_compaction(run_id="run-1")["id"] == second["id"]
    snapshot = store.load_snapshot("run-1")
    assert snapshot["active_run_checkpoint_id"] == second["id"]
    assert snapshot["compacted_message_to"] == 8


def test_thread_compaction_commit_updates_watermark(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Thread compaction", workspace=tmp_path)
    for index in range(6):
        store.append_conversation_message(
            thread_id, role="user", content=f"message-{index}"
        )
    source_hash = canonical_message_hash(
        store.list_conversation_messages(thread_id, limit=None)
    )
    compaction = store.create_compaction(
        scope="thread",
        thread_id=thread_id,
        covered_from=1,
        covered_to=6,
        source_version=0,
        source_hash=source_hash,
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version="analyzer-v1",
        summarizer_prompt_version="summarizer-v1",
    )
    _prepare_validating(
        store,
        compaction["id"],
        source_hash=source_hash,
        covered_to=6,
    )

    store.commit_compaction(
        compaction["id"], expected_source_hash=source_hash
    )

    thread = store.get_thread(thread_id)
    assert thread["active_thread_checkpoint_id"] == compaction["id"]
    assert thread["checkpoint_covered_sequence"] == 6


def test_stale_run_compaction_cannot_replace_newer_checkpoint(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Stale run", workspace=tmp_path)
    store.create_run("run-stale", str(tmp_path), "work", {}, thread_id)
    messages = [
        {"role": "user", "content": f"message-{index}"}
        for index in range(3)
    ]
    store.save_snapshot(
        "run-stale",
        {
            "messages": messages,
            "active_run_checkpoint_id": None,
            "compacted_message_to": -1,
        },
        expected_version=0,
    )
    first_hash = canonical_message_hash(messages[:1])
    stale_hash = canonical_message_hash(messages[:2])
    first = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-stale",
        source_hash=first_hash,
        covered_to=0,
    )
    stale = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-stale",
        source_hash=stale_hash,
        covered_to=1,
    )
    _prepare_validating(store, first["id"], source_hash=first_hash, covered_to=0)
    _prepare_validating(store, stale["id"], source_hash=stale_hash, covered_to=1)

    store.commit_compaction(first["id"], expected_source_hash=first_hash)

    with pytest.raises(ValueError, match="parent checkpoint is stale"):
        store.commit_compaction(stale["id"], expected_source_hash=stale_hash)
    assert store.load_snapshot("run-stale")["compacted_message_to"] == 0


def test_stale_thread_compaction_cannot_replace_newer_checkpoint(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Stale thread", workspace=tmp_path)
    for index in range(3):
        store.append_conversation_message(
            thread_id, role="user", content=f"message-{index}"
        )
    messages = store.list_conversation_messages(thread_id, limit=None)
    first_hash = canonical_message_hash(messages[:1])
    stale_hash = canonical_message_hash(messages[:2])
    common = {
        "scope": "thread",
        "thread_id": thread_id,
        "source_version": 0,
        "model": "deepseek-chat",
        "tokenizer_version": "deepseek-v3-c954ca6f",
        "analyzer_prompt_version": "analyzer-v1",
        "summarizer_prompt_version": "summarizer-v1",
    }
    first = store.create_compaction(
        **common, covered_from=1, covered_to=1, source_hash=first_hash
    )
    stale = store.create_compaction(
        **common, covered_from=1, covered_to=2, source_hash=stale_hash
    )
    _prepare_validating(store, first["id"], source_hash=first_hash, covered_to=1)
    _prepare_validating(store, stale["id"], source_hash=stale_hash, covered_to=2)

    store.commit_compaction(first["id"], expected_source_hash=first_hash)

    with pytest.raises(ValueError, match="parent checkpoint is stale"):
        store.commit_compaction(stale["id"], expected_source_hash=stale_hash)
    assert store.get_thread(thread_id)["checkpoint_covered_sequence"] == 1


def test_compaction_commit_rejects_source_hash_change(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Hash guard", workspace=tmp_path)
    store.create_run("run-1", str(tmp_path), "work", {}, thread_id)
    store.save_snapshot(
        "run-1",
        {"status": "running", "messages": []},
        expected_version=0,
    )
    compaction = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-1",
        source_hash="frozen",
        covered_to=2,
    )
    _prepare_validating(
        store,
        compaction["id"],
        source_hash="frozen",
        covered_to=2,
    )

    with pytest.raises(ValueError, match="source hash"):
        store.commit_compaction(
            compaction["id"], expected_source_hash="mutated"
        )

    assert store.get_compaction(compaction["id"])["status"] == "validating"


def test_compactions_follow_run_and_thread_cascades(tmp_path: Path):
    store = EventStore(tmp_path / "agent.db")
    thread_id = store.create_thread("Cascade", workspace=tmp_path)
    store.create_run("run-1", str(tmp_path), "work", {}, thread_id)
    run_compaction = _create_run_compaction(
        store,
        thread_id=thread_id,
        run_id="run-1",
        source_hash="run-hash",
        covered_to=1,
    )
    thread_compaction = store.create_compaction(
        scope="thread",
        thread_id=thread_id,
        covered_from=1,
        covered_to=2,
        source_version=0,
        source_hash="thread-hash",
        model="deepseek-chat",
        tokenizer_version="deepseek-v3-c954ca6f",
        analyzer_prompt_version="analyzer-v1",
        summarizer_prompt_version="summarizer-v1",
    )

    with store._connect() as connection:
        connection.execute("DELETE FROM runs WHERE id='run-1'")
        connection.commit()

    with pytest.raises(KeyError):
        store.get_compaction(run_compaction["id"])
    assert store.get_compaction(thread_compaction["id"])["scope"] == "thread"

    store.delete_thread(thread_id)
    with pytest.raises(KeyError):
        store.get_compaction(thread_compaction["id"])


def test_v2_database_migrates_without_losing_runtime_data(tmp_path: Path):
    database = tmp_path / "agent.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            workspace TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            runtime_mode TEXT NOT NULL DEFAULT 'agent',
            budget_json TEXT NOT NULL DEFAULT '{}',
            authorized_paths_json TEXT NOT NULL DEFAULT '[]',
            context_items_json TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            last_run_id TEXT,
            version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            workspace TEXT NOT NULL,
            goal TEXT NOT NULL,
            status TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE conversation_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            run_id TEXT,
            sequence INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'message',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(thread_id, sequence)
        );
        INSERT INTO threads(id, title, created_at, updated_at)
        VALUES('thread-v2', 'Existing thread', 'now', 'now');
        INSERT INTO runs(
            id, thread_id, workspace, goal, status, budget_json,
            version, created_at, updated_at
        ) VALUES('run-v2', 'thread-v2', '.', 'Existing goal', 'completed', '{}', 0, 'now', 'now');
        INSERT INTO conversation_messages(
            id, thread_id, run_id, sequence, role, content, created_at
        ) VALUES('message-v2', 'thread-v2', 'run-v2', 1, 'user', 'Existing message', 'now');
        PRAGMA user_version=2;
        """
    )
    connection.commit()
    connection.close()

    store = EventStore(database)

    assert store.get_thread("thread-v2")["title"] == "Existing thread"
    assert store.get_run("run-v2")["goal"] == "Existing goal"
    assert store.list_conversation_messages("thread-v2")[0]["content"] == (
        "Existing message"
    )
    with store._connect() as migrated:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 6
