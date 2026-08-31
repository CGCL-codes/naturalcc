from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .contracts import AgentEvent, now_iso
from .sqlite_migrations import backup_before_schema_v6


class VersionConflict(RuntimeError):
    pass


class ThreadActiveConflict(RuntimeError):
    pass


TERMINAL_RUN_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "budget_exhausted",
}


def _canonical_message_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EventStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        backup_before_schema_v6(self.db_path)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    runtime_mode TEXT NOT NULL DEFAULT 'agent',
                    budget_json TEXT NOT NULL DEFAULT '{}',
                    authorized_paths_json TEXT NOT NULL DEFAULT '[]',
                    context_items_json TEXT NOT NULL DEFAULT '[]',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    last_run_id TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
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
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    idempotency_key TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    UNIQUE (run_id, idempotency_key),
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    run_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS workspace_leases (
                    workspace TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    run_id TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, risk)
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    run_id TEXT,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'message',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(thread_id, sequence),
                    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread
                    ON conversation_messages(thread_id, sequence);
                CREATE TABLE IF NOT EXISTS compactions (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    thread_id TEXT,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    covered_from INTEGER NOT NULL,
                    covered_to INTEGER NOT NULL,
                    source_version INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    parent_compaction_id TEXT,
                    parent_covered_to INTEGER NOT NULL DEFAULT -1,
                    analysis_json TEXT,
                    checkpoint_json TEXT,
                    model TEXT NOT NULL,
                    tokenizer_version TEXT NOT NULL,
                    analyzer_prompt_version TEXT NOT NULL,
                    summarizer_prompt_version TEXT NOT NULL,
                    analyzer_input_tokens INTEGER NOT NULL DEFAULT 0,
                    analyzer_output_tokens INTEGER NOT NULL DEFAULT 0,
                    summarizer_input_tokens INTEGER NOT NULL DEFAULT 0,
                    summarizer_output_tokens INTEGER NOT NULL DEFAULT 0,
                    maintenance_calls INTEGER NOT NULL DEFAULT 0,
                    maintenance_input_tokens INTEGER NOT NULL DEFAULT 0,
                    maintenance_output_tokens INTEGER NOT NULL DEFAULT 0,
                    maintenance_cost_usd REAL NOT NULL DEFAULT 0,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_compactions_run
                    ON compactions(run_id, status, version);
                CREATE INDEX IF NOT EXISTS idx_compactions_thread
                    ON compactions(thread_id, status, version);
                PRAGMA user_version=6;
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
            if "thread_id" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN thread_id TEXT")
            thread_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(threads)").fetchall()
            }
            thread_additions = {
                "workspace": "TEXT NOT NULL DEFAULT ''",
                "model": "TEXT NOT NULL DEFAULT ''",
                "runtime_mode": "TEXT NOT NULL DEFAULT 'agent'",
                "budget_json": "TEXT NOT NULL DEFAULT '{}'",
                "authorized_paths_json": "TEXT NOT NULL DEFAULT '[]'",
                "context_items_json": "TEXT NOT NULL DEFAULT '[]'",
                "capabilities_json": "TEXT NOT NULL DEFAULT '{}'",
                "summary": "TEXT NOT NULL DEFAULT ''",
                "last_run_id": "TEXT",
                "version": "INTEGER NOT NULL DEFAULT 0",
                "active_thread_checkpoint_id": "TEXT",
                "checkpoint_covered_sequence": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in thread_additions.items():
                if name not in thread_columns:
                    connection.execute(f"ALTER TABLE threads ADD COLUMN {name} {definition}")
            compaction_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(compactions)").fetchall()
            }
            compaction_additions = {
                "maintenance_calls": "INTEGER NOT NULL DEFAULT 0",
                "maintenance_input_tokens": "INTEGER NOT NULL DEFAULT 0",
                "maintenance_output_tokens": "INTEGER NOT NULL DEFAULT 0",
                "maintenance_cost_usd": "REAL NOT NULL DEFAULT 0",
                "parent_compaction_id": "TEXT",
                "parent_covered_to": "INTEGER NOT NULL DEFAULT -1",
            }
            for name, definition in compaction_additions.items():
                if name not in compaction_columns:
                    connection.execute(
                        f"ALTER TABLE compactions ADD COLUMN {name} {definition}"
                    )
            connection.commit()

    def create_thread(
        self,
        title: str,
        thread_id: str | None = None,
        *,
        workspace: str | Path | None = None,
        model: str = "",
        runtime_mode: str = "agent",
        budget: dict[str, Any] | None = None,
        authorized_paths: list[str] | None = None,
        context_items: list[dict[str, Any]] | None = None,
        capabilities: dict[str, Any] | None = None,
        summary: str = "",
    ) -> str:
        identifier = thread_id or str(uuid.uuid4())
        timestamp = now_iso()
        normalized_workspace = str(Path(workspace).expanduser().resolve()) if workspace else ""
        normalized_paths = [
            str(Path(path).expanduser().resolve())
            for path in authorized_paths or []
        ]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO threads(
                    id, title, workspace, model, runtime_mode, budget_json,
                    authorized_paths_json, context_items_json, capabilities_json,
                    summary, last_run_id, version, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    identifier,
                    title.strip() or "Untitled task",
                    normalized_workspace,
                    model.strip(),
                    runtime_mode,
                    json.dumps(budget or {}),
                    json.dumps(normalized_paths, ensure_ascii=False),
                    json.dumps(context_items or [], ensure_ascii=False),
                    json.dumps(capabilities or {}, ensure_ascii=False),
                    summary,
                    None,
                    0,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return identifier

    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.*,
                    COUNT(r.id) AS run_count,
                    lr.status AS last_status,
                    (
                        SELECT content FROM conversation_messages cm
                        WHERE cm.thread_id=t.id
                        ORDER BY cm.sequence DESC LIMIT 1
                    ) AS preview
                FROM threads t LEFT JOIN runs r ON r.thread_id=t.id
                LEFT JOIN runs lr ON lr.id=t.last_run_id
                GROUP BY t.id ORDER BY t.updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._decode_thread_row(row) for row in rows]

    def _decode_thread_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["budget"] = json.loads(result.pop("budget_json") or "{}")
        result["authorized_paths"] = json.loads(result.pop("authorized_paths_json") or "[]")
        result["context_items"] = json.loads(result.pop("context_items_json") or "[]")
        result["capabilities"] = json.loads(result.pop("capabilities_json") or "{}")
        return result

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT t.*, lr.status AS last_status
                FROM threads t LEFT JOIN runs lr ON lr.id=t.last_run_id
                WHERE t.id=?
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return self._decode_thread_row(row)

    def delete_thread(self, thread_id: str) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            thread = connection.execute(
                "SELECT 1 FROM threads WHERE id=?",
                (thread_id,),
            ).fetchone()
            if thread is None:
                connection.rollback()
                raise KeyError(f"unknown thread: {thread_id}")

            terminal_placeholders = ", ".join("?" for _ in TERMINAL_RUN_STATUSES)
            active_run = connection.execute(
                f"""
                SELECT id, status FROM runs
                WHERE thread_id=? AND status NOT IN ({terminal_placeholders})
                ORDER BY created_at LIMIT 1
                """,
                (thread_id, *sorted(TERMINAL_RUN_STATUSES)),
            ).fetchone()
            if active_run is not None:
                connection.rollback()
                raise ThreadActiveConflict(
                    "Cancel the active run before deleting this conversation "
                    f"(run {active_run['id']} is {active_run['status']})."
                )

            run_rows = connection.execute(
                "SELECT id FROM runs WHERE thread_id=?",
                (thread_id,),
            ).fetchall()
            run_ids = [row["id"] for row in run_rows]
            if run_ids:
                run_placeholders = ", ".join("?" for _ in run_ids)
                connection.execute(
                    f"DELETE FROM workspace_leases WHERE run_id IN ({run_placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM approvals WHERE run_id IN ({run_placeholders})",
                    run_ids,
                )
                connection.execute(
                    f"DELETE FROM runs WHERE id IN ({run_placeholders})",
                    run_ids,
                )
            connection.execute("DELETE FROM threads WHERE id=?", (thread_id,))
            connection.commit()
        return thread_id

    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        workspace: str | Path | None = None,
        model: str | None = None,
        runtime_mode: str | None = None,
        budget: dict[str, Any] | None = None,
        authorized_paths: list[str] | None = None,
        context_items: list[dict[str, Any]] | None = None,
        capabilities: dict[str, Any] | None = None,
        summary: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if title is not None:
            updates["title"] = title.strip() or "Untitled task"
        if workspace is not None:
            updates["workspace"] = str(Path(workspace).expanduser().resolve())
        if model is not None:
            updates["model"] = model.strip()
        if runtime_mode is not None:
            updates["runtime_mode"] = runtime_mode
        if budget is not None:
            updates["budget_json"] = json.dumps(budget)
        if authorized_paths is not None:
            normalized = [str(Path(path).expanduser().resolve()) for path in authorized_paths]
            updates["authorized_paths_json"] = json.dumps(normalized, ensure_ascii=False)
        if context_items is not None:
            updates["context_items_json"] = json.dumps(context_items, ensure_ascii=False)
        if capabilities is not None:
            updates["capabilities_json"] = json.dumps(capabilities, ensure_ascii=False)
        if summary is not None:
            updates["summary"] = summary
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT version FROM threads WHERE id=?", (thread_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown thread: {thread_id}")
            version = int(row["version"])
            if expected_version is not None and expected_version != version:
                connection.rollback()
                raise VersionConflict(f"expected thread version {expected_version}, found {version}")
            updates["version"] = version + 1
            updates["updated_at"] = now_iso()
            assignments = ", ".join(f"{key}=?" for key in updates)
            connection.execute(
                f"UPDATE threads SET {assignments} WHERE id=?",
                [*updates.values(), thread_id],
            )
            connection.commit()
        return self.get_thread(thread_id)

    def append_conversation_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        run_id: str | None = None,
        kind: str = "message",
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported conversation role: {role}")
        identifier = message_id or str(uuid.uuid4())
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone() is None:
                connection.rollback()
                raise KeyError(f"unknown thread: {thread_id}")
            if run_id is not None:
                run = connection.execute(
                    "SELECT thread_id FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if run is None or run["thread_id"] != thread_id:
                    connection.rollback()
                    raise ValueError("conversation run does not belong to thread")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM conversation_messages WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            sequence = int(row["sequence"]) + 1
            connection.execute(
                """
                INSERT INTO conversation_messages(
                    id, thread_id, run_id, sequence, role, content, kind,
                    metadata_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    identifier,
                    thread_id,
                    run_id,
                    sequence,
                    role,
                    content,
                    kind,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at=? WHERE id=?",
                (timestamp, thread_id),
            )
            connection.commit()
        return {
            "id": identifier,
            "thread_id": thread_id,
            "run_id": run_id,
            "sequence": sequence,
            "role": role,
            "content": content,
            "kind": kind,
            "metadata": metadata or {},
            "created_at": timestamp,
        }

    def list_conversation_messages(
        self,
        thread_id: str,
        *,
        limit: int | None = 500,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone() is None:
                raise KeyError(f"unknown thread: {thread_id}")
            query = """
                SELECT * FROM conversation_messages
                WHERE thread_id=? AND sequence>?
                    AND (? IS NULL OR sequence<=?)
                ORDER BY sequence DESC
            """
            parameters: list[Any] = [
                thread_id,
                after_sequence,
                through_sequence,
                through_sequence,
            ]
            if limit is not None:
                if limit <= 0:
                    raise ValueError("conversation message limit must be positive")
                query += " LIMIT ?"
                parameters.append(limit)
            rows = connection.execute(query, parameters).fetchall()
        messages = []
        for row in reversed(rows):
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            messages.append(item)
        return messages

    def migrate_legacy_runs(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE thread_id IS NULL ORDER BY created_at"
            ).fetchall()
        migrated = 0
        for row in rows:
            run_id = row["id"]
            thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"code-agent:legacy-run:{run_id}"))
            budget = json.loads(row["budget_json"])
            try:
                self.create_thread(
                    row["goal"][:200],
                    thread_id,
                    workspace=row["workspace"],
                    runtime_mode="agent",
                    budget=budget,
                )
            except sqlite3.IntegrityError:
                pass
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT thread_id FROM runs WHERE id=?",
                    (run_id,),
                ).fetchone()
                if current is None or current["thread_id"] is not None:
                    connection.rollback()
                    continue
                connection.execute(
                    "UPDATE runs SET thread_id=? WHERE id=?",
                    (thread_id, run_id),
                )
                connection.execute(
                    "UPDATE threads SET last_run_id=?, updated_at=? WHERE id=?",
                    (run_id, row["updated_at"], thread_id),
                )
                connection.commit()
            snapshot = self.load_snapshot(run_id) or {}
            self.append_conversation_message(
                thread_id,
                role="user",
                content=row["goal"],
                run_id=run_id,
                kind="legacy",
            )
            final_answer = str(snapshot.get("final_answer") or "").strip()
            if not final_answer:
                for message in reversed(snapshot.get("messages", [])):
                    if message.get("role") == "assistant" and str(message.get("content", "")).strip():
                        final_answer = str(message["content"]).strip()
                        break
            if final_answer:
                self.append_conversation_message(
                    thread_id,
                    role="assistant",
                    content=final_answer,
                    run_id=run_id,
                    kind="legacy",
                    metadata={"status": row["status"]},
                )
            with self._connect() as connection:
                connection.execute(
                    "UPDATE threads SET updated_at=? WHERE id=?",
                    (row["updated_at"], thread_id),
                )
                connection.commit()
            migrated += 1
        return migrated

    def create_run(
        self,
        run_id: str,
        workspace: str,
        goal: str,
        budget: dict[str, Any],
        thread_id: str | None = None,
    ) -> None:
        timestamp = now_iso()
        with self._connect() as connection:
            if thread_id and connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone() is None:
                raise KeyError(f"unknown thread: {thread_id}")
            connection.execute(
                "INSERT INTO runs(id, thread_id, workspace, goal, status, budget_json, version, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, thread_id, str(Path(workspace).resolve()), goal, "queued", json.dumps(budget), 0, timestamp, timestamp),
            )
            if thread_id:
                connection.execute(
                    "UPDATE threads SET last_run_id=?, updated_at=? WHERE id=?",
                    (run_id, timestamp, thread_id),
                )
            connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        result = dict(row)
        result["budget"] = json.loads(result.pop("budget_json"))
        return result

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [
            {
                "id": row["id"],
                "thread_id": row["thread_id"],
                "workspace": row["workspace"],
                "goal": row["goal"],
                "status": row["status"],
                "version": row["version"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def update_run_budget(self, run_id: str, budget: dict[str, Any]) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET budget_json=?, updated_at=? WHERE id=?",
                (json.dumps(budget), now_iso(), run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run: {run_id}")
            connection.commit()

    def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
        status: str | None = None,
    ) -> AgentEvent:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM events WHERE run_id=? AND idempotency_key=?",
                    (run_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    connection.rollback()
                    return self._row_to_event(existing)
            row = connection.execute("SELECT version, status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown run: {run_id}")
            version = int(row["version"])
            if expected_version is not None and version != expected_version:
                connection.rollback()
                raise VersionConflict(f"expected version {expected_version}, found {version}")
            sequence = version + 1
            timestamp = now_iso()
            connection.execute(
                "INSERT INTO events(run_id, sequence, type, payload_json, idempotency_key, created_at) VALUES(?,?,?,?,?,?)",
                (run_id, sequence, event_type, json.dumps(payload, ensure_ascii=False), idempotency_key, timestamp),
            )
            connection.execute(
                "UPDATE runs SET version=?, status=?, updated_at=? WHERE id=?",
                (sequence, status or row["status"], timestamp, run_id),
            )
            connection.commit()
        return AgentEvent(run_id, sequence, event_type, payload, timestamp, idempotency_key)

    def _row_to_event(self, row: sqlite3.Row) -> AgentEvent:
        return AgentEvent(
            row["run_id"],
            int(row["sequence"]),
            row["type"],
            json.loads(row["payload_json"]),
            row["created_at"],
            row["idempotency_key"],
        )

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[AgentEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? AND sequence>? ORDER BY sequence",
                (run_id, after_sequence),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def save_snapshot(self, run_id: str, state: dict[str, Any], *, expected_version: int) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT version FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or int(row["version"]) != expected_version:
                connection.rollback()
                found = None if row is None else int(row["version"])
                raise VersionConflict(f"expected version {expected_version}, found {found}")
            connection.execute(
                """
                INSERT INTO snapshots(run_id, version, state_json, updated_at) VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET version=excluded.version, state_json=excluded.state_json, updated_at=excluded.updated_at
                """,
                (run_id, expected_version, json.dumps(state, ensure_ascii=False), now_iso()),
            )
            connection.commit()

    def load_snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT state_json FROM snapshots WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def acquire_workspace_lease(self, workspace: str, run_id: str) -> bool:
        workspace = str(Path(workspace).resolve())
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO workspace_leases(workspace, run_id, acquired_at) VALUES(?,?,?)",
                    (workspace, run_id, now_iso()),
                )
                connection.commit()
                return True
            except sqlite3.IntegrityError:
                row = connection.execute("SELECT run_id FROM workspace_leases WHERE workspace=?", (workspace,)).fetchone()
                return bool(row and row["run_id"] == run_id)

    def release_workspace_lease(self, workspace: str, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM workspace_leases WHERE workspace=? AND run_id=?",
                (str(Path(workspace).resolve()), run_id),
            )
            connection.commit()

    def grant_approval(self, run_id: str, risk: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO approvals(run_id, risk, granted_at) VALUES(?,?,?)",
                (run_id, risk, now_iso()),
            )
            connection.commit()

    def approvals_for(self, run_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT risk FROM approvals WHERE run_id=?", (run_id,)).fetchall()
        return {row["risk"] for row in rows}

    def create_compaction(
        self,
        *,
        scope: str,
        covered_from: int,
        covered_to: int,
        source_version: int,
        source_hash: str,
        model: str,
        tokenizer_version: str,
        analyzer_prompt_version: str,
        summarizer_prompt_version: str,
        thread_id: str | None = None,
        run_id: str | None = None,
        compaction_id: str | None = None,
    ) -> dict[str, Any]:
        if scope not in {"run", "thread"}:
            raise ValueError("compaction scope must be run or thread")
        if covered_from < 0 or covered_to < covered_from:
            raise ValueError("compaction covered range is invalid")
        identifier = compaction_id or str(uuid.uuid4())
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if scope == "run":
                if not run_id:
                    connection.rollback()
                    raise ValueError("run compaction requires run_id")
                run = connection.execute(
                    "SELECT thread_id FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if run is None:
                    connection.rollback()
                    raise KeyError(f"unknown run: {run_id}")
                stored_thread_id = run["thread_id"]
                if thread_id is not None and thread_id != stored_thread_id:
                    connection.rollback()
                    raise ValueError("compaction run does not belong to thread")
                thread_id = thread_id or stored_thread_id
                scope_column = "run_id"
                scope_value = run_id
                snapshot = connection.execute(
                    "SELECT state_json FROM snapshots WHERE run_id=?", (run_id,)
                ).fetchone()
                snapshot_state = json.loads(snapshot["state_json"]) if snapshot else {}
                parent_compaction_id = snapshot_state.get(
                    "active_run_checkpoint_id"
                )
                parent_covered_to = int(
                    snapshot_state.get("compacted_message_to", -1)
                )
            else:
                if not thread_id or run_id is not None:
                    connection.rollback()
                    raise ValueError(
                        "thread compaction requires thread_id and no run_id"
                    )
                if connection.execute(
                    "SELECT 1 FROM threads WHERE id=?", (thread_id,)
                ).fetchone() is None:
                    connection.rollback()
                    raise KeyError(f"unknown thread: {thread_id}")
                scope_column = "thread_id"
                scope_value = thread_id
                thread_state = connection.execute(
                    """
                    SELECT active_thread_checkpoint_id, checkpoint_covered_sequence
                    FROM threads WHERE id=?
                    """,
                    (thread_id,),
                ).fetchone()
                parent_compaction_id = thread_state[
                    "active_thread_checkpoint_id"
                ]
                parent_covered_to = int(
                    thread_state["checkpoint_covered_sequence"]
                )
            row = connection.execute(
                f"SELECT COALESCE(MAX(version), 0) AS version FROM compactions "
                f"WHERE scope=? AND {scope_column}=?",
                (scope, scope_value),
            ).fetchone()
            version = int(row["version"]) + 1
            connection.execute(
                """
                INSERT INTO compactions(
                    id, scope, thread_id, run_id, status, version,
                    covered_from, covered_to, source_version, source_hash,
                    parent_compaction_id, parent_covered_to,
                    analysis_json, checkpoint_json, model, tokenizer_version,
                    analyzer_prompt_version, summarizer_prompt_version,
                    created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    identifier,
                    scope,
                    thread_id,
                    run_id,
                    "analyzing",
                    version,
                    covered_from,
                    covered_to,
                    source_version,
                    source_hash,
                    parent_compaction_id,
                    parent_covered_to,
                    None,
                    None,
                    model,
                    tokenizer_version,
                    analyzer_prompt_version,
                    summarizer_prompt_version,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return self.get_compaction(identifier)

    def _decode_compaction_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for column, decoded_name in (
            ("analysis_json", "analysis"),
            ("checkpoint_json", "checkpoint"),
            ("error_json", "error"),
        ):
            value = result.pop(column)
            result[decoded_name] = json.loads(value) if value else None
        return result

    def get_compaction(self, compaction_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM compactions WHERE id=?", (compaction_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown compaction: {compaction_id}")
        return self._decode_compaction_row(row)

    def reserve_compaction_call(
        self,
        compaction_id: str,
        *,
        max_maintenance_calls: int | None,
    ) -> bool:
        """Durably reserve one maintenance call before contacting the provider."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT maintenance_calls FROM compactions WHERE id=?",
                (compaction_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown compaction: {compaction_id}")
            if (
                max_maintenance_calls is not None
                and int(row["maintenance_calls"]) >= max_maintenance_calls
            ):
                connection.rollback()
                return False
            connection.execute(
                """
                UPDATE compactions
                SET maintenance_calls=maintenance_calls+1, updated_at=?
                WHERE id=?
                """,
                (now_iso(), compaction_id),
            )
            connection.commit()
        return True

    def record_compaction_response_usage(
        self,
        compaction_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> dict[str, Any]:
        if input_tokens < 0 or output_tokens < 0 or cost_usd < 0:
            raise ValueError("compaction response usage cannot be negative")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE compactions
                SET maintenance_input_tokens=maintenance_input_tokens+?,
                    maintenance_output_tokens=maintenance_output_tokens+?,
                    maintenance_cost_usd=maintenance_cost_usd+?,
                    updated_at=?
                WHERE id=?
                """,
                (input_tokens, output_tokens, cost_usd, now_iso(), compaction_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(f"unknown compaction: {compaction_id}")
            connection.commit()
        return self.get_compaction(compaction_id)

    def update_compaction(
        self,
        compaction_id: str,
        *,
        status: str,
        analysis: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        analyzer_input_tokens: int | None = None,
        analyzer_output_tokens: int | None = None,
        summarizer_input_tokens: int | None = None,
        summarizer_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        transitions = {
            "analyzing": {"analyzing", "summarizing", "failed"},
            "summarizing": {"summarizing", "validating", "failed"},
            "validating": {"validating", "failed"},
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM compactions WHERE id=?", (compaction_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown compaction: {compaction_id}")
            current_status = row["status"]
            if status not in transitions.get(current_status, set()):
                connection.rollback()
                raise ValueError(
                    f"invalid compaction transition: {current_status} -> {status}"
                )
            updates: dict[str, Any] = {
                "status": status,
                "updated_at": now_iso(),
            }
            if analysis is not None:
                updates["analysis_json"] = json.dumps(analysis, ensure_ascii=False)
            if checkpoint is not None:
                updates["checkpoint_json"] = json.dumps(
                    checkpoint, ensure_ascii=False
                )
            if error is not None:
                updates["error_json"] = json.dumps(error, ensure_ascii=False)
            for name, value in (
                ("analyzer_input_tokens", analyzer_input_tokens),
                ("analyzer_output_tokens", analyzer_output_tokens),
                ("summarizer_input_tokens", summarizer_input_tokens),
                ("summarizer_output_tokens", summarizer_output_tokens),
            ):
                if value is not None:
                    if value < 0:
                        connection.rollback()
                        raise ValueError(f"{name} cannot be negative")
                    updates[name] = value
            assignments = ", ".join(f"{name}=?" for name in updates)
            connection.execute(
                f"UPDATE compactions SET {assignments} WHERE id=?",
                [*updates.values(), compaction_id],
            )
            connection.commit()
        return self.get_compaction(compaction_id)

    def latest_committed_compaction(
        self,
        *,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any] | None:
        if (run_id is None) == (thread_id is None):
            raise ValueError("provide exactly one of run_id or thread_id")
        column = "run_id" if run_id is not None else "thread_id"
        value = run_id if run_id is not None else thread_id
        scope = "run" if run_id is not None else "thread"
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM compactions
                WHERE scope=? AND {column}=? AND status='committed'
                ORDER BY version DESC LIMIT 1
                """,
                (scope, value),
            ).fetchone()
        return self._decode_compaction_row(row) if row is not None else None

    def latest_incomplete_compaction(self, run_id: str) -> dict[str, Any] | None:
        """Return the newest resumable run compaction, if one exists."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM compactions
                WHERE scope='run' AND run_id=?
                  AND status IN ('analyzing', 'summarizing', 'validating')
                ORDER BY version DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return self._decode_compaction_row(row) if row is not None else None

    def latest_incomplete_thread_compaction(
        self, thread_id: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM compactions
                WHERE scope='thread' AND thread_id=?
                  AND status IN ('analyzing', 'summarizing', 'validating')
                ORDER BY version DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._decode_compaction_row(row) if row is not None else None

    def commit_compaction(
        self,
        compaction_id: str,
        *,
        expected_source_hash: str,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM compactions WHERE id=?", (compaction_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown compaction: {compaction_id}")
            if row["status"] != "validating":
                connection.rollback()
                raise ValueError("only a validating compaction can be committed")
            if row["source_hash"] != expected_source_hash:
                connection.rollback()
                raise ValueError("compaction source hash changed before commit")
            if not row["checkpoint_json"]:
                connection.rollback()
                raise ValueError("compaction has no validated checkpoint")

            expected_from = int(row["parent_covered_to"]) + 1
            if int(row["covered_from"]) != expected_from:
                connection.rollback()
                raise ValueError("compaction coverage is not contiguous with its parent")

            if row["scope"] == "run":
                snapshot = connection.execute(
                    "SELECT state_json FROM snapshots WHERE run_id=?",
                    (row["run_id"],),
                ).fetchone()
                if snapshot is None:
                    connection.rollback()
                    raise ValueError("run compaction requires a persisted snapshot")
                state = json.loads(snapshot["state_json"])
                active_id = state.get("active_run_checkpoint_id")
                active_watermark = int(state.get("compacted_message_to", -1))
                all_messages = state.get("messages", [])
                authoritative_messages = all_messages[
                    int(row["covered_from"]) : int(row["covered_to"]) + 1
                ]
            else:
                thread = connection.execute(
                    """
                    SELECT active_thread_checkpoint_id, checkpoint_covered_sequence
                    FROM threads WHERE id=?
                    """,
                    (row["thread_id"],),
                ).fetchone()
                if thread is None:
                    connection.rollback()
                    raise ValueError("thread compaction requires a persisted thread")
                active_id = thread["active_thread_checkpoint_id"]
                active_watermark = int(thread["checkpoint_covered_sequence"])
                message_rows = connection.execute(
                    """
                    SELECT * FROM conversation_messages
                    WHERE thread_id=? AND sequence>=? AND sequence<=?
                    ORDER BY sequence
                    """,
                    (
                        row["thread_id"],
                        int(row["covered_from"]),
                        int(row["covered_to"]),
                    ),
                ).fetchall()
                authoritative_messages = []
                for message_row in message_rows:
                    item = dict(message_row)
                    item["metadata"] = json.loads(
                        item.pop("metadata_json") or "{}"
                    )
                    authoritative_messages.append(item)

            if (
                active_id != row["parent_compaction_id"]
                or active_watermark != int(row["parent_covered_to"])
            ):
                connection.rollback()
                raise ValueError("compaction parent checkpoint is stale")
            expected_length = int(row["covered_to"]) - int(row["covered_from"]) + 1
            authoritative_hash = _canonical_message_hash(authoritative_messages)
            if (
                len(authoritative_messages) != expected_length
                or authoritative_hash != row["source_hash"]
                or authoritative_hash != expected_source_hash
            ):
                connection.rollback()
                raise ValueError("compaction source changed before atomic commit")

            timestamp = now_iso()
            if row["scope"] == "run":
                scope_column = "run_id"
                scope_value = row["run_id"]
            else:
                scope_column = "thread_id"
                scope_value = row["thread_id"]
            connection.execute(
                f"""
                UPDATE compactions SET status='superseded', updated_at=?
                WHERE scope=? AND {scope_column}=? AND status='committed' AND id<>?
                """,
                (timestamp, row["scope"], scope_value, compaction_id),
            )
            connection.execute(
                "UPDATE compactions SET status='committed', updated_at=? WHERE id=?",
                (timestamp, compaction_id),
            )

            if row["scope"] == "run":
                state["active_run_checkpoint_id"] = compaction_id
                state["compacted_message_to"] = int(row["covered_to"])
                connection.execute(
                    "UPDATE snapshots SET state_json=?, updated_at=? WHERE run_id=?",
                    (json.dumps(state, ensure_ascii=False), timestamp, row["run_id"]),
                )
            else:
                connection.execute(
                    """
                    UPDATE threads
                    SET active_thread_checkpoint_id=?, checkpoint_covered_sequence=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (
                        compaction_id,
                        int(row["covered_to"]),
                        timestamp,
                        row["thread_id"],
                    ),
                )
            connection.commit()
        return self.get_compaction(compaction_id)
