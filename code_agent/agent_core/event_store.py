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


# [maint-note:a073e66dd315]
# 数据与行为导航：VersionConflict。
# 继承接口：RuntimeError。
# [/maint-note:a073e66dd315]
class VersionConflict(RuntimeError):
    pass


# [maint-note:6544ec43edbb]
# 数据与行为导航：ThreadActiveConflict。
# 继承接口：RuntimeError。
# [/maint-note:6544ec43edbb]
class ThreadActiveConflict(RuntimeError):
    pass


TERMINAL_RUN_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "budget_exhausted",
}


# [maint-note:89e3b2c09b19]
# 维护导航：_canonical_message_hash；下列为本函数源码中的接口与路径索引。
# 输入 messages：list[dict[str, Any]]。
# 返回契约：str。
# 返回路径：hashlib.sha256(payload).hexdigest()。
# 直接调用：json.dumps(messages, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode。
# 直接调用：json.dumps。
# 直接调用：hashlib.sha256(payload).hexdigest。
# 直接调用：hashlib.sha256。
# [/maint-note:89e3b2c09b19]
def _canonical_message_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EventStore:
    # [maint-note:16dd1bf4ac0e]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 db_path：str | Path。
    # 返回契约：None。
    # 写入目标：self.db_path。
    # 直接调用：Path(db_path).expanduser().resolve。
    # 直接调用：Path(db_path).expanduser。
    # 直接调用：Path。
    # 直接调用：self.db_path.parent.mkdir。
    # 直接调用：backup_before_schema_v6。
    # 直接调用：self._migrate。
    # [/maint-note:16dd1bf4ac0e]
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        backup_before_schema_v6(self.db_path)
        self._migrate()

    # [maint-note:11de1c99446f]
    # 维护导航：_connect；下列为本函数源码中的接口与路径索引。
    # 返回契约：Iterator[sqlite3.Connection]。
    # 写入目标：connection.row_factory。
    # 直接调用：sqlite3.connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.close。
    # 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
    # 生成器交付：connection。
    # [/maint-note:11de1c99446f]
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

    # [maint-note:08bef544604b]
    # 维护导航：_migrate；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：'thread_id' not in columns。
    # 分支条件：name not in thread_columns。
    # 分支条件：'runtime_model_config_json' not in run_columns。
    # 分支条件：name not in compaction_columns。
    # 遍历边界：(name, definition) ← thread_additions.items()。
    # 遍历边界：(name, definition) ← compaction_additions.items()。
    # 直接调用：self._connect。
    # 直接调用：connection.executescript。
    # 直接调用：connection.execute('PRAGMA table_info(runs)').fetchall。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('PRAGMA table_info(threads)').fetchall。
    # 直接调用：thread_additions.items。
    # 直接调用：connection.execute('PRAGMA table_info(compactions)').fetchall。
    # 直接调用：compaction_additions.items。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 connection.execute('PRAGMA table_info(runs)').fetchall()。
    # 集合派生：row 来自 connection.execute('PRAGMA table_info(threads)').fetchall()。
    # 集合派生：row 来自 connection.execute('PRAGMA table_info(compactions)').fetchall()。
    # [/maint-note:08bef544604b]
    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    runtime_model_config_json TEXT NOT NULL DEFAULT '{}',
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
                    runtime_model_config_json TEXT NOT NULL DEFAULT '{}',
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
                CREATE TABLE IF NOT EXISTS memory_extractions (
                    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                    worker_pid INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    proposal_count INTEGER NOT NULL DEFAULT 0,
                    applied_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
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
                "runtime_model_config_json": "TEXT NOT NULL DEFAULT '{}'",
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
            run_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "runtime_model_config_json" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN runtime_model_config_json TEXT NOT NULL DEFAULT '{}'"
                )
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

    # [maint-note:e2f8358b3518]
    # 维护导航：create_thread；下列为本函数源码中的接口与路径索引。
    # 输入 title：str。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 输入 workspace：str | Path | None；缺省表达式 None。
    # 输入 model：str；缺省表达式 ''。
    # 输入 runtime_model_config：dict[str, Any] | None；缺省表达式 None。
    # 输入 runtime_mode：str；缺省表达式 'agent'。
    # 输入 budget：dict[str, Any] | None；缺省表达式 None。
    # 输入 authorized_paths：list[str] | None；缺省表达式 None。
    # 输入 context_items：list[dict[str, Any]] | None；缺省表达式 None。
    # 输入 capabilities：dict[str, Any] | None；缺省表达式 None。
    # 输入 summary：str；缺省表达式 ''。
    # 返回契约：str。
    # 返回路径：identifier。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：Path(workspace).expanduser().resolve。
    # 直接调用：Path(workspace).expanduser。
    # 直接调用：Path。
    # 直接调用：Path(path).expanduser().resolve。
    # 直接调用：Path(path).expanduser。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：title.strip。
    # 直接调用：model.strip。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 表达式分流：workspace；真值与假值分支分别计算，不同时执行。
    # 集合派生：path 来自 authorized_paths or []。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:e2f8358b3518]
    def create_thread(
        self,
        title: str,
        thread_id: str | None = None,
        *,
        workspace: str | Path | None = None,
        model: str = "",
        runtime_model_config: dict[str, Any] | None = None,
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
                    id, title, workspace, model, runtime_model_config_json, runtime_mode, budget_json,
                    authorized_paths_json, context_items_json, capabilities_json,
                    summary, last_run_id, version, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    identifier,
                    title.strip() or "Untitled task",
                    normalized_workspace,
                    model.strip(),
                    json.dumps(runtime_model_config or {}, ensure_ascii=False),
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

    # [maint-note:6fc63b056bdf]
    # 维护导航：list_threads；下列为本函数源码中的接口与路径索引。
    # 输入 limit：int；缺省表达式 100。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：[self._decode_thread_row(row) for row in rows]。
    # 直接调用：', '.join。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'\n SELECT\n t.*,\n COUNT(r.id) AS run_count,\n lr.status AS last_status,\n (\n SELECT active.id FROM runs active\n WHERE active.thread_id=t.id\n AND active.status NOT IN ({terminal_placeholders})\n ORDER BY active.updat。
    # 直接调用：connection.execute。
    # 直接调用：sorted。
    # 直接调用：self._decode_thread_row。
    # 集合派生：_ 来自 TERMINAL_RUN_STATUSES。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:6fc63b056bdf]
    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        terminal_placeholders = ", ".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    t.*,
                    COUNT(r.id) AS run_count,
                    lr.status AS last_status,
                    (
                        SELECT active.id FROM runs active
                        WHERE active.thread_id=t.id
                          AND active.status NOT IN ({terminal_placeholders})
                        ORDER BY active.updated_at DESC LIMIT 1
                    ) AS active_run_id,
                    (
                        SELECT active.status FROM runs active
                        WHERE active.thread_id=t.id
                          AND active.status NOT IN ({terminal_placeholders})
                        ORDER BY active.updated_at DESC LIMIT 1
                    ) AS active_status,
                    (
                        SELECT content FROM conversation_messages cm
                        WHERE cm.thread_id=t.id
                        ORDER BY cm.sequence DESC LIMIT 1
                    ) AS preview
                FROM threads t LEFT JOIN runs r ON r.thread_id=t.id
                LEFT JOIN runs lr ON lr.id=t.last_run_id
                GROUP BY t.id ORDER BY t.updated_at DESC LIMIT ?
                """,
                (*sorted(TERMINAL_RUN_STATUSES), *sorted(TERMINAL_RUN_STATUSES), limit),
            ).fetchall()
        return [self._decode_thread_row(row) for row in rows]

    # [maint-note:c0351ed1e9df]
    # 维护导航：_decode_thread_row；下列为本函数源码中的接口与路径索引。
    # 输入 row：sqlite3.Row。
    # 返回契约：dict[str, Any]。
    # 返回路径：result。
    # 写入目标：result['budget']。
    # 写入目标：result['authorized_paths']。
    # 写入目标：result['context_items']。
    # 写入目标：result['capabilities']。
    # 写入目标：result['runtime_model_config']。
    # 直接调用：dict。
    # 直接调用：json.loads。
    # 直接调用：result.pop。
    # [/maint-note:c0351ed1e9df]
    def _decode_thread_row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["budget"] = json.loads(result.pop("budget_json") or "{}")
        result["authorized_paths"] = json.loads(result.pop("authorized_paths_json") or "[]")
        result["context_items"] = json.loads(result.pop("context_items_json") or "[]")
        result["capabilities"] = json.loads(result.pop("capabilities_json") or "{}")
        result["runtime_model_config"] = json.loads(
            result.pop("runtime_model_config_json") or "{}"
        )
        return result

    # [maint-note:07a1fa47076a]
    # 维护导航：get_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any]。
    # 分支条件：row is None。
    # 返回路径：self._decode_thread_row(row)。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 直接调用：', '.join。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'\n SELECT\n t.*,\n lr.status AS last_status,\n (\n SELECT active.id FROM runs active\n WHERE active.thread_id=t.id\n AND active.status NOT IN ({terminal_placeholders})\n ORDER BY active.updated_at DESC LIMIT 1\n ) AS ac。
    # 直接调用：connection.execute。
    # 直接调用：sorted。
    # 直接调用：KeyError。
    # 直接调用：self._decode_thread_row。
    # 集合派生：_ 来自 TERMINAL_RUN_STATUSES。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:07a1fa47076a]
    def get_thread(self, thread_id: str) -> dict[str, Any]:
        terminal_placeholders = ", ".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    t.*,
                    lr.status AS last_status,
                    (
                        SELECT active.id FROM runs active
                        WHERE active.thread_id=t.id
                          AND active.status NOT IN ({terminal_placeholders})
                        ORDER BY active.updated_at DESC LIMIT 1
                    ) AS active_run_id,
                    (
                        SELECT active.status FROM runs active
                        WHERE active.thread_id=t.id
                          AND active.status NOT IN ({terminal_placeholders})
                        ORDER BY active.updated_at DESC LIMIT 1
                    ) AS active_status
                FROM threads t LEFT JOIN runs lr ON lr.id=t.last_run_id
                WHERE t.id=?
                """,
                (*sorted(TERMINAL_RUN_STATUSES), *sorted(TERMINAL_RUN_STATUSES), thread_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return self._decode_thread_row(row)

    # [maint-note:d26fb8893705]
    # 维护导航：delete_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：str。
    # 分支条件：thread is None。
    # 分支条件：active_run is not None。
    # 分支条件：run_ids。
    # 返回路径：thread_id。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 显式抛出：ThreadActiveConflict(f"Cancel the active run before deleting this conversation (run {active_run['id']} is {active_run['status']}).")。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：', '.join。
    # 直接调用：connection.execute(f'\n SELECT id, status FROM runs\n WHERE thread_id=? AND status NOT IN ({terminal_placeholders})\n ORDER BY created_at LIMIT 1\n ', (thread_id, *sorted(TERMINAL_RUN_STATUSES))).fetchone。
    # 直接调用：sorted。
    # 直接调用：ThreadActiveConflict。
    # 直接调用：connection.execute('SELECT id FROM runs WHERE thread_id=?', (thread_id,)).fetchall。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：_ 来自 TERMINAL_RUN_STATUSES。
    # 集合派生：row 来自 run_rows。
    # 集合派生：_ 来自 run_ids。
    # [/maint-note:d26fb8893705]
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

    # [maint-note:bde6671e566d]
    # 维护导航：update_thread；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 title：str | None；缺省表达式 None。
    # 输入 workspace：str | Path | None；缺省表达式 None。
    # 输入 model：str | None；缺省表达式 None。
    # 输入 runtime_model_config：dict[str, Any] | None；缺省表达式 None。
    # 输入 runtime_mode：str | None；缺省表达式 None。
    # 输入 budget：dict[str, Any] | None；缺省表达式 None。
    # 输入 authorized_paths：list[str] | None；缺省表达式 None。
    # 输入 context_items：list[dict[str, Any]] | None；缺省表达式 None。
    # 输入 capabilities：dict[str, Any] | None；缺省表达式 None。
    # 输入 summary：str | None；缺省表达式 None。
    # 输入 expected_version：int | None；缺省表达式 None。
    # 返回契约：dict[str, Any]。
    # 分支条件：title is not None。
    # 分支条件：workspace is not None。
    # 分支条件：model is not None。
    # 分支条件：runtime_model_config is not None。
    # 分支条件：runtime_mode is not None。
    # 分支条件：budget is not None。
    # 分支条件：authorized_paths is not None。
    # 分支条件：context_items is not None。
    # 分支条件：capabilities is not None。
    # 分支条件：summary is not None。
    # 分支条件：row is None。
    # 分支条件：expected_version is not None and expected_version != version。
    # 返回路径：self.get_thread(thread_id)。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 显式抛出：VersionConflict(f'expected thread version {expected_version}, found {version}')。
    # 写入目标：updates['title']。
    # 写入目标：updates['workspace']。
    # 写入目标：updates['model']。
    # 写入目标：updates['runtime_model_config_json']。
    # 写入目标：updates['runtime_mode']。
    # 写入目标：updates['budget_json']。
    # 写入目标：updates['authorized_paths_json']。
    # 写入目标：updates['context_items_json']。
    # 写入目标：updates['capabilities_json']。
    # 写入目标：updates['summary']。
    # 写入目标：updates['version']。
    # 写入目标：updates['updated_at']。
    # 直接调用：title.strip。
    # 直接调用：str。
    # 直接调用：Path(workspace).expanduser().resolve。
    # 直接调用：Path(workspace).expanduser。
    # 直接调用：Path。
    # 直接调用：model.strip。
    # 直接调用：json.dumps。
    # 直接调用：Path(path).expanduser().resolve。
    # 直接调用：Path(path).expanduser。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT version FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：VersionConflict。
    # 直接调用：now_iso。
    # 直接调用：', '.join。
    # 直接调用：updates.values。
    # 直接调用：connection.commit。
    # 直接调用：self.get_thread。
    # 集合派生：path 来自 authorized_paths。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：key 来自 updates。
    # [/maint-note:bde6671e566d]
    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        workspace: str | Path | None = None,
        model: str | None = None,
        runtime_model_config: dict[str, Any] | None = None,
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
        if runtime_model_config is not None:
            updates["runtime_model_config_json"] = json.dumps(
                runtime_model_config, ensure_ascii=False
            )
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

    # [maint-note:793c1f7617ce]
    # 维护导航：append_conversation_message；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 role：str。
    # 输入 content：str。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 kind：str；缺省表达式 'message'。
    # 输入 metadata：dict[str, Any] | None；缺省表达式 None。
    # 输入 message_id：str | None；缺省表达式 None。
    # 返回契约：dict[str, Any]。
    # 分支条件：role not in {'user', 'assistant', 'system'}。
    # 分支条件：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone() is None。
    # 分支条件：run_id is not None。
    # 分支条件：run is None or run['thread_id'] != thread_id。
    # 返回路径：{'id': identifier, 'thread_id': thread_id, 'run_id': run_id, 'sequence': sequence, 'role': role, 'content': content, 'kind': kind, 'metadata': metadata or {}, 'created_at': timestamp}。
    # 显式抛出：ValueError(f'unsupported conversation role: {role}')。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 显式抛出：ValueError('conversation run does not belong to thread')。
    # 直接调用：ValueError。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：connection.execute('SELECT thread_id FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：connection.execute('SELECT COALESCE(MAX(sequence), 0) AS sequence FROM conversation_messages WHERE thread_id=?', (thread_id,)).fetchone。
    # 直接调用：int。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:793c1f7617ce]
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

    # [maint-note:6e28c3a129e1]
    # 维护导航：list_conversation_messages；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 limit：int | None；缺省表达式 500。
    # 输入 after_sequence：int；缺省表达式 0。
    # 输入 through_sequence：int | None；缺省表达式 None。
    # 输入 run_id：str | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone() is None。
    # 分支条件：run_id is not None。
    # 分支条件：limit is not None。
    # 分支条件：limit <= 0。
    # 遍历边界：row ← reversed(rows)。
    # 返回路径：messages。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 显式抛出：ValueError('conversation message limit must be positive')。
    # 写入目标：item['metadata']。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：parameters.append。
    # 直接调用：ValueError。
    # 直接调用：connection.execute(query, parameters).fetchall。
    # 直接调用：reversed。
    # 直接调用：dict。
    # 直接调用：json.loads。
    # 直接调用：item.pop。
    # 直接调用：messages.append。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:6e28c3a129e1]
    def list_conversation_messages(
        self,
        thread_id: str,
        *,
        limit: int | None = 500,
        after_sequence: int = 0,
        through_sequence: int | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone() is None:
                raise KeyError(f"unknown thread: {thread_id}")
            query = """
                SELECT * FROM conversation_messages
                WHERE thread_id=? AND sequence>?
                    AND (? IS NULL OR sequence<=?)
            """
            parameters: list[Any] = [
                thread_id,
                after_sequence,
                through_sequence,
                through_sequence,
            ]
            if run_id is not None:
                query += " AND run_id=?"
                parameters.append(run_id)
            query += " ORDER BY sequence DESC"
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

    # [maint-note:97a4760985ed]
    # 维护导航：migrate_legacy_runs；下列为本函数源码中的接口与路径索引。
    # 返回契约：int。
    # 分支条件：current is None or current['thread_id'] is not None。
    # 分支条件：not final_answer。
    # 分支条件：message.get('role') == 'assistant' and str(message.get('content', '')).strip()。
    # 分支条件：final_answer。
    # 遍历边界：row ← rows。
    # 遍历边界：message ← reversed(snapshot.get('messages', []))。
    # 返回路径：migrated。
    # 捕获异常：sqlite3.IntegrityError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM runs WHERE thread_id IS NULL ORDER BY created_at').fetchall。
    # 直接调用：connection.execute。
    # 直接调用：str。
    # 直接调用：uuid.uuid5。
    # 直接调用：json.loads。
    # 直接调用：self.create_thread。
    # 直接调用：connection.execute('SELECT thread_id FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：connection.commit。
    # 直接调用：self.load_snapshot。
    # 直接调用：self.append_conversation_message。
    # 直接调用：str(snapshot.get('final_answer') or '').strip。
    # 直接调用：snapshot.get。
    # 直接调用：reversed。
    # 直接调用：message.get。
    # 直接调用：str(message.get('content', '')).strip。
    # 直接调用：str(message['content']).strip。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:97a4760985ed]
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

    # [maint-note:32b4f078b18b]
    # 维护导航：create_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 workspace：str。
    # 输入 goal：str。
    # 输入 budget：dict[str, Any]。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 输入 runtime_model_config：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：None。
    # 分支条件：thread_id and connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone() is None。
    # 分支条件：thread_id。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 直接调用：now_iso。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：str。
    # 直接调用：Path(workspace).resolve。
    # 直接调用：Path。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:32b4f078b18b]
    def create_run(
        self,
        run_id: str,
        workspace: str,
        goal: str,
        budget: dict[str, Any],
        thread_id: str | None = None,
        runtime_model_config: dict[str, Any] | None = None,
    ) -> None:
        timestamp = now_iso()
        with self._connect() as connection:
            if thread_id and connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone() is None:
                raise KeyError(f"unknown thread: {thread_id}")
            connection.execute(
                "INSERT INTO runs(id, thread_id, workspace, goal, status, budget_json, runtime_model_config_json, version, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, thread_id, str(Path(workspace).resolve()), goal, "queued", json.dumps(budget), json.dumps(runtime_model_config or {}, ensure_ascii=False), 0, timestamp, timestamp),
            )
            if thread_id:
                connection.execute(
                    "UPDATE threads SET last_run_id=?, updated_at=? WHERE id=?",
                    (run_id, timestamp, thread_id),
                )
            connection.commit()

    # [maint-note:81569e708752]
    # 维护导航：get_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any]。
    # 分支条件：row is None。
    # 返回路径：result。
    # 显式抛出：KeyError(f'unknown run: {run_id}')。
    # 写入目标：result['budget']。
    # 写入目标：result['runtime_model_config']。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：dict。
    # 直接调用：json.loads。
    # 直接调用：result.pop。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:81569e708752]
    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        result = dict(row)
        result["budget"] = json.loads(result.pop("budget_json"))
        result["runtime_model_config"] = json.loads(
            result.pop("runtime_model_config_json") or "{}"
        )
        return result

    # [maint-note:e1172b13f878]
    # 维护导航：queue_memory_extraction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 worker_pid：int。
    # 返回契约：bool。
    # 返回路径：inserted.rowcount == 1。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:e1172b13f878]
    def queue_memory_extraction(self, run_id: str, worker_pid: int) -> bool:
        with self._connect() as connection:
            inserted = connection.execute(
                """INSERT OR IGNORE INTO memory_extractions(run_id, worker_pid, status, updated_at)
                   SELECT id, ?, 'queued', ? FROM runs WHERE id=? AND status='completed'""",
                (worker_pid, now_iso(), run_id),
            )
            connection.commit()
            return inserted.rowcount == 1

    # [maint-note:4b75a7e2fc05]
    # 维护导航：update_memory_extraction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 status：str。
    # 输入 reason：str | None；缺省表达式 None。
    # 输入 proposal_count：int；缺省表达式 0。
    # 输入 applied_count：int；缺省表达式 0。
    # 返回契约：None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:4b75a7e2fc05]
    def update_memory_extraction(
        self, run_id: str, status: str, *, reason: str | None = None,
        proposal_count: int = 0, applied_count: int = 0,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE memory_extractions
                   SET status=?, reason=?, proposal_count=?, applied_count=?, updated_at=?
                   WHERE run_id=?""",
                (status, reason, proposal_count, applied_count, now_iso(), run_id),
            )
            connection.commit()

    # [maint-note:6deac22e092c]
    # 维护导航：get_memory_extraction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：dict(row) if row else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT worker_pid, status, reason, proposal_count, applied_count, updated_at\n FROM memory_extractions WHERE run_id=?', (run_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row；真值与假值分支分别计算，不同时执行。
    # [/maint-note:6deac22e092c]
    def get_memory_extraction(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT worker_pid, status, reason, proposal_count, applied_count, updated_at
                   FROM memory_extractions WHERE run_id=?""", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    # [maint-note:0c6751a44c9e]
    # 维护导航：list_runs；下列为本函数源码中的接口与路径索引。
    # 输入 limit：int；缺省表达式 100。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：[{'id': row['id'], 'thread_id': row['thread_id'], 'workspace': row['workspace'], 'goal': row['goal'], 'status': row['status'], 'version': row['version'], 'created_at': row['created_at'], 'updated_at': row['updated_at']} for row in rows]。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?', (limit,)).fetchall。
    # 直接调用：connection.execute。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:0c6751a44c9e]
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

    # [maint-note:28fd68b80e17]
    # 维护导航：update_run_budget；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 budget：dict[str, Any]。
    # 返回契约：None。
    # 分支条件：cursor.rowcount == 0。
    # 显式抛出：KeyError(f'unknown run: {run_id}')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：json.dumps。
    # 直接调用：now_iso。
    # 直接调用：KeyError。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:28fd68b80e17]
    def update_run_budget(self, run_id: str, budget: dict[str, Any]) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET budget_json=?, updated_at=? WHERE id=?",
                (json.dumps(budget), now_iso(), run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run: {run_id}")
            connection.commit()

    # [maint-note:7c4351202e8b]
    # 维护导航：append_event；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 event_type：str。
    # 输入 payload：dict[str, Any]。
    # 输入 expected_version：int | None；缺省表达式 None。
    # 输入 idempotency_key：str | None；缺省表达式 None。
    # 输入 status：str | None；缺省表达式 None。
    # 返回契约：AgentEvent。
    # 分支条件：idempotency_key。
    # 分支条件：existing is not None。
    # 分支条件：row is None。
    # 分支条件：expected_version is not None and version != expected_version。
    # 返回路径：self._row_to_event(existing)。
    # 返回路径：AgentEvent(run_id, sequence, event_type, payload, timestamp, idempotency_key)。
    # 显式抛出：KeyError(f'unknown run: {run_id}')。
    # 显式抛出：VersionConflict(f'expected version {expected_version}, found {version}')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT * FROM events WHERE run_id=? AND idempotency_key=?', (run_id, idempotency_key)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：self._row_to_event。
    # 直接调用：connection.execute('SELECT version, status FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：VersionConflict。
    # 直接调用：now_iso。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 直接调用：AgentEvent。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:7c4351202e8b]
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

    # [maint-note:dd9ba511b6d7]
    # 维护导航：_row_to_event；下列为本函数源码中的接口与路径索引。
    # 输入 row：sqlite3.Row。
    # 返回契约：AgentEvent。
    # 返回路径：AgentEvent(row['run_id'], int(row['sequence']), row['type'], json.loads(row['payload_json']), row['created_at'], row['idempotency_key'])。
    # 直接调用：AgentEvent。
    # 直接调用：int。
    # 直接调用：json.loads。
    # [/maint-note:dd9ba511b6d7]
    def _row_to_event(self, row: sqlite3.Row) -> AgentEvent:
        return AgentEvent(
            row["run_id"],
            int(row["sequence"]),
            row["type"],
            json.loads(row["payload_json"]),
            row["created_at"],
            row["idempotency_key"],
        )

    # [maint-note:5e970f43bfb7]
    # 维护导航：list_events；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 after_sequence：int；缺省表达式 0。
    # 输入 event_type：str | None；缺省表达式 None。
    # 返回契约：list[AgentEvent]。
    # 返回路径：[self._row_to_event(row) for row in rows]。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM events WHERE run_id=? AND sequence>?\n AND (? IS NULL OR type=?) ORDER BY sequence', (run_id, after_sequence, event_type, event_type)).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：self._row_to_event。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:5e970f43bfb7]
    def list_events(
        self, run_id: str, after_sequence: int = 0, *, event_type: str | None = None
    ) -> list[AgentEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM events WHERE run_id=? AND sequence>?
                   AND (? IS NULL OR type=?) ORDER BY sequence""",
                (run_id, after_sequence, event_type, event_type),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    # [maint-note:7b7b7447b75f]
    # 维护导航：save_snapshot；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 state：dict[str, Any]。
    # 输入 expected_version：int。
    # 返回契约：None。
    # 分支条件：row is None or int(row['version']) != expected_version。
    # 显式抛出：VersionConflict(f'expected version {expected_version}, found {found}')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT version FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：int。
    # 直接调用：connection.rollback。
    # 直接调用：VersionConflict。
    # 直接调用：json.dumps。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row is None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:7b7b7447b75f]
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

    # [maint-note:412bce9ddf09]
    # 维护导航：load_snapshot；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：json.loads(row['state_json']) if row else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT state_json FROM snapshots WHERE run_id=?', (run_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：json.loads。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row；真值与假值分支分别计算，不同时执行。
    # [/maint-note:412bce9ddf09]
    def load_snapshot(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT state_json FROM snapshots WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    # [maint-note:8b8aeb7b1adc]
    # 维护导航：acquire_workspace_lease；下列为本函数源码中的接口与路径索引。
    # 输入 workspace：str。
    # 输入 run_id：str。
    # 返回契约：bool。
    # 返回路径：True。
    # 返回路径：bool(row and row['run_id'] == run_id)。
    # 捕获异常：sqlite3.IntegrityError。
    # 直接调用：str。
    # 直接调用：Path(workspace).resolve。
    # 直接调用：Path。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 直接调用：connection.execute('SELECT run_id FROM workspace_leases WHERE workspace=?', (workspace,)).fetchone。
    # 直接调用：bool。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:8b8aeb7b1adc]
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

    # [maint-note:8ad287d54695]
    # 维护导航：release_workspace_lease；下列为本函数源码中的接口与路径索引。
    # 输入 workspace：str。
    # 输入 run_id：str。
    # 返回契约：None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：str。
    # 直接调用：Path(workspace).resolve。
    # 直接调用：Path。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:8ad287d54695]
    def release_workspace_lease(self, workspace: str, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM workspace_leases WHERE workspace=? AND run_id=?",
                (str(Path(workspace).resolve()), run_id),
            )
            connection.commit()

    # [maint-note:122ee37c6438]
    # 维护导航：grant_approval；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 risk：str。
    # 返回契约：None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:122ee37c6438]
    def grant_approval(self, run_id: str, risk: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO approvals(run_id, risk, granted_at) VALUES(?,?,?)",
                (run_id, risk, now_iso()),
            )
            connection.commit()

    # [maint-note:25b910ad92e9]
    # 维护导航：approvals_for；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：set[str]。
    # 返回路径：{row['risk'] for row in rows}。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT risk FROM approvals WHERE run_id=?', (run_id,)).fetchall。
    # 直接调用：connection.execute。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:25b910ad92e9]
    def approvals_for(self, run_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT risk FROM approvals WHERE run_id=?", (run_id,)).fetchall()
        return {row["risk"] for row in rows}

    # [maint-note:0b9f0b1b61d2]
    # 维护导航：create_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 scope：str。
    # 输入 covered_from：int。
    # 输入 covered_to：int。
    # 输入 source_version：int。
    # 输入 source_hash：str。
    # 输入 model：str。
    # 输入 tokenizer_version：str。
    # 输入 analyzer_prompt_version：str。
    # 输入 summarizer_prompt_version：str。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 compaction_id：str | None；缺省表达式 None。
    # 返回契约：dict[str, Any]。
    # 分支条件：scope not in {'run', 'thread'}。
    # 分支条件：covered_from < 0 or covered_to < covered_from。
    # 分支条件：scope == 'run'。
    # 分支条件：not run_id。
    # 分支条件：run is None。
    # 分支条件：thread_id is not None and thread_id != stored_thread_id。
    # 分支条件：not thread_id or run_id is not None。
    # 分支条件：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone() is None。
    # 返回路径：self.get_compaction(identifier)。
    # 显式抛出：ValueError('compaction scope must be run or thread')。
    # 显式抛出：ValueError('compaction covered range is invalid')。
    # 显式抛出：ValueError('run compaction requires run_id')。
    # 显式抛出：KeyError(f'unknown run: {run_id}')。
    # 显式抛出：ValueError('compaction run does not belong to thread')。
    # 显式抛出：ValueError('thread compaction requires thread_id and no run_id')。
    # 显式抛出：KeyError(f'unknown thread: {thread_id}')。
    # 直接调用：ValueError。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.rollback。
    # 直接调用：connection.execute('SELECT thread_id FROM runs WHERE id=?', (run_id,)).fetchone。
    # 直接调用：KeyError。
    # 直接调用：connection.execute('SELECT state_json FROM snapshots WHERE run_id=?', (run_id,)).fetchone。
    # 直接调用：json.loads。
    # 直接调用：snapshot_state.get。
    # 直接调用：int。
    # 直接调用：connection.execute('SELECT 1 FROM threads WHERE id=?', (thread_id,)).fetchone。
    # 直接调用：connection.execute('\n SELECT active_thread_checkpoint_id, checkpoint_covered_sequence\n FROM threads WHERE id=?\n ', (thread_id,)).fetchone。
    # 直接调用：connection.execute(f'SELECT COALESCE(MAX(version), 0) AS version FROM compactions WHERE scope=? AND {scope_column}=?', (scope, scope_value)).fetchone。
    # 直接调用：connection.commit。
    # 直接调用：self.get_compaction。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：snapshot；真值与假值分支分别计算，不同时执行。
    # [/maint-note:0b9f0b1b61d2]
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

    # [maint-note:33f870bc1fc1]
    # 维护导航：_decode_compaction_row；下列为本函数源码中的接口与路径索引。
    # 输入 row：sqlite3.Row。
    # 返回契约：dict[str, Any]。
    # 遍历边界：(column, decoded_name) ← (('analysis_json', 'analysis'), ('checkpoint_json', 'checkpoint'), ('error_json', 'error'))。
    # 返回路径：result。
    # 写入目标：result[decoded_name]。
    # 直接调用：dict。
    # 直接调用：result.pop。
    # 直接调用：json.loads。
    # 表达式分流：value；真值与假值分支分别计算，不同时执行。
    # [/maint-note:33f870bc1fc1]
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

    # [maint-note:f77c6a4552c7]
    # 维护导航：get_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 返回契约：dict[str, Any]。
    # 分支条件：row is None。
    # 返回路径：self._decode_compaction_row(row)。
    # 显式抛出：KeyError(f'unknown compaction: {compaction_id}')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM compactions WHERE id=?', (compaction_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：self._decode_compaction_row。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:f77c6a4552c7]
    def get_compaction(self, compaction_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM compactions WHERE id=?", (compaction_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown compaction: {compaction_id}")
        return self._decode_compaction_row(row)

    # [maint-note:825422df3967]
    # 维护导航：reserve_compaction_call；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 max_maintenance_calls：int | None。
    # 返回契约：bool。
    # 分支条件：row is None。
    # 分支条件：max_maintenance_calls is not None and int(row['maintenance_calls']) >= max_maintenance_calls。
    # 返回路径：False。
    # 返回路径：True。
    # 显式抛出：KeyError(f'unknown compaction: {compaction_id}')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT maintenance_calls FROM compactions WHERE id=?', (compaction_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:825422df3967]
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

    # [maint-note:f8e44200fdf7]
    # 维护导航：record_compaction_response_usage；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 input_tokens：int。
    # 输入 output_tokens：int。
    # 输入 cost_usd：float。
    # 返回契约：dict[str, Any]。
    # 分支条件：input_tokens < 0 or output_tokens < 0 or cost_usd < 0。
    # 分支条件：cursor.rowcount != 1。
    # 返回路径：self.get_compaction(compaction_id)。
    # 显式抛出：ValueError('compaction response usage cannot be negative')。
    # 显式抛出：KeyError(f'unknown compaction: {compaction_id}')。
    # 直接调用：ValueError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：connection.commit。
    # 直接调用：self.get_compaction。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:f8e44200fdf7]
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

    # [maint-note:617d8f6491ef]
    # 维护导航：update_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 status：str。
    # 输入 analysis：dict[str, Any] | None；缺省表达式 None。
    # 输入 checkpoint：dict[str, Any] | None；缺省表达式 None。
    # 输入 error：dict[str, Any] | None；缺省表达式 None。
    # 输入 analyzer_input_tokens：int | None；缺省表达式 None。
    # 输入 analyzer_output_tokens：int | None；缺省表达式 None。
    # 输入 summarizer_input_tokens：int | None；缺省表达式 None。
    # 输入 summarizer_output_tokens：int | None；缺省表达式 None。
    # 返回契约：dict[str, Any]。
    # 分支条件：row is None。
    # 分支条件：status not in transitions.get(current_status, set())。
    # 分支条件：analysis is not None。
    # 分支条件：checkpoint is not None。
    # 分支条件：error is not None。
    # 分支条件：value is not None。
    # 分支条件：value < 0。
    # 遍历边界：(name, value) ← (('analyzer_input_tokens', analyzer_input_tokens), ('analyzer_output_tokens', analyzer_output_tokens), ('summarizer_input_tokens', summarizer_input_tokens), ('summarizer_output_tokens', summarizer_output_tokens))。
    # 返回路径：self.get_compaction(compaction_id)。
    # 显式抛出：KeyError(f'unknown compaction: {compaction_id}')。
    # 显式抛出：ValueError(f'invalid compaction transition: {current_status} -> {status}')。
    # 显式抛出：ValueError(f'{name} cannot be negative')。
    # 写入目标：updates['analysis_json']。
    # 写入目标：updates['checkpoint_json']。
    # 写入目标：updates['error_json']。
    # 写入目标：updates[name]。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT status FROM compactions WHERE id=?', (compaction_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：transitions.get。
    # 直接调用：set。
    # 直接调用：ValueError。
    # 直接调用：now_iso。
    # 直接调用：json.dumps。
    # 直接调用：', '.join。
    # 直接调用：updates.values。
    # 直接调用：connection.commit。
    # 直接调用：self.get_compaction。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：name 来自 updates。
    # [/maint-note:617d8f6491ef]
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

    # [maint-note:aaf6c92f1755]
    # 维护导航：latest_committed_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 返回契约：dict[str, Any] | None。
    # 分支条件：(run_id is None) == (thread_id is None)。
    # 返回路径：self._decode_compaction_row(row) if row is not None else None。
    # 显式抛出：ValueError('provide exactly one of run_id or thread_id')。
    # 直接调用：ValueError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f"\n SELECT * FROM compactions\n WHERE scope=? AND {column}=? AND status='committed'\n ORDER BY version DESC LIMIT 1\n ", (scope, value)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：self._decode_compaction_row。
    # 表达式分流：run_id is not None；真值与假值分支分别计算，不同时执行。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:aaf6c92f1755]
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

    # [maint-note:ab6330ba04aa]
    # 维护导航：latest_incomplete_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：self._decode_compaction_row(row) if row is not None else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("\n SELECT * FROM compactions\n WHERE scope='run' AND run_id=?\n AND status IN ('analyzing', 'summarizing', 'validating')\n ORDER BY version DESC LIMIT 1\n ", (run_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：self._decode_compaction_row。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:ab6330ba04aa]
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

    # [maint-note:542c81ea00f2]
    # 维护导航：latest_incomplete_thread_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：self._decode_compaction_row(row) if row is not None else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("\n SELECT * FROM compactions\n WHERE scope='thread' AND thread_id=?\n AND status IN ('analyzing', 'summarizing', 'validating')\n ORDER BY version DESC LIMIT 1\n ", (thread_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：self._decode_compaction_row。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:542c81ea00f2]
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

    # [maint-note:e96f24c403bb]
    # 维护导航：commit_compaction；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 expected_source_hash：str。
    # 返回契约：dict[str, Any]。
    # 分支条件：row is None。
    # 分支条件：row['status'] != 'validating'。
    # 分支条件：row['source_hash'] != expected_source_hash。
    # 分支条件：not row['checkpoint_json']。
    # 分支条件：int(row['covered_from']) != expected_from。
    # 分支条件：row['scope'] == 'run'。
    # 分支条件：snapshot is None。
    # 分支条件：thread is None。
    # 分支条件：active_id != row['parent_compaction_id'] or active_watermark != int(row['parent_covered_to'])。
    # 分支条件：len(authoritative_messages) != expected_length or authoritative_hash != row['source_hash'] or authoritative_hash != expected_source_hash。
    # 遍历边界：message_row ← message_rows。
    # 返回路径：self.get_compaction(compaction_id)。
    # 显式抛出：KeyError(f'unknown compaction: {compaction_id}')。
    # 显式抛出：ValueError('only a validating compaction can be committed')。
    # 显式抛出：ValueError('compaction source hash changed before commit')。
    # 显式抛出：ValueError('compaction has no validated checkpoint')。
    # 显式抛出：ValueError('compaction coverage is not contiguous with its parent')。
    # 显式抛出：ValueError('run compaction requires a persisted snapshot')。
    # 显式抛出：ValueError('thread compaction requires a persisted thread')。
    # 显式抛出：ValueError('compaction parent checkpoint is stale')。
    # 显式抛出：ValueError('compaction source changed before atomic commit')。
    # 写入目标：item['metadata']。
    # 写入目标：state['active_run_checkpoint_id']。
    # 写入目标：state['compacted_message_to']。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT * FROM compactions WHERE id=?', (compaction_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：ValueError。
    # 直接调用：int。
    # 直接调用：connection.execute('SELECT state_json FROM snapshots WHERE run_id=?', (row['run_id'],)).fetchone。
    # 直接调用：json.loads。
    # 直接调用：state.get。
    # 直接调用：connection.execute('\n SELECT active_thread_checkpoint_id, checkpoint_covered_sequence\n FROM threads WHERE id=?\n ', (row['thread_id'],)).fetchone。
    # 直接调用：connection.execute('\n SELECT * FROM conversation_messages\n WHERE thread_id=? AND sequence>=? AND sequence<=?\n ORDER BY sequence\n ', (row['thread_id'], int(row['covered_from']), int(row['covered_to']))).fetchall。
    # 直接调用：dict。
    # 直接调用：item.pop。
    # 直接调用：authoritative_messages.append。
    # 直接调用：_canonical_message_hash。
    # 直接调用：len。
    # 直接调用：now_iso。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 直接调用：self.get_compaction。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:e96f24c403bb]
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
