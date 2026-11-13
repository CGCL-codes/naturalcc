from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .contracts import now_iso
from .memory_tool_recovery import TOOL_RECOVERY_VERSION
from .sqlite_migrations import backup_before_schema_v6


PINNED_MEMORY_KINDS = frozenset(
    {
        "user_preference",
        "project_constraint",
        "architecture_decision",
        "repository_convention",
        # Legacy aliases retained for databases created before governed proposals.
        "constraint",
        "decision",
    }
)

RECALL_VISIBILITY = """
    m.status='active'
    AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))
    AND (
        m.scope='user'
        OR (m.scope='project' AND m.project_id=?)
        OR (m.scope='thread' AND m.thread_id=?)
        OR (m.scope='run' AND m.source_run_id=?)
    )
"""


# [maint-note:275ec3447b35]
# 数据与行为导航：MemoryProposalConflict。
# 继承接口：RuntimeError。
# [/maint-note:275ec3447b35]
class MemoryProposalConflict(RuntimeError):
    pass


class MemoryStore:
    VALID_STATUSES = {"candidate", "active", "rejected", "superseded", "expired"}

    # [maint-note:19ef5049995f]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 db_path：str | Path。
    # 返回契约：None。
    # 分支条件：'thread_id' not in columns。
    # 分支条件：name not in columns。
    # 遍历边界：(name, definition) ← memory_additions.items()。
    # 写入目标：self.db_path。
    # 直接调用：Path(db_path).expanduser().resolve。
    # 直接调用：Path(db_path).expanduser。
    # 直接调用：Path。
    # 直接调用：self.db_path.parent.mkdir。
    # 直接调用：backup_before_schema_v6。
    # 直接调用：self._connect。
    # 直接调用：connection.executescript。
    # 直接调用：connection.execute('PRAGMA table_info(memories)').fetchall。
    # 直接调用：connection.execute。
    # 直接调用：memory_additions.items。
    # 直接调用：connection.commit。
    # 直接调用：self._migrate_legacy_candidates。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 connection.execute('PRAGMA table_info(memories)').fetchall()。
    # [/maint-note:19ef5049995f]
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        backup_before_schema_v6(self.db_path)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    project_id TEXT,
                    thread_id TEXT,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_run_id TEXT,
                    source_revision TEXT,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, subject, content);
                CREATE TABLE IF NOT EXISTS memory_proposals (
                    id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    target_memory_id TEXT,
                    source_mode TEXT NOT NULL,
                    project_id TEXT,
                    thread_id TEXT,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    canonical_content TEXT NOT NULL,
                    verification TEXT NOT NULL,
                    durability TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    expires_at TEXT,
                    analysis_json TEXT NOT NULL DEFAULT '{}',
                    proposal_json TEXT NOT NULL DEFAULT '{}',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    conflicts_json TEXT NOT NULL DEFAULT '[]',
                    source_hash TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    analyzer_prompt_version TEXT NOT NULL DEFAULT '',
                    composer_prompt_version TEXT NOT NULL DEFAULT '',
                    error_json TEXT,
                    version INTEGER NOT NULL DEFAULT 0,
                    applied_memory_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    FOREIGN KEY (target_memory_id) REFERENCES memories(id) ON DELETE SET NULL,
                    FOREIGN KEY (applied_memory_id) REFERENCES memories(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_proposals_review
                    ON memory_proposals(status, project_id, thread_id, updated_at);
                CREATE TABLE IF NOT EXISTS memory_proposal_evidence (
                    proposal_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    evidence_ref TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_sequence INTEGER,
                    content_hash TEXT NOT NULL,
                    preview TEXT NOT NULL,
                    source_locator TEXT NOT NULL,
                    verification TEXT NOT NULL,
                    PRIMARY KEY (proposal_id, ordinal),
                    UNIQUE (proposal_id, evidence_ref),
                    FOREIGN KEY (proposal_id) REFERENCES memory_proposals(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS memory_review_actions (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    before_json TEXT NOT NULL,
                    after_json TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (proposal_id) REFERENCES memory_proposals(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_review_actions_proposal
                    ON memory_review_actions(proposal_id, created_at);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(memories)").fetchall()}
            if "thread_id" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN thread_id TEXT")
            memory_additions = {
                "verification": "TEXT NOT NULL DEFAULT 'legacy_unverified'",
                "durability": "TEXT NOT NULL DEFAULT 'long_term'",
                "expires_at": "TEXT",
                "last_verified_at": "TEXT",
                "source_proposal_id": "TEXT",
                "evidence_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in memory_additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
            connection.execute("PRAGMA user_version=6")
            connection.commit()
        self._migrate_legacy_candidates()

    # [maint-note:dfb68c0623b7]
    # 维护导航：_connect；下列为本函数源码中的接口与路径索引。
    # 返回契约：Iterator[sqlite3.Connection]。
    # 写入目标：connection.row_factory。
    # 直接调用：sqlite3.connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.close。
    # 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
    # 资源作用域：connection；退出该 with 时调用对应上下文清理。
    # 生成器交付：connection。
    # [/maint-note:dfb68c0623b7]
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    # [maint-note:06278c900f55]
    # 维护导航：_decode_proposal_row；下列为本函数源码中的接口与路径索引。
    # 输入 row：sqlite3.Row。
    # 返回契约：dict[str, Any]。
    # 返回路径：value。
    # 写入目标：value['warnings']。
    # 写入目标：value['conflicts']。
    # 直接调用：dict。
    # 直接调用：json.loads。
    # 直接调用：value.pop。
    # [/maint-note:06278c900f55]
    @staticmethod
    def _decode_proposal_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["warnings"] = json.loads(value.pop("warnings_json") or "[]")
        value["conflicts"] = json.loads(value.pop("conflicts_json") or "[]")
        value.pop("analysis_json", None)
        value.pop("proposal_json", None)
        value.pop("error_json", None)
        return value

    # [maint-note:70d07edf5036]
    # 维护导航：_migrate_legacy_candidates；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 分支条件：existing。
    # 遍历边界：row ← rows。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("SELECT * FROM memories WHERE status='candidate' ORDER BY created_at").fetchall。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute("\n SELECT 1 FROM memory_proposals\n WHERE source_mode='legacy_migration' AND target_memory_id=?\n ", (row['id'],)).fetchone。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：hashlib.sha256(row['content'].encode('utf-8')).hexdigest。
    # 直接调用：hashlib.sha256。
    # 直接调用：row['content'].encode。
    # 直接调用：json.dumps。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:70d07edf5036]
    def _migrate_legacy_candidates(self) -> None:
        """Expose pre-v6 candidates through the governed review queue without data loss."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE status='candidate' ORDER BY created_at"
            ).fetchall()
            for row in rows:
                existing = connection.execute(
                    """
                    SELECT 1 FROM memory_proposals
                    WHERE source_mode='legacy_migration' AND target_memory_id=?
                    """,
                    (row["id"],),
                ).fetchone()
                if existing:
                    continue
                proposal_id = str(uuid.uuid4())
                timestamp = now_iso()
                content_hash = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                raw = {
                    "schema_version": 1,
                    "operation": "create",
                    "scope": row["scope"],
                    "kind": row["kind"],
                    "subject": row["subject"],
                    "canonical_content": row["content"],
                    "evidence_refs": [f"legacy-memory:{row['id']}"],
                }
                connection.execute(
                    """
                    INSERT INTO memory_proposals(
                        id, operation, target_memory_id, source_mode, project_id,
                        thread_id, run_id, status, scope, kind, subject,
                        canonical_content, verification, durability, confidence,
                        expires_at, analysis_json, proposal_json, warnings_json,
                        conflicts_json, source_hash, model, analyzer_prompt_version,
                        composer_prompt_version, version, created_at, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        proposal_id,
                        "create",
                        row["id"],
                        "legacy_migration",
                        row["project_id"],
                        row["thread_id"],
                        row["source_run_id"],
                        "review_ready",
                        row["scope"],
                        row["kind"],
                        row["subject"],
                        row["content"],
                        "legacy_unverified",
                        "long_term",
                        row["confidence"],
                        None,
                        "{}",
                        json.dumps(raw, ensure_ascii=False),
                        json.dumps([
                            "这是从旧版候选记忆迁移的数据，没有精确的原始证据引用。"
                        ], ensure_ascii=False),
                        "[]",
                        content_hash,
                        "legacy",
                        "legacy",
                        "legacy",
                        0,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO memory_proposal_evidence(
                        proposal_id, ordinal, evidence_ref, evidence_type,
                        source_id, source_sequence, content_hash, preview,
                        source_locator, verification
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        proposal_id,
                        0,
                        f"legacy-memory:{row['id']}",
                        "legacy_memory",
                        row["id"],
                        None,
                        content_hash,
                        row["content"][:500],
                        "历史候选记忆",
                        "legacy_unverified",
                    ),
                )
            connection.commit()

    # [maint-note:bcacb76013ba]
    # 维护导航：create_candidate；下列为本函数源码中的接口与路径索引。
    # 输入 scope：str。
    # 输入 kind：str。
    # 输入 project_id：str | None。
    # 输入 subject：str。
    # 输入 content：str。
    # 输入 source_run_id：str | None。
    # 输入 source_revision：str | None。
    # 输入 confidence：float。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 返回契约：str。
    # 分支条件：scope not in {'run', 'thread', 'project', 'user'}。
    # 分支条件：not 0 <= confidence <= 1。
    # 分支条件：duplicate。
    # 返回路径：duplicate['id']。
    # 返回路径：memory_id。
    # 显式抛出：ValueError('invalid memory scope')。
    # 显式抛出：ValueError('confidence must be between 0 and 1')。
    # 直接调用：ValueError。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("\n SELECT id FROM memories\n WHERE scope=?\n AND COALESCE(project_id,'')=COALESCE(?, '')\n AND COALESCE(thread_id,'')=COALESCE(?, '')\n AND subject=? AND content=?\n AND status IN ('candidate','active')\n ", (scope, proj。
    # 直接调用：connection.execute。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:bcacb76013ba]
    def create_candidate(
        self,
        *,
        scope: str,
        kind: str,
        project_id: str | None,
        subject: str,
        content: str,
        source_run_id: str | None,
        source_revision: str | None,
        confidence: float,
        thread_id: str | None = None,
    ) -> str:
        if scope not in {"run", "thread", "project", "user"}:
            raise ValueError("invalid memory scope")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        memory_id = str(uuid.uuid4())
        timestamp = now_iso()
        with self._connect() as connection:
            duplicate = connection.execute(
                """
                SELECT id FROM memories
                WHERE scope=?
                  AND COALESCE(project_id,'')=COALESCE(?, '')
                  AND COALESCE(thread_id,'')=COALESCE(?, '')
                  AND subject=? AND content=?
                  AND status IN ('candidate','active')
                """,
                (scope, project_id, thread_id, subject, content),
            ).fetchone()
            if duplicate:
                return duplicate["id"]
            connection.execute(
                """
                INSERT INTO memories(
                    id, scope, kind, project_id, thread_id, subject, content,
                    source_run_id, source_revision, confidence, status, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    memory_id,
                    scope,
                    kind,
                    project_id,
                    thread_id,
                    subject,
                    content,
                    source_run_id,
                    source_revision,
                    confidence,
                    "candidate",
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return memory_id

    # [maint-note:e0faf2ddc08f]
    # 维护导航：_set_status；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 输入 status：str。
    # 返回契约：None。
    # 分支条件：status not in self.VALID_STATUSES。
    # 分支条件：row is None。
    # 分支条件：status == 'active'。
    # 显式抛出：ValueError('invalid memory status')。
    # 显式抛出：KeyError(memory_id)。
    # 直接调用：ValueError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT subject, content FROM memories WHERE id=?', (memory_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:e0faf2ddc08f]
    def _set_status(self, memory_id: str, status: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError("invalid memory status")
        with self._connect() as connection:
            row = connection.execute("SELECT subject, content FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise KeyError(memory_id)
            connection.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?", (status, now_iso(), memory_id))
            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            if status == "active":
                connection.execute(
                    "INSERT INTO memory_fts(id, subject, content) VALUES(?,?,?)",
                    (memory_id, row["subject"], row["content"]),
                )
            connection.commit()

    # [maint-note:76861372b3c4]
    # 维护导航：activate；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 返回契约：None。
    # 直接调用：self._set_status。
    # [/maint-note:76861372b3c4]
    def activate(self, memory_id: str) -> None:
        self._set_status(memory_id, "active")

    # [maint-note:a9a764dfa8fa]
    # 维护导航：reject；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 返回契约：None。
    # 直接调用：self._set_status。
    # [/maint-note:a9a764dfa8fa]
    def reject(self, memory_id: str) -> None:
        self._set_status(memory_id, "rejected")

    # [maint-note:3a4f882d520b]
    # 维护导航：supersede；下列为本函数源码中的接口与路径索引。
    # 输入 old_id：str。
    # 输入 new_id：str。
    # 返回契约：None。
    # 直接调用：self.activate。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:3a4f882d520b]
    def supersede(self, old_id: str, new_id: str) -> None:
        self.activate(new_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE memories SET status='superseded', supersedes_id=?, updated_at=? WHERE id=?",
                (new_id, now_iso(), old_id),
            )
            connection.execute("DELETE FROM memory_fts WHERE id=?", (old_id,))
            connection.commit()

    # [maint-note:36bd77d1a982]
    # 维护导航：get；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：dict(row) if row else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM memories WHERE id=?', (memory_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row；真值与假值分支分别计算，不同时执行。
    # [/maint-note:36bd77d1a982]
    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

    # [maint-note:f6c24d5af2c6]
    # 维护导航：update；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 输入 subject：str。
    # 输入 content：str。
    # 输入 confidence：float。
    # 返回契约：None。
    # 分支条件：not 0 <= confidence <= 1。
    # 分支条件：row is None。
    # 分支条件：row['status'] == 'active'。
    # 显式抛出：ValueError('confidence must be between 0 and 1')。
    # 显式抛出：KeyError(memory_id)。
    # 直接调用：ValueError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT status FROM memories WHERE id=?', (memory_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:f6c24d5af2c6]
    def update(
        self,
        memory_id: str,
        *,
        subject: str,
        content: str,
        confidence: float,
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise KeyError(memory_id)
            connection.execute(
                "UPDATE memories SET subject=?, content=?, confidence=?, updated_at=? WHERE id=?",
                (subject, content, confidence, now_iso(), memory_id),
            )
            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            if row["status"] == "active":
                connection.execute(
                    "INSERT INTO memory_fts(id, subject, content) VALUES(?,?,?)",
                    (memory_id, subject, content),
                )
            connection.commit()

    # [maint-note:348c9bace0db]
    # 维护导航：search；下列为本函数源码中的接口与路径索引。
    # 输入 query：str。
    # 输入 project_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 limit：int；缺省表达式 5。
    # 输入 exclude_kinds：set[str] | frozenset[str] | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：not terms。
    # 分支条件：exclusions。
    # 返回路径：[]。
    # 返回路径：[dict(row) for row in rows]。
    # 直接调用：query.replace('"', ' ').split。
    # 直接调用：query.replace。
    # 直接调用：' OR '.join。
    # 直接调用：part.replace。
    # 直接调用：chr。
    # 直接调用：sorted。
    # 直接调用：','.join。
    # 直接调用：params.extend。
    # 直接调用：params.append。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'\n SELECT m.*, bm25(memory_fts) AS rank\n FROM memory_fts JOIN memories m ON m.id=memory_fts.id\n WHERE memory_fts MATCH ?\n AND {RECALL_VISIBILITY}\n {exclusion_sql}\n ORDER BY rank, m.confidence DESC, m.updated_at DES。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 集合派生：part 来自 query.replace('"', ' ').split()；过滤 part。
    # 集合派生：part 来自 raw_terms。
    # 集合派生：_ 来自 exclusions。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:348c9bace0db]
    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
        exclude_kinds: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        raw_terms = [part for part in query.replace('"', " ").split() if part]
        terms = " OR ".join(f'"{part.replace(chr(34), chr(34) * 2)}"' for part in raw_terms)
        if not terms:
            return []
        exclusions = sorted(exclude_kinds or ())
        exclusion_sql = ""
        params: list[Any] = [terms, project_id, thread_id, run_id]
        if exclusions:
            placeholders = ",".join("?" for _ in exclusions)
            exclusion_sql = f" AND m.kind NOT IN ({placeholders})"
            params.extend(exclusions)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.*, bm25(memory_fts) AS rank
                FROM memory_fts JOIN memories m ON m.id=memory_fts.id
                WHERE memory_fts MATCH ?
                  AND {RECALL_VISIBILITY}
                  {exclusion_sql}
                ORDER BY rank, m.confidence DESC, m.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    # [maint-note:9e0e6c3cfc07]
    # 维护导航：semantic_recall_candidates；下列为本函数源码中的接口与路径索引。
    # 输入 project_id：str。
    # 输入 thread_id：str | None。
    # 输入 run_id：str。
    # 输入 selected_ids：list[str] | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：selected_ids == []。
    # 分支条件：selected_ids is not None。
    # 分支条件：selected_ids is None。
    # 返回路径：[]。
    # 返回路径：[dict(row) for row in rows]。
    # 返回路径：[by_id[item] for item in selected_ids if item in by_id]。
    # 直接调用：sorted。
    # 直接调用：','.join。
    # 直接调用：params.extend。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'SELECT m.* FROM memories m WHERE {RECALL_VISIBILITY}\n AND m.kind NOT IN ({placeholders}) {id_filter}\n ORDER BY m.updated_at DESC, m.id LIMIT 16', params).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 集合派生：_ 来自 pinned。
    # 集合派生：_ 来自 selected_ids。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # 集合派生：item 来自 selected_ids；过滤 item in by_id。
    # [/maint-note:9e0e6c3cfc07]
    def semantic_recall_candidates(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        run_id: str,
        selected_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Bound a semantic scan and revalidate cached IDs before each use."""
        if selected_ids == []:
            return []
        pinned = sorted(PINNED_MEMORY_KINDS)
        placeholders = ",".join("?" for _ in pinned)
        params: list[Any] = [project_id, thread_id, run_id, *pinned]
        id_filter = ""
        if selected_ids is not None:
            id_filter = f"AND m.id IN ({','.join('?' for _ in selected_ids)})"
            params.extend(selected_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT m.* FROM memories m WHERE {RECALL_VISIBILITY}
                    AND m.kind NOT IN ({placeholders}) {id_filter}
                    ORDER BY m.updated_at DESC, m.id LIMIT 16""",
                params,
            ).fetchall()
        if selected_ids is None:
            return [dict(row) for row in rows]
        by_id = {row["id"]: dict(row) for row in rows}
        return [by_id[item] for item in selected_ids if item in by_id]

    # [maint-note:795480631b51]
    # 维护导航：list_pinned_for_context；下列为本函数源码中的接口与路径索引。
    # 输入 project_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 输入 limit：int；缺省表达式 24。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：[dict(row) for row in rows]。
    # 直接调用：sorted。
    # 直接调用：','.join。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f"\n SELECT m.*\n FROM memories m\n WHERE m.status='active'\n AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))\n AND m.kind IN ({placeholders})\n AND (\n m.scope='user'\n OR (m.scope='project' AND 。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 集合派生：_ 来自 kinds。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:795480631b51]
    def list_pinned_for_context(
        self,
        *,
        project_id: str | None = None,
        thread_id: str | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        """Return deterministic, goal-independent memories for a cacheable prefix."""
        kinds = sorted(PINNED_MEMORY_KINDS)
        placeholders = ",".join("?" for _ in kinds)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.*
                FROM memories m
                WHERE m.status='active'
                  AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))
                  AND m.kind IN ({placeholders})
                  AND (
                    m.scope='user'
                    OR (m.scope='project' AND m.project_id=?)
                    OR (m.scope='thread' AND m.thread_id=?)
                  )
                ORDER BY
                  CASE m.scope
                    WHEN 'user' THEN 0
                    WHEN 'project' THEN 1
                    WHEN 'thread' THEN 2
                    ELSE 3
                  END,
                  m.created_at,
                  m.id
                LIMIT ?
                """,
                [*kinds, project_id, thread_id, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    # [maint-note:74160b03ed37]
    # 维护导航：mark_stale_for_revision；下列为本函数源码中的接口与路径索引。
    # 输入 project_id：str。
    # 输入 current_revision：str。
    # 返回契约：int。
    # 遍历边界：row ← rows。
    # 返回路径：len(rows)。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("SELECT id FROM memories WHERE project_id=? AND status='active' AND source_revision IS NOT NULL AND source_revision<>?", (project_id, current_revision)).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：now_iso。
    # 直接调用：connection.commit。
    # 直接调用：len。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:74160b03ed37]
    def mark_stale_for_revision(self, project_id: str, current_revision: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM memories WHERE project_id=? AND status='active' AND source_revision IS NOT NULL AND source_revision<>?",
                (project_id, current_revision),
            ).fetchall()
            for row in rows:
                connection.execute("UPDATE memories SET status='expired', updated_at=? WHERE id=?", (now_iso(), row["id"]))
                connection.execute("DELETE FROM memory_fts WHERE id=?", (row["id"],))
            connection.commit()
        return len(rows)

    # [maint-note:2e7b26aee07c]
    # 维护导航：delete；下列为本函数源码中的接口与路径索引。
    # 输入 memory_id：str。
    # 返回契约：bool。
    # 返回路径：cursor.rowcount > 0。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:2e7b26aee07c]
    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            connection.commit()
        return cursor.rowcount > 0

    # [maint-note:7cc2c517415c]
    # 维护导航：list；下列为本函数源码中的接口与路径索引。
    # 输入 status：str | None；缺省表达式 None。
    # 输入 project_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：status。
    # 分支条件：project_id。
    # 分支条件：thread_id。
    # 遍历边界：item ← items。
    # 返回路径：items。
    # 写入目标：item['status']。
    # 直接调用：clauses.append。
    # 直接调用：params.append。
    # 直接调用：' AND '.join。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'SELECT *, {effective_status} AS effective_status FROM memories{where} ORDER BY updated_at DESC', params).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 直接调用：item.pop。
    # 表达式分流：clauses；真值与假值分支分别计算，不同时执行。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:7cc2c517415c]
    def list(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        effective_status = """CASE
            WHEN status='active' AND expires_at IS NOT NULL
                AND COALESCE(julianday(expires_at) > julianday('now'), 0)=0
            THEN 'expired' ELSE status END"""
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append(f"{effective_status}=?")
            params.append(status)
        if project_id:
            clauses.append("(project_id=? OR scope='user')")
            params.append(project_id)
        if thread_id:
            clauses.append("(thread_id=? OR scope IN ('project','user'))")
            params.append(thread_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT *, {effective_status} AS effective_status FROM memories{where} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["status"] = item.pop("effective_status")
        return items

    # [maint-note:09a1ce8ea5d1]
    # 维护导航：automatic_memory_manifest；下列为本函数源码中的接口与路径索引。
    # 输入 project_id：str。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：[dict(row) for row in rows]。
    # 直接调用：self._connect。
    # 直接调用：connection.execute("\n SELECT id, scope, kind, subject, substr(content, 1, 180) AS summary, updated_at\n FROM memories\n WHERE status='active' AND (scope='user' OR (scope='project' AND project_id=?))\n AND (expires_at IS NULL OR julianday(e。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:09a1ce8ea5d1]
    def automatic_memory_manifest(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, scope, kind, subject, substr(content, 1, 180) AS summary, updated_at
                FROM memories
                WHERE status='active' AND (scope='user' OR (scope='project' AND project_id=?))
                  AND (expires_at IS NULL OR julianday(expires_at) > julianday('now'))
                ORDER BY updated_at DESC, id LIMIT 20
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # [maint-note:f2c0cc9e4a57]
    # 维护导航：create_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 operation：str。
    # 输入 target_memory_id：str | None。
    # 输入 source_mode：str。
    # 输入 project_id：str | None。
    # 输入 thread_id：str | None。
    # 输入 run_id：str | None。
    # 输入 scope：str。
    # 输入 kind：str。
    # 输入 subject：str。
    # 输入 canonical_content：str。
    # 输入 verification：str。
    # 输入 durability：str。
    # 输入 confidence：float。
    # 输入 expires_at：str | None。
    # 输入 analysis：dict[str, Any]。
    # 输入 raw_proposal：dict[str, Any]。
    # 输入 warnings：list[str]。
    # 输入 conflicts：list[str]。
    # 输入 source_hash：str。
    # 输入 model：str。
    # 输入 analyzer_prompt_version：str。
    # 输入 composer_prompt_version：str。
    # 输入 evidence：list[dict[str, Any]]。
    # 返回契约：str | None。
    # 分支条件：source_mode == 'automatic_run'。
    # 分支条件：duplicate_memory。
    # 分支条件：duplicate。
    # 遍历边界：(ordinal, item) ← enumerate(evidence)。
    # 返回路径：None。
    # 返回路径：duplicate['id']。
    # 返回路径：proposal_id。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：now_iso。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：self._check_completed_source。
    # 直接调用：connection.execute("\n SELECT 1 FROM memories\n WHERE scope=? AND (scope='user' OR project_id=?) AND kind=?\n AND lower(trim(subject))=lower(trim(?)) AND content=? AND ?<>'expire'\n AND status='active'\n AND (expires_at IS NULL OR julianday。
    # 直接调用：connection.execute("\n SELECT id FROM memory_proposals\n WHERE source_hash=? AND source_mode=? AND scope=? AND kind=? AND subject=?\n AND canonical_content=?\n AND status IN ('review_ready','deferred','applied')\n ORDER BY created_at DESC L。
    # 直接调用：connection.rollback。
    # 直接调用：json.dumps。
    # 直接调用：enumerate。
    # 直接调用：item.get。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:f2c0cc9e4a57]
    def create_proposal(
        self,
        *,
        operation: str,
        target_memory_id: str | None,
        source_mode: str,
        project_id: str | None,
        thread_id: str | None,
        run_id: str | None,
        scope: str,
        kind: str,
        subject: str,
        canonical_content: str,
        verification: str,
        durability: str,
        confidence: float,
        expires_at: str | None,
        analysis: dict[str, Any],
        raw_proposal: dict[str, Any],
        warnings: list[str],
        conflicts: list[str],
        source_hash: str,
        model: str,
        analyzer_prompt_version: str,
        composer_prompt_version: str,
        evidence: list[dict[str, Any]],
    ) -> str | None:
        proposal_id = str(uuid.uuid4())
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if source_mode == "automatic_run":
                self._check_completed_source(connection, run_id, thread_id, project_id)
                duplicate_memory = connection.execute(
                    """
                    SELECT 1 FROM memories
                    WHERE scope=? AND (scope='user' OR project_id=?) AND kind=?
                      AND lower(trim(subject))=lower(trim(?)) AND content=? AND ?<>'expire'
                      AND status='active'
                      AND (expires_at IS NULL OR julianday(expires_at) > julianday('now'))
                    UNION ALL
                    SELECT 1 FROM memory_proposals
                    WHERE scope=? AND (scope='user' OR project_id=?) AND kind=?
                      AND lower(trim(subject))=lower(trim(?)) AND canonical_content=?
                      AND operation=? AND target_memory_id IS ?
                      AND status IN ('review_ready','deferred')
                    LIMIT 1
                    """,
                    (scope, project_id, kind, subject, canonical_content, operation,
                     scope, project_id, kind, subject, canonical_content, operation, target_memory_id),
                ).fetchone()
                if duplicate_memory:
                    return None
            duplicate = connection.execute(
                """
                SELECT id FROM memory_proposals
                WHERE source_hash=? AND source_mode=? AND scope=? AND kind=? AND subject=?
                  AND canonical_content=?
                  AND status IN ('review_ready','deferred','applied')
                ORDER BY created_at DESC LIMIT 1
                """,
                (source_hash, source_mode, scope, kind, subject, canonical_content),
            ).fetchone()
            if duplicate:
                connection.rollback()
                return duplicate["id"]
            connection.execute(
                """
                INSERT INTO memory_proposals(
                    id, operation, target_memory_id, source_mode, project_id,
                    thread_id, run_id, status, scope, kind, subject,
                    canonical_content, verification, durability, confidence,
                    expires_at, analysis_json, proposal_json, warnings_json,
                    conflicts_json, source_hash, model, analyzer_prompt_version,
                    composer_prompt_version, version, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    proposal_id,
                    operation,
                    target_memory_id,
                    source_mode,
                    project_id,
                    thread_id,
                    run_id,
                    "review_ready",
                    scope,
                    kind,
                    subject,
                    canonical_content,
                    verification,
                    durability,
                    confidence,
                    expires_at,
                    json.dumps(analysis, ensure_ascii=False),
                    json.dumps(raw_proposal, ensure_ascii=False),
                    json.dumps(warnings, ensure_ascii=False),
                    json.dumps(conflicts, ensure_ascii=False),
                    source_hash,
                    model,
                    analyzer_prompt_version,
                    composer_prompt_version,
                    0,
                    timestamp,
                    timestamp,
                ),
            )
            for ordinal, item in enumerate(evidence):
                connection.execute(
                    """
                    INSERT INTO memory_proposal_evidence(
                        proposal_id, ordinal, evidence_ref, evidence_type,
                        source_id, source_sequence, content_hash, preview,
                        source_locator, verification
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        proposal_id,
                        ordinal,
                        item["ref"],
                        item["evidence_type"],
                        item["source_id"],
                        item.get("source_sequence"),
                        item["content_hash"],
                        item["preview"],
                        item["source_locator"],
                        item["verification"],
                    ),
                )
            connection.commit()
        return proposal_id

    # [maint-note:bd6122537a06]
    # 维护导航：get_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：dict[str, Any] | None。
    # 返回路径：self._decode_proposal_row(row) if row else None。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：self._decode_proposal_row。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row；真值与假值分支分别计算，不同时执行。
    # [/maint-note:bd6122537a06]
    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
        return self._decode_proposal_row(row) if row else None

    # [maint-note:5ff4d095bb95]
    # 维护导航：get_proposal_internal；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：dict[str, Any] | None。
    # 分支条件：row is None。
    # 遍历边界：(source, target, default) ← (('analysis_json', 'analysis', {}), ('proposal_json', 'raw_proposal', {}), ('warnings_json', 'warnings', []), ('conflicts_json', 'conflicts', []), ('error_json', 'error', None))。
    # 返回路径：None。
    # 返回路径：value。
    # 写入目标：value[target]。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 直接调用：value.pop。
    # 直接调用：json.loads。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：raw；真值与假值分支分别计算，不同时执行。
    # [/maint-note:5ff4d095bb95]
    def get_proposal_internal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        for source, target, default in (
            ("analysis_json", "analysis", {}),
            ("proposal_json", "raw_proposal", {}),
            ("warnings_json", "warnings", []),
            ("conflicts_json", "conflicts", []),
            ("error_json", "error", None),
        ):
            raw = value.pop(source)
            value[target] = json.loads(raw) if raw else default
        return value

    # [maint-note:0fead33b1342]
    # 维护导航：list_proposals；下列为本函数源码中的接口与路径索引。
    # 输入 status：str | None；缺省表达式 None。
    # 输入 project_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：status。
    # 分支条件：project_id。
    # 分支条件：thread_id。
    # 返回路径：[self._decode_proposal_row(row) for row in rows]。
    # 直接调用：clauses.append。
    # 直接调用：params.append。
    # 直接调用：' AND '.join。
    # 直接调用：self._connect。
    # 直接调用：connection.execute(f'SELECT * FROM memory_proposals{where} ORDER BY updated_at DESC', params).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：self._decode_proposal_row。
    # 表达式分流：clauses；真值与假值分支分别计算，不同时执行。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:0fead33b1342]
    def list_proposals(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if project_id:
            clauses.append("(project_id=? OR scope='user')")
            params.append(project_id)
        if thread_id:
            clauses.append("(thread_id=? OR scope IN ('project','user'))")
            params.append(thread_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memory_proposals{where} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        return [self._decode_proposal_row(row) for row in rows]

    # [maint-note:9ea0b8092752]
    # 维护导航：get_proposal_evidence；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：connection.execute('SELECT 1 FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone() is None。
    # 返回路径：[dict(row) for row in rows]。
    # 显式抛出：KeyError(proposal_id)。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT 1 FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：KeyError。
    # 直接调用：connection.execute('\n SELECT * FROM memory_proposal_evidence\n WHERE proposal_id=? ORDER BY ordinal\n ', (proposal_id,)).fetchall。
    # 直接调用：dict。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：row 来自 rows。
    # [/maint-note:9ea0b8092752]
    def get_proposal_evidence(self, proposal_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone() is None:
                raise KeyError(proposal_id)
            rows = connection.execute(
                """
                SELECT * FROM memory_proposal_evidence
                WHERE proposal_id=? ORDER BY ordinal
                """,
                (proposal_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # [maint-note:f6ad11ef34c9]
    # 维护导航：_proposal_snapshot；下列为本函数源码中的接口与路径索引。
    # 输入 row：sqlite3.Row | dict[str, Any]。
    # 返回契约：dict[str, Any]。
    # 返回路径：{key: row[key] for key in ('status', 'scope', 'kind', 'subject', 'canonical_content', 'expires_at', 'version', 'applied_memory_id')}。
    # 集合派生：key 来自 ('status', 'scope', 'kind', 'subject', 'canonical_content', 'expires_at', 'version', 'applied_memory_id')。
    # [/maint-note:f6ad11ef34c9]
    @staticmethod
    def _proposal_snapshot(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "status",
                "scope",
                "kind",
                "subject",
                "canonical_content",
                "expires_at",
                "version",
                "applied_memory_id",
            )
        }

    # [maint-note:33639ce3902c]
    # 维护导航：_insert_review_action；下列为本函数源码中的接口与路径索引。
    # 输入 connection：sqlite3.Connection。
    # 输入 proposal_id：str。
    # 输入 action：str。
    # 输入 actor：str。
    # 输入 before：dict[str, Any]。
    # 输入 after：dict[str, Any]。
    # 输入 reason：str。
    # 返回契约：None。
    # 直接调用：connection.execute。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：json.dumps。
    # 直接调用：now_iso。
    # [/maint-note:33639ce3902c]
    @staticmethod
    def _insert_review_action(
        connection: sqlite3.Connection,
        *,
        proposal_id: str,
        action: str,
        actor: str,
        before: dict[str, Any],
        after: dict[str, Any],
        reason: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_review_actions(
                id, proposal_id, action, actor, before_json, after_json,
                reason, created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                proposal_id,
                action,
                actor,
                json.dumps(before, ensure_ascii=False),
                json.dumps(after, ensure_ascii=False),
                reason,
                now_iso(),
            ),
        )

    # [maint-note:8a4f703f2568]
    # 维护导航：update_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 输入 actor：str。
    # 输入 subject：str | None；缺省表达式 None。
    # 输入 canonical_content：str | None；缺省表达式 None。
    # 输入 scope：str | None；缺省表达式 None。
    # 输入 kind：str | None；缺省表达式 None。
    # 输入 expires_at：str | None；缺省表达式 None。
    # 返回契约：None。
    # 分支条件：subject is not None。
    # 分支条件：not subject.strip()。
    # 分支条件：canonical_content is not None。
    # 分支条件：not canonical_content.strip()。
    # 分支条件：scope is not None。
    # 分支条件：scope not in {'run', 'thread', 'project', 'user'}。
    # 分支条件：kind is not None。
    # 分支条件：expires_at is not None。
    # 分支条件：not changes。
    # 分支条件：row is None。
    # 分支条件：int(row['version']) != expected_version。
    # 分支条件：row['status'] not in {'review_ready', 'deferred'}。
    # 返回路径：无返回值。
    # 显式抛出：ValueError('proposal subject cannot be empty')。
    # 显式抛出：ValueError('proposal content cannot be empty')。
    # 显式抛出：ValueError('invalid memory scope')。
    # 显式抛出：KeyError(proposal_id)。
    # 显式抛出：MemoryProposalConflict(f"expected proposal version {expected_version}, found {row['version']}")。
    # 显式抛出：MemoryProposalConflict('proposal is no longer editable')。
    # 写入目标：changes['subject']。
    # 写入目标：changes['canonical_content']。
    # 写入目标：changes['scope']。
    # 写入目标：changes['kind']。
    # 写入目标：changes['expires_at']。
    # 写入目标：changes['status']。
    # 写入目标：changes['version']。
    # 写入目标：changes['updated_at']。
    # 直接调用：subject.strip。
    # 直接调用：ValueError。
    # 直接调用：canonical_content.strip。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT * FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：MemoryProposalConflict。
    # 直接调用：self._proposal_snapshot。
    # 直接调用：now_iso。
    # 直接调用：', '.join。
    # 直接调用：changes.values。
    # 直接调用：self._insert_review_action。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 集合派生：key 来自 changes。
    # [/maint-note:8a4f703f2568]
    def update_proposal(
        self,
        proposal_id: str,
        *,
        expected_version: int,
        actor: str,
        subject: str | None = None,
        canonical_content: str | None = None,
        scope: str | None = None,
        kind: str | None = None,
        expires_at: str | None = None,
    ) -> None:
        changes: dict[str, Any] = {}
        if subject is not None:
            if not subject.strip():
                raise ValueError("proposal subject cannot be empty")
            changes["subject"] = subject.strip()
        if canonical_content is not None:
            if not canonical_content.strip():
                raise ValueError("proposal content cannot be empty")
            changes["canonical_content"] = canonical_content.strip()
        if scope is not None:
            if scope not in {"run", "thread", "project", "user"}:
                raise ValueError("invalid memory scope")
            changes["scope"] = scope
        if kind is not None:
            changes["kind"] = kind
        if expires_at is not None:
            changes["expires_at"] = expires_at or None
        if not changes:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(proposal_id)
            if int(row["version"]) != expected_version:
                connection.rollback()
                raise MemoryProposalConflict(
                    f"expected proposal version {expected_version}, found {row['version']}"
                )
            if row["status"] not in {"review_ready", "deferred"}:
                connection.rollback()
                raise MemoryProposalConflict("proposal is no longer editable")
            before = self._proposal_snapshot(row)
            changes["status"] = "review_ready"
            changes["version"] = int(row["version"]) + 1
            changes["updated_at"] = now_iso()
            assignments = ", ".join(f"{key}=?" for key in changes)
            connection.execute(
                f"UPDATE memory_proposals SET {assignments} WHERE id=?",
                [*changes.values(), proposal_id],
            )
            updated = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            self._insert_review_action(
                connection,
                proposal_id=proposal_id,
                action="edit",
                actor=actor,
                before=before,
                after=self._proposal_snapshot(updated),
                reason="",
            )
            connection.commit()

    # [maint-note:60f987b734cd]
    # 维护导航：review_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 action：str。
    # 输入 expected_version：int。
    # 输入 actor：str。
    # 输入 reason：str；缺省表达式 ''。
    # 返回契约：None。
    # 分支条件：target_status is None。
    # 分支条件：row is None。
    # 分支条件：int(row['version']) != expected_version。
    # 分支条件：row['status'] not in {'review_ready', 'deferred'}。
    # 显式抛出：ValueError('unsupported proposal review action')。
    # 显式抛出：KeyError(proposal_id)。
    # 显式抛出：MemoryProposalConflict(f"expected proposal version {expected_version}, found {row['version']}")。
    # 显式抛出：MemoryProposalConflict('proposal has already been finalized')。
    # 直接调用：{'reject': 'rejected', 'defer': 'deferred'}.get。
    # 直接调用：ValueError。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT * FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：MemoryProposalConflict。
    # 直接调用：self._proposal_snapshot。
    # 直接调用：now_iso。
    # 直接调用：self._insert_review_action。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:60f987b734cd]
    def review_proposal(
        self,
        proposal_id: str,
        *,
        action: str,
        expected_version: int,
        actor: str,
        reason: str = "",
    ) -> None:
        target_status = {"reject": "rejected", "defer": "deferred"}.get(action)
        if target_status is None:
            raise ValueError("unsupported proposal review action")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(proposal_id)
            if int(row["version"]) != expected_version:
                connection.rollback()
                raise MemoryProposalConflict(
                    f"expected proposal version {expected_version}, found {row['version']}"
                )
            if row["status"] not in {"review_ready", "deferred"}:
                connection.rollback()
                raise MemoryProposalConflict("proposal has already been finalized")
            before = self._proposal_snapshot(row)
            timestamp = now_iso()
            connection.execute(
                """
                UPDATE memory_proposals
                SET status=?, version=version+1, reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (target_status, timestamp, timestamp, proposal_id),
            )
            updated = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            self._insert_review_action(
                connection,
                proposal_id=proposal_id,
                action=action,
                actor=actor,
                before=before,
                after=self._proposal_snapshot(updated),
                reason=reason,
            )
            connection.commit()

    # [maint-note:90f0ec389c75]
    # 维护导航：approve_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 输入 actor：str。
    # 返回契约：str | None。
    # 分支条件：row is None。
    # 分支条件：int(row['version']) != expected_version。
    # 分支条件：row['status'] not in {'review_ready', 'deferred'}。
    # 分支条件：row['source_mode'] == 'automatic_run'。
    # 分支条件：row['source_mode'] == 'automatic_run' and row['operation'] != 'create'。
    # 分支条件：target is None or target['updated_at'] != expected_target or target['scope'] != row['scope'] or (target['kind'] != row['kind']) or (row['scope'] == 'project' and target['project_id'] != row['project_id'])。
    # 分支条件：actor == 'auto_memory' and target['subject'].casefold() != row['subject'].casefold()。
    # 分支条件：actor == 'auto_memory'。
    # 分支条件：operation == 'expire'。
    # 分支条件：not target_id。
    # 分支条件：operation == 'create' and target_id。
    # 分支条件：target is None or target['status'] != 'candidate'。
    # 分支条件：operation in {'update', 'supersede'}。
    # 分支条件：target is None or target['status'] != 'active'。
    # 分支条件：target_id and operation in {'update', 'supersede'}。
    # 分支条件：applied_memory_id。
    # 返回路径：applied_memory_id。
    # 显式抛出：KeyError(proposal_id)。
    # 显式抛出：MemoryProposalConflict(f"expected proposal version {expected_version}, found {row['version']}")。
    # 显式抛出：MemoryProposalConflict('proposal has already been finalized')。
    # 显式抛出：MemoryProposalConflict('memory target changed after analysis')。
    # 显式抛出：MemoryProposalConflict('a subject change requires human review')。
    # 显式抛出：MemoryProposalConflict('expire proposal has no target memory')。
    # 显式抛出：MemoryProposalConflict('legacy candidate is no longer available')。
    # 显式抛出：MemoryProposalConflict('target memory is not active')。
    # 直接调用：self._connect。
    # 直接调用：connection.execute。
    # 直接调用：connection.execute('SELECT * FROM memory_proposals WHERE id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.rollback。
    # 直接调用：KeyError。
    # 直接调用：int。
    # 直接调用：MemoryProposalConflict。
    # 直接调用：self._check_completed_source。
    # 直接调用：connection.execute("SELECT * FROM memories WHERE id=? AND status='active'", (row['target_memory_id'],)).fetchone。
    # 直接调用：json.loads(row['proposal_json']).get。
    # 直接调用：json.loads。
    # 直接调用：target['subject'].casefold。
    # 直接调用：row['subject'].casefold。
    # 直接调用：self._check_automatic_approval。
    # 直接调用：self._proposal_snapshot。
    # 直接调用：now_iso。
    # 直接调用：connection.execute('SELECT COUNT(*) AS count FROM memory_proposal_evidence WHERE proposal_id=?', (proposal_id,)).fetchone。
    # 直接调用：connection.execute('SELECT status FROM memories WHERE id=?', (target_id,)).fetchone。
    # 直接调用：connection.execute('SELECT id, status FROM memories WHERE id=?', (target_id,)).fetchone。
    # 直接调用：str。
    # 直接调用：uuid.uuid4。
    # 直接调用：self._insert_review_action。
    # 直接调用：connection.commit。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # 表达式分流：row['verification'] not in {'model_inferred', 'legacy_unverified'}；真值与假值分支分别计算，不同时执行。
    # [/maint-note:90f0ec389c75]
    def approve_proposal(
        self,
        proposal_id: str,
        *,
        expected_version: int,
        actor: str,
    ) -> str | None:
        """Apply a proposal and update FTS atomically."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(proposal_id)
            if int(row["version"]) != expected_version:
                connection.rollback()
                raise MemoryProposalConflict(
                    f"expected proposal version {expected_version}, found {row['version']}"
                )
            if row["status"] not in {"review_ready", "deferred"}:
                connection.rollback()
                raise MemoryProposalConflict("proposal has already been finalized")
            if row["source_mode"] == "automatic_run":
                self._check_completed_source(
                    connection, row["run_id"], row["thread_id"], row["project_id"]
                )
            if row["source_mode"] == "automatic_run" and row["operation"] != "create":
                target = connection.execute(
                    "SELECT * FROM memories WHERE id=? AND status='active'",
                    (row["target_memory_id"],),
                ).fetchone()
                expected_target = json.loads(row["proposal_json"]).get("_auto_target_updated_at")
                if (
                    target is None or target["updated_at"] != expected_target
                    or target["scope"] != row["scope"] or target["kind"] != row["kind"]
                    or (row["scope"] == "project" and target["project_id"] != row["project_id"])
                ):
                    raise MemoryProposalConflict("memory target changed after analysis")
                if actor == "auto_memory" and target["subject"].casefold() != row["subject"].casefold():
                    raise MemoryProposalConflict("a subject change requires human review")
            if actor == "auto_memory":
                self._check_automatic_approval(connection, row)
            before = self._proposal_snapshot(row)
            timestamp = now_iso()
            evidence_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM memory_proposal_evidence WHERE proposal_id=?",
                    (proposal_id,),
                ).fetchone()["count"]
            )
            operation = row["operation"]
            target_id = row["target_memory_id"]
            applied_memory_id: str | None = None
            if operation == "expire":
                if not target_id:
                    connection.rollback()
                    raise MemoryProposalConflict("expire proposal has no target memory")
                connection.execute(
                    "UPDATE memories SET status='expired', updated_at=? WHERE id=?",
                    (timestamp, target_id),
                )
                connection.execute("DELETE FROM memory_fts WHERE id=?", (target_id,))
            elif operation == "create" and target_id:
                target = connection.execute(
                    "SELECT status FROM memories WHERE id=?", (target_id,)
                ).fetchone()
                if target is None or target["status"] != "candidate":
                    connection.rollback()
                    raise MemoryProposalConflict("legacy candidate is no longer available")
                applied_memory_id = target_id
                connection.execute(
                    """
                    UPDATE memories SET
                        scope=?, kind=?, project_id=?, thread_id=?, subject=?, content=?,
                        confidence=?, status='active', verification=?, durability=?,
                        expires_at=?, last_verified_at=?, source_proposal_id=?,
                        evidence_count=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        row["scope"], row["kind"], row["project_id"], row["thread_id"],
                        row["subject"], row["canonical_content"], row["confidence"],
                        row["verification"], row["durability"], row["expires_at"],
                        timestamp if row["verification"] not in {"model_inferred", "legacy_unverified"} else None,
                        proposal_id, evidence_count, timestamp, target_id,
                    ),
                )
            else:
                if operation in {"update", "supersede"}:
                    target = connection.execute(
                        "SELECT id, status FROM memories WHERE id=?", (target_id,)
                    ).fetchone()
                    if target is None or target["status"] != "active":
                        connection.rollback()
                        raise MemoryProposalConflict("target memory is not active")
                applied_memory_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, scope, kind, project_id, thread_id, subject, content,
                        source_run_id, source_revision, confidence, status,
                        supersedes_id, created_at, updated_at, verification,
                        durability, expires_at, last_verified_at,
                        source_proposal_id, evidence_count
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        applied_memory_id, row["scope"], row["kind"], row["project_id"],
                        row["thread_id"], row["subject"], row["canonical_content"],
                        row["run_id"], None, row["confidence"], "active", target_id,
                        timestamp, timestamp, row["verification"], row["durability"],
                        row["expires_at"],
                        timestamp if row["verification"] not in {"model_inferred", "legacy_unverified"} else None,
                        proposal_id, evidence_count,
                    ),
                )
                if target_id and operation in {"update", "supersede"}:
                    connection.execute(
                        "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                        (timestamp, target_id),
                    )
                    connection.execute("DELETE FROM memory_fts WHERE id=?", (target_id,))
            if applied_memory_id:
                connection.execute("DELETE FROM memory_fts WHERE id=?", (applied_memory_id,))
                connection.execute(
                    "INSERT INTO memory_fts(id, subject, content) VALUES(?,?,?)",
                    (applied_memory_id, row["subject"], row["canonical_content"]),
                )
            connection.execute(
                """
                UPDATE memory_proposals SET
                    status='applied', applied_memory_id=?, version=version+1,
                    reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (applied_memory_id, timestamp, timestamp, proposal_id),
            )
            updated = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            self._insert_review_action(
                connection,
                proposal_id=proposal_id,
                action="approve",
                actor=actor,
                before=before,
                after=self._proposal_snapshot(updated),
                reason="",
            )
            connection.commit()
        return applied_memory_id

    # [maint-note:ef8e8a7fcf6d]
    # 维护导航：_check_completed_source；下列为本函数源码中的接口与路径索引。
    # 输入 connection：sqlite3.Connection。
    # 输入 run_id：str | None。
    # 输入 thread_id：str | None。
    # 输入 project_id：str | None。
    # 返回契约：None。
    # 分支条件：connection.execute("SELECT 1 FROM runs\n WHERE id=? AND thread_id=? AND workspace=? AND status='completed'", (run_id, thread_id, project_id)).fetchone() is None。
    # 显式抛出：MemoryProposalConflict('source run is no longer available')。
    # 直接调用：connection.execute("SELECT 1 FROM runs\n WHERE id=? AND thread_id=? AND workspace=? AND status='completed'", (run_id, thread_id, project_id)).fetchone。
    # 直接调用：connection.execute。
    # 直接调用：MemoryProposalConflict。
    # [/maint-note:ef8e8a7fcf6d]
    @staticmethod
    def _check_completed_source(
        connection: sqlite3.Connection, run_id: str | None,
        thread_id: str | None, project_id: str | None,
    ) -> None:
        if connection.execute(
            """SELECT 1 FROM runs
               WHERE id=? AND thread_id=? AND workspace=? AND status='completed'""",
            (run_id, thread_id, project_id),
        ).fetchone() is None:
            raise MemoryProposalConflict("source run is no longer available")

    # [maint-note:beb3bf71e067]
    # 维护导航：_check_automatic_approval；下列为本函数源码中的接口与路径索引。
    # 输入 connection：sqlite3.Connection。
    # 输入 proposal：sqlite3.Row。
    # 返回契约：None。
    # 分支条件：proposal['source_mode'] != 'automatic_run' or proposal['operation'] == 'expire' or (proposal['verification'] not in {'user_asserted', 'test_verified'} and (not verified_recovery)) or (proposal['confidence'] < 0.6) or json.loads(proposal['co。
    # 分支条件：analysis['contradictions'] or analysis['sensitive_findings']。
    # 分支条件：len(cited) < (3 if verified_recovery else 1)。
    # 分支条件：duplicate_subject。
    # 显式抛出：MemoryProposalConflict('proposal requires human review')。
    # 显式抛出：MemoryProposalConflict('evidence analysis requires human review')。
    # 显式抛出：MemoryProposalConflict('proposal lacks matching direct evidence')。
    # 显式抛出：MemoryProposalConflict('another active memory already uses this subject')。
    # 直接调用：json.loads。
    # 直接调用：MemoryProposalConflict。
    # 直接调用：connection.execute('\n SELECT evidence_ref FROM memory_proposal_evidence\n WHERE proposal_id=? AND verification=?\n ', (proposal['id'], proposal['verification'])).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：len。
    # 直接调用：connection.execute("\n SELECT 1 FROM memories\n WHERE scope=? AND (scope='user' OR project_id=?) AND kind=?\n AND lower(trim(subject))=lower(trim(?)) AND id IS NOT ?\n AND status='active'\n AND (expires_at IS NULL OR julianday(expires_at) >。
    # 集合派生：row 来自 matching；过滤 row['evidence_ref'] in evidence。
    # 表达式分流：verified_recovery；真值与假值分支分别计算，不同时执行。
    # [/maint-note:beb3bf71e067]
    @staticmethod
    def _check_automatic_approval(
        connection: sqlite3.Connection, proposal: sqlite3.Row
    ) -> None:
        verified_recovery = (
            proposal["verification"] == "tool_observed"
            and proposal["analyzer_prompt_version"] == TOOL_RECOVERY_VERSION
            and proposal["operation"] == "create"
            and proposal["scope"] == "project"
            and proposal["kind"] == "successful_approach"
        )
        if (
            proposal["source_mode"] != "automatic_run"
            or proposal["operation"] == "expire"
            or (proposal["verification"] not in {"user_asserted", "test_verified"} and not verified_recovery)
            or proposal["confidence"] < 0.6
            or json.loads(proposal["conflicts_json"])
            or json.loads(proposal["warnings_json"])
        ):
            raise MemoryProposalConflict("proposal requires human review")
        analysis = json.loads(proposal["analysis_json"])
        if analysis["contradictions"] or analysis["sensitive_findings"]:
            raise MemoryProposalConflict("evidence analysis requires human review")
        evidence = json.loads(proposal["proposal_json"])["evidence_refs"]
        matching = connection.execute(
            """
            SELECT evidence_ref FROM memory_proposal_evidence
            WHERE proposal_id=? AND verification=?
            """,
            (proposal["id"], proposal["verification"]),
        ).fetchall()
        cited = {row["evidence_ref"] for row in matching if row["evidence_ref"] in evidence}
        if len(cited) < (3 if verified_recovery else 1):
            raise MemoryProposalConflict("proposal lacks matching direct evidence")
        duplicate_subject = connection.execute(
            """
            SELECT 1 FROM memories
            WHERE scope=? AND (scope='user' OR project_id=?) AND kind=?
              AND lower(trim(subject))=lower(trim(?)) AND id IS NOT ?
              AND status='active'
              AND (expires_at IS NULL OR julianday(expires_at) > julianday('now'))
            LIMIT 1
            """,
            (proposal["scope"], proposal["project_id"], proposal["kind"],
             proposal["subject"], proposal["target_memory_id"]),
        ).fetchone()
        if duplicate_subject:
            raise MemoryProposalConflict("another active memory already uses this subject")

    # [maint-note:f702a9d79ba5]
    # 维护导航：list_review_actions；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：list[dict[str, Any]]。
    # 遍历边界：row ← rows。
    # 返回路径：result。
    # 写入目标：value['before']。
    # 写入目标：value['after']。
    # 直接调用：self._connect。
    # 直接调用：connection.execute('SELECT * FROM memory_review_actions WHERE proposal_id=? ORDER BY created_at', (proposal_id,)).fetchall。
    # 直接调用：connection.execute。
    # 直接调用：dict。
    # 直接调用：json.loads。
    # 直接调用：value.pop。
    # 直接调用：result.append。
    # 资源作用域：self._connect()；退出该 with 时调用对应上下文清理。
    # [/maint-note:f702a9d79ba5]
    def list_review_actions(self, proposal_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_review_actions WHERE proposal_id=? ORDER BY created_at",
                (proposal_id,),
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["before"] = json.loads(value.pop("before_json"))
            value["after"] = json.loads(value.pop("after_json"))
            result.append(value)
        return result
