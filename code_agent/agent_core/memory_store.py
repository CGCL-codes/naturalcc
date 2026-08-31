from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .contracts import now_iso
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


class MemoryProposalConflict(RuntimeError):
    pass


class MemoryStore:
    VALID_STATUSES = {"candidate", "active", "rejected", "superseded", "expired"}

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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _decode_proposal_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["warnings"] = json.loads(value.pop("warnings_json") or "[]")
        value["conflicts"] = json.loads(value.pop("conflicts_json") or "[]")
        value.pop("analysis_json", None)
        value.pop("proposal_json", None)
        value.pop("error_json", None)
        return value

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

    def activate(self, memory_id: str) -> None:
        self._set_status(memory_id, "active")

    def reject(self, memory_id: str) -> None:
        self._set_status(memory_id, "rejected")

    def supersede(self, old_id: str, new_id: str) -> None:
        self.activate(new_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE memories SET status='superseded', supersedes_id=?, updated_at=? WHERE id=?",
                (new_id, now_iso(), old_id),
            )
            connection.execute("DELETE FROM memory_fts WHERE id=?", (old_id,))
            connection.commit()

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

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
                  AND m.status='active'
                  AND (
                    m.scope='user'
                    OR (m.scope='project' AND m.project_id=?)
                    OR (m.scope='thread' AND m.thread_id=?)
                    OR (m.scope='run' AND m.source_run_id=?)
                  )
                  {exclusion_sql}
                ORDER BY rank, m.confidence DESC, m.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

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

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            connection.commit()
        return cursor.rowcount > 0

    def list(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
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
            rows = connection.execute(f"SELECT * FROM memories{where} ORDER BY updated_at DESC", params).fetchall()
        return [dict(row) for row in rows]

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
    ) -> str:
        proposal_id = str(uuid.uuid4())
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                """
                SELECT id FROM memory_proposals
                WHERE source_hash=? AND scope=? AND kind=? AND subject=?
                  AND canonical_content=?
                  AND status IN ('review_ready','deferred','applied')
                ORDER BY created_at DESC LIMIT 1
                """,
                (source_hash, scope, kind, subject, canonical_content),
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

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
        return self._decode_proposal_row(row) if row else None

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
