from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_V6_BACKUP_SUFFIX = ".before-v6-memory-proposals.bak"


def backup_before_schema_v6(db_path: str | Path) -> Path | None:
    """Create one consistent, non-overwriting backup before the first v6 migration."""
    source_path = Path(db_path).expanduser().resolve()
    if not source_path.is_file():
        return None
    backup_path = Path(str(source_path) + SCHEMA_V6_BACKUP_SUFFIX)
    if backup_path.exists():
        return backup_path
    source = sqlite3.connect(source_path, timeout=30)
    try:
        user_version = int(source.execute("PRAGMA user_version").fetchone()[0])
        table_count = int(
            source.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchone()[0]
        )
        if user_version >= 6 or table_count == 0:
            return None
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
            destination.commit()
        finally:
            destination.close()
    finally:
        source.close()
    return backup_path
