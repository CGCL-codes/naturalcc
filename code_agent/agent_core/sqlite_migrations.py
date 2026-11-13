from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_V6_BACKUP_SUFFIX = ".before-v6-memory-proposals.bak"


# [maint-note:1034ab49b6ed]
# 维护导航：backup_before_schema_v6；下列为本函数源码中的接口与路径索引。
# 输入 db_path：str | Path。
# 返回契约：Path | None。
# 分支条件：not source_path.is_file()。
# 分支条件：backup_path.exists()。
# 分支条件：user_version >= 6 or table_count == 0。
# 返回路径：None。
# 返回路径：backup_path。
# 直接调用：Path(db_path).expanduser().resolve。
# 直接调用：Path(db_path).expanduser。
# 直接调用：Path。
# 直接调用：source_path.is_file。
# 直接调用：str。
# 直接调用：backup_path.exists。
# 直接调用：sqlite3.connect。
# 直接调用：int。
# 直接调用：source.execute('PRAGMA user_version').fetchone。
# 直接调用：source.execute。
# 直接调用：source.execute("SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','view')").fetchone。
# 直接调用：source.backup。
# 直接调用：destination.commit。
# 直接调用：destination.close。
# 直接调用：source.close。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# [/maint-note:1034ab49b6ed]
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
