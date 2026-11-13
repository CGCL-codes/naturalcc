from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from ...rag.visualize.visualize import GROUP_COLORS, generate_html


DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600
MAX_EXPLORE_CHARS = 60_000


# [maint-note:94663b24e389]
# 维护导航：_env_flag；下列为本函数源码中的接口与路径索引。
# 输入 name：str。
# 输入 default：bool。
# 返回契约：bool。
# 分支条件：value is None。
# 返回路径：default。
# 返回路径：value.strip().casefold() not in {'0', 'false', 'no', 'off'}。
# 直接调用：os.environ.get。
# 直接调用：value.strip().casefold。
# 直接调用：value.strip。
# [/maint-note:94663b24e389]
def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


# [maint-note:448a58db5f84]
# 数据与行为导航：CodeGraphConfig。
# 字段 binary：str；初始表达式 'codegraph'。
# 字段 timeout_seconds：int；初始表达式 DEFAULT_TIMEOUT_SECONDS。
# 字段 no_download：bool；初始表达式 True。
# 字段 no_daemon：bool；初始表达式 True。
# [/maint-note:448a58db5f84]
@dataclass(frozen=True)
class CodeGraphConfig:
    binary: str = "codegraph"
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    no_download: bool = True
    no_daemon: bool = True

    # [maint-note:7d944cefe5eb]
    # 维护导航：from_environment；下列为本函数源码中的接口与路径索引。
    # 返回契约：'CodeGraphConfig'。
    # 返回路径：cls(binary=os.environ.get('CODE_AGENT_CODEGRAPH_BIN', 'codegraph'), timeout_seconds=max(1, min(timeout, MAX_TIMEOUT_SECONDS)), no_download=_env_flag('CODE_AGENT_CODEGRAPH_NO_DOWNLOAD', True), no_daemon=_env_flag('CODE_AGENT_CODEGRAPH_NO_DAE。
    # 直接调用：int。
    # 直接调用：os.environ.get。
    # 直接调用：str。
    # 直接调用：cls。
    # 直接调用：max。
    # 直接调用：min。
    # 直接调用：_env_flag。
    # [/maint-note:7d944cefe5eb]
    @classmethod
    def from_environment(cls) -> "CodeGraphConfig":
        timeout = int(
            os.environ.get(
                "CODE_AGENT_CODEGRAPH_TIMEOUT",
                str(DEFAULT_TIMEOUT_SECONDS),
            )
        )
        return cls(
            binary=os.environ.get("CODE_AGENT_CODEGRAPH_BIN", "codegraph"),
            timeout_seconds=max(1, min(timeout, MAX_TIMEOUT_SECONDS)),
            no_download=_env_flag("CODE_AGENT_CODEGRAPH_NO_DOWNLOAD", True),
            no_daemon=_env_flag("CODE_AGENT_CODEGRAPH_NO_DAEMON", True),
        )


# [maint-note:395c28cb67f3]
# 数据与行为导航：CodeGraphCommandResult。
# 字段 argv：list[str]。
# 字段 returncode：int。
# 字段 stdout：str。
# 字段 stderr：str。
# 字段 timed_out：bool；初始表达式 False。
# [/maint-note:395c28cb67f3]
@dataclass(frozen=True)
class CodeGraphCommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    # [maint-note:162e0def39fb]
    # 维护导航：output；下列为本函数源码中的接口与路径索引。
    # 返回契约：str。
    # 返回路径：self.stdout + ('\n' if self.stdout and self.stderr else '') + self.stderr。
    # 表达式分流：self.stdout and self.stderr；真值与假值分支分别计算，不同时执行。
    # [/maint-note:162e0def39fb]
    @property
    def output(self) -> str:
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr


# [maint-note:433a249065f4]
# 维护导航：normalize_codegraph_capabilities；下列为本函数源码中的接口与路径索引。
# 输入 value：dict[str, Any] | None。
# 返回契约：dict[str, Any]。
# 返回路径：{**source, 'codegraph': {'enabled': bool(raw.get('enabled', False)), 'auto_sync': bool(raw.get('auto_sync', True)), 'hide_workspace_search': bool(raw.get('hide_workspace_search', True))}}。
# 直接调用：isinstance。
# 直接调用：source.get。
# 直接调用：bool。
# 直接调用：raw.get。
# 表达式分流：isinstance(value, dict)；真值与假值分支分别计算，不同时执行。
# 表达式分流：isinstance(source.get('codegraph'), dict)；真值与假值分支分别计算，不同时执行。
# [/maint-note:433a249065f4]
def normalize_codegraph_capabilities(value: dict[str, Any] | None) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    raw = source.get("codegraph") if isinstance(source.get("codegraph"), dict) else {}
    return {
        **source,
        "codegraph": {
            "enabled": bool(raw.get("enabled", False)),
            "auto_sync": bool(raw.get("auto_sync", True)),
            "hide_workspace_search": bool(raw.get("hide_workspace_search", True)),
        },
    }


class CodeGraphClient:
    # [maint-note:928928bb8968]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 workspace：str | Path。
    # 输入 config：CodeGraphConfig | None；缺省表达式 None。
    # 返回契约：None。
    # 分支条件：not self.workspace.is_dir()。
    # 显式抛出：ValueError(f'workspace does not exist: {self.workspace}')。
    # 写入目标：self.workspace。
    # 写入目标：self.config。
    # 直接调用：Path(workspace).expanduser().resolve。
    # 直接调用：Path(workspace).expanduser。
    # 直接调用：Path。
    # 直接调用：self.workspace.is_dir。
    # 直接调用：ValueError。
    # 直接调用：CodeGraphConfig.from_environment。
    # [/maint-note:928928bb8968]
    def __init__(
        self,
        workspace: str | Path,
        config: CodeGraphConfig | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")
        self.config = config or CodeGraphConfig.from_environment()

    # [maint-note:63576f6e3b3b]
    # 维护导航：index_directory；下列为本函数源码中的接口与路径索引。
    # 返回契约：Path。
    # 返回路径：self.workspace / '.codegraph'。
    # [/maint-note:63576f6e3b3b]
    @property
    def index_directory(self) -> Path:
        return self.workspace / ".codegraph"

    # [maint-note:6390156cd54b]
    # 维护导航：database_path；下列为本函数源码中的接口与路径索引。
    # 返回契约：Path。
    # 返回路径：self.index_directory / 'codegraph.db'。
    # [/maint-note:6390156cd54b]
    @property
    def database_path(self) -> Path:
        return self.index_directory / "codegraph.db"

    # [maint-note:39c56180108c]
    # 维护导航：_resolve_binary；下列为本函数源码中的接口与路径索引。
    # 返回契约：str | None。
    # 分支条件：not configured。
    # 分支条件：candidate.is_absolute() or candidate.parent != Path('.')。
    # 返回路径：None。
    # 返回路径：str(candidate.resolve()) if candidate.is_file() else None。
    # 返回路径：shutil.which(configured)。
    # 直接调用：self.config.binary.strip。
    # 直接调用：Path(configured).expanduser。
    # 直接调用：Path。
    # 直接调用：candidate.is_absolute。
    # 直接调用：candidate.is_file。
    # 直接调用：str。
    # 直接调用：candidate.resolve。
    # 直接调用：shutil.which。
    # 表达式分流：candidate.is_file()；真值与假值分支分别计算，不同时执行。
    # [/maint-note:39c56180108c]
    def _resolve_binary(self) -> str | None:
        configured = self.config.binary.strip()
        if not configured:
            return None
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            return str(candidate.resolve()) if candidate.is_file() else None
        return shutil.which(configured)

    # [maint-note:f81d638411a4]
    # 维护导航：_environment；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, str]。
    # 分支条件：self.config.no_download。
    # 分支条件：self.config.no_daemon。
    # 返回路径：env。
    # 写入目标：env['CODEGRAPH_NO_DOWNLOAD']。
    # 写入目标：env['CODEGRAPH_NO_DAEMON']。
    # 直接调用：os.environ.copy。
    # 直接调用：env.setdefault。
    # [/maint-note:f81d638411a4]
    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.no_download:
            env["CODEGRAPH_NO_DOWNLOAD"] = "1"
        if self.config.no_daemon:
            env["CODEGRAPH_NO_DAEMON"] = "1"
        env.setdefault("NO_COLOR", "1")
        return env

    # [maint-note:1f0c932d5151]
    # 维护导航：run；下列为本函数源码中的接口与路径索引。
    # 输入 arguments：list[str]。
    # 输入 timeout_seconds：int | None；缺省表达式 None。
    # 返回契约：CodeGraphCommandResult。
    # 分支条件：executable is None。
    # 分支条件：os.name == 'nt' and Path(executable).suffix.casefold() in {'.cmd', '.bat'}。
    # 返回路径：CodeGraphCommandResult([Path(self.config.binary).name, *arguments], 127, '', 'CodeGraph executable was not found on PATH.')。
    # 返回路径：CodeGraphCommandResult([Path(self.config.binary).name, *arguments], 124, exc.stdout or '', exc.stderr or f'CodeGraph command timed out after {timeout}s.', True)。
    # 返回路径：CodeGraphCommandResult([Path(self.config.binary).name, *arguments], 126, '', str(exc))。
    # 返回路径：CodeGraphCommandResult([Path(self.config.binary).name, *arguments], completed.returncode, completed.stdout or '', completed.stderr or '')。
    # 捕获异常：subprocess.TimeoutExpired。
    # 捕获异常：OSError。
    # 直接调用：self._resolve_binary。
    # 直接调用：CodeGraphCommandResult。
    # 直接调用：Path。
    # 直接调用：Path(executable).suffix.casefold。
    # 直接调用：os.environ.get。
    # 直接调用：max。
    # 直接调用：min。
    # 直接调用：subprocess.run。
    # 直接调用：self._environment。
    # 直接调用：str。
    # [/maint-note:1f0c932d5151]
    def run(
        self,
        arguments: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> CodeGraphCommandResult:
        executable = self._resolve_binary()
        if executable is None:
            return CodeGraphCommandResult(
                [Path(self.config.binary).name, *arguments],
                127,
                "",
                "CodeGraph executable was not found on PATH.",
            )
        launch_argv = [executable, *arguments]
        if os.name == "nt" and Path(executable).suffix.casefold() in {".cmd", ".bat"}:
            command_processor = os.environ.get("COMSPEC", "cmd.exe")
            launch_argv = [command_processor, "/d", "/s", "/c", executable, *arguments]
        timeout = max(
            1,
            min(timeout_seconds or self.config.timeout_seconds, MAX_TIMEOUT_SECONDS),
        )
        try:
            completed = subprocess.run(
                launch_argv,
                cwd=self.workspace,
                env=self._environment(),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CodeGraphCommandResult(
                [Path(self.config.binary).name, *arguments],
                124,
                exc.stdout or "",
                exc.stderr or f"CodeGraph command timed out after {timeout}s.",
                True,
            )
        except OSError as exc:
            return CodeGraphCommandResult(
                [Path(self.config.binary).name, *arguments],
                126,
                "",
                str(exc),
            )
        return CodeGraphCommandResult(
            [Path(self.config.binary).name, *arguments],
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )

    # [maint-note:17a4721629b7]
    # 维护导航：status；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, Any]。
    # 分支条件：binary is None。
    # 分支条件：version.returncode != 0。
    # 分支条件：not initialized。
    # 返回路径：base。
    # 写入目标：base['message']。
    # 写入目标：base['available']。
    # 写入目标：base['version']。
    # 写入目标：base['ready']。
    # 写入目标：base['stale']。
    # 直接调用：self._resolve_binary。
    # 直接调用：self.database_path.is_file。
    # 直接调用：self.database_path.stat。
    # 直接调用：self.run。
    # 直接调用：min。
    # 直接调用：version.output.strip。
    # 直接调用：version.stdout.strip。
    # 直接调用：version.stdout.strip().splitlines。
    # 直接调用：status.output.strip。
    # 直接调用：output.casefold。
    # 表达式分流：initialized；真值与假值分支分别计算，不同时执行。
    # 表达式分流：version.stdout.strip()；真值与假值分支分别计算，不同时执行。
    # 表达式分流：base['ready']；真值与假值分支分别计算，不同时执行。
    # [/maint-note:17a4721629b7]
    def status(self) -> dict[str, Any]:
        binary = self._resolve_binary()
        initialized = self.database_path.is_file()
        index_size = self.database_path.stat().st_size if initialized else 0
        base = {
            "installed": binary is not None,
            "available": False,
            "initialized": initialized,
            "ready": False,
            "stale": False,
            "index_path": ".codegraph/codegraph.db" if initialized else None,
            "index_size_bytes": index_size,
            "version": "",
            "message": "",
        }
        if binary is None:
            base["message"] = "CodeGraph is not installed or is not available on PATH."
            return base

        version = self.run(["version"], timeout_seconds=min(self.config.timeout_seconds, 15))
        if version.returncode != 0:
            base["message"] = version.output.strip() or "CodeGraph could not start."
            return base
        base["available"] = True
        base["version"] = version.stdout.strip().splitlines()[0] if version.stdout.strip() else ""
        if not initialized:
            base["message"] = "CodeGraph is available, but this workspace is not initialized."
            return base

        status = self.run(["status", "."])
        output = status.output.strip()
        base["ready"] = status.returncode == 0
        base["stale"] = "pending sync" in output.casefold()
        base["message"] = output or (
            "CodeGraph is ready." if base["ready"] else "CodeGraph status failed."
        )
        return base

    # [maint-note:7ec0e3f798de]
    # 维护导航：init；下列为本函数源码中的接口与路径索引。
    # 返回契约：CodeGraphCommandResult。
    # 返回路径：self.run(['init', '.'], timeout_seconds=MAX_TIMEOUT_SECONDS)。
    # 直接调用：self.run。
    # [/maint-note:7ec0e3f798de]
    def init(self) -> CodeGraphCommandResult:
        return self.run(["init", "."], timeout_seconds=MAX_TIMEOUT_SECONDS)

    # [maint-note:350a933b80e6]
    # 维护导航：sync；下列为本函数源码中的接口与路径索引。
    # 返回契约：CodeGraphCommandResult。
    # 返回路径：self.run(['sync', '.'], timeout_seconds=MAX_TIMEOUT_SECONDS)。
    # 直接调用：self.run。
    # [/maint-note:350a933b80e6]
    def sync(self) -> CodeGraphCommandResult:
        return self.run(["sync", "."], timeout_seconds=MAX_TIMEOUT_SECONDS)

    # [maint-note:df1881abac00]
    # 维护导航：explore；下列为本函数源码中的接口与路径索引。
    # 输入 query：str。
    # 返回契约：CodeGraphCommandResult。
    # 分支条件：not cleaned。
    # 分支条件：len(cleaned) > 2000。
    # 返回路径：self.run(['explore', cleaned])。
    # 显式抛出：ValueError('query must not be empty')。
    # 显式抛出：ValueError('query must be at most 2000 characters')。
    # 直接调用：query.strip。
    # 直接调用：ValueError。
    # 直接调用：len。
    # 直接调用：self.run。
    # [/maint-note:df1881abac00]
    def explore(self, query: str) -> CodeGraphCommandResult:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        if len(cleaned) > 2_000:
            raise ValueError("query must be at most 2000 characters")
        return self.run(["explore", cleaned])

    # [maint-note:5133dee2ef07]
    # 维护导航：visualize；下列为本函数源码中的接口与路径索引。
    # 输入 output_path：str | Path。
    # 输入 keyword：str | None；缺省表达式 None。
    # 输入 depth：int；缺省表达式 2。
    # 输入 max_nodes：int；缺省表达式 150。
    # 返回契约：dict[str, Any]。
    # 分支条件：not self.database_path.is_file()。
    # 返回路径：{'node_count': len(nodes), 'edge_count': len(edges), 'keyword': keyword or '', 'output': str(output)}。
    # 显式抛出：FileNotFoundError('CodeGraph is not initialized for this workspace')。
    # 直接调用：self.database_path.is_file。
    # 直接调用：FileNotFoundError。
    # 直接调用：Path(output_path).expanduser().resolve。
    # 直接调用：Path(output_path).expanduser。
    # 直接调用：Path。
    # 直接调用：output.parent.mkdir。
    # 直接调用：_load_bounded_graph。
    # 直接调用：max。
    # 直接调用：min。
    # 直接调用：output.write_text。
    # 直接调用：_render_graph_html。
    # 直接调用：len。
    # 直接调用：str。
    # [/maint-note:5133dee2ef07]
    def visualize(
        self,
        output_path: str | Path,
        *,
        keyword: str | None = None,
        depth: int = 2,
        max_nodes: int = 150,
    ) -> dict[str, Any]:
        if not self.database_path.is_file():
            raise FileNotFoundError("CodeGraph is not initialized for this workspace")
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        nodes, edges = _load_bounded_graph(
            self.database_path,
            keyword=keyword,
            depth=max(0, min(depth, 5)),
            max_nodes=max(10, min(max_nodes, 500)),
        )
        output.write_text(_render_graph_html(nodes, edges, keyword), encoding="utf-8")
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "keyword": keyword or "",
            "output": str(output),
        }


# [maint-note:19e336e62266]
# 维护导航：_chunks；下列为本函数源码中的接口与路径索引。
# 输入 values：list[str]。
# 输入 size：int；缺省表达式 400。
# 返回契约：list[list[str]]。
# 返回路径：[values[index:index + size] for index in range(0, len(values), size)]。
# 直接调用：range。
# 直接调用：len。
# 集合派生：index 来自 range(0, len(values), size)。
# [/maint-note:19e336e62266]
def _chunks(values: list[str], size: int = 400) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


# [maint-note:7466600b70ca]
# 维护导航：_load_bounded_graph；下列为本函数源码中的接口与路径索引。
# 输入 database_path：Path。
# 输入 keyword：str | None。
# 输入 depth：int。
# 输入 max_nodes：int。
# 返回契约：tuple[list[dict[str, Any]], list[dict[str, Any]]]。
# 分支条件：keyword and keyword.strip()。
# 分支条件：not selected。
# 分支条件：not frontier or len(selected) >= max_nodes。
# 遍历边界：_ ← range(depth)。
# 遍历边界：chunk ← _chunks(sorted(frontier))。
# 遍历边界：row ← neighbors。
# 遍历边界：chunk ← _chunks(selected_list)。
# 遍历边界：chunk ← _chunks(sorted(selected_ids))。
# 返回路径：(nodes, edges)。
# 显式抛出：ValueError(f'No CodeGraph nodes matched: {keyword}')。
# 写入目标：connection.row_factory。
# 直接调用：sqlite3.connect。
# 直接调用：database_path.as_uri。
# 直接调用：connection.execute。
# 直接调用：keyword.strip。
# 直接调用：connection.execute('\n SELECT id FROM nodes\n WHERE name LIKE ? OR qualified_name LIKE ? OR file_path LIKE ?\n LIMIT 10\n ', (pattern, pattern, pattern)).fetchall。
# 直接调用：str。
# 直接调用：ValueError。
# 直接调用：set。
# 直接调用：range。
# 直接调用：len。
# 直接调用：_chunks。
# 直接调用：sorted。
# 直接调用：','.join。
# 直接调用：neighbors.extend。
# 直接调用：connection.execute(f'SELECT source, target FROM edges WHERE source IN ({placeholders}) OR target IN ({placeholders})', [*chunk, *chunk]).fetchall。
# 直接调用：next_frontier.update。
# 直接调用：selected.update。
# 直接调用：connection.execute('\n SELECT n.id\n FROM nodes n\n LEFT JOIN (\n SELECT node_id, COUNT(*) AS degree FROM (\n SELECT source AS node_id FROM edges\n UNION ALL\n SELECT target AS node_id FROM edges\n ) GROUP BY node_id\n ) d ON d.node_id=n.id。
# 直接调用：node_rows.extend。
# 直接调用：connection.execute(f'SELECT id, name, qualified_name, kind, file_path FROM nodes WHERE id IN ({placeholders})', chunk).fetchall。
# 直接调用：connection.execute(f'SELECT source, target, kind FROM edges WHERE source IN ({placeholders})', chunk).fetchall。
# 直接调用：edge_rows.extend。
# 直接调用：connection.close。
# 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
# 集合派生：row 来自 rows。
# 集合派生：_ 来自 chunk。
# 集合派生：row 来自 node_rows。
# 集合派生：row 来自 candidates；过滤 str(row['target']) in selected_ids。
# 集合派生：row 来自 edge_rows。
# [/maint-note:7466600b70ca]
def _load_bounded_graph(
    database_path: Path,
    *,
    keyword: str | None,
    depth: int,
    max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        if keyword and keyword.strip():
            pattern = f"%{keyword.strip()}%"
            rows = connection.execute(
                """
                SELECT id FROM nodes
                WHERE name LIKE ? OR qualified_name LIKE ? OR file_path LIKE ?
                LIMIT 10
                """,
                (pattern, pattern, pattern),
            ).fetchall()
            selected = {str(row["id"]) for row in rows}
            if not selected:
                raise ValueError(f"No CodeGraph nodes matched: {keyword}")
            frontier = set(selected)
            for _ in range(depth):
                if not frontier or len(selected) >= max_nodes:
                    break
                neighbors: list[sqlite3.Row] = []
                for chunk in _chunks(sorted(frontier)):
                    placeholders = ",".join("?" for _ in chunk)
                    neighbors.extend(
                        connection.execute(
                            f"SELECT source, target FROM edges WHERE source IN ({placeholders}) OR target IN ({placeholders})",
                            [*chunk, *chunk],
                        ).fetchall()
                    )
                next_frontier: set[str] = set()
                for row in neighbors:
                    next_frontier.update((str(row["source"]), str(row["target"])))
                next_frontier -= selected
                room = max_nodes - len(selected)
                frontier = set(sorted(next_frontier)[:room])
                selected.update(frontier)
        else:
            rows = connection.execute(
                """
                SELECT n.id
                FROM nodes n
                LEFT JOIN (
                    SELECT node_id, COUNT(*) AS degree FROM (
                        SELECT source AS node_id FROM edges
                        UNION ALL
                        SELECT target AS node_id FROM edges
                    ) GROUP BY node_id
                ) d ON d.node_id=n.id
                ORDER BY COALESCE(d.degree, 0) DESC, n.id
                LIMIT ?
                """,
                (max_nodes,),
            ).fetchall()
            selected = {str(row["id"]) for row in rows}

        node_rows: list[sqlite3.Row] = []
        edge_rows: list[sqlite3.Row] = []
        selected_list = sorted(selected)[:max_nodes]
        for chunk in _chunks(selected_list):
            placeholders = ",".join("?" for _ in chunk)
            node_rows.extend(
                connection.execute(
                    f"SELECT id, name, qualified_name, kind, file_path FROM nodes WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
            )
        selected_ids = {str(row["id"]) for row in node_rows}
        for chunk in _chunks(sorted(selected_ids)):
            placeholders = ",".join("?" for _ in chunk)
            candidates = connection.execute(
                f"SELECT source, target, kind FROM edges WHERE source IN ({placeholders})",
                chunk,
            ).fetchall()
            edge_rows.extend(
                row for row in candidates if str(row["target"]) in selected_ids
            )
    finally:
        connection.close()

    nodes = [
        {
            "id": str(row["id"]),
            "name": row["name"] or str(row["id"]),
            "qualified_name": row["qualified_name"] or "",
            "kind": row["kind"] or "unknown",
            "file_path": row["file_path"] or "",
        }
        for row in node_rows
    ]
    edges = [
        {
            "source": str(row["source"]),
            "target": str(row["target"]),
            "kind": row["kind"] or "relation",
        }
        for row in edge_rows
    ]
    return nodes, edges


# [maint-note:0759332276e5]
# 维护导航：_render_graph_html；下列为本函数源码中的接口与路径索引。
# 输入 nodes：list[dict[str, Any]]。
# 输入 edges：list[dict[str, Any]]。
# 输入 keyword：str | None。
# 返回契约：str。
# 返回路径：generate_html(vis_nodes, vis_edges, legend_data, stats, title=title)。
# 直接调用：_codegraph_to_vis。
# 直接调用：generate_html。
# 表达式分流：keyword；真值与假值分支分别计算，不同时执行。
# [/maint-note:0759332276e5]
def _render_graph_html(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    keyword: str | None,
) -> str:
    """Render the bounded CodeGraph with the shared Pyvis knowledge-graph UI."""
    vis_nodes, vis_edges, legend_data, stats = _codegraph_to_vis(nodes, edges)
    title = "CodeGraph" + (f" - {keyword}" if keyword else "")
    return generate_html(vis_nodes, vis_edges, legend_data, stats, title=title)


# [maint-note:f3cd8b87dd0d]
# 维护导航：_codegraph_group；下列为本函数源码中的接口与路径索引。
# 输入 kind：str。
# 返回契约：str。
# 返回路径：mapping.get(normalized, str(kind or 'Other') or 'Other')。
# 直接调用：str(kind or '').strip().casefold。
# 直接调用：str(kind or '').strip。
# 直接调用：str。
# 直接调用：mapping.get。
# [/maint-note:f3cd8b87dd0d]
def _codegraph_group(kind: str) -> str:
    normalized = str(kind or "").strip().casefold()
    mapping = {
        "file": "Module",
        "module": "Module",
        "package": "Module",
        "function": "Function",
        "func": "Function",
        "method": "Method",
        "constructor": "Constructor",
        "class": "Class",
        "struct": "Struct",
        "interface": "Interface",
        "field": "Field",
        "property": "Field",
        "variable": "Variable",
        "var": "Variable",
        "constant": "Variable",
        "enum": "Enum",
        "union": "Union",
        "import": "Import",
        "include": "Import",
    }
    return mapping.get(normalized, str(kind or "Other") or "Other")


# [maint-note:1c6683a50cc9]
# 维护导航：_color_for_group；下列为本函数源码中的接口与路径索引。
# 输入 group：str。
# 返回契约：str。
# 返回路径：GROUP_COLORS[index]。
# 直接调用：mapping.get。
# 直接调用：sum。
# 直接调用：ord。
# 直接调用：len。
# 集合派生：char 来自 group。
# [/maint-note:1c6683a50cc9]
def _color_for_group(group: str) -> str:
    mapping = {
        "Module": 0,
        "Function": 1,
        "Method": 1,
        "Class": 2,
        "Struct": 2,
        "Field": 3,
        "Variable": 3,
        "Enum": 4,
        "Union": 4,
        "Interface": 5,
        "Constructor": 6,
        "Import": 7,
        "Record": 8,
        "Other": 9,
    }
    index = mapping.get(group, sum(ord(char) for char in group) % len(GROUP_COLORS))
    return GROUP_COLORS[index]


# [maint-note:48cec99d095b]
# 维护导航：_codegraph_to_vis；下列为本函数源码中的接口与路径索引。
# 输入 nodes：list[dict[str, Any]]。
# 输入 edges：list[dict[str, Any]]。
# 返回契约：tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]。
# 分支条件：file_path。
# 遍历边界：edge ← edges。
# 遍历边界：node ← nodes。
# 返回路径：(vis_nodes, vis_edges, legend_data, stats)。
# 写入目标：degree[source]。
# 写入目标：degree[target]。
# 写入目标：group_counts[group]。
# 直接调用：str。
# 直接调用：degree.get。
# 直接调用：max。
# 直接调用：degree.values。
# 直接调用：node.get。
# 直接调用：_codegraph_group。
# 直接调用：group_counts.get。
# 直接调用：_color_for_group。
# 直接调用：title_parts.append。
# 直接调用：vis_nodes.append。
# 直接调用：'\n'.join。
# 直接调用：round。
# 直接调用：edge.get。
# 直接调用：enumerate。
# 直接调用：sorted。
# 直接调用：group_counts.items。
# 直接调用：len。
# 表达式分流：node_degree >= max_degree * 0.12；真值与假值分支分别计算，不同时执行。
# 表达式分流：index % 2 == 0；真值与假值分支分别计算，不同时执行。
# 集合派生：(index, edge) 来自 enumerate(edges)。
# 集合派生：(group, count) 来自 sorted(group_counts.items(), key=lambda item: -item[1])。
# [/maint-note:48cec99d095b]
def _codegraph_to_vis(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    degree: dict[str, int] = {}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    max_degree = max(degree.values(), default=1) or 1

    group_counts: dict[str, int] = {}
    vis_nodes: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node["id"])
        name = str(node.get("name") or node_id)
        qualified = str(node.get("qualified_name") or "")
        kind = str(node.get("kind") or "Other")
        file_path = str(node.get("file_path") or "")
        group = _codegraph_group(kind)
        group_counts[group] = group_counts.get(group, 0) + 1
        color = _color_for_group(group)
        node_degree = degree.get(node_id, 1)
        title_parts = [qualified or name, f"Kind: {kind}"]
        if file_path:
            title_parts.append(f"File: {file_path}")
        vis_nodes.append(
            {
                "id": node_id,
                "label": name[:120],
                "group": group,
                "source_file": file_path,
                "title": "\n".join(title_parts),
                "degree": node_degree,
                "color": {
                    "background": color,
                    "border": color,
                    "highlight": {"background": "#e5e7eb", "border": color},
                },
                "size": round(10 + 30 * (node_degree / max_degree), 1),
                "font": {"size": 12 if node_degree >= max_degree * 0.12 else 0, "color": "#111827"},
            }
        )

    vis_edges = [
        {
            "from": str(edge["source"]),
            "to": str(edge["target"]),
            "label": str(edge.get("kind") or "relation")[:24],
            "title": str(edge.get("kind") or "relation"),
            "dashes": False,
            "width": 1.5,
            "smooth": {
                "type": "curvedCW" if index % 2 == 0 else "curvedCCW",
                "roundness": 0.28,
            },
            "font": {
                "size": 10,
                "color": "#374151",
                "background": "#ffffff",
                "strokeWidth": 4,
                "strokeColor": "#ffffff",
                "align": "horizontal",
            },
            "color": {"color": "#6b7280", "opacity": 0.75, "highlight": "#111827"},
        }
        for index, edge in enumerate(edges)
    ]
    legend_data = [
        {
            "group": group,
            "label": group,
            "color": _color_for_group(group),
            "count": count,
        }
        for group, count in sorted(group_counts.items(), key=lambda item: -item[1])
    ]
    stats = f"{len(vis_nodes)} nodes &middot; {len(vis_edges)} edges &middot; {len(legend_data)} groups"
    return vis_nodes, vis_edges, legend_data, stats


# [maint-note:39d85f14e926]
# 维护导航：codegraph_artifact_name；下列为本函数源码中的接口与路径索引。
# 输入 thread_id：str。
# 返回契约：str。
# 返回路径：f'codegraph-view-{digest}.html'。
# 直接调用：hashlib.sha256(thread_id.encode('utf-8')).hexdigest。
# 直接调用：hashlib.sha256。
# 直接调用：thread_id.encode。
# [/maint-note:39d85f14e926]
def codegraph_artifact_name(thread_id: str) -> str:
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
    return f"codegraph-view-{digest}.html"


# [maint-note:5f451ad0f7d8]
# 维护导航：_command_failure；下列为本函数源码中的接口与路径索引。
# 输入 result：CodeGraphCommandResult。
# 输入 error_type：str。
# 返回契约：ToolResult。
# 返回路径：ToolResult('timeout' if result.timed_out else 'error', message, data={'stdout': result.stdout, 'stderr': result.stderr}, exit_code=result.returncode, error={'type': error_type, 'message': message})。
# 直接调用：result.output.strip。
# 直接调用：ToolResult。
# 表达式分流：result.timed_out；真值与假值分支分别计算，不同时执行。
# [/maint-note:5f451ad0f7d8]
def _command_failure(result: CodeGraphCommandResult, error_type: str) -> ToolResult:
    message = result.output.strip() or f"CodeGraph exited with {result.returncode}"
    return ToolResult(
        "timeout" if result.timed_out else "error",
        message,
        data={"stdout": result.stdout, "stderr": result.stderr},
        exit_code=result.returncode,
        error={"type": error_type, "message": message},
    )


# [maint-note:af1191936fbd]
# 维护导航：_status；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict[str, Any]。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(status['message'], data={'codegraph': status})。
# 直接调用：CodeGraphClient(context.workspace).status。
# 直接调用：CodeGraphClient。
# 直接调用：ToolResult.success。
# [/maint-note:af1191936fbd]
def _status(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    status = CodeGraphClient(context.workspace).status()
    return ToolResult.success(status["message"], data={"codegraph": status})


# [maint-note:957991993e26]
# 维护导航：_explore；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict[str, Any]。
# 返回契约：ToolResult。
# 分支条件：not status['ready']。
# 分支条件：result.returncode != 0。
# 返回路径：ToolResult.failure(status['message'], 'CodeGraphUnavailable')。
# 返回路径：_command_failure(result, 'CodeGraphExploreFailed')。
# 返回路径：ToolResult('success', content, data={'query': args['query'], 'content': content}, exit_code=result.returncode, truncated=len(original) > len(content))。
# 直接调用：CodeGraphClient。
# 直接调用：client.status。
# 直接调用：ToolResult.failure。
# 直接调用：client.explore。
# 直接调用：_command_failure。
# 直接调用：min。
# 直接调用：args.get。
# 直接调用：ToolResult。
# 直接调用：len。
# [/maint-note:957991993e26]
def _explore(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    client = CodeGraphClient(context.workspace)
    status = client.status()
    if not status["ready"]:
        return ToolResult.failure(status["message"], "CodeGraphUnavailable")
    result = client.explore(args["query"])
    if result.returncode != 0:
        return _command_failure(result, "CodeGraphExploreFailed")
    original = result.stdout or result.stderr
    max_chars = min(args.get("max_chars", 30_000), MAX_EXPLORE_CHARS)
    content = original[:max_chars]
    return ToolResult(
        "success",
        content,
        data={"query": args["query"], "content": content},
        exit_code=result.returncode,
        truncated=len(original) > len(content),
    )


# [maint-note:c8ba4ee029f5]
# 维护导航：_init；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict[str, Any]。
# 返回契约：ToolResult。
# 分支条件：result.returncode != 0。
# 返回路径：_command_failure(result, 'CodeGraphInitFailed')。
# 返回路径：ToolResult.success(result.output.strip() or 'CodeGraph initialized.', exit_code=0)。
# 直接调用：CodeGraphClient(context.workspace).init。
# 直接调用：CodeGraphClient。
# 直接调用：_command_failure。
# 直接调用：ToolResult.success。
# 直接调用：result.output.strip。
# [/maint-note:c8ba4ee029f5]
def _init(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    result = CodeGraphClient(context.workspace).init()
    if result.returncode != 0:
        return _command_failure(result, "CodeGraphInitFailed")
    return ToolResult.success(result.output.strip() or "CodeGraph initialized.", exit_code=0)


# [maint-note:01c123161f1a]
# 维护导航：_sync；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 _args：dict[str, Any]。
# 返回契约：ToolResult。
# 分支条件：result.returncode != 0。
# 返回路径：_command_failure(result, 'CodeGraphSyncFailed')。
# 返回路径：ToolResult.success(result.output.strip() or 'CodeGraph synchronized.', exit_code=0)。
# 直接调用：CodeGraphClient(context.workspace).sync。
# 直接调用：CodeGraphClient。
# 直接调用：_command_failure。
# 直接调用：ToolResult.success。
# 直接调用：result.output.strip。
# [/maint-note:01c123161f1a]
def _sync(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    result = CodeGraphClient(context.workspace).sync()
    if result.returncode != 0:
        return _command_failure(result, "CodeGraphSyncFailed")
    return ToolResult.success(result.output.strip() or "CodeGraph synchronized.", exit_code=0)


# [maint-note:fe4b82b4bd55]
# 维护导航：_visualize；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict[str, Any]。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(f"Generated CodeGraph visualization with {data['node_count']} nodes.", data={**data, 'output': 'codegraph-view.html'}, artifacts=[str(output)])。
# 直接调用：CodeGraphClient(context.workspace).visualize。
# 直接调用：CodeGraphClient。
# 直接调用：args.get。
# 直接调用：ToolResult.success。
# 直接调用：str。
# [/maint-note:fe4b82b4bd55]
def _visualize(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    output = context.artifact_root / "codegraph-view.html"
    data = CodeGraphClient(context.workspace).visualize(
        output,
        keyword=args.get("keyword") or None,
        depth=args.get("depth", 2),
        max_nodes=args.get("max_nodes", 150),
    )
    return ToolResult.success(
        f"Generated CodeGraph visualization with {data['node_count']} nodes.",
        data={**data, "output": "codegraph-view.html"},
        artifacts=[str(output)],
    )


# [maint-note:ce6b1a62ea08]
# 维护导航：_empty_schema；下列为本函数源码中的接口与路径索引。
# 返回契约：dict[str, Any]。
# 返回路径：{'type': 'object', 'properties': {}, 'additionalProperties': False}。
# [/maint-note:ce6b1a62ea08]
def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


# [maint-note:860168b6837d]
# 维护导航：codegraph_tool_specs；下列为本函数源码中的接口与路径索引。
# 返回契约：list[ToolSpec]。
# 返回路径：[ToolSpec('codegraph.status', 'Check whether CodeGraph is installed, initialized, current, and usable for this workspace.', _empty_schema(), RiskLevel.READ, _status), ToolSpec('codegraph.explore', "Read relevant symbols' line-numbered sourc。
# 直接调用：ToolSpec。
# 直接调用：_empty_schema。
# [/maint-note:860168b6837d]
def codegraph_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "codegraph.status",
            "Check whether CodeGraph is installed, initialized, current, and usable for this workspace.",
            _empty_schema(),
            RiskLevel.READ,
            _status,
        ),
        ToolSpec(
            "codegraph.explore",
            "Read relevant symbols' line-numbered source and their call paths from the workspace CodeGraph. Use this before workspace search for code structure, symbol lookup, callers, callees, and impact analysis.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            RiskLevel.READ,
            _explore,
            max_output_chars=MAX_EXPLORE_CHARS,
        ),
        ToolSpec(
            "codegraph.init",
            "Initialize and fully index CodeGraph for the current workspace. This writes only the workspace's .codegraph index and requires user approval.",
            _empty_schema(),
            RiskLevel.EXECUTE,
            _init,
            idempotent=False,
            parallel_safe=False,
            default_timeout_seconds=MAX_TIMEOUT_SECONDS,
        ),
        ToolSpec(
            "codegraph.sync",
            "Incrementally update the current workspace CodeGraph. This writes only the .codegraph index and requires user approval when requested by the model.",
            _empty_schema(),
            RiskLevel.EXECUTE,
            _sync,
            idempotent=False,
            parallel_safe=False,
            default_timeout_seconds=MAX_TIMEOUT_SECONDS,
        ),
        ToolSpec(
            "codegraph.visualize",
            "Generate a bounded interactive HTML visualization from the current workspace CodeGraph index.",
            {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "depth": {"type": "integer"},
                    "max_nodes": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            RiskLevel.READ,
            _visualize,
            parallel_safe=False,
        ),
    ]
