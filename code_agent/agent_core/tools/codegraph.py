from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec


DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600
MAX_EXPLORE_CHARS = 60_000


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class CodeGraphConfig:
    binary: str = "codegraph"
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    no_download: bool = True
    no_daemon: bool = True

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


@dataclass(frozen=True)
class CodeGraphCommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr


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
    def __init__(
        self,
        workspace: str | Path,
        config: CodeGraphConfig | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")
        self.config = config or CodeGraphConfig.from_environment()

    @property
    def index_directory(self) -> Path:
        return self.workspace / ".codegraph"

    @property
    def database_path(self) -> Path:
        return self.index_directory / "codegraph.db"

    def _resolve_binary(self) -> str | None:
        configured = self.config.binary.strip()
        if not configured:
            return None
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            return str(candidate.resolve()) if candidate.is_file() else None
        return shutil.which(configured)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.no_download:
            env["CODEGRAPH_NO_DOWNLOAD"] = "1"
        if self.config.no_daemon:
            env["CODEGRAPH_NO_DAEMON"] = "1"
        env.setdefault("NO_COLOR", "1")
        return env

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

    def init(self) -> CodeGraphCommandResult:
        return self.run(["init", "."], timeout_seconds=MAX_TIMEOUT_SECONDS)

    def sync(self) -> CodeGraphCommandResult:
        return self.run(["sync", "."], timeout_seconds=MAX_TIMEOUT_SECONDS)

    def explore(self, query: str) -> CodeGraphCommandResult:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        if len(cleaned) > 2_000:
            raise ValueError("query must be at most 2000 characters")
        return self.run(["explore", cleaned])

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


def _chunks(values: list[str], size: int = 400) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


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


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _render_graph_html(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    keyword: str | None,
) -> str:
    title = "CodeGraph" + (f" - {html.escape(keyword)}" if keyword else "")
    payload = _safe_json({"nodes": nodes, "edges": edges})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; overflow: hidden; background: #171717; color: #ececec; }}
    header {{ height: 52px; display: flex; align-items: center; gap: 18px; padding: 0 18px; border-bottom: 1px solid #343434; background: #202020; }}
    header strong {{ font-size: 14px; }}
    header span {{ color: #aaa; font-size: 12px; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; height: calc(100vh - 52px); }}
    canvas {{ width: 100%; height: 100%; cursor: grab; }}
    aside {{ padding: 16px; overflow: auto; border-left: 1px solid #343434; background: #202020; }}
    aside h2 {{ margin: 0 0 12px; font-size: 14px; }}
    aside dl {{ display: grid; grid-template-columns: 82px 1fr; gap: 8px; margin: 0; font-size: 12px; }}
    aside dt {{ color: #929292; }}
    aside dd {{ margin: 0; overflow-wrap: anywhere; }}
    .empty {{ color: #929292; font-size: 12px; line-height: 1.5; }}
  </style>
</head>
<body>
  <header><strong>{title}</strong><span>{len(nodes)} nodes</span><span>{len(edges)} edges</span></header>
  <main><canvas id="graph"></canvas><aside id="details"><p class="empty">Select a node to inspect its symbol and file.</p></aside></main>
  <script>
    const graph = {payload};
    const canvas = document.getElementById('graph');
    const ctx = canvas.getContext('2d');
    const details = document.getElementById('details');
    const palette = ['#72a7d7','#d97757','#72b88c','#d4a84f','#b48bd0','#d77c94','#86b9b0'];
    const colorFor = kind => palette[Math.abs([...kind].reduce((a,c)=>a+c.charCodeAt(0),0)) % palette.length];
    const byId = new Map(graph.nodes.map((node, index) => [node.id, {{...node, x: Math.cos(index*2.399)*Math.sqrt(index+1)*34, y: Math.sin(index*2.399)*Math.sqrt(index+1)*34}}]));
    let scale = 1, offsetX = 0, offsetY = 0, dragging = false, lastX = 0, lastY = 0;
    function resize() {{ const r=canvas.getBoundingClientRect(); canvas.width=Math.max(1,r.width*devicePixelRatio); canvas.height=Math.max(1,r.height*devicePixelRatio); draw(); }}
    function project(n) {{ return [canvas.width/2 + (n.x+offsetX)*scale*devicePixelRatio, canvas.height/2 + (n.y+offsetY)*scale*devicePixelRatio]; }}
    function draw() {{
      ctx.clearRect(0,0,canvas.width,canvas.height); ctx.lineWidth=devicePixelRatio; ctx.font=`${{11*devicePixelRatio}}px Inter,system-ui`;
      ctx.strokeStyle='#4a4a4a';
      for (const edge of graph.edges) {{ const a=byId.get(edge.source), b=byId.get(edge.target); if(!a||!b) continue; const [ax,ay]=project(a), [bx,by]=project(b); ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke(); }}
      for (const node of byId.values()) {{ const [x,y]=project(node); const radius=5.5*devicePixelRatio; ctx.fillStyle=colorFor(node.kind); ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill(); if(scale>0.7) {{ ctx.fillStyle='#ddd'; ctx.fillText(node.name.slice(0,28),x+9*devicePixelRatio,y+4*devicePixelRatio); }} }}
    }}
    function hit(x,y) {{ let best=null, distance=Infinity; for(const node of byId.values()) {{ const [nx,ny]=project(node); const d=Math.hypot(nx-x*devicePixelRatio,ny-y*devicePixelRatio); if(d<14*devicePixelRatio&&d<distance) {{best=node;distance=d;}} }} return best; }}
    function escapeText(value) {{ const el=document.createElement('span'); el.textContent=value||''; return el.innerHTML; }}
    canvas.addEventListener('click', e => {{ const node=hit(e.offsetX,e.offsetY); if(!node) return; details.innerHTML=`<h2>${{escapeText(node.name)}}</h2><dl><dt>Kind</dt><dd>${{escapeText(node.kind)}}</dd><dt>Qualified</dt><dd>${{escapeText(node.qualified_name)}}</dd><dt>File</dt><dd>${{escapeText(node.file_path)}}</dd></dl>`; }});
    canvas.addEventListener('pointerdown', e => {{ dragging=true; lastX=e.clientX; lastY=e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor='grabbing'; }});
    canvas.addEventListener('pointermove', e => {{ if(!dragging)return; offsetX+=(e.clientX-lastX)/scale; offsetY+=(e.clientY-lastY)/scale; lastX=e.clientX; lastY=e.clientY; draw(); }});
    canvas.addEventListener('pointerup', e => {{ dragging=false; canvas.releasePointerCapture(e.pointerId); canvas.style.cursor='grab'; }});
    canvas.addEventListener('wheel', e => {{ e.preventDefault(); scale=Math.max(.25,Math.min(4,scale*Math.exp(-e.deltaY*.001))); draw(); }}, {{passive:false}});
    addEventListener('resize',resize); resize();
  </script>
</body>
</html>"""


def codegraph_artifact_name(thread_id: str) -> str:
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
    return f"codegraph-view-{digest}.html"


def _command_failure(result: CodeGraphCommandResult, error_type: str) -> ToolResult:
    message = result.output.strip() or f"CodeGraph exited with {result.returncode}"
    return ToolResult(
        "timeout" if result.timed_out else "error",
        message,
        data={"stdout": result.stdout, "stderr": result.stderr},
        exit_code=result.returncode,
        error={"type": error_type, "message": message},
    )


def _status(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    status = CodeGraphClient(context.workspace).status()
    return ToolResult.success(status["message"], data={"codegraph": status})


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


def _init(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    result = CodeGraphClient(context.workspace).init()
    if result.returncode != 0:
        return _command_failure(result, "CodeGraphInitFailed")
    return ToolResult.success(result.output.strip() or "CodeGraph initialized.", exit_code=0)


def _sync(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
    result = CodeGraphClient(context.workspace).sync()
    if result.returncode != 0:
        return _command_failure(result, "CodeGraphSyncFailed")
    return ToolResult.success(result.output.strip() or "CodeGraph synchronized.", exit_code=0)


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


def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


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
