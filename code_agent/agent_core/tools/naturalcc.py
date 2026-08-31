from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec


_CACHE: dict[tuple[str, str, str], Any] = {}
_CACHE_LOCK = Lock()


def _source_revision(workspace: Path) -> str:
    digest = hashlib.sha256()
    extensions = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".java"}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        stat = path.stat()
        digest.update(path.relative_to(workspace).as_posix().encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:20]


def _load_agent(context: ToolContext, language: str):
    from code_agent.completion_prompt_agent import CompletionPromptAgent

    revision = _source_revision(context.workspace)
    key = (str(context.workspace), language, revision)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached, revision
    agent = CompletionPromptAgent()
    agent.load_project(str(context.workspace), language)
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE[key] = agent
    return agent, revision


def _parse(context: ToolContext, args: dict) -> ToolResult:
    language = args.get("language", "c")
    agent, revision = _load_agent(context, language)
    parsed = agent.parse_res or {}
    return ToolResult.success(
        f"NaturalCC parsed {len(parsed)} source units",
        data={"language": language, "source_units": len(parsed), "revision": revision},
    )


def _walk_matches(value: Any, symbol: str, path: list[str], matches: list[dict], limit: int) -> None:
    if len(matches) >= limit:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = [*path, str(key)]
            if str(key) == symbol:
                matches.append({"path": ".".join(next_path), "value": str(child)[:1000]})
            _walk_matches(child, symbol, next_path, matches, limit)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_matches(child, symbol, [*path, str(index)], matches, limit)


def _symbol_search(context: ToolContext, args: dict) -> ToolResult:
    language = args.get("language", "c")
    symbol = args["symbol"]
    limit = args.get("limit", 20)
    agent, revision = _load_agent(context, language)
    matches: list[dict] = []
    _walk_matches(agent.parse_res or {}, symbol, [], matches, limit)
    return ToolResult.success(
        f"NaturalCC found {len(matches)} matches for {symbol}",
        data={"symbol": symbol, "matches": matches, "revision": revision},
    )


def naturalcc_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "naturalcc.parse",
            "Parse C/C++ or Java sources in the locked workspace with NaturalCC.",
            {
                "type": "object",
                "properties": {"language": {"type": "string", "enum": ["c", "java"]}},
                "additionalProperties": False,
            },
            RiskLevel.READ,
            _parse,
            default_timeout_seconds=300,
        ),
        ToolSpec(
            "naturalcc.symbol_search",
            "Search the versioned NaturalCC project graph for a symbol.",
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "language": {"type": "string", "enum": ["c", "java"]},
                    "limit": {"type": "integer"},
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
            RiskLevel.READ,
            _symbol_search,
            default_timeout_seconds=300,
        ),
    ]
