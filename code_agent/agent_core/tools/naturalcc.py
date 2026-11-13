from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec


_CACHE: dict[tuple[str, str, str], Any] = {}
_CACHE_LOCK = Lock()


# [maint-note:6f090c9b97b0]
# 维护导航：_source_revision；下列为本函数源码中的接口与路径索引。
# 输入 workspace：Path。
# 返回契约：str。
# 分支条件：not path.is_file() or path.suffix.lower() not in extensions。
# 遍历边界：path ← sorted(workspace.rglob('*'))。
# 返回路径：digest.hexdigest()[:20]。
# 直接调用：hashlib.sha256。
# 直接调用：sorted。
# 直接调用：workspace.rglob。
# 直接调用：path.is_file。
# 直接调用：path.suffix.lower。
# 直接调用：path.stat。
# 直接调用：digest.update。
# 直接调用：path.relative_to(workspace).as_posix().encode。
# 直接调用：path.relative_to(workspace).as_posix。
# 直接调用：path.relative_to。
# 直接调用：str(stat.st_size).encode。
# 直接调用：str。
# 直接调用：str(stat.st_mtime_ns).encode。
# 直接调用：digest.hexdigest。
# [/maint-note:6f090c9b97b0]
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


# [maint-note:0c8aa33da37b]
# 维护导航：_load_agent；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 language：str。
# 分支条件：cached is not None。
# 返回路径：(cached, revision)。
# 返回路径：(agent, revision)。
# 写入目标：_CACHE[key]。
# 直接调用：_source_revision。
# 直接调用：str。
# 直接调用：_CACHE.get。
# 直接调用：CompletionPromptAgent。
# 直接调用：agent.load_project。
# 直接调用：_CACHE.clear。
# 资源作用域：_CACHE_LOCK；退出该 with 时调用对应上下文清理。
# [/maint-note:0c8aa33da37b]
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


# [maint-note:d27b39d9465f]
# 维护导航：_parse；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(f'NaturalCC parsed {len(parsed)} source units', data={'language': language, 'source_units': len(parsed), 'revision': revision})。
# 直接调用：args.get。
# 直接调用：_load_agent。
# 直接调用：ToolResult.success。
# 直接调用：len。
# [/maint-note:d27b39d9465f]
def _parse(context: ToolContext, args: dict) -> ToolResult:
    language = args.get("language", "c")
    agent, revision = _load_agent(context, language)
    parsed = agent.parse_res or {}
    return ToolResult.success(
        f"NaturalCC parsed {len(parsed)} source units",
        data={"language": language, "source_units": len(parsed), "revision": revision},
    )


# [maint-note:0d6594a4e8b9]
# 维护导航：_walk_matches；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 symbol：str。
# 输入 path：list[str]。
# 输入 matches：list[dict]。
# 输入 limit：int。
# 返回契约：None。
# 分支条件：len(matches) >= limit。
# 分支条件：isinstance(value, dict)。
# 分支条件：str(key) == symbol。
# 分支条件：isinstance(value, list)。
# 遍历边界：(key, child) ← value.items()。
# 遍历边界：(index, child) ← enumerate(value)。
# 返回路径：无返回值。
# 直接调用：len。
# 直接调用：isinstance。
# 直接调用：value.items。
# 直接调用：str。
# 直接调用：matches.append。
# 直接调用：'.'.join。
# 直接调用：_walk_matches。
# 直接调用：enumerate。
# [/maint-note:0d6594a4e8b9]
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


# [maint-note:5dd8bd051f62]
# 维护导航：_symbol_search；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(f'NaturalCC found {len(matches)} matches for {symbol}', data={'symbol': symbol, 'matches': matches, 'revision': revision})。
# 直接调用：args.get。
# 直接调用：_load_agent。
# 直接调用：_walk_matches。
# 直接调用：ToolResult.success。
# 直接调用：len。
# [/maint-note:5dd8bd051f62]
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


# [maint-note:23087c0c75ad]
# 维护导航：naturalcc_tool_specs；下列为本函数源码中的接口与路径索引。
# 返回契约：list[ToolSpec]。
# 返回路径：[ToolSpec('naturalcc.parse', 'Parse C/C++ or Java sources in the locked workspace with NaturalCC.', {'type': 'object', 'properties': {'language': {'type': 'string', 'enum': ['c', 'java']}}, 'additionalProperties': False}, RiskLevel.READ, _p。
# 直接调用：ToolSpec。
# [/maint-note:23087c0c75ad]
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
