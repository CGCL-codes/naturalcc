from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec


IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
SENSITIVE_DIRS = {".git", ".ssh", ".aws", ".gnupg"}
SENSITIVE_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "secrets.json",
}


# [maint-note:d96a45d9456e]
# 维护导航：_is_sensitive；下列为本函数源码中的接口与路径索引。
# 输入 path：Path。
# 返回契约：bool。
# 返回路径：bool(lowered_parts & SENSITIVE_DIRS) or name in SENSITIVE_NAMES or name.startswith('.env.') or (path.suffix.casefold() in {'.pem', '.p12', '.pfx', '.key'})。
# 直接调用：part.casefold。
# 直接调用：path.name.casefold。
# 直接调用：bool。
# 直接调用：name.startswith。
# 直接调用：path.suffix.casefold。
# 集合派生：part 来自 path.parts。
# [/maint-note:d96a45d9456e]
def _is_sensitive(path: Path) -> bool:
    lowered_parts = {part.casefold() for part in path.parts}
    name = path.name.casefold()
    return (
        bool(lowered_parts & SENSITIVE_DIRS)
        or name in SENSITIVE_NAMES
        or name.startswith(".env.")
        or path.suffix.casefold() in {".pem", ".p12", ".pfx", ".key"}
    )


# [maint-note:7a0fc4ff0662]
# 维护导航：resolve_workspace_path；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 value：str。
# 输入 must_exist：bool；缺省表达式 False。
# 返回契约：Path。
# 分支条件：candidate.is_absolute()。
# 分支条件：resolved == authorized。
# 分支条件：authorized.is_dir()。
# 分支条件：relative is None。
# 分支条件：_is_sensitive(relative)。
# 分支条件：must_exist and (not resolved.exists())。
# 遍历边界：authorized ← context.authorized_paths。
# 返回路径：resolved。
# 显式抛出：ValueError(f'path escapes workspace and explicit authorization: {value}')。
# 显式抛出：ValueError(f'sensitive path is not available to tools: {value}')。
# 显式抛出：FileNotFoundError(f'path does not exist: {value}')。
# 捕获异常：ValueError。
# 直接调用：Path(value).expanduser。
# 直接调用：Path。
# 直接调用：candidate.is_absolute。
# 直接调用：candidate.resolve。
# 直接调用：(context.workspace / candidate).resolve。
# 直接调用：resolved.relative_to。
# 直接调用：authorized.is_dir。
# 直接调用：ValueError。
# 直接调用：_is_sensitive。
# 直接调用：resolved.exists。
# 直接调用：FileNotFoundError。
# [/maint-note:7a0fc4ff0662]
def resolve_workspace_path(context: ToolContext, value: str, *, must_exist: bool = False) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (context.workspace / candidate).resolve()
    try:
        relative = resolved.relative_to(context.workspace)
    except ValueError:
        relative = None
        for authorized in context.authorized_paths:
            if resolved == authorized:
                relative = Path(resolved.name)
                break
            if authorized.is_dir():
                try:
                    relative = resolved.relative_to(authorized)
                    break
                except ValueError:
                    continue
        if relative is None:
            raise ValueError(f"path escapes workspace and explicit authorization: {value}")
    if _is_sensitive(relative):
        raise ValueError(f"sensitive path is not available to tools: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"path does not exist: {value}")
    return resolved


# [maint-note:91553c57cc5c]
# 维护导航：display_workspace_path；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 path：Path。
# 返回契约：str。
# 返回路径：resolved.relative_to(context.workspace).as_posix()。
# 返回路径：resolved.as_posix()。
# 捕获异常：ValueError。
# 直接调用：path.resolve。
# 直接调用：resolved.relative_to(context.workspace).as_posix。
# 直接调用：resolved.relative_to。
# 直接调用：resolved.as_posix。
# [/maint-note:91553c57cc5c]
def display_workspace_path(context: ToolContext, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(context.workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


# [maint-note:6fbe1de8a90d]
# 维护导航：_list；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not root.is_dir()。
# 分支条件：any((part in IGNORED_DIRS or part.startswith('.agent-') for part in relative_parts)) or _is_sensitive(path.relative_to(root))。
# 分支条件：len(entries) >= max_entries。
# 遍历边界：path ← sorted(root.rglob('*'), key=lambda item: item.as_posix().casefold())。
# 返回路径：ToolResult.success(f'Listed {len(entries)} entries', data={'entries': entries})。
# 显式抛出：ValueError('workspace.list path must be a directory')。
# 直接调用：resolve_workspace_path。
# 直接调用：args.get。
# 直接调用：root.is_dir。
# 直接调用：ValueError。
# 直接调用：sorted。
# 直接调用：root.rglob。
# 直接调用：path.relative_to。
# 直接调用：any。
# 直接调用：part.startswith。
# 直接调用：_is_sensitive。
# 直接调用：entries.append。
# 直接调用：display_workspace_path。
# 直接调用：path.is_dir。
# 直接调用：path.is_file。
# 直接调用：path.stat。
# 直接调用：len。
# 直接调用：ToolResult.success。
# 集合派生：part 来自 relative_parts。
# 表达式分流：path.is_dir()；真值与假值分支分别计算，不同时执行。
# 表达式分流：path.is_file()；真值与假值分支分别计算，不同时执行。
# [/maint-note:6fbe1de8a90d]
def _list(context: ToolContext, args: dict) -> ToolResult:
    root = resolve_workspace_path(context, args.get("path", "."), must_exist=True)
    if not root.is_dir():
        raise ValueError("workspace.list path must be a directory")
    max_entries = args.get("max_entries", 200)
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        relative_parts = path.relative_to(root).parts
        if (
            any(part in IGNORED_DIRS or part.startswith(".agent-") for part in relative_parts)
            or _is_sensitive(path.relative_to(root))
        ):
            continue
        entries.append(
            {
                "path": display_workspace_path(context, path),
                "type": "directory" if path.is_dir() else "file",
                "size": path.stat().st_size if path.is_file() else None,
            }
        )
        if len(entries) >= max_entries:
            break
    return ToolResult.success(f"Listed {len(entries)} entries", data={"entries": entries})


# [maint-note:34e5f4e54d2b]
# 维护导航：_read；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not path.is_file()。
# 返回路径：ToolResult.success(content, data={'path': display_workspace_path(context, path), 'content': content, 'chars': len(content), 'truncated': len(original) > len(content)})。
# 显式抛出：ValueError('workspace.read path must be a file')。
# 直接调用：resolve_workspace_path。
# 直接调用：path.is_file。
# 直接调用：ValueError。
# 直接调用：args.get。
# 直接调用：path.read_text。
# 直接调用：ToolResult.success。
# 直接调用：display_workspace_path。
# 直接调用：len。
# [/maint-note:34e5f4e54d2b]
def _read(context: ToolContext, args: dict) -> ToolResult:
    path = resolve_workspace_path(context, args["path"], must_exist=True)
    if not path.is_file():
        raise ValueError("workspace.read path must be a file")
    max_chars = args.get("max_chars", 20_000)
    original = path.read_text(encoding="utf-8", errors="replace")
    content = original[:max_chars]
    return ToolResult.success(
        content,
        data={
            "path": display_workspace_path(context, path),
            "content": content,
            "chars": len(content),
            "truncated": len(original) > len(content),
        },
    )


# [maint-note:114a14e007ef]
# 维护导航：_iter_text_files；下列为本函数源码中的接口与路径索引。
# 输入 root：Path。
# 返回契约：Iterable[Path]。
# 分支条件：root.is_file()。
# 分支条件：not _is_sensitive(path.relative_to(root))。
# 遍历边界：(dirpath, dirnames, filenames) ← os.walk(root)。
# 遍历边界：filename ← filenames。
# 返回路径：无返回值。
# 写入目标：dirnames[:]。
# 直接调用：root.is_file。
# 直接调用：os.walk。
# 直接调用：name.startswith。
# 直接调用：name.casefold。
# 直接调用：Path。
# 直接调用：_is_sensitive。
# 直接调用：path.relative_to。
# 生成器交付：root。
# 集合派生：name 来自 dirnames；过滤 name not in IGNORED_DIRS and (not name.startswith('.agent-')) and (name.casefold() not in SENSITIVE_DIRS)。
# 生成器交付：path。
# [/maint-note:114a14e007ef]
def _iter_text_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORED_DIRS
            and not name.startswith(".agent-")
            and name.casefold() not in SENSITIVE_DIRS
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
            if not _is_sensitive(path.relative_to(root)):
                yield path


# [maint-note:af8dbb84c5b7]
# 维护导航：_search；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：query.casefold() in line.casefold()。
# 分支条件：len(matches) >= max_matches。
# 遍历边界：path ← _iter_text_files(root)。
# 遍历边界：(line_number, line) ← enumerate(lines, 1)。
# 返回路径：ToolResult.success(f'Found {len(matches)} matches', data={'matches': matches})。
# 捕获异常：OSError。
# 直接调用：resolve_workspace_path。
# 直接调用：args.get。
# 直接调用：_iter_text_files。
# 直接调用：path.read_text(encoding='utf-8', errors='replace').splitlines。
# 直接调用：path.read_text。
# 直接调用：enumerate。
# 直接调用：query.casefold。
# 直接调用：line.casefold。
# 直接调用：matches.append。
# 直接调用：display_workspace_path。
# 直接调用：len。
# 直接调用：ToolResult.success。
# [/maint-note:af8dbb84c5b7]
def _search(context: ToolContext, args: dict) -> ToolResult:
    query = args["query"]
    root = resolve_workspace_path(context, args.get("path", "."), must_exist=True)
    max_matches = args.get("max_matches", 100)
    matches = []
    for path in _iter_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, 1):
            if query.casefold() in line.casefold():
                matches.append(
                    {
                        "path": display_workspace_path(context, path),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) >= max_matches:
                    return ToolResult.success(f"Found {len(matches)} matches", data={"matches": matches})
    return ToolResult.success(f"Found {len(matches)} matches", data={"matches": matches})


# [maint-note:7ab87b50f2d1]
# 维护导航：_stat；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 返回路径：ToolResult.success(f"{display_path} is {('a directory' if path.is_dir() else 'a file')}", data={'path': display_path, 'type': 'directory' if path.is_dir() else 'file', 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})。
# 直接调用：resolve_workspace_path。
# 直接调用：path.stat。
# 直接调用：display_workspace_path。
# 直接调用：ToolResult.success。
# 直接调用：path.is_dir。
# 表达式分流：path.is_dir()；真值与假值分支分别计算，不同时执行。
# [/maint-note:7ab87b50f2d1]
def _stat(context: ToolContext, args: dict) -> ToolResult:
    path = resolve_workspace_path(context, args["path"], must_exist=True)
    stat = path.stat()
    display_path = display_workspace_path(context, path)
    return ToolResult.success(
        f"{display_path} is {'a directory' if path.is_dir() else 'a file'}",
        data={
            "path": display_path,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
    )


# [maint-note:ffa1b8d3eb4f]
# 维护导航：_schema；下列为本函数源码中的接口与路径索引。
# 输入 properties：dict。
# 输入 required：list[str] | None；缺省表达式 None。
# 返回契约：dict。
# 返回路径：{'type': 'object', 'properties': properties, 'required': required or [], 'additionalProperties': False}。
# [/maint-note:ffa1b8d3eb4f]
def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


# [maint-note:1a9d53d990ac]
# 维护导航：workspace_tool_specs；下列为本函数源码中的接口与路径索引。
# 返回契约：list[ToolSpec]。
# 返回路径：[ToolSpec('workspace.list', 'List files and directories inside the locked workspace.', _schema({'path': {'type': 'string'}, 'max_entries': {'type': 'integer'}}), RiskLevel.READ, _list), ToolSpec('workspace.read', 'Read a UTF-8 text file ins。
# 直接调用：ToolSpec。
# 直接调用：_schema。
# [/maint-note:1a9d53d990ac]
def workspace_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "workspace.list",
            "List files and directories inside the locked workspace.",
            _schema(
                {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer"},
                }
            ),
            RiskLevel.READ,
            _list,
        ),
        ToolSpec(
            "workspace.read",
            "Read a UTF-8 text file inside the locked workspace.",
            _schema(
                {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                ["path"],
            ),
            RiskLevel.READ,
            _read,
        ),
        ToolSpec(
            "workspace.search",
            "Search text files inside the locked workspace.",
            _schema(
                {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "max_matches": {"type": "integer"},
                },
                ["query"],
            ),
            RiskLevel.READ,
            _search,
        ),
        ToolSpec(
            "workspace.stat",
            "Inspect a file or directory inside the locked workspace.",
            _schema({"path": {"type": "string"}}, ["path"]),
            RiskLevel.READ,
            _stat,
        ),
    ]
