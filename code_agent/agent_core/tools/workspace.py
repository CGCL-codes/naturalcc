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


def _is_sensitive(path: Path) -> bool:
    lowered_parts = {part.casefold() for part in path.parts}
    name = path.name.casefold()
    return (
        bool(lowered_parts & SENSITIVE_DIRS)
        or name in SENSITIVE_NAMES
        or name.startswith(".env.")
        or path.suffix.casefold() in {".pem", ".p12", ".pfx", ".key"}
    )


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


def display_workspace_path(context: ToolContext, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(context.workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


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


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


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
