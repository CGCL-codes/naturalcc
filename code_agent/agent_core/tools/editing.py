from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .workspace import display_workspace_path, resolve_workspace_path


def _snapshot_path(context: ToolContext, relative: str) -> Path:
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
    return context.artifact_root / "snapshots" / f"{digest}.json"


MAX_NEW_FILE_BYTES = 1_000_000


def _write_snapshot_once(
    context: ToolContext,
    relative: str,
    content: str | None,
    *,
    kind: str = "file",
) -> Path:
    snapshot_path = _snapshot_path(context, relative)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        snapshot_path.write_text(
            json.dumps(
                {
                    "path": relative,
                    "kind": kind,
                    "existed": content is not None,
                    "content": content or "",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return snapshot_path


def _unified_diff(relative: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def _apply_exact_patch(context: ToolContext, args: dict) -> ToolResult:
    path = resolve_workspace_path(context, args["path"], must_exist=True)
    if not path.is_file():
        raise ValueError("patch target must be a file")
    old_text = args["old_text"]
    new_text = args["new_text"]
    content = path.read_text(encoding="utf-8")
    occurrences = content.count(old_text)
    if occurrences != 1:
        raise ValueError(f"old_text must match exactly once; matched {occurrences} times")

    relative = display_workspace_path(context, path)
    snapshot_path = _write_snapshot_once(context, relative, content)
    updated = content.replace(old_text, new_text, 1)
    path.write_text(updated, encoding="utf-8")
    return ToolResult.success(
        f"Updated {relative}",
        artifacts=[str(snapshot_path)],
        changed_files=[relative],
        data={
            "path": relative,
            "snapshot": str(snapshot_path),
            "diff": _unified_diff(relative, content, updated),
        },
    )


def _create_file(context: ToolContext, args: dict) -> ToolResult:
    target = resolve_workspace_path(context, args["path"])
    if target.exists():
        raise FileExistsError(f"path already exists: {args['path']}")
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"parent directory does not exist: {display_workspace_path(context, target.parent)}"
        )
    content = args["content"]
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_NEW_FILE_BYTES:
        raise ValueError(
            f"new file exceeds {MAX_NEW_FILE_BYTES} UTF-8 bytes"
        )
    relative = display_workspace_path(context, target)
    snapshot_path = _write_snapshot_once(context, relative, None)
    with target.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return ToolResult.success(
        f"Created {relative}",
        artifacts=[str(snapshot_path)],
        changed_files=[relative],
        data={
            "path": relative,
            "snapshot": str(snapshot_path),
            "created": True,
            "bytes": len(encoded),
            "diff": _unified_diff(relative, "", content),
        },
    )


def _create_directory(context: ToolContext, args: dict) -> ToolResult:
    target = resolve_workspace_path(context, args["path"])
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"path already exists and is not a directory: {args['path']}")
        relative = display_workspace_path(context, target)
        return ToolResult.success(
            f"Directory already exists: {relative}",
            data={
                "path": relative,
                "created": False,
                "created_directories": [],
            },
        )

    parents = bool(args.get("parents", False))
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(
            f"existing parent is not a directory: {display_workspace_path(context, cursor)}"
        )
    if not parents and len(missing) > 1:
        raise FileNotFoundError(
            f"parent directory does not exist: {display_workspace_path(context, target.parent)}"
        )

    creation_order = list(reversed(missing))
    relative_paths = [display_workspace_path(context, path) for path in creation_order]
    snapshots = [
        str(
            _write_snapshot_once(
                context,
                relative,
                None,
                kind="directory",
            )
        )
        for relative in relative_paths
    ]
    target.mkdir(parents=parents, exist_ok=False)
    return ToolResult.success(
        f"Created directory {display_workspace_path(context, target)}",
        artifacts=snapshots,
        data={
            "path": display_workspace_path(context, target),
            "created": True,
            "created_directories": relative_paths,
            "snapshots": snapshots,
        },
    )


def _restore_snapshot(context: ToolContext, args: dict) -> ToolResult:
    target = resolve_workspace_path(context, args["path"])
    relative = display_workspace_path(context, target)
    snapshot_path = _snapshot_path(context, relative)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"no snapshot exists for {relative}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if payload.get("path") != relative:
        raise ValueError("snapshot path does not match requested target")
    if payload.get("existed", True) is False:
        if not target.exists():
            return ToolResult.success(
                f"{relative} is already absent",
                artifacts=[str(snapshot_path)],
                data={"path": relative, "restored_absence": True},
            )
        if payload.get("kind", "file") == "directory":
            if not target.is_dir():
                raise ValueError("snapshot target is no longer a directory")
            try:
                target.rmdir()
            except OSError as exc:
                raise OSError(
                    f"cannot restore directory absence while it is not empty: {relative}"
                ) from exc
            return ToolResult.success(
                f"Removed newly created directory {relative}",
                artifacts=[str(snapshot_path)],
                data={
                    "path": relative,
                    "restored_absence": True,
                    "removed_directories": [relative],
                },
            )
        if not target.is_file():
            raise ValueError("snapshot target is no longer a file")
        current = target.read_text(encoding="utf-8", errors="replace")
        target.unlink()
        return ToolResult.success(
            f"Removed newly created {relative}",
            artifacts=[str(snapshot_path)],
            changed_files=[relative],
            data={
                "path": relative,
                "restored_absence": True,
                "diff": _unified_diff(relative, current, ""),
            },
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload["content"], encoding="utf-8")
    return ToolResult.success(
        f"Restored {relative}",
        artifacts=[str(snapshot_path)],
        changed_files=[relative],
        data={"path": relative},
    )


def _aider_edit(context: ToolContext, args: dict) -> ToolResult:
    from code_agent.aider_runner import run_aider_stream

    target_files: list[str] = []
    before: dict[str, tuple[Path, str]] = {}
    snapshots: list[str] = []
    for value in args["target_files"]:
        path = resolve_workspace_path(context, value, must_exist=True)
        if not path.is_file():
            raise ValueError(f"Aider target must be a file: {value}")
        relative = display_workspace_path(context, path)
        content = path.read_text(encoding="utf-8", errors="replace")
        before[relative] = (path, content)
        target_files.append(str(path))
        snapshots.append(str(_write_snapshot_once(context, relative, content)))
    logs: list[str] = []
    previous = ""
    for cumulative in run_aider_stream(
        target_files=target_files,
        user_instruction=args["instruction"],
        model=context.metadata.get("model", "deepseek/deepseek-chat"),
        api_key=context.metadata.get("api_key"),
        project_dir=str(context.workspace),
    ):
        delta = cumulative[len(previous) :] if cumulative.startswith(previous) else cumulative
        if delta:
            logs.append(delta)
        previous = cumulative
    text = "".join(logs)
    success = "任务圆满完成" in text and "❌" not in text
    artifact = context.artifact_root / "aider.log"
    artifact.write_text(text, encoding="utf-8")
    changed_files: list[str] = []
    diffs: list[str] = []
    for relative, (path, original) in before.items():
        current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if current != original:
            changed_files.append(relative)
            diffs.append(_unified_diff(relative, original, current))
    artifacts = [str(artifact), *snapshots]
    if not success:
        message = text[-2000:] or "Aider failed"
        return ToolResult(
            "error",
            message,
            data={"log_tail": message, "diff": "\n".join(diffs)},
            artifacts=artifacts,
            changed_files=changed_files,
            error={"type": "AiderError", "message": message},
        )
    return ToolResult.success(
        "Aider edit completed",
        artifacts=artifacts,
        changed_files=changed_files,
        data={"log_tail": text[-2000:], "diff": "\n".join(diffs)},
    )


def editing_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            "workspace.create_directory",
            (
                "Create a directory inside the locked workspace. Set parents=true to create "
                "a missing parent chain. Existing directories are an idempotent success."
            ),
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "parents": {"type": "boolean"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            RiskLevel.WRITE,
            _create_directory,
            idempotent=True,
            parallel_safe=False,
        ),
        ToolSpec(
            "workspace.create_file",
            (
                "Create one new UTF-8 text file inside the locked workspace. "
                "The parent directory must already exist and existing files are never overwritten."
            ),
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            RiskLevel.WRITE,
            _create_file,
            idempotent=False,
            parallel_safe=False,
        ),
        ToolSpec(
            "workspace.apply_patch",
            "Replace one exact text occurrence in a workspace file and create a recoverable snapshot.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            RiskLevel.WRITE,
            _apply_exact_patch,
            idempotent=False,
            parallel_safe=False,
        ),
        ToolSpec(
            "aider.edit",
            "Delegate an approved multi-file code edit to Aider inside the locked workspace.",
            {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "target_files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["instruction", "target_files"],
                "additionalProperties": False,
            },
            RiskLevel.WRITE,
            _aider_edit,
            idempotent=False,
            parallel_safe=False,
            default_timeout_seconds=900,
        ),
        ToolSpec(
            "workspace.restore_snapshot",
            "Restore one workspace path to its pre-run file or directory state.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            RiskLevel.WRITE,
            _restore_snapshot,
            idempotent=True,
            parallel_safe=False,
        ),
    ]
