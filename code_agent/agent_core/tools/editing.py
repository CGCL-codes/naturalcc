from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from ..contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .workspace import display_workspace_path, resolve_workspace_path


# [maint-note:09cf9e9d34ee]
# 维护导航：_snapshot_path；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 relative：str。
# 返回契约：Path。
# 返回路径：context.artifact_root / 'snapshots' / f'{digest}.json'。
# 直接调用：hashlib.sha256(relative.encode('utf-8')).hexdigest。
# 直接调用：hashlib.sha256。
# 直接调用：relative.encode。
# [/maint-note:09cf9e9d34ee]
def _snapshot_path(context: ToolContext, relative: str) -> Path:
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
    return context.artifact_root / "snapshots" / f"{digest}.json"


MAX_NEW_FILE_BYTES = 1_000_000


# [maint-note:8eb8cbeaf804]
# 维护导航：_write_snapshot_once；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 relative：str。
# 输入 content：str | None。
# 输入 kind：str；缺省表达式 'file'。
# 返回契约：Path。
# 分支条件：not snapshot_path.exists()。
# 返回路径：snapshot_path。
# 直接调用：_snapshot_path。
# 直接调用：snapshot_path.parent.mkdir。
# 直接调用：snapshot_path.exists。
# 直接调用：snapshot_path.write_text。
# 直接调用：json.dumps。
# [/maint-note:8eb8cbeaf804]
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


# [maint-note:775b217ae946]
# 维护导航：_unified_diff；下列为本函数源码中的接口与路径索引。
# 输入 relative：str。
# 输入 before：str。
# 输入 after：str。
# 返回契约：str。
# 返回路径：''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=f'a/{relative}', tofile=f'b/{relative}'))。
# 直接调用：''.join。
# 直接调用：difflib.unified_diff。
# 直接调用：before.splitlines。
# 直接调用：after.splitlines。
# [/maint-note:775b217ae946]
def _unified_diff(relative: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


# [maint-note:049a36568f09]
# 维护导航：_apply_exact_patch；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not path.is_file()。
# 分支条件：occurrences != 1。
# 返回路径：ToolResult.success(f'Updated {relative}', artifacts=[str(snapshot_path)], changed_files=[relative], data={'path': relative, 'snapshot': str(snapshot_path), 'diff': _unified_diff(relative, content, updated)})。
# 显式抛出：ValueError('patch target must be a file')。
# 显式抛出：ValueError(f'old_text must match exactly once; matched {occurrences} times')。
# 直接调用：resolve_workspace_path。
# 直接调用：path.is_file。
# 直接调用：ValueError。
# 直接调用：path.read_text。
# 直接调用：content.count。
# 直接调用：display_workspace_path。
# 直接调用：_write_snapshot_once。
# 直接调用：content.replace。
# 直接调用：path.write_text。
# 直接调用：ToolResult.success。
# 直接调用：str。
# 直接调用：_unified_diff。
# [/maint-note:049a36568f09]
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


# [maint-note:c58a3ce35f3a]
# 维护导航：_create_file；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：target.exists()。
# 分支条件：not target.parent.is_dir()。
# 分支条件：len(encoded) > MAX_NEW_FILE_BYTES。
# 返回路径：ToolResult.success(f'Created {relative}', artifacts=[str(snapshot_path)], changed_files=[relative], data={'path': relative, 'snapshot': str(snapshot_path), 'created': True, 'bytes': len(encoded), 'diff': _unified_diff(relative, '', content)。
# 显式抛出：FileExistsError(f"path already exists: {args['path']}")。
# 显式抛出：FileNotFoundError(f'parent directory does not exist: {display_workspace_path(context, target.parent)}')。
# 显式抛出：ValueError(f'new file exceeds {MAX_NEW_FILE_BYTES} UTF-8 bytes')。
# 直接调用：resolve_workspace_path。
# 直接调用：target.exists。
# 直接调用：FileExistsError。
# 直接调用：target.parent.is_dir。
# 直接调用：FileNotFoundError。
# 直接调用：display_workspace_path。
# 直接调用：content.encode。
# 直接调用：len。
# 直接调用：ValueError。
# 直接调用：_write_snapshot_once。
# 直接调用：target.open。
# 直接调用：handle.write。
# 直接调用：ToolResult.success。
# 直接调用：str。
# 直接调用：_unified_diff。
# 资源作用域：target.open('x', encoding='utf-8', newline='')；退出该 with 时调用对应上下文清理。
# [/maint-note:c58a3ce35f3a]
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


# [maint-note:eba955925e9e]
# 维护导航：_create_directory；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：target.exists()。
# 分支条件：not target.is_dir()。
# 分支条件：not cursor.is_dir()。
# 分支条件：not parents and len(missing) > 1。
# 遍历边界：not cursor.exists()。
# 返回路径：ToolResult.success(f'Directory already exists: {relative}', data={'path': relative, 'created': False, 'created_directories': []})。
# 返回路径：ToolResult.success(f'Created directory {display_workspace_path(context, target)}', artifacts=snapshots, data={'path': display_workspace_path(context, target), 'created': True, 'created_directories': relative_paths, 'snapshots': snapshots})。
# 显式抛出：FileExistsError(f"path already exists and is not a directory: {args['path']}")。
# 显式抛出：NotADirectoryError(f'existing parent is not a directory: {display_workspace_path(context, cursor)}')。
# 显式抛出：FileNotFoundError(f'parent directory does not exist: {display_workspace_path(context, target.parent)}')。
# 直接调用：resolve_workspace_path。
# 直接调用：target.exists。
# 直接调用：target.is_dir。
# 直接调用：FileExistsError。
# 直接调用：display_workspace_path。
# 直接调用：ToolResult.success。
# 直接调用：bool。
# 直接调用：args.get。
# 直接调用：cursor.exists。
# 直接调用：missing.append。
# 直接调用：cursor.is_dir。
# 直接调用：NotADirectoryError。
# 直接调用：len。
# 直接调用：FileNotFoundError。
# 直接调用：list。
# 直接调用：reversed。
# 直接调用：str。
# 直接调用：_write_snapshot_once。
# 直接调用：target.mkdir。
# 集合派生：path 来自 creation_order。
# 集合派生：relative 来自 relative_paths。
# [/maint-note:eba955925e9e]
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


# [maint-note:a4444dda955c]
# 维护导航：_restore_snapshot；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not snapshot_path.is_file()。
# 分支条件：payload.get('path') != relative。
# 分支条件：payload.get('existed', True) is False。
# 分支条件：not target.exists()。
# 分支条件：payload.get('kind', 'file') == 'directory'。
# 分支条件：not target.is_dir()。
# 分支条件：not target.is_file()。
# 返回路径：ToolResult.success(f'{relative} is already absent', artifacts=[str(snapshot_path)], data={'path': relative, 'restored_absence': True})。
# 返回路径：ToolResult.success(f'Removed newly created directory {relative}', artifacts=[str(snapshot_path)], data={'path': relative, 'restored_absence': True, 'removed_directories': [relative]})。
# 返回路径：ToolResult.success(f'Removed newly created {relative}', artifacts=[str(snapshot_path)], changed_files=[relative], data={'path': relative, 'restored_absence': True, 'diff': _unified_diff(relative, current, '')})。
# 返回路径：ToolResult.success(f'Restored {relative}', artifacts=[str(snapshot_path)], changed_files=[relative], data={'path': relative})。
# 显式抛出：FileNotFoundError(f'no snapshot exists for {relative}')。
# 显式抛出：ValueError('snapshot path does not match requested target')。
# 显式抛出：ValueError('snapshot target is no longer a directory')。
# 显式抛出：OSError(f'cannot restore directory absence while it is not empty: {relative}')。
# 显式抛出：ValueError('snapshot target is no longer a file')。
# 捕获异常：OSError。
# 直接调用：resolve_workspace_path。
# 直接调用：display_workspace_path。
# 直接调用：_snapshot_path。
# 直接调用：snapshot_path.is_file。
# 直接调用：FileNotFoundError。
# 直接调用：json.loads。
# 直接调用：snapshot_path.read_text。
# 直接调用：payload.get。
# 直接调用：ValueError。
# 直接调用：target.exists。
# 直接调用：ToolResult.success。
# 直接调用：str。
# 直接调用：target.is_dir。
# 直接调用：target.rmdir。
# 直接调用：OSError。
# 直接调用：target.is_file。
# 直接调用：target.read_text。
# 直接调用：target.unlink。
# 直接调用：_unified_diff。
# 直接调用：target.parent.mkdir。
# 直接调用：target.write_text。
# [/maint-note:a4444dda955c]
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


# [maint-note:f55f9c85a5ba]
# 维护导航：_local_aider_runtime；下列为本函数源码中的接口与路径索引。
# 输入 metadata：dict。
# 返回契约：tuple[str, str] | None。
# 分支条件：not isinstance(runtime_config, dict) or runtime_config.get('provider') != 'ollama'。
# 返回路径：None。
# 返回路径：(f'ollama_chat/{config.model}', config.base_url)。
# 直接调用：metadata.get。
# 直接调用：isinstance。
# 直接调用：runtime_config.get。
# 直接调用：RuntimeModelConfig.from_dict。
# [/maint-note:f55f9c85a5ba]
def _local_aider_runtime(metadata: dict) -> tuple[str, str] | None:
    runtime_config = metadata.get("runtime_model_config")
    if not isinstance(runtime_config, dict) or runtime_config.get("provider") != "ollama":
        return None
    from ..model_gateway import RuntimeModelConfig

    config = RuntimeModelConfig.from_dict(runtime_config)
    return f"ollama_chat/{config.model}", config.base_url


# [maint-note:314014c277b7]
# 维护导航：_aider_edit；下列为本函数源码中的接口与路径索引。
# 输入 context：ToolContext。
# 输入 args：dict。
# 返回契约：ToolResult。
# 分支条件：not path.is_file()。
# 分支条件：delta。
# 分支条件：current != original。
# 分支条件：not success。
# 遍历边界：value ← args['target_files']。
# 遍历边界：cumulative ← run_aider_stream(target_files=target_files, user_instruction=args['instruction'], model=local_runtime[0] if local_runtime else context.metadata.get('model', 'deepseek/deepseek-chat'), api_key=None if local_runtime else context.metadata.get(。
# 遍历边界：(relative, (path, original)) ← before.items()。
# 返回路径：ToolResult('error', message, data={'log_tail': message, 'diff': '\n'.join(diffs)}, artifacts=artifacts, changed_files=changed_files, error={'type': 'AiderError', 'message': message})。
# 返回路径：ToolResult.success('Aider edit completed', artifacts=artifacts, changed_files=changed_files, data={'log_tail': text[-2000:], 'diff': '\n'.join(diffs)})。
# 显式抛出：ValueError(f'Aider target must be a file: {value}')。
# 写入目标：before[relative]。
# 直接调用：resolve_workspace_path。
# 直接调用：path.is_file。
# 直接调用：ValueError。
# 直接调用：display_workspace_path。
# 直接调用：path.read_text。
# 直接调用：target_files.append。
# 直接调用：str。
# 直接调用：snapshots.append。
# 直接调用：_write_snapshot_once。
# 直接调用：_local_aider_runtime。
# 直接调用：run_aider_stream。
# 直接调用：context.metadata.get。
# 直接调用：cumulative.startswith。
# 直接调用：len。
# 直接调用：logs.append。
# 直接调用：''.join。
# 直接调用：artifact.write_text。
# 直接调用：before.items。
# 直接调用：changed_files.append。
# 直接调用：diffs.append。
# 直接调用：_unified_diff。
# 直接调用：ToolResult。
# 直接调用：'\n'.join。
# 直接调用：ToolResult.success。
# 表达式分流：local_runtime；真值与假值分支分别计算，不同时执行。
# 表达式分流：cumulative.startswith(previous)；真值与假值分支分别计算，不同时执行。
# 表达式分流：path.is_file()；真值与假值分支分别计算，不同时执行。
# [/maint-note:314014c277b7]
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
    local_runtime = _local_aider_runtime(context.metadata)
    for cumulative in run_aider_stream(
        target_files=target_files,
        user_instruction=args["instruction"],
        model=local_runtime[0] if local_runtime else context.metadata.get("model", "deepseek/deepseek-chat"),
        api_key=None if local_runtime else context.metadata.get("api_key"),
        base_url=local_runtime[1] if local_runtime else None,
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


# [maint-note:4c95a1e16c6f]
# 维护导航：editing_tool_specs；下列为本函数源码中的接口与路径索引。
# 返回契约：list[ToolSpec]。
# 返回路径：[ToolSpec('workspace.create_directory', 'Create a directory inside the locked workspace. Set parents=true to create a missing parent chain. Existing directories are an idempotent success.', {'type': 'object', 'properties': {'path': {'type':。
# 直接调用：ToolSpec。
# [/maint-note:4c95a1e16c6f]
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
