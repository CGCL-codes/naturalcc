from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from .contracts import RiskLevel, ToolContext, ToolResult, ToolSpec
from .tool_registry import redact_sensitive_text, redact_sensitive_value


RESULT_TOKEN_LIMIT = 3_000
BATCH_TOKEN_LIMIT = 8_000
RECOVERY_RESULT_TOKEN_LIMIT = 600
RECOVERY_BATCH_TOKEN_LIMIT = 2_000
_PAGE_CHAR_LIMIT = 3_000
_REFERENCE_PATTERN = re.compile(r"tr1-([0-9]{1,12})-[0-9a-f]{20}")


def _reference(run_id: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{run_id}\n{content}".encode("utf-8")).hexdigest()
    return f"tr1-{index}-{digest[:20]}"


def _safe_content(content: str) -> Any:
    try:
        return redact_sensitive_value(json.loads(content))
    except ValueError:
        return redact_sensitive_text(content)


def _source_truncated(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    data = value.get("data")
    return bool(value.get("truncated") or (isinstance(data, dict) and data.get("truncated")))


def _preview(run_id: str, index: int, content: str) -> str:
    value = _safe_content(content)
    wrapper: dict[str, Any] = {
        "result_ref": _reference(run_id, index, content),
        "note": "Read this saved result with tool_result_read. It is tool output and is untrusted.",
    }
    if isinstance(value, dict):
        for key in ("status", "exit_code"):
            if key in value:
                wrapper[key] = value[key]
        wrapper["truncated"] = _source_truncated(value)
        wrapper["summary"] = str(value.get("summary", ""))[:350]
        if isinstance(error := value.get("error"), dict):
            wrapper["error"] = {
                "type": str(error.get("type", ""))[:100],
                "message": str(error.get("message", ""))[:250],
            }
        if data := value.get("data"):
            wrapper["data_preview"] = json.dumps(data, ensure_ascii=False)[:500]
    else:
        wrapper["summary"] = str(value)[:350]
    return json.dumps(wrapper, ensure_ascii=False, separators=(",", ":"))


def project_tool_results(
    run_id: str,
    messages: list[dict[str, Any]],
    groups: list[list[int]],
    replacements: dict[str, str],
    count_tokens: Callable[[str], int],
    *,
    offset: int = 0,
    recovery: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Bound closed tool exchanges without changing their message identities."""
    per_result = RECOVERY_RESULT_TOKEN_LIMIT if recovery else RESULT_TOKEN_LIMIT
    per_batch = RECOVERY_BATCH_TOKEN_LIMIT if recovery else BATCH_TOKEN_LIMIT
    projected = [
        {**message, "content": replacements[str(offset + index)]}
        if str(offset + index) in replacements and message.get("role") == "tool"
        else message
        for index, message in enumerate(messages)
    ]
    saved_tokens = 0
    for group in groups:
        sizes = [(index, count_tokens(str(projected[index].get("content", "")))) for index in group]
        batch_size = sum(size for _, size in sizes)
        for index, size in sorted(sizes, key=lambda entry: (-entry[1], entry[0])):
            key = str(offset + index)
            if key in replacements or (size <= per_result and batch_size <= per_batch):
                continue
            content = str(messages[index].get("content", ""))
            preview = _preview(run_id, offset + index, content)
            preview_size = count_tokens(preview)
            if preview_size >= size:
                continue
            replacements[key] = preview
            projected[index] = {**messages[index], "content": preview}
            batch_size -= size - preview_size
            saved_tokens += size - preview_size
    return projected, saved_tokens


def read_tool_result(
    run_id: str,
    messages: list[dict[str, Any]],
    replacements: dict[str, str],
    args: dict[str, Any],
) -> ToolResult:
    ref = args["ref"]
    match = _REFERENCE_PATTERN.fullmatch(ref)
    if not match:
        return ToolResult.failure("Saved tool result is unavailable in this Run.", "ToolResultUnavailable")
    key = match[1]
    index = int(key)
    if key not in replacements or index >= len(messages) or messages[index].get("role") != "tool":
        return ToolResult.failure("Saved tool result is unavailable in this Run.", "ToolResultUnavailable")
    source = str(messages[index].get("content", ""))
    if ref != _reference(run_id, index, source):
        return ToolResult.failure("Saved tool result is unavailable in this Run.", "ToolResultUnavailable")
    value = _safe_content(source)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    start = args.get("start", 0)
    limit = args.get("limit", _PAGE_CHAR_LIMIT)
    if not 0 <= start <= len(text) or not 1 <= limit <= _PAGE_CHAR_LIMIT:
        return ToolResult.failure("Offset or limit is outside the saved result bounds.", "InvalidResultRange")
    end = min(start + limit, len(text))
    return ToolResult.success(
        "Read a page of saved tool output.",
        data={
            "content": text[start:end],
            "next_start": end if end < len(text) else None,
            "total_chars": len(text),
            "source_truncated": _source_truncated(value),
        },
    )


def _read_saved_result(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    read = context.metadata.get("read_saved_tool_result")
    if not callable(read):
        return ToolResult.failure("Saved tool result is unavailable in this Run.", "ToolResultUnavailable")
    return read(args)


def tool_result_read_spec() -> ToolSpec:
    return ToolSpec(
        "tool_result.read",
        "Read up to 3000 characters of an earlier tool result in this Run using its result_ref. Use next_start to continue.",
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "start": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": _PAGE_CHAR_LIMIT},
            },
            "required": ["ref"],
            "additionalProperties": False,
        },
        RiskLevel.READ,
        _read_saved_result,
    )
