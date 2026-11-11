from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from .contracts import AgentEvent


TOOL_RECOVERY_VERSION = "tool-argument-recovery-v1"
_MISSING_ARGUMENTS = re.compile(
    r"missing required arguments: ([A-Za-z_][A-Za-z_0-9]*(?:, [A-Za-z_][A-Za-z_0-9]*)*)"
)
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,120}")


@dataclass(frozen=True)
class ToolArgumentRecovery:
    tool: str
    arguments: tuple[str, ...]
    evidence: tuple[AgentEvent, ...]


def find_tool_argument_recoveries(events: list[AgentEvent]) -> list[ToolArgumentRecovery]:
    """Find repeated argument-validation errors followed by a corrected successful call."""
    started: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[tuple[str, ...], deque[AgentEvent]]] = {}
    recoveries: dict[tuple[str, tuple[str, ...]], ToolArgumentRecovery] = {}
    for event in events:
        if event.type == "tool.started":
            call = event.payload["tool_call"]
            started[call["id"]] = call
            continue
        if event.type != "tool.finished":
            continue
        call = started.pop(event.payload.get("tool_call_id"), None)
        tool = event.payload.get("tool")
        if not call or tool != call.get("name") or not isinstance(tool, str) or not _TOOL_NAME.fullmatch(tool):
            continue
        args = call.get("args")
        result = event.payload.get("result") or {}
        if not isinstance(args, dict):
            continue
        if result.get("status") == "success":
            for missing, recent in failures.pop(tool, {}).items():
                key = (tool, missing)
                if len(recent) == 2 and set(missing) <= args.keys() and key not in recoveries:
                    recoveries[key] = ToolArgumentRecovery(tool, missing, (*recent, event))
                    if len(recoveries) == 3:
                        return list(recoveries.values())
            continue
        error = result.get("error") or {}
        if result.get("status") != "error" or error.get("type") != "ValueError":
            continue
        match = _MISSING_ARGUMENTS.fullmatch(str(error.get("message") or ""))
        if not match:
            continue
        missing = tuple(sorted(set(match[1].split(", "))))
        if set(missing) & args.keys():
            continue
        by_signature = failures.setdefault(tool, {})
        by_signature.setdefault(missing, deque(maxlen=2)).append(event)
    return list(recoveries.values())
