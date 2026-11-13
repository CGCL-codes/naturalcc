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


# [maint-note:ef6c711a9462]
# 数据与行为导航：ToolArgumentRecovery。
# 字段 tool：str。
# 字段 arguments：tuple[str, ...]。
# 字段 evidence：tuple[AgentEvent, ...]。
# [/maint-note:ef6c711a9462]
@dataclass(frozen=True)
class ToolArgumentRecovery:
    tool: str
    arguments: tuple[str, ...]
    evidence: tuple[AgentEvent, ...]


# [maint-note:421405076076]
# 维护导航：find_tool_argument_recoveries；下列为本函数源码中的接口与路径索引。
# 输入 events：list[AgentEvent]。
# 返回契约：list[ToolArgumentRecovery]。
# 分支条件：event.type == 'tool.started'。
# 分支条件：event.type != 'tool.finished'。
# 分支条件：not call or tool != call.get('name') or (not isinstance(tool, str)) or (not _TOOL_NAME.fullmatch(tool))。
# 分支条件：not isinstance(args, dict)。
# 分支条件：result.get('status') == 'success'。
# 分支条件：len(recent) == 2 and set(missing) <= args.keys() and (key not in recoveries)。
# 分支条件：len(recoveries) == 3。
# 分支条件：result.get('status') != 'error' or error.get('type') != 'ValueError'。
# 分支条件：not match。
# 分支条件：set(missing) & args.keys()。
# 遍历边界：event ← events。
# 遍历边界：(missing, recent) ← failures.pop(tool, {}).items()。
# 返回路径：list(recoveries.values())。
# 写入目标：started[call['id']]。
# 写入目标：recoveries[key]。
# 直接调用：started.pop。
# 直接调用：event.payload.get。
# 直接调用：call.get。
# 直接调用：isinstance。
# 直接调用：_TOOL_NAME.fullmatch。
# 直接调用：result.get。
# 直接调用：failures.pop(tool, {}).items。
# 直接调用：failures.pop。
# 直接调用：len。
# 直接调用：set。
# 直接调用：args.keys。
# 直接调用：ToolArgumentRecovery。
# 直接调用：list。
# 直接调用：recoveries.values。
# 直接调用：error.get。
# 直接调用：_MISSING_ARGUMENTS.fullmatch。
# 直接调用：str。
# 直接调用：tuple。
# 直接调用：sorted。
# 直接调用：match[1].split。
# 直接调用：failures.setdefault。
# 直接调用：by_signature.setdefault(missing, deque(maxlen=2)).append。
# 直接调用：by_signature.setdefault。
# 直接调用：deque。
# [/maint-note:421405076076]
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
