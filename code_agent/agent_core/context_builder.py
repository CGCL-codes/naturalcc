from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from typing import Any

from .model_gateway import DeepSeekRequestSerializer
from .token_budget import (
    ContextHardLimitExceeded,
    DeepSeekModelProfile,
    DeepSeekTokenCounter,
    TokenBreakdown,
)


@dataclass(frozen=True)
class ContextPlan:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    breakdown: TokenBreakdown
    system_tokens: int
    working_state_tokens: int
    checkpoint_tokens: int
    recent_history_tokens: int
    memory_tokens: int
    pinned_memory_tokens: int
    retrieved_memory_tokens: int
    tool_schema_tokens: int
    estimated_input_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int
    requires_compaction: bool
    compactable_from: int | None
    compactable_to: int | None
    recent_tail_from: int
    retained_message_indexes: list[int]


@dataclass(frozen=True)
class _ExchangeGroup:
    start: int
    end: int
    messages: list[dict[str, Any]]
    closed: bool = True


def _format_memory_items(memories: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for memory in memories:
        attributes = {
            "id": memory.get("id", ""),
            "scope": memory.get("scope", ""),
            "kind": memory.get("kind", ""),
            "verification": memory.get("verification", ""),
        }
        rendered_attributes = " ".join(
            f'{name}="{escape(str(value), quote=True)}"'
            for name, value in attributes.items()
        )
        subject = escape(str(memory.get("subject", "")))
        content = escape(str(memory.get("content", "")))
        rows.append(
            f'<memory {rendered_attributes} untrusted="true">'
            f"<subject>{subject}</subject><content>{content}</content></memory>"
        )
    return "\n".join(rows)


def _format_runtime_authorization(value: dict[str, Any]) -> str:
    workspace = escape(str(value.get("workspace", "")))
    target_files = "".join(
        f"<path>{escape(str(path))}</path>"
        for path in value.get("target_files", [])
    )
    authorized_paths = "".join(
        f"<path>{escape(str(path))}</path>"
        for path in value.get("authorized_paths", [])
    )
    return (
        f"<workspace>{workspace}</workspace>"
        f"<target_files>{target_files}</target_files>"
        '<authorized_paths description="user-authorized external paths">'
        f"{authorized_paths}</authorized_paths>"
    )


class ContextPlanner:
    def __init__(
        self,
        counter: DeepSeekTokenCounter,
        serializer: DeepSeekRequestSerializer,
        profile: DeepSeekModelProfile,
    ) -> None:
        self.counter = counter
        self.serializer = serializer
        self.profile = profile

    def plan(
        self,
        *,
        system_rules: str,
        goal: str,
        messages: list[dict[str, Any]],
        active_state: dict[str, Any],
        memories: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        checkpoint: dict[str, Any] | None = None,
        reserved_output_tokens: int | None = None,
        pinned_memories: list[dict[str, Any]] | None = None,
        runtime_authorization: dict[str, Any] | None = None,
    ) -> ContextPlan:
        return self._build_plan(
            system_rules=system_rules,
            goal=goal,
            messages=messages,
            active_state=active_state,
            memories=memories,
            pinned_memories=list(pinned_memories or []),
            tools=tools,
            checkpoint=checkpoint,
            reserved_output_tokens=reserved_output_tokens,
            runtime_authorization=runtime_authorization or {},
        )

    def _build_plan(
        self,
        *,
        system_rules: str,
        goal: str,
        messages: list[dict[str, Any]],
        active_state: dict[str, Any],
        memories: list[dict[str, Any]],
        pinned_memories: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        checkpoint: dict[str, Any] | None,
        reserved_output_tokens: int | None,
        runtime_authorization: dict[str, Any],
    ) -> ContextPlan:
        retained_pinned_memories = list(pinned_memories)
        retained_memories = list(memories)
        system_message, category_text = self._system_message(
            system_rules,
            goal,
            active_state,
            retained_pinned_memories,
            retained_memories,
            checkpoint,
            runtime_authorization,
        )

        while retained_pinned_memories or retained_memories:
            protected = self.counter.count_request(
                [system_message],
                tools,
                self.serializer,
                self.profile,
                reserved_output_tokens=reserved_output_tokens,
            )
            if protected.total_with_reserve <= protected.context_window_tokens:
                break
            if retained_memories:
                retained_memories.pop()
            else:
                retained_pinned_memories.pop()
            system_message, category_text = self._system_message(
                system_rules,
                goal,
                active_state,
                retained_pinned_memories,
                retained_memories,
                checkpoint,
                runtime_authorization,
            )

        self.counter.assert_fits(
            [system_message],
            tools,
            self.serializer,
            self.profile,
            reserved_output_tokens=reserved_output_tokens,
        )

        full_messages = [system_message, *messages]
        full_breakdown = self.counter.count_request(
            full_messages,
            tools,
            self.serializer,
            self.profile,
            reserved_output_tokens=reserved_output_tokens,
        )
        trigger_tokens = int(
            self.profile.context_window_tokens
            * self.profile.compaction_trigger_ratio
        )
        requires_compaction = full_breakdown.total_with_reserve >= trigger_tokens
        selection_limit = self.profile.context_window_tokens
        if requires_compaction:
            selection_limit = int(
                self.profile.context_window_tokens
                * self.profile.compaction_target_ratio
            )

        groups = self._exchange_groups(messages)
        selected_groups: list[_ExchangeGroup] = []
        if full_breakdown.total_with_reserve <= selection_limit:
            selected_groups = groups
        else:
            for group in reversed(groups):
                candidate_groups = [group, *selected_groups]
                candidate_messages = [
                    message
                    for candidate_group in candidate_groups
                    for message in candidate_group.messages
                ]
                candidate_breakdown = self.counter.count_request(
                    [system_message, *candidate_messages],
                    tools,
                    self.serializer,
                    self.profile,
                    reserved_output_tokens=reserved_output_tokens,
                )
                if candidate_breakdown.total_with_reserve > selection_limit:
                    break
                selected_groups = candidate_groups

        open_groups = [group for group in groups if not group.closed]
        if open_groups:
            earliest_open = min(group.start for group in open_groups)
            selected_start = (
                selected_groups[0].start if selected_groups else len(messages)
            )
            if selected_start > earliest_open:
                selected_groups = [
                    group for group in groups if group.start >= earliest_open
                ]

        retained = [
            message for group in selected_groups for message in group.messages
        ]
        retained_start = (
            selected_groups[0].start if selected_groups else len(messages)
        )
        retained_indexes = list(range(retained_start, len(messages)))
        planned_messages = [system_message, *retained]
        breakdown = self.counter.assert_fits(
            planned_messages,
            tools,
            self.serializer,
            self.profile,
            reserved_output_tokens=reserved_output_tokens,
        )

        compactable_from: int | None = None
        compactable_to: int | None = None
        if retained_start > 0:
            omitted_groups = [group for group in groups if group.end < retained_start]
            if any(not group.closed for group in omitted_groups):
                raise ContextHardLimitExceeded(breakdown)
            compactable_from = 0
            compactable_to = retained_start - 1

        recent_history_text = json.dumps(
            retained,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ContextPlan(
            messages=planned_messages,
            tools=self.serializer.serialize_tools(tools),
            breakdown=breakdown,
            system_tokens=self.counter.count_text(category_text["system"]),
            working_state_tokens=self.counter.count_text(category_text["working_state"]),
            checkpoint_tokens=self.counter.count_text(category_text["checkpoint"]),
            recent_history_tokens=self.counter.count_text(recent_history_text),
            memory_tokens=self.counter.count_text(category_text["memories"]),
            pinned_memory_tokens=self.counter.count_text(
                category_text["pinned_memories"]
            ),
            retrieved_memory_tokens=self.counter.count_text(
                category_text["retrieved_memories"]
            ),
            tool_schema_tokens=breakdown.tool_schema_tokens,
            estimated_input_tokens=breakdown.input_tokens,
            reserved_output_tokens=breakdown.reserved_output_tokens,
            safety_margin_tokens=breakdown.safety_margin_tokens,
            context_window_tokens=breakdown.context_window_tokens,
            requires_compaction=requires_compaction,
            compactable_from=compactable_from,
            compactable_to=compactable_to,
            recent_tail_from=retained_start,
            retained_message_indexes=retained_indexes,
        )

    def _system_message(
        self,
        system_rules: str,
        goal: str,
        active_state: dict[str, Any],
        pinned_memories: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        checkpoint: dict[str, Any] | None,
        runtime_authorization: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        pinned_memory_block = (
            "[PINNED_MEMORY_REFERENCES]\n"
            f"{self._format_memories(pinned_memories)}\n"
            "[/PINNED_MEMORY_REFERENCES]"
        )
        checkpoint_text = ""
        if checkpoint is not None:
            checkpoint_text = (
                "[CONTEXT_CHECKPOINT]\n"
                f"{json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)}\n"
                "[/CONTEXT_CHECKPOINT]"
            )
        runtime_authorization_text = (
            "[RUNTIME_AUTHORIZATION]\n"
            f"{_format_runtime_authorization(runtime_authorization)}\n"
            "[/RUNTIME_AUTHORIZATION]"
        )
        goal_text = f"[USER_GOAL]\n{goal}\n[/USER_GOAL]"
        retrieved_memory_block = (
            "[RETRIEVED_MEMORY_REFERENCES]\n"
            f"{self._format_memories(memories)}\n"
            "[/RETRIEVED_MEMORY_REFERENCES]"
        )
        active_state_text = (
            "[ACTIVE_STATE]\n"
            f"{json.dumps(active_state, ensure_ascii=False, sort_keys=True)}\n"
            "[/ACTIVE_STATE]"
        )
        parts = [system_rules, pinned_memory_block]
        if checkpoint_text:
            parts.append(checkpoint_text)
        parts.extend(
            [
                runtime_authorization_text,
                goal_text,
                retrieved_memory_block,
                active_state_text,
            ]
        )
        return (
            {"role": "system", "content": "\n\n".join(parts)},
            {
                "system": system_rules,
                "working_state": "\n\n".join(
                    [runtime_authorization_text, goal_text, active_state_text]
                ),
                "checkpoint": checkpoint_text,
                "pinned_memories": pinned_memory_block,
                "retrieved_memories": retrieved_memory_block,
                "memories": "\n\n".join(
                    [pinned_memory_block, retrieved_memory_block]
                ),
            },
        )

    def _exchange_groups(
        self, messages: list[dict[str, Any]]
    ) -> list[_ExchangeGroup]:
        groups: list[_ExchangeGroup] = []
        index = 0
        while index < len(messages):
            start = index
            message = messages[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                call_ids = {
                    str(call.get("id"))
                    for call in message.get("tool_calls", [])
                    if call.get("id") is not None
                }
                seen_ids: set[str] = set()
                group_messages = [message]
                index += 1
                while index < len(messages):
                    candidate = messages[index]
                    tool_call_id = str(candidate.get("tool_call_id"))
                    if (
                        candidate.get("role") != "tool"
                        or tool_call_id not in call_ids
                    ):
                        break
                    group_messages.append(candidate)
                    seen_ids.add(tool_call_id)
                    index += 1
                groups.append(
                    _ExchangeGroup(
                        start=start,
                        end=index - 1,
                        messages=group_messages,
                        closed=bool(call_ids) and seen_ids == call_ids,
                    )
                )
            else:
                groups.append(
                    _ExchangeGroup(
                        start=start,
                        end=start,
                        messages=[message],
                    )
                )
                index += 1
        return groups

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        return _format_memory_items(memories)


class ContextBuilder:
    def __init__(self, max_chars: int = 40_000) -> None:
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        self.max_chars = max_chars

    def build(
        self,
        *,
        system_rules: str,
        goal: str,
        messages: list[dict[str, Any]],
        active_state: dict[str, Any],
        memories: list[dict[str, Any]],
        pinned_memories: list[dict[str, Any]] | None = None,
        runtime_authorization: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pinned_memory_block = (
            "[PINNED_MEMORY_REFERENCES]\n"
            f"{self._format_memories(list(pinned_memories or []))}\n"
            "[/PINNED_MEMORY_REFERENCES]"
        )
        retrieved_memory_block = (
            "[RETRIEVED_MEMORY_REFERENCES]\n"
            f"{self._format_memories(memories)}\n"
            "[/RETRIEVED_MEMORY_REFERENCES]"
        )
        system = {
            "role": "system",
            "content": (
                f"{system_rules}\n\n"
                f"{pinned_memory_block}\n\n"
                "[RUNTIME_AUTHORIZATION]\n"
                f"{_format_runtime_authorization(runtime_authorization or {})}\n"
                "[/RUNTIME_AUTHORIZATION]\n\n"
                f"[USER_GOAL]\n{goal}\n[/USER_GOAL]\n\n"
                f"{retrieved_memory_block}\n\n"
                f"[ACTIVE_STATE]\n{json.dumps(active_state, ensure_ascii=False, sort_keys=True)}\n[/ACTIVE_STATE]\n\n"
            ),
        }
        budget = max(0, self.max_chars - len(system["content"]))
        groups = self._exchange_groups(messages)
        selected: list[list[dict[str, Any]]] = []
        used = 0
        for group in reversed(groups):
            size = len(json.dumps(group, ensure_ascii=False))
            if selected and used + size > budget:
                continue
            selected.append(group)
            used += size
            if used >= budget:
                break
        selected.reverse()
        flattened = [message for group in selected for message in group]
        selected_message_ids = {id(message) for message in flattened}
        omitted_messages = [message for message in messages if id(message) not in selected_message_ids]
        if omitted_messages:
            flattened.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "[HISTORY_SUMMARY]\n"
                        f"{self._summarize_messages(omitted_messages)}\n"
                        "[/HISTORY_SUMMARY]\n"
                        "This is a derived summary; authoritative events remain persisted."
                    ),
                },
            )
        return [system, *flattened]

    def _exchange_groups(self, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                call_ids = {call.get("id") for call in message.get("tool_calls", [])}
                group = [message]
                index += 1
                while index < len(messages):
                    candidate = messages[index]
                    if candidate.get("role") != "tool" or candidate.get("tool_call_id") not in call_ids:
                        break
                    group.append(candidate)
                    index += 1
                if len(group) == 1:
                    continue
                groups.append(group)
            else:
                groups.append([message])
                index += 1
        return groups

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        return _format_memory_items(memories)

    def _summarize_messages(self, messages: list[dict[str, Any]], max_chars: int = 1_200) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = " ".join(str(message.get("content", "")).split())
            if role == "assistant" and message.get("tool_calls"):
                names = ", ".join(
                    str(call.get("name", "unknown")) for call in message.get("tool_calls", [])
                )
                line = f"- assistant requested tools: {names}"
            elif role == "tool":
                name = str(message.get("name", "unknown"))
                line = f"- tool {name}: {content[:160]}"
            else:
                line = f"- {role}: {content[:320]}"
            if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
                lines.append("- additional history omitted")
                break
            lines.append(line)
        return "\n".join(lines) or "- no textual history"
