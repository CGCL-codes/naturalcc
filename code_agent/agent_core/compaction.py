from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Any

from .compaction_prompts import (
    ANALYZER_PROMPT_VERSION,
    CHECKPOINT_SCHEMA,
    COMPACTION_ANALYSIS_SCHEMA,
    SUMMARIZER_PROMPT_VERSION,
    TOKENIZER_VERSION,
    analyzer_messages,
    repair_messages,
    summarizer_messages,
)
from .contracts import ModelRequest
from .event_store import EventStore
from .model_gateway import DeepSeekRequestSerializer, ModelGateway
from .token_budget import DeepSeekModelProfile, DeepSeekTokenCounter


# [maint-note:bfc4746abe17]
# 数据与行为导航：CompactionValidationError。
# 继承接口：ValueError。
# [/maint-note:bfc4746abe17]
class CompactionValidationError(ValueError):
    """Raised when model-produced compaction data violates its contract."""


# [maint-note:d61da1a002a3]
# 数据与行为导航：MaintenanceBudgetExhausted。
# 继承接口：RuntimeError。
# [/maint-note:d61da1a002a3]
class MaintenanceBudgetExhausted(RuntimeError):
    """Raised internally before a compaction model call exceeds its allowance."""


# [maint-note:cafb04d51a86]
# 维护导航：canonical_message_hash；下列为本函数源码中的接口与路径索引。
# 输入 messages：list[dict[str, Any]]。
# 返回契约：str。
# 返回路径：hashlib.sha256(payload).hexdigest()。
# 直接调用：json.dumps(messages, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode。
# 直接调用：json.dumps。
# 直接调用：hashlib.sha256(payload).hexdigest。
# 直接调用：hashlib.sha256。
# [/maint-note:cafb04d51a86]
def canonical_message_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# [maint-note:46cbecb6bd7d]
# 数据与行为导航：FrozenCompactionSource。
# 字段 run_id：str。
# 字段 thread_id：str | None。
# 字段 source_snapshot_version：int。
# 字段 covered_from：int。
# 字段 covered_to：int。
# 字段 recent_tail_from：int。
# 字段 covered_messages：list[dict[str, Any]]。
# 字段 working_state：dict[str, Any]。
# 字段 old_checkpoint：dict[str, Any] | None。
# 字段 source_hash：str。
# [/maint-note:46cbecb6bd7d]
@dataclass(frozen=True)
class FrozenCompactionSource:
    run_id: str
    thread_id: str | None
    source_snapshot_version: int
    covered_from: int
    covered_to: int
    recent_tail_from: int
    covered_messages: list[dict[str, Any]]
    working_state: dict[str, Any]
    old_checkpoint: dict[str, Any] | None
    source_hash: str

    # [maint-note:996cf37f88d4]
    # 维护导航：freeze；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 thread_id：str | None。
    # 输入 source_snapshot_version：int。
    # 输入 covered_from：int。
    # 输入 covered_to：int。
    # 输入 recent_tail_from：int。
    # 输入 messages：list[dict[str, Any]]。
    # 输入 working_state：dict[str, Any]。
    # 输入 old_checkpoint：dict[str, Any] | None。
    # 返回契约：'FrozenCompactionSource'。
    # 分支条件：covered_from < 0 or covered_to < covered_from。
    # 分支条件：recent_tail_from != covered_to + 1。
    # 分支条件：covered_to >= len(messages)。
    # 返回路径：cls(run_id=run_id, thread_id=thread_id, source_snapshot_version=source_snapshot_version, covered_from=covered_from, covered_to=covered_to, recent_tail_from=recent_tail_from, covered_messages=covered, working_state=frozen_state, old_checkpoi。
    # 显式抛出：ValueError('frozen compaction range is invalid')。
    # 显式抛出：ValueError('recent tail must immediately follow covered range')。
    # 显式抛出：ValueError('frozen compaction range exceeds message history')。
    # 直接调用：ValueError。
    # 直接调用：len。
    # 直接调用：json.loads。
    # 直接调用：json.dumps。
    # 直接调用：cls。
    # 直接调用：canonical_message_hash。
    # 表达式分流：old_checkpoint is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:996cf37f88d4]
    @classmethod
    def freeze(
        cls,
        *,
        run_id: str,
        thread_id: str | None,
        source_snapshot_version: int,
        covered_from: int,
        covered_to: int,
        recent_tail_from: int,
        messages: list[dict[str, Any]],
        working_state: dict[str, Any],
        old_checkpoint: dict[str, Any] | None,
    ) -> "FrozenCompactionSource":
        if covered_from < 0 or covered_to < covered_from:
            raise ValueError("frozen compaction range is invalid")
        if recent_tail_from != covered_to + 1:
            raise ValueError("recent tail must immediately follow covered range")
        if covered_to >= len(messages):
            raise ValueError("frozen compaction range exceeds message history")
        covered = json.loads(
            json.dumps(
                messages[covered_from : covered_to + 1],
                ensure_ascii=False,
            )
        )
        frozen_state = json.loads(json.dumps(working_state, ensure_ascii=False))
        frozen_checkpoint = (
            json.loads(json.dumps(old_checkpoint, ensure_ascii=False))
            if old_checkpoint is not None
            else None
        )
        return cls(
            run_id=run_id,
            thread_id=thread_id,
            source_snapshot_version=source_snapshot_version,
            covered_from=covered_from,
            covered_to=covered_to,
            recent_tail_from=recent_tail_from,
            covered_messages=covered,
            working_state=frozen_state,
            old_checkpoint=frozen_checkpoint,
            source_hash=canonical_message_hash(covered),
        )

    # [maint-note:c592c71d0ec2]
    # 维护导航：covered_range；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, int]。
    # 返回路径：{'from': self.covered_from, 'to': self.covered_to}。
    # [/maint-note:c592c71d0ec2]
    @property
    def covered_range(self) -> dict[str, int]:
        return {"from": self.covered_from, "to": self.covered_to}


# [maint-note:22325c4cf295]
# 数据与行为导航：FrozenThreadCompactionSource。
# 字段 thread_id：str。
# 字段 source_thread_version：int。
# 字段 covered_from：int。
# 字段 covered_to：int。
# 字段 covered_messages：list[dict[str, Any]]。
# 字段 working_state：dict[str, Any]。
# 字段 old_checkpoint：dict[str, Any] | None。
# 字段 source_hash：str。
# [/maint-note:22325c4cf295]
@dataclass(frozen=True)
class FrozenThreadCompactionSource:
    thread_id: str
    source_thread_version: int
    covered_from: int
    covered_to: int
    covered_messages: list[dict[str, Any]]
    working_state: dict[str, Any]
    old_checkpoint: dict[str, Any] | None
    source_hash: str

    # [maint-note:9a160d790e47]
    # 维护导航：freeze；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 source_thread_version：int。
    # 输入 covered_messages：list[dict[str, Any]]。
    # 输入 working_state：dict[str, Any]。
    # 输入 old_checkpoint：dict[str, Any] | None。
    # 返回契约：'FrozenThreadCompactionSource'。
    # 分支条件：not covered_messages。
    # 分支条件：sequences != list(range(sequences[0], sequences[-1] + 1))。
    # 返回路径：cls(thread_id=thread_id, source_thread_version=source_thread_version, covered_from=sequences[0], covered_to=sequences[-1], covered_messages=frozen, working_state=json.loads(json.dumps(working_state, ensure_ascii=False)), old_checkpoint=json。
    # 显式抛出：ValueError('thread compaction requires conversation messages')。
    # 显式抛出：ValueError('thread compaction messages must be a continuous sequence')。
    # 直接调用：ValueError。
    # 直接调用：json.loads。
    # 直接调用：json.dumps。
    # 直接调用：int。
    # 直接调用：list。
    # 直接调用：range。
    # 直接调用：cls。
    # 直接调用：canonical_message_hash。
    # 集合派生：message 来自 frozen。
    # 表达式分流：old_checkpoint is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:9a160d790e47]
    @classmethod
    def freeze(
        cls,
        *,
        thread_id: str,
        source_thread_version: int,
        covered_messages: list[dict[str, Any]],
        working_state: dict[str, Any],
        old_checkpoint: dict[str, Any] | None,
    ) -> "FrozenThreadCompactionSource":
        if not covered_messages:
            raise ValueError("thread compaction requires conversation messages")
        frozen = json.loads(json.dumps(covered_messages, ensure_ascii=False))
        sequences = [int(message["sequence"]) for message in frozen]
        if sequences != list(range(sequences[0], sequences[-1] + 1)):
            raise ValueError("thread compaction messages must be a continuous sequence")
        return cls(
            thread_id=thread_id,
            source_thread_version=source_thread_version,
            covered_from=sequences[0],
            covered_to=sequences[-1],
            covered_messages=frozen,
            working_state=json.loads(json.dumps(working_state, ensure_ascii=False)),
            old_checkpoint=(
                json.loads(json.dumps(old_checkpoint, ensure_ascii=False))
                if old_checkpoint is not None
                else None
            ),
            source_hash=canonical_message_hash(frozen),
        )

    # [maint-note:4f344b43c626]
    # 维护导航：covered_range；下列为本函数源码中的接口与路径索引。
    # 返回契约：dict[str, int]。
    # 返回路径：{'from': self.covered_from, 'to': self.covered_to}。
    # [/maint-note:4f344b43c626]
    @property
    def covered_range(self) -> dict[str, int]:
        return {"from": self.covered_from, "to": self.covered_to}


# [maint-note:fc2c845b20d2]
# 数据与行为导航：CompactionOutcome。
# 字段 compaction_id：str。
# 字段 checkpoint：dict[str, Any]。
# 字段 maintenance_calls：int。
# 字段 input_tokens：int。
# 字段 output_tokens：int。
# 字段 cost_usd：float。
# 字段 fallback_used：bool；初始表达式 False。
# [/maint-note:fc2c845b20d2]
@dataclass(frozen=True)
class CompactionOutcome:
    compaction_id: str
    checkpoint: dict[str, Any]
    maintenance_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fallback_used: bool = False


class CompactionService:
    # [maint-note:bd8f2ed6ce3d]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 store：EventStore。
    # 输入 gateway：ModelGateway。
    # 输入 counter：DeepSeekTokenCounter。
    # 输入 serializer：DeepSeekRequestSerializer。
    # 输入 profile：DeepSeekModelProfile。
    # 返回契约：None。
    # 写入目标：self.store。
    # 写入目标：self.gateway。
    # 写入目标：self.counter。
    # 写入目标：self.serializer。
    # 写入目标：self.profile。
    # [/maint-note:bd8f2ed6ce3d]
    def __init__(
        self,
        *,
        store: EventStore,
        gateway: ModelGateway,
        counter: DeepSeekTokenCounter,
        serializer: DeepSeekRequestSerializer,
        profile: DeepSeekModelProfile,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.counter = counter
        self.serializer = serializer
        self.profile = profile

    # [maint-note:296b3b322543]
    # 维护导航：with_profile；下列为本函数源码中的接口与路径索引。
    # 输入 profile：DeepSeekModelProfile。
    # 返回契约：'CompactionService'。
    # 返回路径：CompactionService(store=self.store, gateway=self.gateway, counter=self.counter, serializer=self.serializer, profile=profile)。
    # 直接调用：CompactionService。
    # [/maint-note:296b3b322543]
    def with_profile(self, profile: DeepSeekModelProfile) -> "CompactionService":
        """Return an isolated view for a Run-specific context profile.

        The service instance is shared by concurrent runs, so mutating
        ``self.profile`` per request would leak provider settings across runs.
        """
        return CompactionService(
            store=self.store,
            gateway=self.gateway,
            counter=self.counter,
            serializer=self.serializer,
            profile=profile,
        )

    # [maint-note:a7821f057c09]
    # 维护导航：_preflight_maintenance_call；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 request：ModelRequest。
    # 输入 max_maintenance_input_tokens：int | None。
    # 输入 max_maintenance_output_tokens：int | None。
    # 输入 deadline_epoch：float | None。
    # 返回契约：None。
    # 分支条件：deadline_epoch is not None and time.time() >= deadline_epoch。
    # 分支条件：max_maintenance_input_tokens is not None and int(usage.get('maintenance_input_tokens', 0)) + breakdown.input_tokens > max_maintenance_input_tokens。
    # 分支条件：max_maintenance_output_tokens is not None and int(usage.get('maintenance_output_tokens', 0)) + request.max_output_tokens > max_maintenance_output_tokens。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance time budget exhausted')。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance input-token budget exhausted')。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance output-token budget exhausted')。
    # 直接调用：time.time。
    # 直接调用：MaintenanceBudgetExhausted。
    # 直接调用：self.counter.assert_fits。
    # 直接调用：self.store.get_compaction。
    # 直接调用：int。
    # 直接调用：usage.get。
    # [/maint-note:a7821f057c09]
    def _preflight_maintenance_call(
        self,
        compaction_id: str,
        request: ModelRequest,
        *,
        max_maintenance_input_tokens: int | None,
        max_maintenance_output_tokens: int | None,
        deadline_epoch: float | None,
    ) -> None:
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise MaintenanceBudgetExhausted(
                "compaction maintenance time budget exhausted"
            )
        breakdown = self.counter.assert_fits(
            request.messages,
            request.tools,
            self.serializer,
            self.profile,
            reserved_output_tokens=request.max_output_tokens,
        )
        usage = self.store.get_compaction(compaction_id)
        if (
            max_maintenance_input_tokens is not None
            and int(usage.get("maintenance_input_tokens", 0))
            + breakdown.input_tokens
            > max_maintenance_input_tokens
        ):
            raise MaintenanceBudgetExhausted(
                "compaction maintenance input-token budget exhausted"
            )
        if (
            max_maintenance_output_tokens is not None
            and int(usage.get("maintenance_output_tokens", 0))
            + request.max_output_tokens
            > max_maintenance_output_tokens
        ):
            raise MaintenanceBudgetExhausted(
                "compaction maintenance output-token budget exhausted"
            )

    # [maint-note:33372cc49af2]
    # 维护导航：compact_run；下列为本函数源码中的接口与路径索引。
    # 输入 source：FrozenCompactionSource。
    # 输入 max_maintenance_calls：int | None；缺省表达式 None。
    # 输入 max_maintenance_input_tokens：int | None；缺省表达式 None。
    # 输入 max_maintenance_output_tokens：int | None；缺省表达式 None。
    # 输入 deadline_epoch：float | None；缺省表达式 None。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：CompactionOutcome。
    # 分支条件：not self.store.reserve_compaction_call(record['id'], max_maintenance_calls=max_maintenance_calls)。
    # 返回路径：response。
    # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
    # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
    # 返回路径：self._commit_fallback(record_id=record['id'], source=source, calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, error=exc)。
    # 返回路径：CompactionOutcome(compaction_id=record['id'], checkpoint=checkpoint, maintenance_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
    # 捕获异常：CompactionValidationError。
    # 捕获异常：Exception。
    # 直接调用：self.store.create_compaction。
    # 直接调用：self._preflight_maintenance_call。
    # 直接调用：self.store.reserve_compaction_call。
    # 直接调用：MaintenanceBudgetExhausted。
    # 直接调用：self.gateway.generate。
    # 直接调用：replace。
    # 直接调用：self.store.record_compaction_response_usage。
    # 直接调用：_parse_and_validate。
    # 直接调用：ModelRequest。
    # 直接调用：repair_messages。
    # 直接调用：issue。
    # 直接调用：analyzer_messages。
    # 直接调用：enumerate。
    # 直接调用：parse_with_one_repair。
    # 直接调用：self._commit_fallback。
    # 直接调用：self.store.update_compaction。
    # 直接调用：summarizer_messages。
    # 直接调用：self._validate_source_and_commit。
    # 直接调用：CompactionOutcome。
    # 集合派生：(index, message) 来自 enumerate(source.covered_messages)。
    # [/maint-note:33372cc49af2]
    def compact_run(
        self,
        source: FrozenCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> CompactionOutcome:
        record = self.store.create_compaction(
            scope="run",
            thread_id=source.thread_id,
            run_id=source.run_id,
            covered_from=source.covered_from,
            covered_to=source.covered_to,
            source_version=source.source_snapshot_version,
            source_hash=source.source_hash,
            model=self.profile.model,
            tokenizer_version=TOKENIZER_VERSION,
            analyzer_prompt_version=ANALYZER_PROMPT_VERSION,
            summarizer_prompt_version=SUMMARIZER_PROMPT_VERSION,
        )
        calls = 0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        # [maint-note:97ea7d4ce8e0]
        # 维护导航：issue；下列为本函数源码中的接口与路径索引。
        # 输入 request：ModelRequest。
        # 分支条件：not self.store.reserve_compaction_call(record['id'], max_maintenance_calls=max_maintenance_calls)。
        # 返回路径：response。
        # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
        # 直接调用：self._preflight_maintenance_call。
        # 直接调用：self.store.reserve_compaction_call。
        # 直接调用：MaintenanceBudgetExhausted。
        # 直接调用：self.gateway.generate。
        # 直接调用：replace。
        # 直接调用：self.store.record_compaction_response_usage。
        # [/maint-note:97ea7d4ce8e0]
        def issue(request: ModelRequest):
            nonlocal calls, input_tokens, output_tokens, cost_usd
            self._preflight_maintenance_call(
                record["id"],
                request,
                max_maintenance_input_tokens=max_maintenance_input_tokens,
                max_maintenance_output_tokens=max_maintenance_output_tokens,
                deadline_epoch=deadline_epoch,
            )
            if not self.store.reserve_compaction_call(
                record["id"], max_maintenance_calls=max_maintenance_calls
            ):
                raise MaintenanceBudgetExhausted(
                    "compaction maintenance call budget exhausted"
                )
            calls += 1
            response = self.gateway.generate(
                replace(request, metadata={**(request_metadata or {}), **request.metadata})
            )
            self.store.record_compaction_response_usage(
                record["id"],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cost_usd += response.cost_usd
            return response

        # [maint-note:e2996fac2a0d]
        # 维护导航：parse_with_one_repair；下列为本函数源码中的接口与路径索引。
        # 输入 content：str。
        # 输入 schema：dict[str, Any]。
        # 输入 max_output_tokens：int。
        # 返回契约：dict[str, Any]。
        # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
        # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
        # 捕获异常：CompactionValidationError。
        # 直接调用：_parse_and_validate。
        # 直接调用：ModelRequest。
        # 直接调用：repair_messages。
        # 直接调用：issue。
        # [/maint-note:e2996fac2a0d]
        def parse_with_one_repair(
            content: str,
            schema: dict[str, Any],
            max_output_tokens: int,
        ) -> dict[str, Any]:
            try:
                return _parse_and_validate(
                    content,
                    schema,
                    source.covered_range,
                )
            except CompactionValidationError:
                repair_request = ModelRequest(
                    messages=repair_messages(
                        invalid_content=content,
                        schema=schema,
                        covered_range=source.covered_range,
                    ),
                    tools=[],
                    purpose="compaction_json_repair",
                    max_output_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                )
                repaired = issue(repair_request)
                return _parse_and_validate(
                    repaired.content,
                    schema,
                    source.covered_range,
                )

        analysis_request = ModelRequest(
            messages=analyzer_messages(
                {
                    "covered_range": source.covered_range,
                    "old_checkpoint": source.old_checkpoint,
                    "history": [
                        {
                            "ref": f"message:{source.covered_from + index}",
                            "message": message,
                        }
                        for index, message in enumerate(source.covered_messages)
                    ],
                    "working_state": source.working_state,
                }
            ),
            tools=[],
            purpose="compaction_analysis",
            max_output_tokens=self.profile.analyzer_output_tokens,
            response_format={"type": "json_object"},
        )
        try:
            analysis_response = issue(analysis_request)
            analysis = parse_with_one_repair(
                analysis_response.content,
                COMPACTION_ANALYSIS_SCHEMA,
                self.profile.analyzer_output_tokens,
            )
        except Exception as exc:
            return self._commit_fallback(
                record_id=record["id"],
                source=source,
                calls=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                error=exc,
            )
        self.store.update_compaction(
            record["id"],
            status="summarizing",
            analysis=analysis,
            analyzer_input_tokens=analysis_response.input_tokens,
            analyzer_output_tokens=analysis_response.output_tokens,
        )

        summary_request = ModelRequest(
            messages=summarizer_messages(
                {
                    "covered_range": source.covered_range,
                    "old_checkpoint": source.old_checkpoint,
                    "analysis": analysis,
                    "working_state": source.working_state,
                }
            ),
            tools=[],
            purpose="compaction_summary",
            max_output_tokens=self.profile.summarizer_output_tokens,
            response_format={"type": "json_object"},
        )
        try:
            summary_response = issue(summary_request)
            checkpoint = parse_with_one_repair(
                summary_response.content,
                CHECKPOINT_SCHEMA,
                self.profile.summarizer_output_tokens,
            )
        except Exception as exc:
            return self._commit_fallback(
                record_id=record["id"],
                source=source,
                calls=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                error=exc,
            )
        self.store.update_compaction(
            record["id"],
            status="validating",
            checkpoint=checkpoint,
            summarizer_input_tokens=summary_response.input_tokens,
            summarizer_output_tokens=summary_response.output_tokens,
        )

        self._validate_source_and_commit(record["id"], source)
        return CompactionOutcome(
            compaction_id=record["id"],
            checkpoint=checkpoint,
            maintenance_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    # [maint-note:910647077f47]
    # 维护导航：resume_run；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 source：FrozenCompactionSource。
    # 输入 max_maintenance_calls：int | None；缺省表达式 None。
    # 输入 max_maintenance_input_tokens：int | None；缺省表达式 None。
    # 输入 max_maintenance_output_tokens：int | None；缺省表达式 None。
    # 输入 deadline_epoch：float | None；缺省表达式 None。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：CompactionOutcome。
    # 分支条件：record['scope'] != 'run' or record['run_id'] != source.run_id。
    # 分支条件：record['covered_from'] != source.covered_from or record['covered_to'] != source.covered_to or record['source_hash'] != source.source_hash。
    # 分支条件：record['status'] == 'committed'。
    # 分支条件：record['status'] in {'failed', 'superseded'}。
    # 分支条件：not self.store.reserve_compaction_call(compaction_id, max_maintenance_calls=max_maintenance_calls)。
    # 分支条件：record['status'] == 'analyzing'。
    # 分支条件：record['status'] == 'summarizing'。
    # 返回路径：CompactionOutcome(compaction_id=compaction_id, checkpoint=record['checkpoint'], maintenance_calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)。
    # 返回路径：response。
    # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
    # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
    # 返回路径：self._commit_fallback(record_id=compaction_id, source=source, calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, error=exc)。
    # 返回路径：CompactionOutcome(compaction_id=compaction_id, checkpoint=checkpoint, maintenance_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)。
    # 显式抛出：ValueError('compaction does not belong to the frozen run source')。
    # 显式抛出：ValueError('stored compaction does not match frozen source')。
    # 显式抛出：ValueError(f"cannot resume {record['status']} compaction")。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
    # 捕获异常：CompactionValidationError。
    # 捕获异常：Exception。
    # 直接调用：self.store.get_compaction。
    # 直接调用：ValueError。
    # 直接调用：CompactionOutcome。
    # 直接调用：self._preflight_maintenance_call。
    # 直接调用：self.store.reserve_compaction_call。
    # 直接调用：MaintenanceBudgetExhausted。
    # 直接调用：self.gateway.generate。
    # 直接调用：replace。
    # 直接调用：self.store.record_compaction_response_usage。
    # 直接调用：_parse_and_validate。
    # 直接调用：issue。
    # 直接调用：ModelRequest。
    # 直接调用：repair_messages。
    # 直接调用：analyzer_messages。
    # 直接调用：enumerate。
    # 直接调用：parse_with_one_repair。
    # 直接调用：self._commit_fallback。
    # 直接调用：self.store.update_compaction。
    # 直接调用：_validate_object。
    # 直接调用：summarizer_messages。
    # 直接调用：self._validate_source_and_commit。
    # 集合派生：(index, message) 来自 enumerate(source.covered_messages)。
    # [/maint-note:910647077f47]
    def resume_run(
        self,
        compaction_id: str,
        source: FrozenCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> CompactionOutcome:
        record = self.store.get_compaction(compaction_id)
        if record["scope"] != "run" or record["run_id"] != source.run_id:
            raise ValueError("compaction does not belong to the frozen run source")
        if (
            record["covered_from"] != source.covered_from
            or record["covered_to"] != source.covered_to
            or record["source_hash"] != source.source_hash
        ):
            raise ValueError("stored compaction does not match frozen source")
        if record["status"] == "committed":
            return CompactionOutcome(
                compaction_id=compaction_id,
                checkpoint=record["checkpoint"],
                maintenance_calls=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )
        if record["status"] in {"failed", "superseded"}:
            raise ValueError(f"cannot resume {record['status']} compaction")

        calls = 0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        # [maint-note:76139014fbf2]
        # 维护导航：issue；下列为本函数源码中的接口与路径索引。
        # 输入 request：ModelRequest。
        # 分支条件：not self.store.reserve_compaction_call(compaction_id, max_maintenance_calls=max_maintenance_calls)。
        # 返回路径：response。
        # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
        # 直接调用：self._preflight_maintenance_call。
        # 直接调用：self.store.reserve_compaction_call。
        # 直接调用：MaintenanceBudgetExhausted。
        # 直接调用：self.gateway.generate。
        # 直接调用：replace。
        # 直接调用：self.store.record_compaction_response_usage。
        # [/maint-note:76139014fbf2]
        def issue(request: ModelRequest):
            nonlocal calls, input_tokens, output_tokens, cost_usd
            self._preflight_maintenance_call(
                compaction_id,
                request,
                max_maintenance_input_tokens=max_maintenance_input_tokens,
                max_maintenance_output_tokens=max_maintenance_output_tokens,
                deadline_epoch=deadline_epoch,
            )
            if not self.store.reserve_compaction_call(
                compaction_id, max_maintenance_calls=max_maintenance_calls
            ):
                raise MaintenanceBudgetExhausted(
                    "compaction maintenance call budget exhausted"
                )
            calls += 1
            response = self.gateway.generate(
                replace(request, metadata={**(request_metadata or {}), **request.metadata})
            )
            self.store.record_compaction_response_usage(
                compaction_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cost_usd += response.cost_usd
            return response

        # [maint-note:913b2772d307]
        # 维护导航：parse_with_one_repair；下列为本函数源码中的接口与路径索引。
        # 输入 content：str。
        # 输入 schema：dict[str, Any]。
        # 输入 max_output_tokens：int。
        # 返回契约：dict[str, Any]。
        # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
        # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
        # 捕获异常：CompactionValidationError。
        # 直接调用：_parse_and_validate。
        # 直接调用：issue。
        # 直接调用：ModelRequest。
        # 直接调用：repair_messages。
        # [/maint-note:913b2772d307]
        def parse_with_one_repair(
            content: str,
            schema: dict[str, Any],
            max_output_tokens: int,
        ) -> dict[str, Any]:
            try:
                return _parse_and_validate(
                    content, schema, source.covered_range
                )
            except CompactionValidationError:
                repaired = issue(
                    ModelRequest(
                        messages=repair_messages(
                            invalid_content=content,
                            schema=schema,
                            covered_range=source.covered_range,
                        ),
                        tools=[],
                        purpose="compaction_json_repair",
                        max_output_tokens=max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                )
                return _parse_and_validate(
                    repaired.content, schema, source.covered_range
                )

        if record["status"] == "analyzing":
            request = ModelRequest(
                messages=analyzer_messages(
                    {
                        "covered_range": source.covered_range,
                        "old_checkpoint": source.old_checkpoint,
                        "history": [
                            {
                                "ref": f"message:{source.covered_from + index}",
                                "message": message,
                            }
                            for index, message in enumerate(
                                source.covered_messages
                            )
                        ],
                        "working_state": source.working_state,
                    }
                ),
                tools=[],
                purpose="compaction_analysis",
                max_output_tokens=self.profile.analyzer_output_tokens,
                response_format={"type": "json_object"},
            )
            try:
                response = issue(request)
                analysis = parse_with_one_repair(
                    response.content,
                    COMPACTION_ANALYSIS_SCHEMA,
                    self.profile.analyzer_output_tokens,
                )
            except Exception as exc:
                return self._commit_fallback(
                    record_id=compaction_id,
                    source=source,
                    calls=calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    error=exc,
                )
            self.store.update_compaction(
                compaction_id,
                status="summarizing",
                analysis=analysis,
                analyzer_input_tokens=response.input_tokens,
                analyzer_output_tokens=response.output_tokens,
            )
            record = self.store.get_compaction(compaction_id)

        if record["status"] == "summarizing":
            analysis = _validate_object(
                record["analysis"],
                COMPACTION_ANALYSIS_SCHEMA,
                source.covered_range,
            )
            request = ModelRequest(
                messages=summarizer_messages(
                    {
                        "covered_range": source.covered_range,
                        "old_checkpoint": source.old_checkpoint,
                        "analysis": analysis,
                        "working_state": source.working_state,
                    }
                ),
                tools=[],
                purpose="compaction_summary",
                max_output_tokens=self.profile.summarizer_output_tokens,
                response_format={"type": "json_object"},
            )
            try:
                response = issue(request)
                checkpoint = parse_with_one_repair(
                    response.content,
                    CHECKPOINT_SCHEMA,
                    self.profile.summarizer_output_tokens,
                )
            except Exception as exc:
                return self._commit_fallback(
                    record_id=compaction_id,
                    source=source,
                    calls=calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    error=exc,
                )
            self.store.update_compaction(
                compaction_id,
                status="validating",
                checkpoint=checkpoint,
                summarizer_input_tokens=response.input_tokens,
                summarizer_output_tokens=response.output_tokens,
            )
            record = self.store.get_compaction(compaction_id)

        checkpoint = _validate_object(
            record["checkpoint"],
            CHECKPOINT_SCHEMA,
            source.covered_range,
        )
        self._validate_source_and_commit(compaction_id, source)
        return CompactionOutcome(
            compaction_id=compaction_id,
            checkpoint=checkpoint,
            maintenance_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    # [maint-note:82325a79ff7c]
    # 维护导航：compact_thread；下列为本函数源码中的接口与路径索引。
    # 输入 source：FrozenThreadCompactionSource。
    # 输入 max_maintenance_calls：int | None；缺省表达式 None。
    # 输入 max_maintenance_input_tokens：int | None；缺省表达式 None。
    # 输入 max_maintenance_output_tokens：int | None；缺省表达式 None。
    # 输入 deadline_epoch：float | None；缺省表达式 None。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：CompactionOutcome。
    # 分支条件：not self.store.reserve_compaction_call(record['id'], max_maintenance_calls=max_maintenance_calls)。
    # 分支条件：current['status'] == 'analyzing'。
    # 分支条件：len(current) != len(source.covered_messages) or current_hash != source.source_hash。
    # 返回路径：response。
    # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
    # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
    # 返回路径：CompactionOutcome(compaction_id=record['id'], checkpoint=checkpoint, maintenance_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, fallback_used=fallback_used)。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
    # 显式抛出：CompactionValidationError('frozen conversation prefix changed before commit')。
    # 捕获异常：CompactionValidationError。
    # 捕获异常：Exception。
    # 直接调用：self.store.create_compaction。
    # 直接调用：self._preflight_maintenance_call。
    # 直接调用：self.store.reserve_compaction_call。
    # 直接调用：MaintenanceBudgetExhausted。
    # 直接调用：self.gateway.generate。
    # 直接调用：replace。
    # 直接调用：self.store.record_compaction_response_usage。
    # 直接调用：_parse_and_validate。
    # 直接调用：issue。
    # 直接调用：ModelRequest。
    # 直接调用：repair_messages。
    # 直接调用：analyzer_messages。
    # 直接调用：parse_or_repair。
    # 直接调用：self.store.update_compaction。
    # 直接调用：summarizer_messages。
    # 直接调用：self.store.get_compaction。
    # 直接调用：_deterministic_thread_analysis。
    # 直接调用：type。
    # 直接调用：str。
    # 直接调用：_deterministic_thread_checkpoint。
    # 直接调用：self.counter.assert_fits。
    # 直接调用：json.dumps。
    # 直接调用：self.store.list_conversation_messages。
    # 直接调用：len。
    # 直接调用：canonical_message_hash。
    # 直接调用：CompactionValidationError。
    # 直接调用：self.store.commit_compaction。
    # 直接调用：CompactionOutcome。
    # 集合派生：message 来自 source.covered_messages。
    # [/maint-note:82325a79ff7c]
    def compact_thread(
        self,
        source: FrozenThreadCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> CompactionOutcome:
        record = self.store.create_compaction(
            scope="thread",
            thread_id=source.thread_id,
            covered_from=source.covered_from,
            covered_to=source.covered_to,
            source_version=source.source_thread_version,
            source_hash=source.source_hash,
            model=self.profile.model,
            tokenizer_version=TOKENIZER_VERSION,
            analyzer_prompt_version=ANALYZER_PROMPT_VERSION,
            summarizer_prompt_version=SUMMARIZER_PROMPT_VERSION,
        )
        calls = 0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        # [maint-note:d2736bb38296]
        # 维护导航：issue；下列为本函数源码中的接口与路径索引。
        # 输入 request：ModelRequest。
        # 分支条件：not self.store.reserve_compaction_call(record['id'], max_maintenance_calls=max_maintenance_calls)。
        # 返回路径：response。
        # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
        # 直接调用：self._preflight_maintenance_call。
        # 直接调用：self.store.reserve_compaction_call。
        # 直接调用：MaintenanceBudgetExhausted。
        # 直接调用：self.gateway.generate。
        # 直接调用：replace。
        # 直接调用：self.store.record_compaction_response_usage。
        # [/maint-note:d2736bb38296]
        def issue(request: ModelRequest):
            nonlocal calls, input_tokens, output_tokens, cost_usd
            self._preflight_maintenance_call(
                record["id"],
                request,
                max_maintenance_input_tokens=max_maintenance_input_tokens,
                max_maintenance_output_tokens=max_maintenance_output_tokens,
                deadline_epoch=deadline_epoch,
            )
            if not self.store.reserve_compaction_call(
                record["id"], max_maintenance_calls=max_maintenance_calls
            ):
                raise MaintenanceBudgetExhausted(
                    "compaction maintenance call budget exhausted"
                )
            calls += 1
            response = self.gateway.generate(
                replace(request, metadata={**(request_metadata or {}), **request.metadata})
            )
            self.store.record_compaction_response_usage(
                record["id"],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cost_usd += response.cost_usd
            return response

        # [maint-note:30fbca8cc237]
        # 维护导航：parse_or_repair；下列为本函数源码中的接口与路径索引。
        # 输入 content：str。
        # 输入 schema：dict[str, Any]。
        # 输入 max_output_tokens：int。
        # 返回契约：dict[str, Any]。
        # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
        # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
        # 捕获异常：CompactionValidationError。
        # 直接调用：_parse_and_validate。
        # 直接调用：issue。
        # 直接调用：ModelRequest。
        # 直接调用：repair_messages。
        # [/maint-note:30fbca8cc237]
        def parse_or_repair(
            content: str,
            schema: dict[str, Any],
            max_output_tokens: int,
        ) -> dict[str, Any]:
            try:
                return _parse_and_validate(content, schema, source.covered_range)
            except CompactionValidationError:
                repaired = issue(
                    ModelRequest(
                        messages=repair_messages(
                            invalid_content=content,
                            schema=schema,
                            covered_range=source.covered_range,
                        ),
                        tools=[],
                        purpose="compaction_json_repair",
                        max_output_tokens=max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                )
                return _parse_and_validate(
                    repaired.content, schema, source.covered_range
                )

        try:
            analysis_response = issue(
                ModelRequest(
                    messages=analyzer_messages(
                        {
                            "covered_range": source.covered_range,
                            "old_checkpoint": source.old_checkpoint,
                            "history": [
                                {
                                    "ref": f"conversation:{message['sequence']}",
                                    "message": {
                                        "role": message["role"],
                                        "content": message["content"],
                                    },
                                }
                                for message in source.covered_messages
                            ],
                            "working_state": source.working_state,
                        }
                    ),
                    tools=[],
                    purpose="compaction_analysis",
                    max_output_tokens=self.profile.analyzer_output_tokens,
                    response_format={"type": "json_object"},
                )
            )
            analysis = parse_or_repair(
                analysis_response.content,
                COMPACTION_ANALYSIS_SCHEMA,
                self.profile.analyzer_output_tokens,
            )
            self.store.update_compaction(
                record["id"],
                status="summarizing",
                analysis=analysis,
                analyzer_input_tokens=analysis_response.input_tokens,
                analyzer_output_tokens=analysis_response.output_tokens,
            )
            summary_response = issue(
                ModelRequest(
                    messages=summarizer_messages(
                        {
                            "covered_range": source.covered_range,
                            "old_checkpoint": source.old_checkpoint,
                            "analysis": analysis,
                            "working_state": source.working_state,
                        }
                    ),
                    tools=[],
                    purpose="compaction_summary",
                    max_output_tokens=self.profile.summarizer_output_tokens,
                    response_format={"type": "json_object"},
                )
            )
            checkpoint = parse_or_repair(
                summary_response.content,
                CHECKPOINT_SCHEMA,
                self.profile.summarizer_output_tokens,
            )
            self.store.update_compaction(
                record["id"],
                status="validating",
                checkpoint=checkpoint,
                summarizer_input_tokens=summary_response.input_tokens,
                summarizer_output_tokens=summary_response.output_tokens,
            )
        except Exception as exc:
            current = self.store.get_compaction(record["id"])
            if current["status"] == "analyzing":
                self.store.update_compaction(
                    record["id"],
                    status="summarizing",
                    analysis=_deterministic_thread_analysis(source),
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            checkpoint = _deterministic_thread_checkpoint(source)
            self.counter.assert_fits(
                [
                    {
                        "role": "system",
                        "content": json.dumps(checkpoint, ensure_ascii=False),
                    }
                ],
                [],
                self.serializer,
                self.profile,
            )
            self.store.update_compaction(
                record["id"],
                status="validating",
                checkpoint=checkpoint,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            fallback_used = True
        else:
            fallback_used = False

        current = self.store.list_conversation_messages(
            source.thread_id,
            limit=len(source.covered_messages) + 1,
            after_sequence=source.covered_from - 1,
            through_sequence=source.covered_to,
        )
        current_hash = canonical_message_hash(current)
        if (
            len(current) != len(source.covered_messages)
            or current_hash != source.source_hash
        ):
            self.store.update_compaction(
                record["id"],
                status="failed",
                error={
                    "type": "SourceMutation",
                    "message": "frozen conversation prefix changed before commit",
                },
            )
            raise CompactionValidationError(
                "frozen conversation prefix changed before commit"
            )
        self.store.commit_compaction(
            record["id"], expected_source_hash=current_hash
        )
        return CompactionOutcome(
            compaction_id=record["id"],
            checkpoint=checkpoint,
            maintenance_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            fallback_used=fallback_used,
        )

    # [maint-note:4d4e0ee7ad8a]
    # 维护导航：resume_thread；下列为本函数源码中的接口与路径索引。
    # 输入 compaction_id：str。
    # 输入 source：FrozenThreadCompactionSource。
    # 输入 max_maintenance_calls：int | None；缺省表达式 None。
    # 输入 max_maintenance_input_tokens：int | None；缺省表达式 None。
    # 输入 max_maintenance_output_tokens：int | None；缺省表达式 None。
    # 输入 deadline_epoch：float | None；缺省表达式 None。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 返回契约：CompactionOutcome。
    # 分支条件：record['scope'] != 'thread' or record['thread_id'] != source.thread_id。
    # 分支条件：int(record['covered_from']) != source.covered_from or int(record['covered_to']) != source.covered_to or record['source_hash'] != source.source_hash。
    # 分支条件：record['status'] == 'committed'。
    # 分支条件：record['status'] in {'failed', 'superseded'}。
    # 分支条件：not self.store.reserve_compaction_call(compaction_id, max_maintenance_calls=max_maintenance_calls)。
    # 分支条件：record['status'] == 'analyzing'。
    # 分支条件：record['status'] == 'summarizing'。
    # 分支条件：current['status'] == 'analyzing'。
    # 返回路径：CompactionOutcome(compaction_id=compaction_id, checkpoint=record['checkpoint'], maintenance_calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)。
    # 返回路径：response。
    # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
    # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
    # 返回路径：CompactionOutcome(compaction_id=compaction_id, checkpoint=checkpoint, maintenance_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, fallback_used=fallback_used)。
    # 显式抛出：ValueError('compaction does not belong to the frozen thread source')。
    # 显式抛出：ValueError('stored compaction does not match frozen thread source')。
    # 显式抛出：ValueError(f"cannot resume {record['status']} compaction")。
    # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
    # 捕获异常：CompactionValidationError。
    # 捕获异常：Exception。
    # 直接调用：self.store.get_compaction。
    # 直接调用：ValueError。
    # 直接调用：int。
    # 直接调用：CompactionOutcome。
    # 直接调用：self._preflight_maintenance_call。
    # 直接调用：self.store.reserve_compaction_call。
    # 直接调用：MaintenanceBudgetExhausted。
    # 直接调用：self.gateway.generate。
    # 直接调用：replace。
    # 直接调用：self.store.record_compaction_response_usage。
    # 直接调用：_parse_and_validate。
    # 直接调用：issue。
    # 直接调用：ModelRequest。
    # 直接调用：repair_messages。
    # 直接调用：analyzer_messages。
    # 直接调用：parse_or_repair。
    # 直接调用：self.store.update_compaction。
    # 直接调用：summarizer_messages。
    # 直接调用：_validate_object。
    # 直接调用：_deterministic_thread_analysis。
    # 直接调用：type。
    # 直接调用：str。
    # 直接调用：_deterministic_thread_checkpoint。
    # 直接调用：self.counter.assert_fits。
    # 直接调用：json.dumps。
    # 直接调用：self.store.commit_compaction。
    # 集合派生：message 来自 source.covered_messages。
    # [/maint-note:4d4e0ee7ad8a]
    def resume_thread(
        self,
        compaction_id: str,
        source: FrozenThreadCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> CompactionOutcome:
        record = self.store.get_compaction(compaction_id)
        if record["scope"] != "thread" or record["thread_id"] != source.thread_id:
            raise ValueError("compaction does not belong to the frozen thread source")
        if (
            int(record["covered_from"]) != source.covered_from
            or int(record["covered_to"]) != source.covered_to
            or record["source_hash"] != source.source_hash
        ):
            raise ValueError("stored compaction does not match frozen thread source")
        if record["status"] == "committed":
            return CompactionOutcome(
                compaction_id=compaction_id,
                checkpoint=record["checkpoint"],
                maintenance_calls=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )
        if record["status"] in {"failed", "superseded"}:
            raise ValueError(f"cannot resume {record['status']} compaction")

        calls = 0
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        # [maint-note:b18c7aa8c0c6]
        # 维护导航：issue；下列为本函数源码中的接口与路径索引。
        # 输入 request：ModelRequest。
        # 分支条件：not self.store.reserve_compaction_call(compaction_id, max_maintenance_calls=max_maintenance_calls)。
        # 返回路径：response。
        # 显式抛出：MaintenanceBudgetExhausted('compaction maintenance call budget exhausted')。
        # 直接调用：self._preflight_maintenance_call。
        # 直接调用：self.store.reserve_compaction_call。
        # 直接调用：MaintenanceBudgetExhausted。
        # 直接调用：self.gateway.generate。
        # 直接调用：replace。
        # 直接调用：self.store.record_compaction_response_usage。
        # [/maint-note:b18c7aa8c0c6]
        def issue(request: ModelRequest):
            nonlocal calls, input_tokens, output_tokens, cost_usd
            self._preflight_maintenance_call(
                compaction_id,
                request,
                max_maintenance_input_tokens=max_maintenance_input_tokens,
                max_maintenance_output_tokens=max_maintenance_output_tokens,
                deadline_epoch=deadline_epoch,
            )
            if not self.store.reserve_compaction_call(
                compaction_id, max_maintenance_calls=max_maintenance_calls
            ):
                raise MaintenanceBudgetExhausted(
                    "compaction maintenance call budget exhausted"
                )
            calls += 1
            response = self.gateway.generate(
                replace(request, metadata={**(request_metadata or {}), **request.metadata})
            )
            self.store.record_compaction_response_usage(
                compaction_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cost_usd += response.cost_usd
            return response

        # [maint-note:3a0c47a33a80]
        # 维护导航：parse_or_repair；下列为本函数源码中的接口与路径索引。
        # 输入 content：str。
        # 输入 schema：dict[str, Any]。
        # 输入 max_output_tokens：int。
        # 返回契约：dict[str, Any]。
        # 返回路径：_parse_and_validate(content, schema, source.covered_range)。
        # 返回路径：_parse_and_validate(repaired.content, schema, source.covered_range)。
        # 捕获异常：CompactionValidationError。
        # 直接调用：_parse_and_validate。
        # 直接调用：issue。
        # 直接调用：ModelRequest。
        # 直接调用：repair_messages。
        # [/maint-note:3a0c47a33a80]
        def parse_or_repair(
            content: str,
            schema: dict[str, Any],
            max_output_tokens: int,
        ) -> dict[str, Any]:
            try:
                return _parse_and_validate(content, schema, source.covered_range)
            except CompactionValidationError:
                repaired = issue(
                    ModelRequest(
                        messages=repair_messages(
                            invalid_content=content,
                            schema=schema,
                            covered_range=source.covered_range,
                        ),
                        tools=[],
                        purpose="compaction_json_repair",
                        max_output_tokens=max_output_tokens,
                        response_format={"type": "json_object"},
                    )
                )
                return _parse_and_validate(
                    repaired.content, schema, source.covered_range
                )

        try:
            if record["status"] == "analyzing":
                response = issue(
                    ModelRequest(
                        messages=analyzer_messages(
                            {
                                "covered_range": source.covered_range,
                                "old_checkpoint": source.old_checkpoint,
                                "history": [
                                    {
                                        "ref": f"conversation:{message['sequence']}",
                                        "message": {
                                            "role": message["role"],
                                            "content": message["content"],
                                        },
                                    }
                                    for message in source.covered_messages
                                ],
                                "working_state": source.working_state,
                            }
                        ),
                        tools=[],
                        purpose="compaction_analysis",
                        max_output_tokens=self.profile.analyzer_output_tokens,
                        response_format={"type": "json_object"},
                    )
                )
                analysis = parse_or_repair(
                    response.content,
                    COMPACTION_ANALYSIS_SCHEMA,
                    self.profile.analyzer_output_tokens,
                )
                self.store.update_compaction(
                    compaction_id,
                    status="summarizing",
                    analysis=analysis,
                    analyzer_input_tokens=response.input_tokens,
                    analyzer_output_tokens=response.output_tokens,
                )
                record = self.store.get_compaction(compaction_id)

            if record["status"] == "summarizing":
                response = issue(
                    ModelRequest(
                        messages=summarizer_messages(
                            {
                                "covered_range": source.covered_range,
                                "old_checkpoint": source.old_checkpoint,
                                "analysis": record["analysis"],
                                "working_state": source.working_state,
                            }
                        ),
                        tools=[],
                        purpose="compaction_summary",
                        max_output_tokens=self.profile.summarizer_output_tokens,
                        response_format={"type": "json_object"},
                    )
                )
                checkpoint = parse_or_repair(
                    response.content,
                    CHECKPOINT_SCHEMA,
                    self.profile.summarizer_output_tokens,
                )
                self.store.update_compaction(
                    compaction_id,
                    status="validating",
                    checkpoint=checkpoint,
                    summarizer_input_tokens=response.input_tokens,
                    summarizer_output_tokens=response.output_tokens,
                )
                record = self.store.get_compaction(compaction_id)

            checkpoint = _validate_object(
                record["checkpoint"], CHECKPOINT_SCHEMA, source.covered_range
            )
            fallback_used = False
        except Exception as exc:
            current = self.store.get_compaction(compaction_id)
            if current["status"] == "analyzing":
                self.store.update_compaction(
                    compaction_id,
                    status="summarizing",
                    analysis=_deterministic_thread_analysis(source),
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            checkpoint = _deterministic_thread_checkpoint(source)
            self.counter.assert_fits(
                [{"role": "system", "content": json.dumps(checkpoint, ensure_ascii=False)}],
                [],
                self.serializer,
                self.profile,
            )
            self.store.update_compaction(
                compaction_id,
                status="validating",
                checkpoint=checkpoint,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            fallback_used = True

        self.store.commit_compaction(
            compaction_id, expected_source_hash=source.source_hash
        )

        return CompactionOutcome(
            compaction_id=compaction_id,
            checkpoint=checkpoint,
            maintenance_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            fallback_used=fallback_used,
        )

    # [maint-note:74791a20bbc0]
    # 维护导航：_validate_source_and_commit；下列为本函数源码中的接口与路径索引。
    # 输入 record_id：str。
    # 输入 source：FrozenCompactionSource。
    # 返回契约：None。
    # 分支条件：len(current_covered) != len(source.covered_messages) or current_hash != source.source_hash。
    # 显式抛出：CompactionValidationError('frozen message prefix changed before commit')。
    # 直接调用：self.store.load_snapshot。
    # 直接调用：current_snapshot.get。
    # 直接调用：canonical_message_hash。
    # 直接调用：len。
    # 直接调用：self.store.update_compaction。
    # 直接调用：CompactionValidationError。
    # 直接调用：self.store.commit_compaction。
    # [/maint-note:74791a20bbc0]
    def _validate_source_and_commit(
        self,
        record_id: str,
        source: FrozenCompactionSource,
    ) -> None:
        current_snapshot = self.store.load_snapshot(source.run_id) or {}
        current_messages = current_snapshot.get("messages", [])
        current_covered = current_messages[
            source.covered_from : source.covered_to + 1
        ]
        current_hash = canonical_message_hash(current_covered)
        if (
            len(current_covered) != len(source.covered_messages)
            or current_hash != source.source_hash
        ):
            self.store.update_compaction(
                record_id,
                status="failed",
                error={
                    "type": "SourceMutation",
                    "message": "frozen message prefix changed before commit",
                },
            )
            raise CompactionValidationError(
                "frozen message prefix changed before commit"
            )
        self.store.commit_compaction(
            record_id, expected_source_hash=current_hash
        )

    # [maint-note:a247f3ce94e4]
    # 维护导航：_commit_fallback；下列为本函数源码中的接口与路径索引。
    # 输入 record_id：str。
    # 输入 source：FrozenCompactionSource。
    # 输入 calls：int。
    # 输入 input_tokens：int。
    # 输入 output_tokens：int。
    # 输入 cost_usd：float。
    # 输入 error：Exception。
    # 返回契约：CompactionOutcome。
    # 分支条件：record['status'] == 'analyzing'。
    # 返回路径：CompactionOutcome(compaction_id=record_id, checkpoint=checkpoint, maintenance_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd, fallback_used=True)。
    # 直接调用：self.store.get_compaction。
    # 直接调用：self.store.update_compaction。
    # 直接调用：_deterministic_analysis。
    # 直接调用：type。
    # 直接调用：str。
    # 直接调用：_deterministic_checkpoint。
    # 直接调用：json.dumps。
    # 直接调用：self.counter.assert_fits。
    # 直接调用：self._validate_source_and_commit。
    # 直接调用：CompactionOutcome。
    # [/maint-note:a247f3ce94e4]
    def _commit_fallback(
        self,
        *,
        record_id: str,
        source: FrozenCompactionSource,
        calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        error: Exception,
    ) -> CompactionOutcome:
        record = self.store.get_compaction(record_id)
        if record["status"] == "analyzing":
            self.store.update_compaction(
                record_id,
                status="summarizing",
                analysis=_deterministic_analysis(source),
                error={"type": type(error).__name__, "message": str(error)},
            )
        checkpoint = _deterministic_checkpoint(source)
        checkpoint_message = [
            {
                "role": "system",
                "content": "[CONTEXT_CHECKPOINT]\n"
                + json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)
                + "\n[/CONTEXT_CHECKPOINT]",
            }
        ]
        self.counter.assert_fits(
            checkpoint_message,
            [],
            self.serializer,
            self.profile,
        )
        self.store.update_compaction(
            record_id,
            status="validating",
            checkpoint=checkpoint,
            error={"type": type(error).__name__, "message": str(error)},
        )
        self._validate_source_and_commit(record_id, source)
        return CompactionOutcome(
            compaction_id=record_id,
            checkpoint=checkpoint,
            maintenance_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            fallback_used=True,
        )


# [maint-note:1513a06471a2]
# 维护导航：_bounded_items；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 输入 limit：int；缺省表达式 20。
# 输入 width：int；缺省表达式 400。
# 返回契约：list[Any]。
# 分支条件：value is None。
# 分支条件：isinstance(item, (dict, list))。
# 遍历边界：item ← items[:limit]。
# 返回路径：[]。
# 返回路径：bounded。
# 直接调用：isinstance。
# 直接调用：json.dumps。
# 直接调用：bounded.append。
# 直接调用：str。
# 表达式分流：isinstance(value, list)；真值与假值分支分别计算，不同时执行。
# [/maint-note:1513a06471a2]
def _bounded_items(value: Any, *, limit: int = 20, width: int = 400) -> list[Any]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    bounded: list[Any] = []
    for item in items[:limit]:
        if isinstance(item, (dict, list)):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
            bounded.append(text[:width])
        else:
            bounded.append(str(item)[:width])
    return bounded


# [maint-note:a2732a626b29]
# 维护导航：_merge_checkpoint_items；下列为本函数源码中的接口与路径索引。
# 输入 old：Any。
# 输入 new：Any。
# 输入 limit：int；缺省表达式 20。
# 返回契约：list[Any]。
# 分支条件：key not in seen。
# 分支条件：len(old_items) + len(new_items) <= limit。
# 遍历边界：item ← items。
# 返回路径：result。
# 返回路径：unique([*old_items, *new_items])。
# 返回路径：unique([*old_items[:old_quota], *new_items[:new_quota]])。
# 直接调用：set。
# 直接调用：json.dumps。
# 直接调用：result.append。
# 直接调用：seen.add。
# 直接调用：unique。
# 直接调用：_bounded_items。
# 直接调用：len。
# 直接调用：min。
# 直接调用：max。
# [/maint-note:a2732a626b29]
def _merge_checkpoint_items(old: Any, new: Any, *, limit: int = 20) -> list[Any]:
    # [maint-note:73b8ca9fbd2a]
    # 维护导航：unique；下列为本函数源码中的接口与路径索引。
    # 输入 items：list[Any]。
    # 返回契约：list[Any]。
    # 分支条件：key not in seen。
    # 遍历边界：item ← items。
    # 返回路径：result。
    # 直接调用：set。
    # 直接调用：json.dumps。
    # 直接调用：result.append。
    # 直接调用：seen.add。
    # [/maint-note:73b8ca9fbd2a]
    def unique(items: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result

    old_items = unique(_bounded_items(old, limit=limit))
    new_items = unique(_bounded_items(new, limit=limit))
    if len(old_items) + len(new_items) <= limit:
        return unique([*old_items, *new_items])
    new_quota = min(len(new_items), max(1, limit // 2))
    old_quota = min(len(old_items), limit - new_quota)
    remaining = limit - old_quota - new_quota
    extra_new = min(remaining, len(new_items) - new_quota)
    new_quota += extra_new
    remaining -= extra_new
    old_quota += min(remaining, len(old_items) - old_quota)
    return unique([*old_items[:old_quota], *new_items[:new_quota]])


# [maint-note:4cc44e279701]
# 维护导航：_deterministic_analysis；下列为本函数源码中的接口与路径索引。
# 输入 source：FrozenCompactionSource。
# 返回契约：dict[str, Any]。
# 分支条件：message.get('role') == 'user' and content。
# 分支条件：message.get('role') == 'tool'。
# 分支条件：message.get('role') == 'assistant' and message.get('tool_calls')。
# 遍历边界：(index, message) ← enumerate(source.covered_messages)。
# 返回路径：{'covered_range': source.covered_range, 'active_goals': _bounded_items(working.get('current_objective')), 'user_constraints': user_constraints[:20], 'decisions': _bounded_items(working.get('decisions')), 'completed_actions': completed_actio。
# 直接调用：enumerate。
# 直接调用：' '.join。
# 直接调用：str(message.get('content', '')).split。
# 直接调用：str。
# 直接调用：message.get。
# 直接调用：user_constraints.append。
# 直接调用：tool_findings.append。
# 直接调用：completed_actions.extend。
# 直接调用：call.get。
# 直接调用：_bounded_items。
# 直接调用：working.get。
# 集合派生：call 来自 message.get('tool_calls', [])[:20]。
# [/maint-note:4cc44e279701]
def _deterministic_analysis(source: FrozenCompactionSource) -> dict[str, Any]:
    user_constraints = []
    completed_actions = []
    tool_findings = []
    for index, message in enumerate(source.covered_messages):
        absolute_index = source.covered_from + index
        content = " ".join(str(message.get("content", "")).split())[:400]
        if message.get("role") == "user" and content:
            user_constraints.append(
                {"claim": content, "source_ref": f"message:{absolute_index}"}
            )
        elif message.get("role") == "tool":
            tool_findings.append(
                {
                    "tool": str(message.get("name", "unknown")),
                    "result": content,
                    "source_ref": f"message:{absolute_index}",
                }
            )
        elif message.get("role") == "assistant" and message.get("tool_calls"):
            completed_actions.extend(
                {
                    "tool": str(call.get("name", "unknown")),
                    "source_ref": f"message:{absolute_index}",
                }
                for call in message.get("tool_calls", [])[:20]
            )
    working = source.working_state
    return {
        "covered_range": source.covered_range,
        "active_goals": _bounded_items(working.get("current_objective")),
        "user_constraints": user_constraints[:20],
        "decisions": _bounded_items(working.get("decisions")),
        "completed_actions": completed_actions[:20],
        "changed_files": _bounded_items(working.get("changed_files")),
        "symbols_touched": _bounded_items(working.get("symbols_touched")),
        "tool_findings": tool_findings[:20],
        "failed_attempts": _bounded_items(working.get("last_failure")),
        "verification_results": _bounded_items(working.get("verification")),
        "unresolved_questions": _bounded_items(
            working.get("unresolved_questions")
        ),
        "next_steps": _bounded_items(working.get("next_expected_action")),
        "contradictions": [],
        "discarded_noise": [],
    }


# [maint-note:84e1bc3690fc]
# 维护导航：_deterministic_checkpoint；下列为本函数源码中的接口与路径索引。
# 输入 source：FrozenCompactionSource。
# 返回契约：dict[str, Any]。
# 返回路径：{'version': 1, 'covered_range': source.covered_range, 'task_objective': objective or str(old.get('task_objective', ''))[:800], 'must_preserve': _merge_checkpoint_items(old.get('must_preserve'), must_preserve), 'decisions': _merge_checkpoint。
# 直接调用：_deterministic_analysis。
# 直接调用：str。
# 直接调用：working.get。
# 直接调用：item.get。
# 直接调用：_bounded_items。
# 直接调用：old.get。
# 直接调用：_merge_checkpoint_items。
# 直接调用：range。
# 直接调用：min。
# 集合派生：item 来自 analysis['user_constraints']；过滤 item.get('claim')。
# 集合派生：item 来自 _bounded_items(working.get('changed_files'))。
# 集合派生：item 来自 _bounded_items(working.get('inspected_files'))。
# 集合派生：index 来自 range(source.covered_from, min(source.covered_to + 1, source.covered_from + 20))。
# 表达式分流：working.get('repository_revision') is not None；真值与假值分支分别计算，不同时执行。
# [/maint-note:84e1bc3690fc]
def _deterministic_checkpoint(source: FrozenCompactionSource) -> dict[str, Any]:
    analysis = _deterministic_analysis(source)
    working = source.working_state
    objective = str(working.get("current_objective") or "")[:800]
    must_preserve = [
        str(item.get("claim", ""))[:400]
        for item in analysis["user_constraints"]
        if item.get("claim")
    ]
    repository_state = [
        *[f"changed: {item}" for item in _bounded_items(working.get("changed_files"))],
        *[
            f"inspected: {item}"
            for item in _bounded_items(working.get("inspected_files"))
        ],
    ][:20]
    open_work = [
        *_bounded_items(working.get("pending_tool_calls")),
        *_bounded_items(working.get("next_expected_action")),
    ][:20]
    artifacts = [
        *_bounded_items(working.get("important_artifacts")),
        *_bounded_items(working.get("artifacts")),
    ][:20]
    old = source.old_checkpoint or {}
    return {
        "version": 1,
        "covered_range": source.covered_range,
        "task_objective": objective or str(old.get("task_objective", ""))[:800],
        "must_preserve": _merge_checkpoint_items(
            old.get("must_preserve"), must_preserve
        ),
        "decisions": _merge_checkpoint_items(
            old.get("decisions"), working.get("decisions")
        ),
        "current_repository_state": _merge_checkpoint_items(
            old.get("current_repository_state"), repository_state
        ),
        "completed_work": _merge_checkpoint_items(
            old.get("completed_work"), working.get("completed_tool_calls")
        ),
        "failed_approaches": _merge_checkpoint_items(
            old.get("failed_approaches"), working.get("last_failure")
        ),
        "verification_state": _merge_checkpoint_items(
            old.get("verification_state"), working.get("verification")
        ),
        "open_work": _merge_checkpoint_items(old.get("open_work"), open_work),
        "important_artifacts": _merge_checkpoint_items(
            old.get("important_artifacts"), artifacts
        ),
        "source_refs": _merge_checkpoint_items(
            old.get("source_refs"),
            [
                f"message:{index}"
                for index in range(
                    source.covered_from,
                    min(source.covered_to + 1, source.covered_from + 20),
                )
            ],
        ),
        "repository_revision": (
            str(working["repository_revision"])[:200]
            if working.get("repository_revision") is not None
            else old.get("repository_revision")
        ),
    }


# [maint-note:825876b8b13f]
# 维护导航：_deterministic_thread_analysis；下列为本函数源码中的接口与路径索引。
# 输入 source：FrozenThreadCompactionSource。
# 返回契约：dict[str, Any]。
# 分支条件：message.get('role') == 'user' and content。
# 遍历边界：message ← source.covered_messages。
# 返回路径：{'covered_range': source.covered_range, 'active_goals': _bounded_items(working.get('current_objective')), 'user_constraints': user_constraints[:20], 'decisions': _bounded_items(working.get('decisions')), 'completed_actions': _bounded_items(。
# 直接调用：' '.join。
# 直接调用：str(message.get('content', '')).split。
# 直接调用：str。
# 直接调用：message.get。
# 直接调用：user_constraints.append。
# 直接调用：_bounded_items。
# 直接调用：working.get。
# [/maint-note:825876b8b13f]
def _deterministic_thread_analysis(
    source: FrozenThreadCompactionSource,
) -> dict[str, Any]:
    user_constraints: list[dict[str, str]] = []
    for message in source.covered_messages:
        content = " ".join(str(message.get("content", "")).split())[:400]
        if message.get("role") == "user" and content:
            user_constraints.append(
                {
                    "claim": content,
                    "source_ref": f"conversation:{message['sequence']}",
                }
            )
    working = source.working_state
    return {
        "covered_range": source.covered_range,
        "active_goals": _bounded_items(working.get("current_objective")),
        "user_constraints": user_constraints[:20],
        "decisions": _bounded_items(working.get("decisions")),
        "completed_actions": _bounded_items(working.get("completed_work")),
        "changed_files": _bounded_items(working.get("changed_files")),
        "symbols_touched": _bounded_items(working.get("symbols_touched")),
        "tool_findings": [],
        "failed_attempts": _bounded_items(working.get("last_failure")),
        "verification_results": _bounded_items(working.get("verification")),
        "unresolved_questions": _bounded_items(
            working.get("unresolved_questions")
        ),
        "next_steps": _bounded_items(working.get("next_expected_action")),
        "contradictions": [],
        "discarded_noise": [],
    }


# [maint-note:7514ea4229d1]
# 维护导航：_deterministic_thread_checkpoint；下列为本函数源码中的接口与路径索引。
# 输入 source：FrozenThreadCompactionSource。
# 返回契约：dict[str, Any]。
# 返回路径：{'version': 1, 'covered_range': source.covered_range, 'task_objective': str(working.get('current_objective') or old.get('task_objective') or '')[:800], 'must_preserve': must_preserve, 'decisions': _merge_checkpoint_items(old.get('decisions'。
# 直接调用：_deterministic_thread_analysis。
# 直接调用：_merge_checkpoint_items。
# 直接调用：old.get。
# 直接调用：str。
# 直接调用：item.get。
# 直接调用：working.get。
# 集合派生：item 来自 analysis['user_constraints']；过滤 item.get('claim')。
# 集合派生：message 来自 source.covered_messages[:20]。
# 表达式分流：working.get('repository_revision') is not None；真值与假值分支分别计算，不同时执行。
# [/maint-note:7514ea4229d1]
def _deterministic_thread_checkpoint(
    source: FrozenThreadCompactionSource,
) -> dict[str, Any]:
    analysis = _deterministic_thread_analysis(source)
    working = source.working_state
    old = source.old_checkpoint or {}
    must_preserve = _merge_checkpoint_items(
        old.get("must_preserve"),
        [
            str(item.get("claim", ""))[:400]
            for item in analysis["user_constraints"]
            if item.get("claim")
        ],
    )
    return {
        "version": 1,
        "covered_range": source.covered_range,
        "task_objective": str(
            working.get("current_objective")
            or old.get("task_objective")
            or ""
        )[:800],
        "must_preserve": must_preserve,
        "decisions": _merge_checkpoint_items(
            old.get("decisions"), working.get("decisions")
        ),
        "current_repository_state": _merge_checkpoint_items(
            old.get("current_repository_state"), working.get("repository_state")
        ),
        "completed_work": _merge_checkpoint_items(
            old.get("completed_work"), working.get("completed_work")
        ),
        "failed_approaches": _merge_checkpoint_items(
            old.get("failed_approaches"), working.get("last_failure")
        ),
        "verification_state": _merge_checkpoint_items(
            old.get("verification_state"), working.get("verification")
        ),
        "open_work": _merge_checkpoint_items(
            old.get("open_work"), working.get("next_expected_action")
        ),
        "important_artifacts": _merge_checkpoint_items(
            old.get("important_artifacts"), working.get("important_artifacts")
        ),
        "source_refs": _merge_checkpoint_items(
            old.get("source_refs"),
            [
                f"conversation:{message['sequence']}"
                for message in source.covered_messages[:20]
            ],
        ),
        "repository_revision": (
            str(working["repository_revision"])[:200]
            if working.get("repository_revision") is not None
            else old.get("repository_revision")
        ),
    }


# [maint-note:f66daf131fe3]
# 维护导航：_parse_and_validate；下列为本函数源码中的接口与路径索引。
# 输入 content：str。
# 输入 schema：dict[str, Any]。
# 输入 covered_range：dict[str, int]。
# 返回契约：dict[str, Any]。
# 分支条件：not isinstance(value, dict)。
# 分支条件：missing。
# 分支条件：value.get('covered_range') != covered_range。
# 分支条件：expected_type == 'array' and (not isinstance(item, list))。
# 分支条件：expected_type == 'string' and (not isinstance(item, str))。
# 分支条件：expected_type == 'integer' and (not isinstance(item, int))。
# 分支条件：'const' in contract and item != contract['const']。
# 分支条件：isinstance(expected_type, list)。
# 分支条件：not isinstance(item, allowed)。
# 分支条件：schema.get('additionalProperties') is False。
# 分支条件：extra。
# 遍历边界：(name, contract) ← properties.items()。
# 返回路径：value。
# 显式抛出：CompactionValidationError(f'invalid compaction JSON: {exc}')。
# 显式抛出：CompactionValidationError('compaction output must be a JSON object')。
# 显式抛出：CompactionValidationError('compaction output is missing required fields: ' + ', '.join(missing))。
# 显式抛出：CompactionValidationError('compaction covered_range does not match source')。
# 显式抛出：CompactionValidationError(f'compaction field {name} must be an array')。
# 显式抛出：CompactionValidationError(f'compaction field {name} must be a string')。
# 显式抛出：CompactionValidationError(f'compaction field {name} must be an integer')。
# 显式抛出：CompactionValidationError(f"compaction field {name} must equal {contract['const']}")。
# 显式抛出：CompactionValidationError(f'compaction field {name} has an invalid type')。
# 显式抛出：CompactionValidationError('compaction output has unsupported fields: ' + ', '.join(extra))。
# 捕获异常：json.JSONDecodeError。
# 直接调用：json.loads。
# 直接调用：CompactionValidationError。
# 直接调用：isinstance。
# 直接调用：set。
# 直接调用：sorted。
# 直接调用：', '.join。
# 直接调用：value.get。
# 直接调用：properties.items。
# 直接调用：contract.get。
# 直接调用：type。
# 直接调用：schema.get。
# 表达式分流：expected_type == ['string']；真值与假值分支分别计算，不同时执行。
# [/maint-note:f66daf131fe3]
def _parse_and_validate(
    content: str,
    schema: dict[str, Any],
    covered_range: dict[str, int],
) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CompactionValidationError(f"invalid compaction JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CompactionValidationError("compaction output must be a JSON object")
    required = set(schema["required"])
    missing = sorted(required - set(value))
    if missing:
        raise CompactionValidationError(
            "compaction output is missing required fields: " + ", ".join(missing)
        )
    if value.get("covered_range") != covered_range:
        raise CompactionValidationError("compaction covered_range does not match source")
    properties = schema["properties"]
    for name, contract in properties.items():
        item = value.get(name)
        expected_type = contract.get("type")
        if expected_type == "array" and not isinstance(item, list):
            raise CompactionValidationError(f"compaction field {name} must be an array")
        if expected_type == "string" and not isinstance(item, str):
            raise CompactionValidationError(f"compaction field {name} must be a string")
        if expected_type == "integer" and not isinstance(item, int):
            raise CompactionValidationError(f"compaction field {name} must be an integer")
        if "const" in contract and item != contract["const"]:
            raise CompactionValidationError(
                f"compaction field {name} must equal {contract['const']}"
            )
        if isinstance(expected_type, list):
            allowed = (
                (str,) if expected_type == ["string"] else (str, type(None))
            )
            if not isinstance(item, allowed):
                raise CompactionValidationError(
                    f"compaction field {name} has an invalid type"
                )
    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        if extra:
            raise CompactionValidationError(
                "compaction output has unsupported fields: " + ", ".join(extra)
            )
    return value


# [maint-note:7ab450fa94f5]
# 维护导航：_validate_object；下列为本函数源码中的接口与路径索引。
# 输入 value：dict[str, Any] | None。
# 输入 schema：dict[str, Any]。
# 输入 covered_range：dict[str, int]。
# 返回契约：dict[str, Any]。
# 分支条件：value is None。
# 返回路径：_parse_and_validate(json.dumps(value, ensure_ascii=False), schema, covered_range)。
# 显式抛出：CompactionValidationError('stored compaction data is missing')。
# 直接调用：CompactionValidationError。
# 直接调用：_parse_and_validate。
# 直接调用：json.dumps。
# [/maint-note:7ab450fa94f5]
def _validate_object(
    value: dict[str, Any] | None,
    schema: dict[str, Any],
    covered_range: dict[str, int],
) -> dict[str, Any]:
    if value is None:
        raise CompactionValidationError("stored compaction data is missing")
    return _parse_and_validate(
        json.dumps(value, ensure_ascii=False),
        schema,
        covered_range,
    )
