from __future__ import annotations

from dataclasses import dataclass
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


class CompactionValidationError(ValueError):
    """Raised when model-produced compaction data violates its contract."""


class MaintenanceBudgetExhausted(RuntimeError):
    """Raised internally before a compaction model call exceeds its allowance."""


def canonical_message_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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

    @property
    def covered_range(self) -> dict[str, int]:
        return {"from": self.covered_from, "to": self.covered_to}


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

    @property
    def covered_range(self) -> dict[str, int]:
        return {"from": self.covered_from, "to": self.covered_to}


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

    def compact_run(
        self,
        source: FrozenCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
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
            response = self.gateway.generate(request)
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

    def resume_run(
        self,
        compaction_id: str,
        source: FrozenCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
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
            response = self.gateway.generate(request)
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

    def compact_thread(
        self,
        source: FrozenThreadCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
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
            response = self.gateway.generate(request)
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

    def resume_thread(
        self,
        compaction_id: str,
        source: FrozenThreadCompactionSource,
        *,
        max_maintenance_calls: int | None = None,
        max_maintenance_input_tokens: int | None = None,
        max_maintenance_output_tokens: int | None = None,
        deadline_epoch: float | None = None,
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
            response = self.gateway.generate(request)
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


def _merge_checkpoint_items(old: Any, new: Any, *, limit: int = 20) -> list[Any]:
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
