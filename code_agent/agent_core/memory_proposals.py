from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from queue import Full, Queue
from threading import Thread
from typing import Any

from .contracts import AgentEvent, ModelRequest
from .event_store import EventStore
from .memory_prompts import (
    ANALYZER_RESPONSE_FORMAT,
    COMPOSER_RESPONSE_FORMAT,
    MEMORY_ANALYZER_PROMPT_VERSION,
    MEMORY_COMPOSER_PROMPT_VERSION,
    build_analyzer_messages,
    build_composer_messages,
    build_repair_messages,
)
from .memory_projection import project_memory_review
from .memory_store import MemoryProposalConflict, MemoryStore
from .memory_tool_recovery import TOOL_RECOVERY_VERSION, ToolArgumentRecovery, find_tool_argument_recoveries
from .model_gateway import ModelGateway, RuntimeModelConfig
from .token_budget import ContextHardLimitExceeded
from .tool_registry import redact_sensitive_text


logger = logging.getLogger(__name__)
VALID_OPERATIONS = {"create", "update", "supersede", "expire"}
VALID_SCOPES = {"run", "thread", "project", "user"}
VALID_KINDS = {
    "user_preference",
    "project_constraint",
    "architecture_decision",
    "verified_fact",
    "repository_convention",
    "workflow",
    "failure_pattern",
    "successful_approach",
    "important_artifact",
    "task_summary",
}
VALID_VERIFICATIONS = {
    "user_asserted",
    "tool_observed",
    "test_verified",
    "diff_verified",
    "model_inferred",
    "legacy_unverified",
}
VALID_DURABILITIES = {"session", "temporary", "long_term"}
MAX_EVIDENCE_ITEMS = 50
MAX_PROPOSALS = 8
MAX_SUBJECT_CHARS = 200
MAX_CONTENT_CHARS = 4_000

_DURABLE_USER_SIGNAL = re.compile(
    r"^(?:请\s*)?(?:(?:为(?:本|当前)?(?:项目|仓库|团队)|为我|帮我|替我|把|将).{0,22})?"
    r"(?:记住|记录|保存|提议保存|纠正|更正)"
    r"|(?:纠正|更正|更新|替换).{0,26}(?:记忆|原规则|之前|规则|规范|约束|偏好)"
    r"|(?:长期|以后|今后|从今往后).{0,30}(?:规则|规范|约束|偏好|经验|要求|验收|必须|先)"
    r"|^(?:please\s+)?(?:remember|always|never|from now on|going forward)\b"
    r"|\b(?:all changes must|correction to|correct the previous)\b",
    re.IGNORECASE,
)


def _has_durable_user_signal(content: str) -> bool:
    for sentence in re.split(r"(?<=[。！？!?;；])|\n+", content):
        sentence = sentence.strip()
        if sentence.endswith(("?", "？")) or re.match(
            r"^(?:请\s*)?(?:我猜|假设|复述|回忆|查询|查找|告诉我|列出|总结|"
            r"maybe\b|suppose\b|recall\b|summarize\b|what\b)",
            sentence, re.I,
        ):
            continue
        if _DURABLE_USER_SIGNAL.search(sentence):
            return True
    return False


class ProposalValidationError(ValueError):
    pass


class MemoryProposalGenerationError(RuntimeError):
    pass


class AutoMemoryWorker:
    """Process one bounded queue of completed runs without delaying responses."""

    def __init__(self, service: MemoryProposalService, *, auto_apply: bool) -> None:
        self._service = service
        self._auto_apply = auto_apply
        self._pending: Queue[tuple[str, dict[str, Any]]] = Queue(maxsize=8)
        Thread(target=self._work, name="naturalcc-auto-memory", daemon=True).start()

    def submit(self, run_id: str, request_metadata: dict[str, Any]) -> None:
        try:
            if not self._service.event_store.queue_memory_extraction(run_id, os.getpid()):
                return
            self._pending.put_nowait((run_id, request_metadata))
        except Full:
            self._record(run_id, "skipped", reason="queue_full")
        except sqlite3.Error as exc:
            logger.warning("Unable to queue auto memory for %s (%s)", run_id, type(exc).__name__)

    def _record(self, run_id: str, status: str, **details: Any) -> None:
        try:
            self._service.event_store.update_memory_extraction(run_id, status, **details)
        except sqlite3.Error as exc:
            logger.warning("Unable to record auto memory for %s (%s)", run_id, type(exc).__name__)

    def _work(self) -> None:
        while True:
            run_id, request_metadata = self._pending.get()
            try:
                self._record(run_id, "running")
                result = self._service.extract_completed_run(
                    run_id, request_metadata, auto_apply=self._auto_apply
                )
                self._record(run_id, **result)
            except Exception as exc:
                cause = exc.__cause__ if isinstance(exc, MemoryProposalGenerationError) else None
                reason = type(cause if cause is not None else exc).__name__
                self._record(run_id, "failed", reason=reason)
                logger.warning("Auto memory failed for run %s (%s)", run_id, type(exc).__name__)
            finally:
                del request_metadata
                self._pending.task_done()


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return _content_hash(payload)


def _parse_json_object(content: str) -> dict[str, Any]:
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProposalValidationError(f"model returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ProposalValidationError("model response must be a JSON object")
    return parsed


class MemoryProposalService:
    def __init__(
        self,
        *,
        event_store: EventStore,
        memory_store: MemoryStore,
        gateway: ModelGateway,
        model_name: str = "",
        context_planner: Any | None = None,
    ) -> None:
        self.event_store = event_store
        self.memory_store = memory_store
        self.gateway = gateway
        self.model_name = model_name or getattr(gateway, "model", "") or "configured-model"
        self.context_planner = context_planner

    def create_from_selection(
        self,
        *,
        thread_id: str,
        project_id: str | None,
        evidence_refs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        thread = self.event_store.get_thread(thread_id)
        expected_project = str(thread.get("workspace") or "")
        effective_project = str(project_id or expected_project)
        if expected_project and effective_project != expected_project:
            raise ProposalValidationError("project_id does not match the selected thread")
        evidence = self._freeze_evidence(thread_id, evidence_refs)
        if not evidence:
            raise ProposalValidationError("select at least one evidence item")
        runtime_model_config = RuntimeModelConfig.from_dict(
            thread.get("runtime_model_config"),
            default_model=str(thread.get("model") or "deepseek-chat"),
        )
        proposals, _ = self._create_proposals(
            thread_id=thread_id,
            project_id=effective_project or None,
            evidence=evidence,
            request_metadata={"runtime_model_config": runtime_model_config.to_dict()},
        )
        return proposals

    def extract_completed_run(
        self, run_id: str, request_metadata: dict[str, Any], *, auto_apply: bool
    ) -> dict[str, Any]:
        run = self.event_store.get_run(run_id)
        thread_id = run["thread_id"]
        if run["status"] != "completed" or not thread_id:
            return {"status": "skipped", "reason": "no_completed_thread"}
        messages = self.event_store.list_conversation_messages(thread_id, run_id=run_id, limit=6)
        user_evidence_ids = {
            message["id"] for message in messages
            if message["role"] == "user" and _has_durable_user_signal(message["content"])
        }
        verified = [
            event for event in self.event_store.list_events(run_id, event_type="verification.finished")
            if event.payload.get("passed") is True and event.payload.get("tool")
        ][-2:]
        finished = self.event_store.list_events(run_id, event_type="tool.finished")
        recoveries = []
        if len(finished) >= 3:
            started = self.event_store.list_events(run_id, event_type="tool.started")
            recoveries = find_tool_argument_recoveries(sorted(
                [*started, *finished], key=lambda event: event.sequence
            ))
        reviews = self._propose_tool_recoveries(
            run, recoveries, request_key=request_metadata.get("api_key")
        )
        reason = "no_new_valid_proposals" if recoveries else "no_durable_evidence"
        if user_evidence_ids or verified:
            try:
                extracted, reason = self._propose_from_run_evidence(
                    run, messages, user_evidence_ids, verified, request_metadata
                )
                reviews.extend(extracted)
            except (MemoryProposalGenerationError, ContextHardLimitExceeded) as exc:
                if not reviews:
                    raise
                reason = "additional_model_proposals_failed"
                logger.warning("Additional memory extraction failed for %s (%s)", run_id, type(exc).__name__)

        if not reviews:
            return {
                "status": "abstained" if reason in {"analyzer_abstained", "composer_abstained"} else "skipped",
                "reason": reason,
            }
        applied_count = 0
        if auto_apply:
            for review in reviews:
                if review["status"] != "review_ready":
                    continue
                proposal_id = review["proposal_id"]
                try:
                    self._assert_evidence_stable(proposal_id)
                    self.memory_store.approve_proposal(
                        proposal_id, expected_version=0, actor="auto_memory"
                    )
                    applied_count += 1
                except MemoryProposalConflict:
                    # Ambiguous or concurrently edited proposals remain visible for review.
                    continue
        return {
            "status": "completed",
            "reason": reason if reason == "additional_model_proposals_failed" else None,
            "proposal_count": len(reviews),
            "applied_count": applied_count,
        }

    def _propose_tool_recoveries(
        self, run: dict[str, Any], recoveries: list[ToolArgumentRecovery], *,
        request_key: str | None,
    ) -> list[dict[str, Any]]:
        run_id = run["id"]
        reviews = []
        for recovery in recoveries:
            arguments = ", ".join(recovery.arguments)
            subject = f"{recovery.tool} 参数检查（{arguments}）"
            content = (
                f"调用 {recovery.tool} 前，确认已提供必填参数 {arguments}。"
                "遇到 missing required arguments 时，补齐提示中的参数再重试。"
                "同一任务中补齐这些参数后，该工具调用成功。"
            )
            if len(subject) > MAX_SUBJECT_CHARS:
                continue
            evidence = self._freeze_evidence(
                run["thread_id"],
                [{"type": "run_event", "run_id": run_id, "sequence": event.sequence}
                 for event in recovery.evidence],
                run_id=run_id, request_key=request_key,
            )
            if any(item["redacted"] for item in evidence):
                continue
            refs = [item["ref"] for item in evidence]
            proposal_id = self.memory_store.create_proposal(
                operation="create",
                target_memory_id=None,
                source_mode="automatic_run",
                project_id=run["workspace"],
                thread_id=run["thread_id"],
                run_id=run_id,
                scope="project",
                kind="successful_approach",
                subject=subject,
                canonical_content=content,
                verification="tool_observed",
                durability="long_term",
                confidence=0.95,
                expires_at=None,
                analysis={"contradictions": [], "sensitive_findings": []},
                raw_proposal={"evidence_refs": refs},
                warnings=[],
                conflicts=[],
                source_hash=_canonical_hash([
                    {"ref": item["ref"], "hash": item["content_hash"]} for item in evidence
                ]),
                model="runtime",
                analyzer_prompt_version=TOOL_RECOVERY_VERSION,
                composer_prompt_version=TOOL_RECOVERY_VERSION,
                evidence=evidence,
            )
            if proposal_id:
                reviews.append(self.get_review(proposal_id))
        return reviews

    def _propose_from_run_evidence(
        self,
        run: dict[str, Any],
        messages: list[dict[str, Any]],
        user_evidence_ids: set[str],
        verified: list[AgentEvent],
        request_metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None]:
        run_id = run["id"]
        thread_id = run["thread_id"]
        direct_refs = {f"conversation:{item}" for item in user_evidence_ids}
        direct_refs.update(f"event:{run_id}:{event.sequence}" for event in verified)
        refs = [
            {"type": "conversation_message", "source_id": message["id"]}
            for message in messages
            if message["id"] in user_evidence_ids or (
                verified and (
                    message["role"] == "user"
                    or (message["role"] == "assistant" and message["kind"] == "final")
                )
            )
        ]
        refs.extend(
            {"type": "run_event", "run_id": run_id, "sequence": event.sequence}
            for event in verified
        )
        evidence = self._freeze_evidence(
            thread_id, refs, run_id=run_id, request_key=request_metadata.get("api_key")
        )
        # Existing notes guide deduplication and corrections. They cannot serve as fresh evidence.
        existing = self.memory_store.automatic_memory_manifest(run["workspace"])
        for item in existing:
            item["subject"] = redact_sensitive_text(item["subject"])
            item["summary"] = redact_sensitive_text(item["summary"])
        return self._create_proposals(
            thread_id=thread_id,
            project_id=run["workspace"],
            evidence=evidence,
            request_metadata={**request_metadata, "timeout_seconds": 25},
            run_id=run_id,
            existing_memories=existing,
            direct_refs=direct_refs,
            explicit_only=not verified,
        )

    def _create_proposals(
        self,
        *,
        thread_id: str,
        project_id: str | None,
        evidence: list[dict[str, Any]],
        request_metadata: dict[str, Any],
        run_id: str | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        direct_refs: set[str] | None = None,
        explicit_only: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        runtime_model_config = RuntimeModelConfig.from_dict(request_metadata["runtime_model_config"])
        profile = self._profile_for(runtime_model_config)
        composer_tokens = 1536 if run_id else 3072
        if explicit_only:
            # Preserve the user's actual statement through composition instead of
            # asking a second model pass to infer whether to keep it.
            analysis = {
                "mode": "explicit_user_guidance",
                "claims": [],
                "contradictions": [],
                "sensitive_findings": [],
                "abstain": False,
            }
        else:
            analysis = self._analyze_evidence(
                evidence, direct_refs=direct_refs, max_output_tokens=1536 if run_id else 4096,
                request_metadata=request_metadata, profile=profile,
            )
        if analysis["abstain"]:
            return [], "analyzer_abstained"

        composer_response = self._generate(
            ModelRequest(
                messages=build_composer_messages(evidence, analysis, existing_memories, direct_refs),
                tools=[],
                purpose="memory_proposal",
                max_output_tokens=composer_tokens,
                response_format={"type": "json_object"},
            ),
            request_metadata=request_metadata,
            profile=profile,
        )
        allowed_refs = (
            direct_refs if direct_refs is not None else {item["ref"] for item in evidence}
        )
        evidence_by_ref = {item["ref"]: item for item in evidence}
        try:
            validated_proposals = self._parse_proposals(
                composer_response.content, allowed_refs, evidence_by_ref
            )
        except ProposalValidationError as first_error:
            repaired = self._repair_output(
                stage="memory_proposal",
                invalid_content=composer_response.content,
                validation_error=first_error,
                required_schema=COMPOSER_RESPONSE_FORMAT["json_schema"]["schema"],
                max_output_tokens=composer_tokens,
                request_metadata=request_metadata,
                profile=profile,
            )
            try:
                validated_proposals = self._parse_proposals(
                    repaired.content, allowed_refs, evidence_by_ref
                )
            except ProposalValidationError as exc:
                raise MemoryProposalGenerationError(
                    f"memory_proposal returned invalid output after one repair: {exc}"
                ) from exc

        source_hash = _canonical_hash(
            [{"ref": item["ref"], "hash": item["content_hash"]} for item in evidence]
        )
        created: list[dict[str, Any]] = []
        targets = {item["id"]: item for item in existing_memories or []}
        for raw, proposal in validated_proposals:
            target = targets.get(proposal["target_memory_id"])
            proposal_evidence = evidence
            if run_id:
                if (
                    proposal["scope"] not in {"project", "user"}
                    or proposal["durability"] != "long_term"
                    or proposal["kind"] == "task_summary"
                ):
                    continue
                if not any(
                    evidence_by_ref[ref]["verification"] == proposal["verification"]
                    for ref in raw["evidence_refs"]
                ):
                    continue
                if proposal["target_memory_id"] and (
                    target is None or target["scope"] != proposal["scope"]
                    or target["kind"] != proposal["kind"]
                ):
                    continue
                # Keep server metadata separate from the model's claimed verification.
                raw = {**raw, "_auto_target_updated_at": target["updated_at"] if target else None}
                cited = set(raw["evidence_refs"])
                proposal_evidence = [item for item in evidence if item["ref"] in cited]
            proposal_id = self.memory_store.create_proposal(
                operation=proposal["operation"],
                target_memory_id=proposal["target_memory_id"],
                source_mode="automatic_run" if run_id else "manual_selection",
                project_id=project_id,
                thread_id=thread_id,
                run_id=run_id,
                scope=proposal["scope"],
                kind=proposal["kind"],
                subject=proposal["subject"],
                canonical_content=proposal["canonical_content"],
                verification=proposal["verification"],
                durability=proposal["durability"],
                confidence=proposal["confidence"],
                expires_at=proposal["expires_at"],
                analysis=analysis,
                raw_proposal=raw,
                warnings=(
                    ["证据中的敏感片段已在模型分析前脱敏。"]
                    if any(item["redacted"] for item in evidence)
                    else []
                ),
                conflicts=proposal["conflicts_with"],
                source_hash=source_hash,
                model=runtime_model_config.model if run_id else self.model_name,
                analyzer_prompt_version=(
                    "explicit-user-v1" if explicit_only else MEMORY_ANALYZER_PROMPT_VERSION
                ),
                composer_prompt_version=MEMORY_COMPOSER_PROMPT_VERSION,
                evidence=proposal_evidence,
            )
            if proposal_id:
                created.append(self.get_review(proposal_id))
        reason = None
        if not created:
            reason = "no_new_valid_proposals" if validated_proposals else "composer_abstained"
        return created, reason

    def _analyze_evidence(
        self,
        evidence: list[dict[str, Any]],
        *,
        direct_refs: set[str] | None,
        max_output_tokens: int,
        request_metadata: dict[str, Any],
        profile: Any,
    ) -> dict[str, Any]:
        response = self._generate(
            ModelRequest(
                messages=build_analyzer_messages(evidence, direct_refs),
                purpose="memory_analysis", max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            ),
            request_metadata=request_metadata, profile=profile,
        )
        try:
            return self._parse_analysis(response.content)
        except ProposalValidationError as first_error:
            repaired = self._repair_output(
                stage="memory_analysis",
                invalid_content=response.content,
                validation_error=first_error,
                required_schema=ANALYZER_RESPONSE_FORMAT["json_schema"]["schema"],
                max_output_tokens=max_output_tokens,
                request_metadata=request_metadata,
                profile=profile,
            )
            try:
                return self._parse_analysis(repaired.content)
            except ProposalValidationError as exc:
                raise MemoryProposalGenerationError(
                    f"memory_analysis returned invalid output after one repair: {exc}"
                ) from exc

    def _profile_for(self, config: RuntimeModelConfig):
        if self.context_planner is None:
            return None
        from dataclasses import replace

        return replace(
            self.context_planner.profile,
            model=config.model,
            context_window_tokens=config.context_window_tokens,
            safety_margin_tokens=config.safety_margin_tokens,
        )

    def _generate(
        self,
        request: ModelRequest,
        *,
        request_metadata: dict[str, Any] | None = None,
        profile: Any | None = None,
    ):
        if self.context_planner is not None:
            self.context_planner.counter.assert_fits(
                request.messages,
                request.tools,
                self.context_planner.serializer,
                profile or self.context_planner.profile,
                reserved_output_tokens=request.max_output_tokens,
            )
        try:
            return self.gateway.generate(
                ModelRequest(
                    messages=request.messages,
                    tools=request.tools,
                    purpose=request.purpose,
                    max_output_tokens=request.max_output_tokens,
                    response_format=request.response_format,
                    metadata={**(request_metadata or {}), **request.metadata},
                )
            )
        except Exception as exc:
            detail = redact_sensitive_text(str(exc)) or type(exc).__name__
            raise MemoryProposalGenerationError(
                f"{request.purpose} model request failed: {detail}"
            ) from exc

    def _repair_output(
        self,
        *,
        stage: str,
        invalid_content: str,
        validation_error: ProposalValidationError,
        required_schema: dict[str, Any],
        max_output_tokens: int,
        request_metadata: dict[str, Any] | None = None,
        profile: Any | None = None,
    ):
        return self._generate(
            ModelRequest(
                messages=build_repair_messages(
                    stage=stage,
                    invalid_content=redact_sensitive_text(invalid_content),
                    validation_error=redact_sensitive_text(str(validation_error)),
                    required_schema=required_schema,
                ),
                tools=[],
                purpose=f"{stage}_repair",
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            ),
            request_metadata=request_metadata,
            profile=profile,
        )

    def _parse_analysis(self, content: str) -> dict[str, Any]:
        analysis = _parse_json_object(content)
        self._validate_analysis(analysis)
        return analysis

    def _parse_proposals(
        self,
        content: str,
        allowed_refs: set[str],
        evidence_by_ref: dict[str, dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        envelope = _parse_json_object(content)
        raw_proposals = envelope.get("proposals")
        if not isinstance(raw_proposals, list):
            raise ProposalValidationError(
                "proposal response must contain a proposals array"
            )
        if len(raw_proposals) > MAX_PROPOSALS:
            raise ProposalValidationError(
                f"at most {MAX_PROPOSALS} proposals are allowed"
            )
        validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for raw in raw_proposals:
            proposal = self._validate_proposal(raw, allowed_refs, evidence_by_ref)
            if proposal is not None:
                validated.append((raw, proposal))
        return validated

    def _freeze_evidence(
        self, thread_id: str, references: list[dict[str, Any]], *,
        run_id: str | None = None, request_key: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(references) > MAX_EVIDENCE_ITEMS:
            raise ProposalValidationError(
                f"at most {MAX_EVIDENCE_ITEMS} evidence items are allowed"
            )
        frozen: list[dict[str, Any]] = []
        seen: set[str] = set()
        messages = {
            item["id"]: item
            for item in self.event_store.list_conversation_messages(thread_id, limit=None, run_id=run_id)
        }
        for raw in references:
            evidence_type = str(raw.get("type") or "")
            if evidence_type == "conversation_message":
                source_id = str(raw.get("source_id") or "")
                message = messages.get(source_id)
                if message is None:
                    raise ProposalValidationError(
                        "conversation evidence does not belong to the selected thread"
                    )
                ref = f"conversation:{source_id}"
                content = str(message["content"])
                verification = "user_asserted" if message["role"] == "user" else "model_inferred"
                locator = f"对话 #{message['sequence']}"
                role = message["role"]
                sequence = int(message["sequence"])
            elif evidence_type == "run_event":
                evidence_run_id = str(raw.get("run_id") or "")
                sequence = int(raw.get("sequence") or 0)
                run = self.event_store.get_run(evidence_run_id)
                if run.get("thread_id") != thread_id or (run_id and evidence_run_id != run_id):
                    raise ProposalValidationError(
                        "run-event evidence does not belong to the selected thread"
                    )
                event = next(
                    (
                        item
                        for item in self.event_store.list_events(evidence_run_id, sequence - 1)
                        if item.sequence == sequence
                    ),
                    None,
                )
                if event is None:
                    raise ProposalValidationError("unknown run-event evidence")
                ref = f"event:{evidence_run_id}:{sequence}"
                content = json.dumps(
                    {"type": event.type, "payload": event.payload}, ensure_ascii=False
                )
                evidence_type, verification = self._classify_event(event.type)
                locator = f"任务 {evidence_run_id[:8]} · 事件 #{sequence}"
                source_id = f"{evidence_run_id}:{sequence}"
                role = None
            else:
                raise ProposalValidationError(f"unsupported evidence type: {evidence_type}")
            if ref in seen:
                continue
            seen.add(ref)
            model_content = redact_sensitive_text(content)
            if request_key:
                model_content = model_content.replace(request_key, "[REDACTED]")
            model_content_limit = (2_000 if role else 800) if run_id else 12_000
            frozen.append(
                {
                    "ref": ref,
                    "evidence_type": evidence_type,
                    "source_id": source_id,
                    "source_sequence": sequence,
                    "content_hash": _content_hash(content),
                    "preview": model_content[:500],
                    "model_content": model_content[:model_content_limit],
                    "source_locator": locator,
                    "verification": verification,
                    "role": role,
                    "redacted": model_content != content,
                }
            )
        return frozen

    @staticmethod
    def _classify_event(event_type: str) -> tuple[str, str]:
        if event_type == "verification.finished":
            return "verification_result", "test_verified"
        if event_type == "tool.finished":
            return "tool_result", "tool_observed"
        return "run_event", "tool_observed"

    @staticmethod
    def _validate_analysis(analysis: dict[str, Any]) -> None:
        for field in ("claims", "contradictions", "sensitive_findings"):
            if not isinstance(analysis.get(field), list):
                raise ProposalValidationError(f"analysis.{field} must be an array")
        if not isinstance(analysis.get("abstain"), bool):
            raise ProposalValidationError("analysis.abstain must be boolean")

    def _validate_proposal(
        self,
        raw: Any,
        allowed_refs: set[str],
        evidence_by_ref: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            raise ProposalValidationError("each proposal must be an object")
        if raw.get("abstained"):
            return None
        if raw.get("schema_version") != 1:
            raise ProposalValidationError("unsupported proposal schema_version")
        operation = str(raw.get("operation") or "")
        scope = str(raw.get("scope") or "")
        kind = str(raw.get("kind") or "")
        verification = str(raw.get("verification") or "")
        durability = str(raw.get("durability") or "")
        if operation not in VALID_OPERATIONS:
            raise ProposalValidationError("invalid proposal operation")
        if scope not in VALID_SCOPES:
            raise ProposalValidationError("invalid proposal scope")
        if kind not in VALID_KINDS:
            raise ProposalValidationError("invalid proposal kind")
        if verification not in VALID_VERIFICATIONS:
            raise ProposalValidationError("invalid verification level")
        if durability not in VALID_DURABILITIES:
            raise ProposalValidationError("invalid durability")
        subject = str(raw.get("subject") or "").strip()
        content = str(raw.get("canonical_content") or "").strip()
        if not subject or len(subject) > MAX_SUBJECT_CHARS:
            raise ProposalValidationError("proposal subject is empty or too long")
        if not content or len(content) > MAX_CONTENT_CHARS:
            raise ProposalValidationError("proposal content is empty or too long")
        if raw.get("sensitive"):
            raise ProposalValidationError("proposal contains sensitive information")
        if redact_sensitive_text(subject) != subject or redact_sensitive_text(content) != content:
            raise ProposalValidationError("proposal contains a credential-like secret")
        evidence_refs = raw.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ProposalValidationError("proposal must cite evidence")
        if not set(evidence_refs) <= allowed_refs:
            raise ProposalValidationError("proposal cites evidence outside the frozen bundle")
        verification_rank = {
            "legacy_unverified": 0,
            "model_inferred": 1,
            "user_asserted": 2,
            "tool_observed": 2,
            "test_verified": 3,
            "diff_verified": 3,
        }
        strongest_evidence = max(
            verification_rank[evidence_by_ref[ref]["verification"]]
            for ref in evidence_refs
        )
        if verification_rank[verification] > strongest_evidence:
            raise ProposalValidationError(
                "proposal verification level is stronger than its cited evidence"
            )
        if scope == "user" and verification != "user_asserted":
            raise ProposalValidationError("user-scope memory requires explicit user evidence")
        if scope in {"project", "user"} and verification == "model_inferred":
            raise ProposalValidationError(
                "model-inferred memory cannot use project or user scope"
            )
        confidence = raw.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ProposalValidationError("confidence must be between 0 and 1")
        target_memory_id = raw.get("target_memory_id")
        if operation == "create" and target_memory_id is not None:
            raise ProposalValidationError("create proposal cannot target an existing memory")
        if operation != "create":
            if not target_memory_id or self.memory_store.get(str(target_memory_id)) is None:
                raise ProposalValidationError("proposal target memory does not exist")
        conflicts = raw.get("conflicts_with") or []
        if not isinstance(conflicts, list):
            raise ProposalValidationError("conflicts_with must be an array")
        return {
            "operation": operation,
            "target_memory_id": target_memory_id,
            "scope": scope,
            "kind": kind,
            "subject": subject,
            "canonical_content": content,
            "verification": verification,
            "durability": durability,
            "expires_at": raw.get("expires_at"),
            "confidence": float(confidence),
            "conflicts_with": [str(item) for item in conflicts],
        }

    def get_review(self, proposal_id: str) -> dict[str, Any]:
        proposal = self.memory_store.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        evidence = self.memory_store.get_proposal_evidence(proposal_id)
        target = (
            self.memory_store.get(proposal["target_memory_id"])
            if proposal.get("target_memory_id")
            else None
        )
        return project_memory_review(proposal, evidence, target_memory=target)

    def list_reviews(
        self,
        *,
        status: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.get_review(item["id"])
            for item in self.memory_store.list_proposals(
                status=status, project_id=project_id, thread_id=thread_id
            )
        ]

    def update_review(
        self, proposal_id: str, *, expected_version: int, **changes: Any
    ) -> dict[str, Any]:
        self.memory_store.update_proposal(
            proposal_id, expected_version=expected_version, actor="user", **changes
        )
        return self.get_review(proposal_id)

    def approve(self, proposal_id: str, *, expected_version: int) -> dict[str, Any]:
        self._assert_evidence_stable(proposal_id)
        self.memory_store.approve_proposal(
            proposal_id, expected_version=expected_version, actor="user"
        )
        return self.get_review(proposal_id)

    def _assert_evidence_stable(self, proposal_id: str) -> None:
        proposal = self.memory_store.get_proposal(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        messages = {
            item["id"]: item
            for item in self.event_store.list_conversation_messages(
                proposal["thread_id"], limit=None, run_id=proposal["run_id"]
            )
        } if proposal.get("thread_id") else {}
        for item in self.memory_store.get_proposal_evidence(proposal_id):
            if item["evidence_type"] == "legacy_memory":
                continue
            if item["evidence_type"] == "conversation_message":
                message = messages.get(item["source_id"])
                content = str(message["content"]) if message else ""
            else:
                try:
                    run_id, sequence_text = item["source_id"].rsplit(":", 1)
                    sequence = int(sequence_text)
                    event = next(
                        (
                            event
                            for event in self.event_store.list_events(run_id, sequence - 1)
                            if event.sequence == sequence
                        ),
                        None,
                    )
                except (KeyError, ValueError):
                    event = None
                content = (
                    json.dumps(
                        {"type": event.type, "payload": event.payload},
                        ensure_ascii=False,
                    )
                    if event
                    else ""
                )
            if not content or _content_hash(content) != item["content_hash"]:
                raise MemoryProposalConflict(
                    "proposal evidence changed after analysis; regenerate before approval"
                )

    def reject(
        self, proposal_id: str, *, expected_version: int, reason: str = ""
    ) -> dict[str, Any]:
        self.memory_store.review_proposal(
            proposal_id,
            action="reject",
            expected_version=expected_version,
            actor="user",
            reason=reason,
        )
        return self.get_review(proposal_id)

    def defer(self,
        proposal_id: str,
        *,
        expected_version: int,
        reason: str = "",
    ) -> dict[str, Any]:
        self.memory_store.review_proposal(
            proposal_id,
            action="defer",
            expected_version=expected_version,
            actor="user",
            reason=reason,
        )
        return self.get_review(proposal_id)
