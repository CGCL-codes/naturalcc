from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .contracts import ModelRequest
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
from .model_gateway import ModelGateway
from .tool_registry import redact_sensitive_text


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


class ProposalValidationError(ValueError):
    pass


class MemoryProposalGenerationError(RuntimeError):
    pass


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

        analyzer_response = self._generate(
            ModelRequest(
                messages=build_analyzer_messages(evidence),
                tools=[],
                purpose="memory_analysis",
                max_output_tokens=4096,
                response_format={"type": "json_object"},
            )
        )
        try:
            analysis = self._parse_analysis(analyzer_response.content)
        except ProposalValidationError as first_error:
            repaired = self._repair_output(
                stage="memory_analysis",
                invalid_content=analyzer_response.content,
                validation_error=first_error,
                required_schema=ANALYZER_RESPONSE_FORMAT["json_schema"]["schema"],
                max_output_tokens=4096,
            )
            try:
                analysis = self._parse_analysis(repaired.content)
            except ProposalValidationError as exc:
                raise MemoryProposalGenerationError(
                    f"memory_analysis returned invalid output after one repair: {exc}"
                ) from exc
        if analysis["abstain"]:
            return []

        composer_response = self._generate(
            ModelRequest(
                messages=build_composer_messages(evidence, analysis),
                tools=[],
                purpose="memory_proposal",
                max_output_tokens=3072,
                response_format={"type": "json_object"},
            )
        )
        allowed_refs = {item["ref"] for item in evidence}
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
                max_output_tokens=3072,
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
        for raw, proposal in validated_proposals:
            proposal_id = self.memory_store.create_proposal(
                operation=proposal["operation"],
                target_memory_id=proposal["target_memory_id"],
                source_mode="manual_selection",
                project_id=effective_project or None,
                thread_id=thread_id,
                run_id=None,
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
                model=self.model_name,
                analyzer_prompt_version=MEMORY_ANALYZER_PROMPT_VERSION,
                composer_prompt_version=MEMORY_COMPOSER_PROMPT_VERSION,
                evidence=evidence,
            )
            created.append(self.get_review(proposal_id))
        return created

    def _generate(self, request: ModelRequest):
        if self.context_planner is not None:
            self.context_planner.counter.assert_fits(
                request.messages,
                request.tools,
                self.context_planner.serializer,
                self.context_planner.profile,
                reserved_output_tokens=request.max_output_tokens,
            )
        try:
            return self.gateway.generate(request)
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
            )
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
        self, thread_id: str, references: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(references) > MAX_EVIDENCE_ITEMS:
            raise ProposalValidationError(
                f"at most {MAX_EVIDENCE_ITEMS} evidence items are allowed"
            )
        frozen: list[dict[str, Any]] = []
        seen: set[str] = set()
        messages = {
            item["id"]: item
            for item in self.event_store.list_conversation_messages(thread_id, limit=None)
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
                run_id = str(raw.get("run_id") or "")
                sequence = int(raw.get("sequence") or 0)
                run = self.event_store.get_run(run_id)
                if run.get("thread_id") != thread_id:
                    raise ProposalValidationError(
                        "run-event evidence does not belong to the selected thread"
                    )
                event = next(
                    (
                        item
                        for item in self.event_store.list_events(run_id, sequence - 1)
                        if item.sequence == sequence
                    ),
                    None,
                )
                if event is None:
                    raise ProposalValidationError("unknown run-event evidence")
                ref = f"event:{run_id}:{sequence}"
                content = json.dumps(
                    {"type": event.type, "payload": event.payload}, ensure_ascii=False
                )
                evidence_type, verification = self._classify_event(event.type)
                locator = f"任务 {run_id[:8]} · 事件 #{sequence}"
                source_id = f"{run_id}:{sequence}"
                role = None
            else:
                raise ProposalValidationError(f"unsupported evidence type: {evidence_type}")
            if ref in seen:
                continue
            seen.add(ref)
            model_content = redact_sensitive_text(content)
            frozen.append(
                {
                    "ref": ref,
                    "evidence_type": evidence_type,
                    "source_id": source_id,
                    "source_sequence": sequence,
                    "content_hash": _content_hash(content),
                    "preview": model_content[:500],
                    "model_content": model_content[:12_000],
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
                proposal["thread_id"], limit=None
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
