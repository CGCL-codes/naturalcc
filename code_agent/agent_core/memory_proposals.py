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


# [maint-note:463755bc06c2]
# 维护导航：_has_durable_user_signal；下列为本函数源码中的接口与路径索引。
# 输入 content：str。
# 返回契约：bool。
# 分支条件：sentence.endswith(('?', '？')) or re.match('^(?:请\\s*)?(?:我猜|假设|复述|回忆|查询|查找|告诉我|列出|总结|maybe\\b|suppose\\b|recall\\b|summarize\\b|what\\b)', sentence, re.I)。
# 分支条件：_DURABLE_USER_SIGNAL.search(sentence)。
# 遍历边界：sentence ← re.split('(?<=[。！？!?;；])|\\n+', content)。
# 返回路径：True。
# 返回路径：False。
# 直接调用：re.split。
# 直接调用：sentence.strip。
# 直接调用：sentence.endswith。
# 直接调用：re.match。
# 直接调用：_DURABLE_USER_SIGNAL.search。
# [/maint-note:463755bc06c2]
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


# [maint-note:3bc5c6f77b35]
# 数据与行为导航：ProposalValidationError。
# 继承接口：ValueError。
# [/maint-note:3bc5c6f77b35]
class ProposalValidationError(ValueError):
    pass


# [maint-note:33adfb429304]
# 数据与行为导航：MemoryProposalGenerationError。
# 继承接口：RuntimeError。
# [/maint-note:33adfb429304]
class MemoryProposalGenerationError(RuntimeError):
    pass


class AutoMemoryWorker:
    """Process one bounded queue of completed runs without delaying responses."""

    # [maint-note:397d6b35cb76]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 service：MemoryProposalService。
    # 输入 auto_apply：bool。
    # 返回契约：None。
    # 写入目标：self._service。
    # 写入目标：self._auto_apply。
    # 写入目标：self._pending。
    # 直接调用：Queue。
    # 直接调用：Thread(target=self._work, name='naturalcc-auto-memory', daemon=True).start。
    # 直接调用：Thread。
    # [/maint-note:397d6b35cb76]
    def __init__(self, service: MemoryProposalService, *, auto_apply: bool) -> None:
        self._service = service
        self._auto_apply = auto_apply
        self._pending: Queue[tuple[str, dict[str, Any]]] = Queue(maxsize=8)
        Thread(target=self._work, name="naturalcc-auto-memory", daemon=True).start()

    # [maint-note:21bc76f33445]
    # 维护导航：submit；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 request_metadata：dict[str, Any]。
    # 返回契约：None。
    # 分支条件：not self._service.event_store.queue_memory_extraction(run_id, os.getpid())。
    # 返回路径：无返回值。
    # 捕获异常：Full。
    # 捕获异常：sqlite3.Error。
    # 直接调用：self._service.event_store.queue_memory_extraction。
    # 直接调用：os.getpid。
    # 直接调用：self._pending.put_nowait。
    # 直接调用：self._record。
    # 直接调用：logger.warning。
    # 直接调用：type。
    # [/maint-note:21bc76f33445]
    def submit(self, run_id: str, request_metadata: dict[str, Any]) -> None:
        try:
            if not self._service.event_store.queue_memory_extraction(run_id, os.getpid()):
                return
            self._pending.put_nowait((run_id, request_metadata))
        except Full:
            self._record(run_id, "skipped", reason="queue_full")
        except sqlite3.Error as exc:
            logger.warning("Unable to queue auto memory for %s (%s)", run_id, type(exc).__name__)

    # [maint-note:ee62093055f0]
    # 维护导航：_record；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 status：str。
    # 扩展输入 details：收集调用方的额外参数。
    # 返回契约：None。
    # 捕获异常：sqlite3.Error。
    # 直接调用：self._service.event_store.update_memory_extraction。
    # 直接调用：logger.warning。
    # 直接调用：type。
    # [/maint-note:ee62093055f0]
    def _record(self, run_id: str, status: str, **details: Any) -> None:
        try:
            self._service.event_store.update_memory_extraction(run_id, status, **details)
        except sqlite3.Error as exc:
            logger.warning("Unable to record auto memory for %s (%s)", run_id, type(exc).__name__)

    # [maint-note:af7c78dd2a19]
    # 维护导航：_work；下列为本函数源码中的接口与路径索引。
    # 返回契约：None。
    # 遍历边界：True。
    # 捕获异常：Exception。
    # 直接调用：self._pending.get。
    # 直接调用：self._record。
    # 直接调用：self._service.extract_completed_run。
    # 直接调用：isinstance。
    # 直接调用：type。
    # 直接调用：logger.warning。
    # 直接调用：self._pending.task_done。
    # 清理边界：本函数含 finally；正常返回和异常展开都经过清理块。
    # 表达式分流：isinstance(exc, MemoryProposalGenerationError)；真值与假值分支分别计算，不同时执行。
    # 表达式分流：cause is not None；真值与假值分支分别计算，不同时执行。
    # [/maint-note:af7c78dd2a19]
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


# [maint-note:2ab0c00ff3ee]
# 维护导航：_content_hash；下列为本函数源码中的接口与路径索引。
# 输入 value：str。
# 返回契约：str。
# 返回路径：hashlib.sha256(value.encode('utf-8')).hexdigest()。
# 直接调用：hashlib.sha256(value.encode('utf-8')).hexdigest。
# 直接调用：hashlib.sha256。
# 直接调用：value.encode。
# [/maint-note:2ab0c00ff3ee]
def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# [maint-note:d29bad3fc28f]
# 维护导航：_canonical_hash；下列为本函数源码中的接口与路径索引。
# 输入 value：Any。
# 返回契约：str。
# 返回路径：_content_hash(payload)。
# 直接调用：json.dumps。
# 直接调用：_content_hash。
# [/maint-note:d29bad3fc28f]
def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return _content_hash(payload)


# [maint-note:58dfcc87bba1]
# 维护导航：_parse_json_object；下列为本函数源码中的接口与路径索引。
# 输入 content：str。
# 返回契约：dict[str, Any]。
# 分支条件：fenced。
# 分支条件：not isinstance(parsed, dict)。
# 返回路径：parsed。
# 显式抛出：ProposalValidationError(f'model returned invalid JSON: {exc.msg}')。
# 显式抛出：ProposalValidationError('model response must be a JSON object')。
# 捕获异常：json.JSONDecodeError。
# 直接调用：content.strip。
# 直接调用：re.fullmatch。
# 直接调用：fenced.group。
# 直接调用：json.loads。
# 直接调用：ProposalValidationError。
# 直接调用：isinstance。
# [/maint-note:58dfcc87bba1]
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
    # [maint-note:d988c9659382]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 event_store：EventStore。
    # 输入 memory_store：MemoryStore。
    # 输入 gateway：ModelGateway。
    # 输入 model_name：str；缺省表达式 ''。
    # 输入 context_planner：Any | None；缺省表达式 None。
    # 返回契约：None。
    # 写入目标：self.event_store。
    # 写入目标：self.memory_store。
    # 写入目标：self.gateway。
    # 写入目标：self.model_name。
    # 写入目标：self.context_planner。
    # 直接调用：getattr。
    # [/maint-note:d988c9659382]
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

    # [maint-note:96ff7cb5e58b]
    # 维护导航：create_from_selection；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 project_id：str | None。
    # 输入 evidence_refs：list[dict[str, Any]]。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：expected_project and effective_project != expected_project。
    # 分支条件：not evidence。
    # 返回路径：proposals。
    # 显式抛出：ProposalValidationError('project_id does not match the selected thread')。
    # 显式抛出：ProposalValidationError('select at least one evidence item')。
    # 直接调用：self.event_store.get_thread。
    # 直接调用：str。
    # 直接调用：thread.get。
    # 直接调用：ProposalValidationError。
    # 直接调用：self._freeze_evidence。
    # 直接调用：RuntimeModelConfig.from_dict。
    # 直接调用：self._create_proposals。
    # 直接调用：runtime_model_config.to_dict。
    # [/maint-note:96ff7cb5e58b]
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

    # [maint-note:cec55ee2d2c0]
    # 维护导航：extract_completed_run；下列为本函数源码中的接口与路径索引。
    # 输入 run_id：str。
    # 输入 request_metadata：dict[str, Any]。
    # 输入 auto_apply：bool。
    # 返回契约：dict[str, Any]。
    # 分支条件：run['status'] != 'completed' or not thread_id。
    # 分支条件：len(finished) >= 3。
    # 分支条件：user_evidence_ids or verified。
    # 分支条件：not reviews。
    # 分支条件：auto_apply。
    # 分支条件：review['status'] != 'review_ready'。
    # 遍历边界：review ← reviews。
    # 返回路径：{'status': 'skipped', 'reason': 'no_completed_thread'}。
    # 返回路径：{'status': 'abstained' if reason in {'analyzer_abstained', 'composer_abstained'} else 'skipped', 'reason': reason}。
    # 返回路径：{'status': 'completed', 'reason': reason if reason == 'additional_model_proposals_failed' else None, 'proposal_count': len(reviews), 'applied_count': applied_count}。
    # 显式抛出：继续抛出当前异常。
    # 捕获异常：(MemoryProposalGenerationError, ContextHardLimitExceeded)。
    # 捕获异常：MemoryProposalConflict。
    # 直接调用：self.event_store.get_run。
    # 直接调用：self.event_store.list_conversation_messages。
    # 直接调用：_has_durable_user_signal。
    # 直接调用：self.event_store.list_events。
    # 直接调用：event.payload.get。
    # 直接调用：len。
    # 直接调用：find_tool_argument_recoveries。
    # 直接调用：sorted。
    # 直接调用：self._propose_tool_recoveries。
    # 直接调用：request_metadata.get。
    # 直接调用：self._propose_from_run_evidence。
    # 直接调用：reviews.extend。
    # 直接调用：logger.warning。
    # 直接调用：type。
    # 直接调用：self._assert_evidence_stable。
    # 直接调用：self.memory_store.approve_proposal。
    # 集合派生：message 来自 messages；过滤 message['role'] == 'user' and _has_durable_user_signal(message['content'])。
    # 集合派生：event 来自 self.event_store.list_events(run_id, event_type='verification.finished')；过滤 event.payload.get('passed') is True and event.payload.get('tool')。
    # 表达式分流：recoveries；真值与假值分支分别计算，不同时执行。
    # 表达式分流：reason in {'analyzer_abstained', 'composer_abstained'}；真值与假值分支分别计算，不同时执行。
    # 表达式分流：reason == 'additional_model_proposals_failed'；真值与假值分支分别计算，不同时执行。
    # [/maint-note:cec55ee2d2c0]
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

    # [maint-note:9df1cf341fd3]
    # 维护导航：_propose_tool_recoveries；下列为本函数源码中的接口与路径索引。
    # 输入 run：dict[str, Any]。
    # 输入 recoveries：list[ToolArgumentRecovery]。
    # 输入 request_key：str | None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：len(subject) > MAX_SUBJECT_CHARS。
    # 分支条件：any((item['redacted'] for item in evidence))。
    # 分支条件：proposal_id。
    # 遍历边界：recovery ← recoveries。
    # 返回路径：reviews。
    # 直接调用：', '.join。
    # 直接调用：len。
    # 直接调用：self._freeze_evidence。
    # 直接调用：any。
    # 直接调用：self.memory_store.create_proposal。
    # 直接调用：_canonical_hash。
    # 直接调用：reviews.append。
    # 直接调用：self.get_review。
    # 集合派生：event 来自 recovery.evidence。
    # 集合派生：item 来自 evidence。
    # [/maint-note:9df1cf341fd3]
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

    # [maint-note:b8f50c8014cd]
    # 维护导航：_propose_from_run_evidence；下列为本函数源码中的接口与路径索引。
    # 输入 run：dict[str, Any]。
    # 输入 messages：list[dict[str, Any]]。
    # 输入 user_evidence_ids：set[str]。
    # 输入 verified：list[AgentEvent]。
    # 输入 request_metadata：dict[str, Any]。
    # 返回契约：tuple[list[dict[str, Any]], str | None]。
    # 遍历边界：item ← existing。
    # 返回路径：self._create_proposals(thread_id=thread_id, project_id=run['workspace'], evidence=evidence, request_metadata={**request_metadata, 'timeout_seconds': 25}, run_id=run_id, existing_memories=existing, direct_refs=direct_refs, explicit_only=not 。
    # 写入目标：item['subject']。
    # 写入目标：item['summary']。
    # 直接调用：direct_refs.update。
    # 直接调用：refs.extend。
    # 直接调用：self._freeze_evidence。
    # 直接调用：request_metadata.get。
    # 直接调用：self.memory_store.automatic_memory_manifest。
    # 直接调用：redact_sensitive_text。
    # 直接调用：self._create_proposals。
    # 集合派生：item 来自 user_evidence_ids。
    # 集合派生：event 来自 verified。
    # 集合派生：message 来自 messages；过滤 message['id'] in user_evidence_ids or (verified and (message['role'] == 'user' or (message['role'] == 'assistant' and message['kind'] == 'final')))。
    # [/maint-note:b8f50c8014cd]
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

    # [maint-note:aa977172330d]
    # 维护导航：_create_proposals；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 project_id：str | None。
    # 输入 evidence：list[dict[str, Any]]。
    # 输入 request_metadata：dict[str, Any]。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 existing_memories：list[dict[str, Any]] | None；缺省表达式 None。
    # 输入 direct_refs：set[str] | None；缺省表达式 None。
    # 输入 explicit_only：bool；缺省表达式 False。
    # 返回契约：tuple[list[dict[str, Any]], str | None]。
    # 分支条件：explicit_only。
    # 分支条件：analysis['abstain']。
    # 分支条件：run_id。
    # 分支条件：proposal['scope'] not in {'project', 'user'} or proposal['durability'] != 'long_term' or proposal['kind'] == 'task_summary'。
    # 分支条件：not any((evidence_by_ref[ref]['verification'] == proposal['verification'] for ref in raw['evidence_refs']))。
    # 分支条件：proposal['target_memory_id'] and (target is None or target['scope'] != proposal['scope'] or target['kind'] != proposal['kind'])。
    # 分支条件：proposal_id。
    # 分支条件：not created。
    # 遍历边界：(raw, proposal) ← validated_proposals。
    # 返回路径：([], 'analyzer_abstained')。
    # 返回路径：(created, reason)。
    # 显式抛出：MemoryProposalGenerationError(f'memory_proposal returned invalid output after one repair: {exc}')。
    # 捕获异常：ProposalValidationError。
    # 直接调用：RuntimeModelConfig.from_dict。
    # 直接调用：self._profile_for。
    # 直接调用：self._analyze_evidence。
    # 直接调用：self._generate。
    # 直接调用：ModelRequest。
    # 直接调用：build_composer_messages。
    # 直接调用：self._parse_proposals。
    # 直接调用：self._repair_output。
    # 直接调用：MemoryProposalGenerationError。
    # 直接调用：_canonical_hash。
    # 直接调用：targets.get。
    # 直接调用：any。
    # 直接调用：set。
    # 直接调用：self.memory_store.create_proposal。
    # 直接调用：created.append。
    # 直接调用：self.get_review。
    # 表达式分流：run_id；真值与假值分支分别计算，不同时执行。
    # 表达式分流：direct_refs is not None；真值与假值分支分别计算，不同时执行。
    # 集合派生：item 来自 evidence。
    # 集合派生：item 来自 existing_memories or []。
    # 集合派生：ref 来自 raw['evidence_refs']。
    # 表达式分流：target；真值与假值分支分别计算，不同时执行。
    # 集合派生：item 来自 evidence；过滤 item['ref'] in cited。
    # 表达式分流：any((item['redacted'] for item in evidence))；真值与假值分支分别计算，不同时执行。
    # 表达式分流：explicit_only；真值与假值分支分别计算，不同时执行。
    # 表达式分流：validated_proposals；真值与假值分支分别计算，不同时执行。
    # [/maint-note:aa977172330d]
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

    # [maint-note:a6cc5c59f75d]
    # 维护导航：_analyze_evidence；下列为本函数源码中的接口与路径索引。
    # 输入 evidence：list[dict[str, Any]]。
    # 输入 direct_refs：set[str] | None。
    # 输入 max_output_tokens：int。
    # 输入 request_metadata：dict[str, Any]。
    # 输入 profile：Any。
    # 返回契约：dict[str, Any]。
    # 返回路径：self._parse_analysis(response.content)。
    # 返回路径：self._parse_analysis(repaired.content)。
    # 显式抛出：MemoryProposalGenerationError(f'memory_analysis returned invalid output after one repair: {exc}')。
    # 捕获异常：ProposalValidationError。
    # 直接调用：self._generate。
    # 直接调用：ModelRequest。
    # 直接调用：build_analyzer_messages。
    # 直接调用：self._parse_analysis。
    # 直接调用：self._repair_output。
    # 直接调用：MemoryProposalGenerationError。
    # [/maint-note:a6cc5c59f75d]
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

    # [maint-note:9042135a5166]
    # 维护导航：_profile_for；下列为本函数源码中的接口与路径索引。
    # 输入 config：RuntimeModelConfig。
    # 分支条件：self.context_planner is None。
    # 返回路径：None。
    # 返回路径：replace(self.context_planner.profile, model=config.model, context_window_tokens=config.context_window_tokens, safety_margin_tokens=config.safety_margin_tokens)。
    # 直接调用：replace。
    # [/maint-note:9042135a5166]
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

    # [maint-note:5ea59d4870de]
    # 维护导航：_generate；下列为本函数源码中的接口与路径索引。
    # 输入 request：ModelRequest。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 输入 profile：Any | None；缺省表达式 None。
    # 分支条件：self.context_planner is not None。
    # 返回路径：self.gateway.generate(ModelRequest(messages=request.messages, tools=request.tools, purpose=request.purpose, max_output_tokens=request.max_output_tokens, response_format=request.response_format, metadata={**(request_metadata or {}), **reques。
    # 显式抛出：MemoryProposalGenerationError(f'{request.purpose} model request failed: {detail}')。
    # 捕获异常：Exception。
    # 直接调用：self.context_planner.counter.assert_fits。
    # 直接调用：self.gateway.generate。
    # 直接调用：ModelRequest。
    # 直接调用：redact_sensitive_text。
    # 直接调用：str。
    # 直接调用：type。
    # 直接调用：MemoryProposalGenerationError。
    # [/maint-note:5ea59d4870de]
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

    # [maint-note:b3c54e109327]
    # 维护导航：_repair_output；下列为本函数源码中的接口与路径索引。
    # 输入 stage：str。
    # 输入 invalid_content：str。
    # 输入 validation_error：ProposalValidationError。
    # 输入 required_schema：dict[str, Any]。
    # 输入 max_output_tokens：int。
    # 输入 request_metadata：dict[str, Any] | None；缺省表达式 None。
    # 输入 profile：Any | None；缺省表达式 None。
    # 返回路径：self._generate(ModelRequest(messages=build_repair_messages(stage=stage, invalid_content=redact_sensitive_text(invalid_content), validation_error=redact_sensitive_text(str(validation_error)), required_schema=required_schema), tools=[], purpo。
    # 直接调用：self._generate。
    # 直接调用：ModelRequest。
    # 直接调用：build_repair_messages。
    # 直接调用：redact_sensitive_text。
    # 直接调用：str。
    # [/maint-note:b3c54e109327]
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

    # [maint-note:43be9da7db36]
    # 维护导航：_parse_analysis；下列为本函数源码中的接口与路径索引。
    # 输入 content：str。
    # 返回契约：dict[str, Any]。
    # 返回路径：analysis。
    # 直接调用：_parse_json_object。
    # 直接调用：self._validate_analysis。
    # [/maint-note:43be9da7db36]
    def _parse_analysis(self, content: str) -> dict[str, Any]:
        analysis = _parse_json_object(content)
        self._validate_analysis(analysis)
        return analysis

    # [maint-note:2e963ee529e0]
    # 维护导航：_parse_proposals；下列为本函数源码中的接口与路径索引。
    # 输入 content：str。
    # 输入 allowed_refs：set[str]。
    # 输入 evidence_by_ref：dict[str, dict[str, Any]]。
    # 返回契约：list[tuple[dict[str, Any], dict[str, Any]]]。
    # 分支条件：not isinstance(raw_proposals, list)。
    # 分支条件：len(raw_proposals) > MAX_PROPOSALS。
    # 分支条件：proposal is not None。
    # 遍历边界：raw ← raw_proposals。
    # 返回路径：validated。
    # 显式抛出：ProposalValidationError('proposal response must contain a proposals array')。
    # 显式抛出：ProposalValidationError(f'at most {MAX_PROPOSALS} proposals are allowed')。
    # 直接调用：_parse_json_object。
    # 直接调用：envelope.get。
    # 直接调用：isinstance。
    # 直接调用：ProposalValidationError。
    # 直接调用：len。
    # 直接调用：self._validate_proposal。
    # 直接调用：validated.append。
    # [/maint-note:2e963ee529e0]
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

    # [maint-note:056462c61a23]
    # 维护导航：_freeze_evidence；下列为本函数源码中的接口与路径索引。
    # 输入 thread_id：str。
    # 输入 references：list[dict[str, Any]]。
    # 输入 run_id：str | None；缺省表达式 None。
    # 输入 request_key：str | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 分支条件：len(references) > MAX_EVIDENCE_ITEMS。
    # 分支条件：evidence_type == 'conversation_message'。
    # 分支条件：message is None。
    # 分支条件：evidence_type == 'run_event'。
    # 分支条件：run.get('thread_id') != thread_id or (run_id and evidence_run_id != run_id)。
    # 分支条件：event is None。
    # 分支条件：ref in seen。
    # 分支条件：request_key。
    # 遍历边界：raw ← references。
    # 返回路径：frozen。
    # 显式抛出：ProposalValidationError(f'at most {MAX_EVIDENCE_ITEMS} evidence items are allowed')。
    # 显式抛出：ProposalValidationError('conversation evidence does not belong to the selected thread')。
    # 显式抛出：ProposalValidationError('run-event evidence does not belong to the selected thread')。
    # 显式抛出：ProposalValidationError('unknown run-event evidence')。
    # 显式抛出：ProposalValidationError(f'unsupported evidence type: {evidence_type}')。
    # 直接调用：len。
    # 直接调用：ProposalValidationError。
    # 直接调用：set。
    # 直接调用：self.event_store.list_conversation_messages。
    # 直接调用：str。
    # 直接调用：raw.get。
    # 直接调用：messages.get。
    # 直接调用：int。
    # 直接调用：self.event_store.get_run。
    # 直接调用：run.get。
    # 直接调用：next。
    # 直接调用：self.event_store.list_events。
    # 直接调用：json.dumps。
    # 直接调用：self._classify_event。
    # 直接调用：seen.add。
    # 直接调用：redact_sensitive_text。
    # 直接调用：model_content.replace。
    # 直接调用：frozen.append。
    # 直接调用：_content_hash。
    # 集合派生：item 来自 self.event_store.list_conversation_messages(thread_id, limit=None, run_id=run_id)。
    # 表达式分流：message['role'] == 'user'；真值与假值分支分别计算，不同时执行。
    # 集合派生：item 来自 self.event_store.list_events(evidence_run_id, sequence - 1)；过滤 item.sequence == sequence。
    # 表达式分流：run_id；真值与假值分支分别计算，不同时执行。
    # 表达式分流：role；真值与假值分支分别计算，不同时执行。
    # [/maint-note:056462c61a23]
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

    # [maint-note:8f34b0f94196]
    # 维护导航：_classify_event；下列为本函数源码中的接口与路径索引。
    # 输入 event_type：str。
    # 返回契约：tuple[str, str]。
    # 分支条件：event_type == 'verification.finished'。
    # 分支条件：event_type == 'tool.finished'。
    # 返回路径：('verification_result', 'test_verified')。
    # 返回路径：('tool_result', 'tool_observed')。
    # 返回路径：('run_event', 'tool_observed')。
    # [/maint-note:8f34b0f94196]
    @staticmethod
    def _classify_event(event_type: str) -> tuple[str, str]:
        if event_type == "verification.finished":
            return "verification_result", "test_verified"
        if event_type == "tool.finished":
            return "tool_result", "tool_observed"
        return "run_event", "tool_observed"

    # [maint-note:ee9e3c380b60]
    # 维护导航：_validate_analysis；下列为本函数源码中的接口与路径索引。
    # 输入 analysis：dict[str, Any]。
    # 返回契约：None。
    # 分支条件：not isinstance(analysis.get(field), list)。
    # 分支条件：not isinstance(analysis.get('abstain'), bool)。
    # 遍历边界：field ← ('claims', 'contradictions', 'sensitive_findings')。
    # 显式抛出：ProposalValidationError(f'analysis.{field} must be an array')。
    # 显式抛出：ProposalValidationError('analysis.abstain must be boolean')。
    # 直接调用：isinstance。
    # 直接调用：analysis.get。
    # 直接调用：ProposalValidationError。
    # [/maint-note:ee9e3c380b60]
    @staticmethod
    def _validate_analysis(analysis: dict[str, Any]) -> None:
        for field in ("claims", "contradictions", "sensitive_findings"):
            if not isinstance(analysis.get(field), list):
                raise ProposalValidationError(f"analysis.{field} must be an array")
        if not isinstance(analysis.get("abstain"), bool):
            raise ProposalValidationError("analysis.abstain must be boolean")

    # [maint-note:7b113977ae23]
    # 维护导航：_validate_proposal；下列为本函数源码中的接口与路径索引。
    # 输入 raw：Any。
    # 输入 allowed_refs：set[str]。
    # 输入 evidence_by_ref：dict[str, dict[str, Any]]。
    # 返回契约：dict[str, Any] | None。
    # 分支条件：not isinstance(raw, dict)。
    # 分支条件：raw.get('abstained')。
    # 分支条件：raw.get('schema_version') != 1。
    # 分支条件：operation not in VALID_OPERATIONS。
    # 分支条件：scope not in VALID_SCOPES。
    # 分支条件：kind not in VALID_KINDS。
    # 分支条件：verification not in VALID_VERIFICATIONS。
    # 分支条件：durability not in VALID_DURABILITIES。
    # 分支条件：not subject or len(subject) > MAX_SUBJECT_CHARS。
    # 分支条件：not content or len(content) > MAX_CONTENT_CHARS。
    # 分支条件：raw.get('sensitive')。
    # 分支条件：redact_sensitive_text(subject) != subject or redact_sensitive_text(content) != content。
    # 分支条件：not isinstance(evidence_refs, list) or not evidence_refs。
    # 分支条件：not set(evidence_refs) <= allowed_refs。
    # 分支条件：verification_rank[verification] > strongest_evidence。
    # 分支条件：scope == 'user' and verification != 'user_asserted'。
    # 分支条件：scope in {'project', 'user'} and verification == 'model_inferred'。
    # 分支条件：not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1。
    # 分支条件：operation == 'create' and target_memory_id is not None。
    # 分支条件：operation != 'create'。
    # 分支条件：not target_memory_id or self.memory_store.get(str(target_memory_id)) is None。
    # 分支条件：not isinstance(conflicts, list)。
    # 返回路径：None。
    # 返回路径：{'operation': operation, 'target_memory_id': target_memory_id, 'scope': scope, 'kind': kind, 'subject': subject, 'canonical_content': content, 'verification': verification, 'durability': durability, 'expires_at': raw.get('expires_at'), 'con。
    # 显式抛出：ProposalValidationError('each proposal must be an object')。
    # 显式抛出：ProposalValidationError('unsupported proposal schema_version')。
    # 显式抛出：ProposalValidationError('invalid proposal operation')。
    # 显式抛出：ProposalValidationError('invalid proposal scope')。
    # 显式抛出：ProposalValidationError('invalid proposal kind')。
    # 显式抛出：ProposalValidationError('invalid verification level')。
    # 显式抛出：ProposalValidationError('invalid durability')。
    # 显式抛出：ProposalValidationError('proposal subject is empty or too long')。
    # 显式抛出：ProposalValidationError('proposal content is empty or too long')。
    # 显式抛出：ProposalValidationError('proposal contains sensitive information')。
    # 显式抛出：ProposalValidationError('proposal contains a credential-like secret')。
    # 显式抛出：ProposalValidationError('proposal must cite evidence')。
    # 显式抛出：ProposalValidationError('proposal cites evidence outside the frozen bundle')。
    # 显式抛出：ProposalValidationError('proposal verification level is stronger than its cited evidence')。
    # 显式抛出：ProposalValidationError('user-scope memory requires explicit user evidence')。
    # 显式抛出：ProposalValidationError('model-inferred memory cannot use project or user scope')。
    # 显式抛出：ProposalValidationError('confidence must be between 0 and 1')。
    # 显式抛出：ProposalValidationError('create proposal cannot target an existing memory')。
    # 显式抛出：ProposalValidationError('proposal target memory does not exist')。
    # 显式抛出：ProposalValidationError('conflicts_with must be an array')。
    # 直接调用：isinstance。
    # 直接调用：ProposalValidationError。
    # 直接调用：raw.get。
    # 直接调用：str。
    # 直接调用：str(raw.get('subject') or '').strip。
    # 直接调用：str(raw.get('canonical_content') or '').strip。
    # 直接调用：len。
    # 直接调用：redact_sensitive_text。
    # 直接调用：set。
    # 直接调用：max。
    # 直接调用：self.memory_store.get。
    # 直接调用：float。
    # 集合派生：ref 来自 evidence_refs。
    # 集合派生：item 来自 conflicts。
    # [/maint-note:7b113977ae23]
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

    # [maint-note:4fd647196ecf]
    # 维护导航：get_review；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：dict[str, Any]。
    # 分支条件：proposal is None。
    # 返回路径：project_memory_review(proposal, evidence, target_memory=target)。
    # 显式抛出：KeyError(proposal_id)。
    # 直接调用：self.memory_store.get_proposal。
    # 直接调用：KeyError。
    # 直接调用：self.memory_store.get_proposal_evidence。
    # 直接调用：proposal.get。
    # 直接调用：self.memory_store.get。
    # 直接调用：project_memory_review。
    # 表达式分流：proposal.get('target_memory_id')；真值与假值分支分别计算，不同时执行。
    # [/maint-note:4fd647196ecf]
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

    # [maint-note:d5b0de4b79fc]
    # 维护导航：list_reviews；下列为本函数源码中的接口与路径索引。
    # 输入 status：str | None；缺省表达式 None。
    # 输入 project_id：str | None；缺省表达式 None。
    # 输入 thread_id：str | None；缺省表达式 None。
    # 返回契约：list[dict[str, Any]]。
    # 返回路径：[self.get_review(item['id']) for item in self.memory_store.list_proposals(status=status, project_id=project_id, thread_id=thread_id)]。
    # 直接调用：self.get_review。
    # 直接调用：self.memory_store.list_proposals。
    # 集合派生：item 来自 self.memory_store.list_proposals(status=status, project_id=project_id, thread_id=thread_id)。
    # [/maint-note:d5b0de4b79fc]
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

    # [maint-note:fad7bf508da2]
    # 维护导航：update_review；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 扩展输入 changes：收集调用方的额外参数。
    # 返回契约：dict[str, Any]。
    # 返回路径：self.get_review(proposal_id)。
    # 直接调用：self.memory_store.update_proposal。
    # 直接调用：self.get_review。
    # [/maint-note:fad7bf508da2]
    def update_review(
        self, proposal_id: str, *, expected_version: int, **changes: Any
    ) -> dict[str, Any]:
        self.memory_store.update_proposal(
            proposal_id, expected_version=expected_version, actor="user", **changes
        )
        return self.get_review(proposal_id)

    # [maint-note:11e0dfef903b]
    # 维护导航：approve；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 返回契约：dict[str, Any]。
    # 返回路径：self.get_review(proposal_id)。
    # 直接调用：self._assert_evidence_stable。
    # 直接调用：self.memory_store.approve_proposal。
    # 直接调用：self.get_review。
    # [/maint-note:11e0dfef903b]
    def approve(self, proposal_id: str, *, expected_version: int) -> dict[str, Any]:
        self._assert_evidence_stable(proposal_id)
        self.memory_store.approve_proposal(
            proposal_id, expected_version=expected_version, actor="user"
        )
        return self.get_review(proposal_id)

    # [maint-note:e30ec37c627e]
    # 维护导航：_assert_evidence_stable；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 返回契约：None。
    # 分支条件：proposal is None。
    # 分支条件：item['evidence_type'] == 'legacy_memory'。
    # 分支条件：item['evidence_type'] == 'conversation_message'。
    # 分支条件：not content or _content_hash(content) != item['content_hash']。
    # 遍历边界：item ← self.memory_store.get_proposal_evidence(proposal_id)。
    # 显式抛出：KeyError(proposal_id)。
    # 显式抛出：MemoryProposalConflict('proposal evidence changed after analysis; regenerate before approval')。
    # 捕获异常：(KeyError, ValueError)。
    # 直接调用：self.memory_store.get_proposal。
    # 直接调用：KeyError。
    # 直接调用：proposal.get。
    # 直接调用：self.event_store.list_conversation_messages。
    # 直接调用：self.memory_store.get_proposal_evidence。
    # 直接调用：messages.get。
    # 直接调用：str。
    # 直接调用：item['source_id'].rsplit。
    # 直接调用：int。
    # 直接调用：next。
    # 直接调用：self.event_store.list_events。
    # 直接调用：json.dumps。
    # 直接调用：_content_hash。
    # 直接调用：MemoryProposalConflict。
    # 表达式分流：proposal.get('thread_id')；真值与假值分支分别计算，不同时执行。
    # 集合派生：item 来自 self.event_store.list_conversation_messages(proposal['thread_id'], limit=None, run_id=proposal['run_id'])。
    # 表达式分流：message；真值与假值分支分别计算，不同时执行。
    # 集合派生：event 来自 self.event_store.list_events(run_id, sequence - 1)；过滤 event.sequence == sequence。
    # 表达式分流：event；真值与假值分支分别计算，不同时执行。
    # [/maint-note:e30ec37c627e]
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

    # [maint-note:8d760953beff]
    # 维护导航：reject；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 输入 reason：str；缺省表达式 ''。
    # 返回契约：dict[str, Any]。
    # 返回路径：self.get_review(proposal_id)。
    # 直接调用：self.memory_store.review_proposal。
    # 直接调用：self.get_review。
    # [/maint-note:8d760953beff]
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

    # [maint-note:dc4d4c8ad854]
    # 维护导航：defer；下列为本函数源码中的接口与路径索引。
    # 输入 proposal_id：str。
    # 输入 expected_version：int。
    # 输入 reason：str；缺省表达式 ''。
    # 返回契约：dict[str, Any]。
    # 返回路径：self.get_review(proposal_id)。
    # 直接调用：self.memory_store.review_proposal。
    # 直接调用：self.get_review。
    # [/maint-note:dc4d4c8ad854]
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
