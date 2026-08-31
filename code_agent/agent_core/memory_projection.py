from __future__ import annotations

from typing import Any


OPERATION_LABELS = {
    "create": "新增记忆",
    "update": "更新记忆",
    "supersede": "替代旧记忆",
    "expire": "使记忆失效",
}
SCOPE_LABELS = {
    "run": "仅当前任务",
    "thread": "仅当前对话",
    "project": "仅当前项目",
    "user": "所有项目",
}
KIND_LABELS = {
    "user_preference": "用户偏好",
    "project_constraint": "项目约束",
    "architecture_decision": "架构决策",
    "verified_fact": "已验证事实",
    "repository_convention": "仓库规范",
    "workflow": "工作流程",
    "failure_pattern": "失败模式",
    "successful_approach": "有效方法",
    "important_artifact": "重要产物",
    "task_summary": "任务归档",
}
VERIFICATION_LABELS = {
    "user_asserted": "用户明确提出",
    "tool_observed": "工具已观测",
    "test_verified": "测试已验证",
    "diff_verified": "代码差异已验证",
    "model_inferred": "模型推断，需谨慎",
    "legacy_unverified": "历史数据，未精确验证",
}
DURABILITY_LABELS = {
    "session": "仅本次会话",
    "temporary": "短期有效",
    "long_term": "长期有效",
}


def _confidence_label(value: float) -> str:
    if value >= 0.85:
        return "高可信"
    if value >= 0.6:
        return "中等可信"
    return "低可信"


def project_memory_review(
    proposal: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    target_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the internal protocol into a stable, user-facing review DTO."""
    status = proposal["status"]
    warnings = list(proposal.get("warnings") or [])
    conflicts = list(proposal.get("conflicts") or [])
    if proposal["verification"] in {"model_inferred", "legacy_unverified"}:
        warnings.append(VERIFICATION_LABELS[proposal["verification"]])
    if proposal["confidence"] < 0.6:
        warnings.append("该建议可信度较低，请核对证据。")

    allowed_actions: list[str] = []
    if status in {"review_ready", "deferred"}:
        allowed_actions = ["approve", "edit", "reject", "defer"]
    impact = {
        "run": "接受后只在当前任务的相关上下文中使用。",
        "thread": "接受后会在当前对话的后续任务中使用。",
        "project": "接受后会在当前项目的相关任务中加入模型上下文。",
        "user": "接受后可在用户的所有项目中使用，影响范围较大。",
    }[proposal["scope"]]
    return {
        "proposal_id": proposal["id"],
        "version": proposal["version"],
        "status": status,
        "title": proposal["subject"],
        "summary": proposal["canonical_content"],
        "operation": proposal["operation"],
        "operation_label": OPERATION_LABELS[proposal["operation"]],
        "scope": proposal["scope"],
        "scope_label": SCOPE_LABELS[proposal["scope"]],
        "kind": proposal["kind"],
        "kind_label": KIND_LABELS.get(proposal["kind"], proposal["kind"]),
        "verification": proposal["verification"],
        "verification_label": VERIFICATION_LABELS[proposal["verification"]],
        "confidence": proposal["confidence"],
        "confidence_label": _confidence_label(float(proposal["confidence"])),
        "durability": proposal["durability"],
        "durability_label": DURABILITY_LABELS[proposal["durability"]],
        "expires_at": proposal.get("expires_at"),
        "impact": impact,
        "evidence": [
            {
                "ref": item["evidence_ref"],
                "label": {
                    "conversation_message": "来自对话消息",
                    "run_event": "来自任务事件",
                    "tool_result": "来自工具结果",
                    "verification_result": "来自验证结果",
                    "changed_file": "来自文件变更",
                    "manual_note": "来自用户备注",
                    "legacy_memory": "来自历史候选记忆",
                }.get(item["evidence_type"], "来自证据"),
                "preview": item["preview"],
                "source_locator": item["source_locator"],
                "verification_label": VERIFICATION_LABELS.get(
                    item["verification"], item["verification"]
                ),
                "verified": item["verification"] not in {"model_inferred", "legacy_unverified"},
            }
            for item in evidence
        ],
        "warnings": list(dict.fromkeys(warnings)),
        "conflicts": conflicts,
        "editable": {
            "title": True,
            "summary": True,
            "scope": True,
            "kind": True,
            "expires_at": True,
        },
        "allowed_actions": allowed_actions,
        "target_memory": (
            {
                "id": target_memory["id"],
                "subject": target_memory["subject"],
                "content": target_memory["content"],
            }
            if target_memory
            else None
        ),
    }
