const REVIEW_FIELDS = [
  "proposal_id",
  "version",
  "status",
  "title",
  "summary",
  "operation",
  "operation_label",
  "scope",
  "scope_label",
  "kind",
  "kind_label",
  "verification",
  "verification_label",
  "confidence",
  "confidence_label",
  "durability",
  "durability_label",
  "expires_at",
  "impact",
  "warnings",
  "conflicts",
  "editable",
  "allowed_actions",
  "target_memory"
];


// [maint-note:7df5a7b958e9]
// 维护导航：normalizeMemoryReview；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束；缺省表达式 {}。
// 遍历边界：const field ← REVIEW_FIELDS。
// 返回路径：review。
// 写入目标：review[field]。
// 写入目标：review.warnings。
// 写入目标：review.conflicts。
// 写入目标：review.allowed_actions。
// 写入目标：review.evidence。
// 直接调用：Array.isArray。
// 直接调用：value.warnings.map。
// 直接调用：value.conflicts.map。
// 直接调用：value.allowed_actions.map。
// 直接调用：value.evidence.map。
// [/maint-note:7df5a7b958e9]
export function normalizeMemoryReview(value = {}) {
  const review = {};
  for (const field of REVIEW_FIELDS) {
    review[field] = value[field];
  }
  review.warnings = Array.isArray(value.warnings) ? value.warnings.map(String) : [];
  review.conflicts = Array.isArray(value.conflicts) ? value.conflicts.map(String) : [];
  review.allowed_actions = Array.isArray(value.allowed_actions) ? value.allowed_actions.map(String) : [];
  review.evidence = Array.isArray(value.evidence)
    ? value.evidence.map((item) => ({
        ref: String(item.ref || ""),
        label: String(item.label || "Evidence"),
        preview: String(item.preview || ""),
        source_locator: String(item.source_locator || ""),
        verification_label: String(item.verification_label || ""),
        verified: Boolean(item.verified)
      }))
    : [];
  return review;
}


// [maint-note:53d7b892c549]
// 维护导航：toggleEvidenceSelection；下列为本函数源码中的接口与路径索引。
// 输入 selected：类型由调用方与函数体约束；缺省表达式 []。
// 输入 messageId：类型由调用方与函数体约束。
// 分支条件：!messageId。
// 返回路径：selected。
// 返回路径：selected.includes(messageId) ? selected.filter((item) => item !== messageId) : [...selected, messageId]。
// 直接调用：selected.includes。
// 直接调用：selected.filter。
// [/maint-note:53d7b892c549]
export function toggleEvidenceSelection(selected = [], messageId) {
  if (!messageId) return selected;
  return selected.includes(messageId)
    ? selected.filter((item) => item !== messageId)
    : [...selected, messageId];
}


// [maint-note:a55b638ab180]
// 维护导航：buildMemoryProposalSelection；下列为本函数源码中的接口与路径索引。
// 输入 threadId：类型由调用方与函数体约束。
// 输入 projectId：类型由调用方与函数体约束。
// 输入 messageIds：类型由调用方与函数体约束；缺省表达式 []。
// 返回路径：{ thread_id: threadId, project_id: projectId || null, evidence_refs: Array.from(new Set(messageIds)).map((sourceId) => ({ type: "conversation_message", source_id: sourceId })) }。
// 直接调用：Array.from(new Set(messageIds)).map。
// 直接调用：Array.from。
// [/maint-note:a55b638ab180]
export function buildMemoryProposalSelection(threadId, projectId, messageIds = []) {
  return {
    thread_id: threadId,
    project_id: projectId || null,
    evidence_refs: Array.from(new Set(messageIds)).map((sourceId) => ({
      type: "conversation_message",
      source_id: sourceId
    }))
  };
}


// [maint-note:08099433b0b7]
// 维护导航：groupMemoryReviews；下列为本函数源码中的接口与路径索引。
// 输入 reviews：类型由调用方与函数体约束；缺省表达式 []。
// 返回路径：{ reviewReady: reviews.filter((item) => item.status === "review_ready"), deferred: reviews.filter((item) => item.status === "deferred"), history: reviews.filter((item) => !["review_ready", "deferred"].includes(item.status)) }。
// 直接调用：reviews.filter。
// [/maint-note:08099433b0b7]
export function groupMemoryReviews(reviews = []) {
  return {
    reviewReady: reviews.filter((item) => item.status === "review_ready"),
    deferred: reviews.filter((item) => item.status === "deferred"),
    history: reviews.filter((item) => !["review_ready", "deferred"].includes(item.status))
  };
}
