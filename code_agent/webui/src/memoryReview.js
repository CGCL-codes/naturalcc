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


export function toggleEvidenceSelection(selected = [], messageId) {
  if (!messageId) return selected;
  return selected.includes(messageId)
    ? selected.filter((item) => item !== messageId)
    : [...selected, messageId];
}


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


export function groupMemoryReviews(reviews = []) {
  return {
    reviewReady: reviews.filter((item) => item.status === "review_ready"),
    deferred: reviews.filter((item) => item.status === "deferred"),
    history: reviews.filter((item) => !["review_ready", "deferred"].includes(item.status))
  };
}
