const TERMINAL_STATUS_MAP = {
  completed: "complete",
  failed: "error",
  cancelled: "error",
  budget_exhausted: "error"
};

const ACTIVE_RUN_STATUSES = new Set(["queued", "running", "waiting_approval", "paused"]);


export function groupThreads(threads = [], now = new Date()) {
  const buckets = [
    { label: "Today", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Earlier", items: [] }
  ];
  const dayMs = 24 * 60 * 60 * 1000;
  for (const thread of threads) {
    const updated = new Date(thread.updated_at);
    if (Number.isNaN(updated.getTime())) {
      buckets[2].items.push(thread);
      continue;
    }
    if (updated.toDateString() === now.toDateString()) {
      buckets[0].items.push(thread);
    } else if (now.getTime() - updated.getTime() < 7 * dayMs) {
      buckets[1].items.push(thread);
    } else {
      buckets[2].items.push(thread);
    }
  }
  return buckets.filter((bucket) => bucket.items.length > 0);
}


export function hydrateConversationMessages(records = []) {
  return records.map((record) => ({
    id: record.id,
    type: record.role === "user" ? "user" : "assistant",
    content: record.content || "",
    timestamp: new Date(record.created_at),
    status: TERMINAL_STATUS_MAP[record.metadata?.status] || record.metadata?.status || "complete",
    runId: record.run_id || null,
    kind: record.kind || "message",
    metadata: record.metadata || {}
  }));
}


export function selectThreadAfterDeletion(threads = [], deletedId, activeId) {
  if (deletedId !== activeId) {
    return activeId || null;
  }
  return threads.find((thread) => thread.id !== deletedId)?.id || null;
}


export function threadNeedsCancellation(status) {
  return ACTIVE_RUN_STATUSES.has(status);
}


export function detectContextCandidate(text, cursor = text.length) {
  const beforeCursor = String(text || "").slice(0, cursor);
  const trimmed = beforeCursor.trim();
  if (/^(?:[a-zA-Z]:[\\/]|\/)/.test(trimmed)) {
    return { kind: "absolute", value: trimmed, query: trimmed };
  }
  const match = beforeCursor.match(/(?:^|\s)(@[^\s@]*)$/);
  if (!match) {
    return null;
  }
  return {
    kind: "mention",
    value: match[1],
    query: match[1].slice(1)
  };
}


export function buildThreadMessagePayload(
  content,
  contextItems = [],
  budget = null,
  capabilities = null,
  apiKey = ""
) {
  const targetFiles = contextItems.map((item) => item.path).filter(Boolean);
  const authorizedPaths = contextItems
    .filter((item) => item.external && item.absolute_path)
    .map((item) => item.absolute_path);
  return {
    content,
    target_files: Array.from(new Set(targetFiles)),
    authorized_paths: Array.from(new Set(authorizedPaths)),
    context_items: contextItems.map((item) => ({ ...item })),
    ...(budget ? { budget } : {}),
    ...(capabilities ? { capabilities } : {}),
    ...(apiKey ? { api_key: apiKey } : {})
  };
}
