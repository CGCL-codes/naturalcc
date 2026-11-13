const TERMINAL_STATUS_MAP = {
  completed: "complete",
  failed: "error",
  cancelled: "error",
  budget_exhausted: "error"
};

const ACTIVE_RUN_STATUSES = new Set(["queued", "running", "waiting_approval", "paused"]);


// [maint-note:d8cc6b28248a]
// 维护导航：groupThreads；下列为本函数源码中的接口与路径索引。
// 输入 threads：类型由调用方与函数体约束；缺省表达式 []。
// 输入 now：类型由调用方与函数体约束；缺省表达式 new Date()。
// 分支条件：Number.isNaN(updated.getTime())。
// 分支条件：updated.toDateString() === now.toDateString()。
// 分支条件：now.getTime() - updated.getTime() < 7 * dayMs。
// 遍历边界：const thread ← threads。
// 返回路径：buckets.filter((bucket) => bucket.items.length > 0)。
// 直接调用：Number.isNaN。
// 直接调用：updated.getTime。
// 直接调用：buckets[2].items.push。
// 直接调用：updated.toDateString。
// 直接调用：now.toDateString。
// 直接调用：buckets[0].items.push。
// 直接调用：now.getTime。
// 直接调用：buckets[1].items.push。
// 直接调用：buckets.filter。
// [/maint-note:d8cc6b28248a]
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


// [maint-note:c22f5ebf0627]
// 维护导航：hydrateConversationMessages；下列为本函数源码中的接口与路径索引。
// 输入 records：类型由调用方与函数体约束；缺省表达式 []。
// 输入 lastRun：类型由调用方与函数体约束；缺省表达式 null。
// 分支条件：lastRun?.run_id && ["queued", "running", "waiting_approval", "paused"].includes(lastRun.status) && !messages.some((message) => message.type === "assistant" && message.runId === lastRun.run_id)。
// 返回路径：messages。
// 直接调用：records.map。
// 直接调用：["queued", "running", "waiting_approval", "paused"].includes。
// 直接调用：messages.some。
// 直接调用：messages.push。
// 直接调用：["queued", "running"].includes。
// [/maint-note:c22f5ebf0627]
export function hydrateConversationMessages(records = [], lastRun = null) {
  const messages = records.map((record) => ({
    id: record.id,
    type: record.role === "user" ? "user" : "assistant",
    content: record.content || "",
    timestamp: new Date(record.created_at),
    status: TERMINAL_STATUS_MAP[record.metadata?.status] || record.metadata?.status || "complete",
    runId: record.run_id || null,
    kind: record.kind || "message",
    metadata: record.metadata || {}
  }));
  if (lastRun?.run_id && ["queued", "running", "waiting_approval", "paused"].includes(lastRun.status)
      && !messages.some((message) => message.type === "assistant" && message.runId === lastRun.run_id)) {
    messages.push({
      id: `pending-${lastRun.run_id}`,
      type: "assistant",
      content: "Agent run in progress...",
      timestamp: new Date(),
      status: ["queued", "running"].includes(lastRun.status) ? "running" : "attention",
      runId: lastRun.run_id,
      approval: lastRun.status === "waiting_approval" ? lastRun.pending_approval : null
    });
  }
  return messages;
}


// [maint-note:c52b39f6d205]
// 维护导航：selectThreadAfterDeletion；下列为本函数源码中的接口与路径索引。
// 输入 threads：类型由调用方与函数体约束；缺省表达式 []。
// 输入 deletedId：类型由调用方与函数体约束。
// 输入 activeId：类型由调用方与函数体约束。
// 分支条件：deletedId !== activeId。
// 返回路径：activeId || null。
// 返回路径：threads.find((thread) => thread.id !== deletedId)?.id || null。
// 直接调用：threads.find。
// [/maint-note:c52b39f6d205]
export function selectThreadAfterDeletion(threads = [], deletedId, activeId) {
  if (deletedId !== activeId) {
    return activeId || null;
  }
  return threads.find((thread) => thread.id !== deletedId)?.id || null;
}


// [maint-note:616f1c6375ee]
// 维护导航：threadNeedsCancellation；下列为本函数源码中的接口与路径索引。
// 输入 status：类型由调用方与函数体约束。
// 返回路径：ACTIVE_RUN_STATUSES.has(status)。
// 直接调用：ACTIVE_RUN_STATUSES.has。
// [/maint-note:616f1c6375ee]
export function threadNeedsCancellation(status) {
  return ACTIVE_RUN_STATUSES.has(status);
}


// [maint-note:57bcf6a8b872]
// 维护导航：detectContextCandidate；下列为本函数源码中的接口与路径索引。
// 输入 text：类型由调用方与函数体约束。
// 输入 cursor：类型由调用方与函数体约束；缺省表达式 text.length。
// 分支条件：/^(?:[a-zA-Z]:[\\/]|\/)/.test(trimmed)。
// 分支条件：!match。
// 返回路径：{ kind: "absolute", value: trimmed, query: trimmed }。
// 返回路径：null。
// 返回路径：{ kind: "mention", value: match[1], query: match[1].slice(1) }。
// 直接调用：String(text || "").slice。
// 直接调用：String。
// 直接调用：beforeCursor.trim。
// 直接调用：/^(?:[a-zA-Z]:[\\/]|\/)/.test。
// 直接调用：beforeCursor.match。
// 直接调用：match[1].slice。
// [/maint-note:57bcf6a8b872]
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


// [maint-note:1bd14b243bc4]
// 维护导航：buildThreadMessagePayload；下列为本函数源码中的接口与路径索引。
// 输入 content：类型由调用方与函数体约束。
// 输入 contextItems：类型由调用方与函数体约束；缺省表达式 []。
// 输入 budget：类型由调用方与函数体约束；缺省表达式 null。
// 输入 capabilities：类型由调用方与函数体约束；缺省表达式 null。
// 输入 apiKey：类型由调用方与函数体约束；缺省表达式 ""。
// 返回路径：{ content, target_files: Array.from(new Set(targetFiles)), authorized_paths: Array.from(new Set(authorizedPaths)), context_items: contextItems.map((item) => ({ ...item })), ...(budget ? { budget } : {}), ...(capabilities ? { capabilities } 。
// 直接调用：contextItems.map((item) => item.path).filter。
// 直接调用：contextItems.map。
// 直接调用：contextItems .filter((item) => item.external && item.absolute_path) .map。
// 直接调用：contextItems .filter。
// 直接调用：Array.from。
// [/maint-note:1bd14b243bc4]
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
