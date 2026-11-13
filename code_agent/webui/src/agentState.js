export const initialAgentState = {
  runId: null,
  status: "idle",
  events: [],
  lastSequence: 0,
  thinking: [],
  streamedContent: "",
  streamedContentCall: 0,
  pendingApproval: null,
  changedFiles: [],
  verification: { required: false, results: [] },
  finalAnswer: "",
  budget: null,
  budgetExhausted: null,
  uncertainTool: null
};


// [maint-note:6ccf112c34b8]
// 维护导航：reduceAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 state：类型由调用方与函数体约束。
// 输入 event：类型由调用方与函数体约束。
// 分支条件：event.sequence && event.sequence <= state.lastSequence。
// 分支条件：event.type === "run.created"。
// 分支条件：event.type === "run.started" || event.type === "run.resumed"。
// 分支条件：event.type === "approval.requested"。
// 分支条件：event.type === "approval.resolved"。
// 分支条件：event.type === "tool.finished"。
// 分支条件：event.type === "model.reasoning.delta"。
// 分支条件：event.type === "model.content.delta"。
// 分支条件：event.type === "verification.required"。
// 分支条件：event.type === "verification.finished"。
// 分支条件：event.type === "run.completed"。
// 分支条件：event.type === "run.failed"。
// 分支条件：event.type === "run.cancelled"。
// 分支条件：event.type === "run.paused"。
// 分支条件：event.type === "tool.uncertain"。
// 分支条件：event.type === "run.budget_exhausted"。
// 分支条件：next.status !== "waiting_approval"。
// 返回路径：state。
// 返回路径：next。
// 写入目标：next.status。
// 写入目标：next.budget。
// 写入目标：next.pendingApproval。
// 写入目标：next.changedFiles。
// 写入目标：next.thinking。
// 写入目标：next.streamedContent。
// 写入目标：next.streamedContentCall。
// 写入目标：next.verification。
// 写入目标：next.finalAnswer。
// 写入目标：next.uncertainTool。
// 写入目标：next.budgetExhausted。
// 直接调用：Array.from。
// 直接调用：Number。
// 直接调用：next.thinking.find。
// 直接调用：next.thinking.map。
// [/maint-note:6ccf112c34b8]
export function reduceAgentEvent(state, event) {
  if (event.sequence && event.sequence <= state.lastSequence) return state;
  const next = {
    ...state,
    events: [...state.events, event],
    lastSequence: event.sequence || state.lastSequence
  };
  if (event.type === "run.created") {
    next.status = "queued";
    next.budget = event.payload?.budget || null;
  } else if (event.type === "run.started" || event.type === "run.resumed") {
    next.status = "running";
  } else if (event.type === "approval.requested") {
    next.status = "waiting_approval";
    next.pendingApproval = event.payload;
  } else if (event.type === "approval.resolved") {
    next.status = "running";
    next.pendingApproval = null;
  } else if (event.type === "tool.finished") {
    next.changedFiles = Array.from(new Set([
      ...next.changedFiles,
      ...(event.payload?.result?.changed_files || [])
    ]));
  } else if (event.type === "model.reasoning.delta") {
    const callIndex = Number(event.payload?.call_index || 1);
    const current = next.thinking.find((item) => item.callIndex === callIndex);
    next.thinking = current
      ? next.thinking.map((item) => item.callIndex === callIndex
        ? { ...item, text: item.text + (event.payload?.text || "") }
        : item)
      : [...next.thinking, { callIndex, text: event.payload?.text || "" }];
  } else if (event.type === "model.content.delta") {
    const callIndex = Number(event.payload?.call_index || 1);
    next.streamedContent = (state.streamedContentCall === callIndex ? state.streamedContent : "")
      + (event.payload?.text || "");
    next.streamedContentCall = callIndex;
  } else if (event.type === "verification.required") {
    next.verification = { ...next.verification, required: true };
  } else if (event.type === "verification.finished") {
    next.verification = {
      required: event.payload?.passed === false,
      results: [...next.verification.results, event.payload]
    };
  } else if (event.type === "run.completed") {
    next.status = "completed";
    next.pendingApproval = null;
    next.finalAnswer = event.payload?.final_answer || "";
  } else if (event.type === "run.failed") {
    next.status = "failed";
  } else if (event.type === "run.cancelled") {
    next.status = "cancelled";
  } else if (event.type === "run.paused") {
    next.status = "paused";
  } else if (event.type === "tool.uncertain") {
    next.status = "paused";
    next.uncertainTool = event.payload || {};
  } else if (event.type === "run.budget_exhausted") {
    next.status = "budget_exhausted";
    next.budgetExhausted = event.payload || {};
  }
  if (next.status !== "waiting_approval") {
    next.pendingApproval = null;
  }
  return next;
}


// [maint-note:4d81989ed2e1]
// 维护导航：reduceAgentEvents；下列为本函数源码中的接口与路径索引。
// 输入 state：类型由调用方与函数体约束。
// 输入 events：类型由调用方与函数体约束。
// 返回路径：events.reduce(reduceAgentEvent, state)。
// 直接调用：events.reduce。
// [/maint-note:4d81989ed2e1]
export function reduceAgentEvents(state, events) {
  return events.reduce(reduceAgentEvent, state);
}

// [maint-note:08ae9b9689dc]
// 维护导航：hydrateAgentState；下列为本函数源码中的接口与路径索引。
// 输入 runId：类型由调用方与函数体约束。
// 输入 events：类型由调用方与函数体约束。
// 输入 snapshot：类型由调用方与函数体约束。
// 返回路径：{ ...state, status: snapshot.status, pendingApproval: snapshot.status === "waiting_approval" ? snapshot.pending_approval : null }。
// 直接调用：reduceAgentEvents。
// [/maint-note:08ae9b9689dc]
export function hydrateAgentState(runId, events, snapshot) {
  const state = reduceAgentEvents({ ...initialAgentState, runId }, events);
  return {
    ...state,
    status: snapshot.status,
    pendingApproval: snapshot.status === "waiting_approval" ? snapshot.pending_approval : null
  };
}
