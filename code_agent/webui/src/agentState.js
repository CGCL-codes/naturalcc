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


export function reduceAgentEvents(state, events) {
  return events.reduce(reduceAgentEvent, state);
}

export function hydrateAgentState(runId, events, snapshot) {
  const state = reduceAgentEvents({ ...initialAgentState, runId }, events);
  return {
    ...state,
    status: snapshot.status,
    pendingApproval: snapshot.status === "waiting_approval" ? snapshot.pending_approval : null
  };
}
