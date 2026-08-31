export const initialAgentState = {
  runId: null,
  status: "idle",
  events: [],
  pendingApproval: null,
  changedFiles: [],
  verification: { required: false, results: [] },
  finalAnswer: "",
  budget: null,
  budgetExhausted: null,
  uncertainTool: null
};


export function reduceAgentEvent(state, event) {
  const next = {
    ...state,
    events: [...state.events, event]
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
  return next;
}


export function reduceAgentEvents(state, events) {
  return events.reduce(reduceAgentEvent, state);
}
