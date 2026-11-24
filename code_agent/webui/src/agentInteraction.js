const uncertaintyOutcomes = new Set(["completed", "failed", "unknown"]);
export const MAX_USER_ANSWER_CHARS = 4000;

export function validUserAnswer(request, answer) {
  switch (request?.answer_type) {
    case "text":
      return typeof answer === "string" && Boolean(answer.trim()) && answer.length <= MAX_USER_ANSWER_CHARS;
    case "choice":
      return typeof answer === "string" && (request.options || []).some((option) => option.id === answer);
    case "confirm":
      return typeof answer === "boolean";
    default:
      return false;
  }
}

export function resolveRunInteraction(requestJson, runId, interaction, value, note, apiKey) {
  const request = interaction.request;
  const body = { expected_version: request.version };
  let endpoint;
  if (interaction.kind === "input") {
    if (!validUserAnswer(request, value)) throw new Error("Select or enter an answer.");
    endpoint = "input";
    body.request_id = request.request_id;
    body.answer = value;
  } else {
    if (!uncertaintyOutcomes.has(value)) throw new Error("Select the observed outcome.");
    endpoint = "uncertain/resolve";
    body.tool_call_id = request.tool_call.id;
    body.outcome = value;
    if (note.trim()) body.note = note.trim();
  }
  if (apiKey.trim()) body.api_key = apiKey.trim();
  return requestJson(`/api/agent/runs/${encodeURIComponent(runId)}/${endpoint}`, {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function getRunInteraction(state) {
  const input = state.pendingUserInput;
  if (state.status === "waiting_input" && input?.request_id && Number.isSafeInteger(input.version)) {
    return {
      kind: "input",
      request: input,
      key: JSON.stringify([state.runId, "input", input.request_id, input.version])
    };
  }
  const uncertain = state.uncertainTool;
  if (state.status === "paused" && uncertain?.status === "pending"
      && uncertain.tool_call?.id && Number.isSafeInteger(uncertain.version)) {
    return {
      kind: "uncertain",
      request: uncertain,
      key: JSON.stringify([state.runId, "uncertain", uncertain.tool_call.id, uncertain.version])
    };
  }
  return null;
}

export function needsInteractionContinuation(state) {
  if (!["queued", "running"].includes(state?.status)) return false;
  const relevant = (state.events || []).findLast(({ type }) => (
    type === "user_input.resolved" || type === "tool.uncertain_resolved"
    || type === "model.requested" || type === "tool.started"
    || type.startsWith("context.")
    || (type.startsWith("run.") && type !== "run.budget_updated")
  ));
  return relevant?.type === "user_input.resolved" || relevant?.type === "tool.uncertain_resolved";
}

export function approvalContext(approval) {
  if (!approval) return "";
  const details = [];
  if (approval.scope === "call") details.push("This call only");
  if (approval.scope === "run_risk") details.push("This risk for the current run");
  if (approval.status === "expired") details.push("Expired");
  if (approval.status === "revoked") details.push("Revoked");
  if (approval.expires_at && approval.status !== "expired") {
    const expires = new Date(approval.expires_at);
    if (!Number.isNaN(expires.getTime())) details.push(`Expires ${expires.toLocaleString()}`);
  }
  return details.join(" · ");
}
