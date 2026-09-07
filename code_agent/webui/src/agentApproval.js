export async function approveAgentAction(requestJson, runId, approval) {
  const state = await requestJson(`/api/agent/runs/${runId}`);
  const pending = state.pending_approval;
  if (
    state.status !== "waiting_approval"
    || !pending?.tool_call?.id
    || pending.tool_call.id !== approval?.tool_call?.id
    || pending.risk !== approval?.risk
  ) {
    const error = new Error("This approval has changed or is no longer pending. Review the current run.");
    error.status = 409;
    throw error;
  }
  return requestJson(`/api/agent/runs/${runId}/approve`, {
    method: "POST",
    body: JSON.stringify({ risk: pending.risk, tool_call_id: pending.tool_call.id })
  });
}

export function approvalErrorMessage(error) {
  if (error.status === 404) {
    return "This run is not available in the connected service. Reload the conversation and check that the original service and runtime database are in use. If the run was deleted, submit the instruction again.";
  }
  return error.message;
}
