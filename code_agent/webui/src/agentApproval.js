// [maint-note:575d3adca70b]
// 维护导航：reviewAgentAction；下列为本函数源码中的接口与路径索引。
// 输入 requestJson：类型由调用方与函数体约束。
// 输入 runId：类型由调用方与函数体约束。
// 输入 approval：类型由调用方与函数体约束。
// 输入 action：approve 或 reject。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：state.status !== "waiting_approval" || !pending?.tool_call?.id || pending.tool_call.id !== approval?.tool_call?.id || pending.risk !== approval?.risk。
// 返回路径：向对应的审批或拒绝端点发送已核对的工具调用 ID。
// 显式抛出：error。
// 写入目标：error.status。
// 直接调用：requestJson。
// 直接调用：JSON.stringify。
// [/maint-note:575d3adca70b]
async function reviewAgentAction(requestJson, runId, approval, action) {
  const state = await requestJson(`/api/agent/runs/${runId}`);
  const pending = state.pending_approval;
  if (
    state.status !== "waiting_approval"
    || !pending?.tool_call?.id
    || pending.tool_call.id !== approval?.tool_call?.id
    || pending.risk !== approval?.risk
    || (action === "approve" && ["expired", "revoked"].includes(pending.status))
  ) {
    const error = new Error("This approval has changed or is no longer pending. Review the current run.");
    error.status = 409;
    throw error;
  }
  const body = { tool_call_id: pending.tool_call.id };
  if (action === "approve") body.risk = pending.risk;
  return requestJson(`/api/agent/runs/${runId}/${action}`, {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function approveAgentAction(requestJson, runId, approval) {
  return reviewAgentAction(requestJson, runId, approval, "approve");
}

export function rejectAgentAction(requestJson, runId, approval) {
  return reviewAgentAction(requestJson, runId, approval, "reject");
}

// [maint-note:becedee996c4]
// 维护导航：approvalErrorMessage；下列为本函数源码中的接口与路径索引。
// 输入 error：类型由调用方与函数体约束。
// 分支条件：error.status === 404。
// 返回路径："This run is not available in the connected service. Reload the conversation and check that the original service and runtime database are in use. If the run was deleted, submit the instruction again."。
// 返回路径：error.message。
// [/maint-note:becedee996c4]
export function approvalErrorMessage(error) {
  if (error.status === 404) {
    return "This run is not available in the connected service. Reload the conversation and check that the original service and runtime database are in use. If the run was deleted, submit the instruction again.";
  }
  return error.message;
}
