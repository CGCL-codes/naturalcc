// [maint-note:332bf783bcbe]
// 维护导航：shouldAppendUserMessage；下列为本函数源码中的接口与路径索引。
// 输入 messageAlreadyQueued：类型由调用方与函数体约束。
// 返回路径：!messageAlreadyQueued。
// [/maint-note:332bf783bcbe]
export function shouldAppendUserMessage(messageAlreadyQueued) {
  return !messageAlreadyQueued;
}


// [maint-note:b77836a9a2a5]
// 维护导航：modelInputValue；下列为本函数源码中的接口与路径索引。
// 输入 model：类型由调用方与函数体约束。
// 返回路径：typeof model === "string" ? model : ""。
// [/maint-note:b77836a9a2a5]
export function modelInputValue(model) {
  return typeof model === "string" ? model : "";
}


const TERMINAL_AGENT_STATUSES = new Set(["completed", "failed", "cancelled", "budget_exhausted"]);


// [maint-note:c7f69a59c79f]
// 维护导航：canControlAgentRun；下列为本函数源码中的接口与路径索引。
// 输入 status：类型由调用方与函数体约束。
// 返回路径：Boolean(status) && !TERMINAL_AGENT_STATUSES.has(status)。
// 直接调用：Boolean。
// 直接调用：TERMINAL_AGENT_STATUSES.has。
// [/maint-note:c7f69a59c79f]
export function canControlAgentRun(status) {
  return Boolean(status) && !TERMINAL_AGENT_STATUSES.has(status);
}


// [maint-note:5775556ff270]
// 维护导航：hasAgentRuntimeContent；下列为本函数源码中的接口与路径索引。
// 输入 runtimeMode：类型由调用方与函数体约束。
// 输入 agentState：类型由调用方与函数体约束；缺省表达式 {}。
// 输入 agentRuns：类型由调用方与函数体约束；缺省表达式 []。
// 输入 agentMemories：类型由调用方与函数体约束；缺省表达式 []。
// 分支条件：runtimeMode !== "agent"。
// 返回路径：false。
// 返回路径：Boolean( agentState?.runId || (Array.isArray(agentRuns) && agentRuns.length > 0) || (Array.isArray(agentMemories) && agentMemories.length > 0) )。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:5775556ff270]
export function hasAgentRuntimeContent(runtimeMode, agentState = {}, agentRuns = [], agentMemories = []) {
  if (runtimeMode !== "agent") {
    return false;
  }
  return Boolean(
    agentState?.runId
    || (Array.isArray(agentRuns) && agentRuns.length > 0)
    || (Array.isArray(agentMemories) && agentMemories.length > 0)
  );
}


// [maint-note:019807ceb7f6]
// 维护导航：buildCreateRunPayload；下列为本函数源码中的接口与路径索引。
// 输入 workspace：类型由调用方与函数体约束。
// 输入 goal：类型由调用方与函数体约束。
// 输入 targetFiles：类型由调用方与函数体约束；缺省表达式 []。
// 返回路径：{ workspace, goal, target_files: Array.isArray(targetFiles) ? targetFiles : [] }。
// 直接调用：Array.isArray。
// [/maint-note:019807ceb7f6]
export function buildCreateRunPayload(workspace, goal, targetFiles = []) {
  return {
    workspace,
    goal,
    target_files: Array.isArray(targetFiles) ? targetFiles : []
  };
}


// [maint-note:f6afd58276f2]
// 维护导航：formatToolCall；下列为本函数源码中的接口与路径索引。
// 输入 toolCall：类型由调用方与函数体约束。
// 分支条件：!toolCall。
// 分支条件：name === "command.run" && Array.isArray(args.argv)。
// 返回路径："Tool: unknown"。
// 返回路径：`Tool: ${name}\nCommand: ${args.argv.join(" ")}`。
// 返回路径：`Tool: ${name}\nArguments: ${argText}`。
// 直接调用：Array.isArray。
// 直接调用：args.argv.join。
// 直接调用：Object.keys。
// 直接调用：JSON.stringify。
// [/maint-note:f6afd58276f2]
export function formatToolCall(toolCall) {
  if (!toolCall) {
    return "Tool: unknown";
  }
  const name = toolCall.name || "unknown";
  const args = toolCall.args || {};
  if (name === "command.run" && Array.isArray(args.argv)) {
    return `Tool: ${name}\nCommand: ${args.argv.join(" ")}`;
  }
  const argText = Object.keys(args).length
    ? JSON.stringify(args, null, 2)
    : "{}";
  return `Tool: ${name}\nArguments: ${argText}`;
}


// [maint-note:1de4471de5d0]
// 维护导航：buildAgentRunMessage；下列为本函数源码中的接口与路径索引。
// 输入 runState：类型由调用方与函数体约束。
// 输入 agentState：类型由调用方与函数体约束。
// 分支条件：runState?.final_answer。
// 分支条件：runState?.status === "waiting_approval"。
// 分支条件：runState?.status === "budget_exhausted"。
// 分支条件：proposal。
// 分支条件：runState?.status === "failed"。
// 返回路径：{ content: runState.final_answer, status: "complete", approval: null }。
// 返回路径：{ content: [ `Approval required: ${risk}`, formatToolCall(approval?.tool_call), "Review this action before continuing." ].join("\n\n"), status: "waiting_approval", approval }。
// 返回路径：{ content: lines.join("\n\n"), status: "budget_exhausted", approval: null }。
// 返回路径：{ content: `Run failed: ${failure?.payload?.error?.message || "unknown error"}`, status: "failed", approval: null }。
// 返回路径：{ content: `Agent stopped with status: ${runState?.status || "unknown"}`, status: runState?.status || "unknown", approval: null }。
// 直接调用：[ `Approval required: ${risk}`, formatToolCall(approval?.tool_call), "Review this action before continuing." ].join。
// 直接调用：formatToolCall。
// 直接调用：lines.push。
// 直接调用：lines.join。
// 直接调用：[...(agentState?.events || [])].reverse().find。
// 直接调用：[...(agentState?.events || [])].reverse。
// [/maint-note:1de4471de5d0]
export function buildAgentRunMessage(runState, agentState) {
  if (runState?.final_answer) {
    return {
      content: runState.final_answer,
      status: "complete",
      approval: null
    };
  }
  if (runState?.status === "waiting_approval") {
    const approval = agentState?.pendingApproval || null;
    const risk = approval?.risk || "write/execute";
    return {
      content: [
        `Approval required: ${risk}`,
        formatToolCall(approval?.tool_call),
        "Review this action before continuing."
      ].join("\n\n"),
      status: "waiting_approval",
      approval
    };
  }
  if (runState?.status === "budget_exhausted") {
    const budget = agentState?.budgetExhausted || {};
    const proposal = budget.unexecuted_tool_calls?.[0] || null;
    const lines = [`Budget exhausted: ${budget.reason || "unknown"}`];
    if (proposal) {
      lines.push("Unexecuted proposal:", formatToolCall(proposal));
    }
    return {
      content: lines.join("\n\n"),
      status: "budget_exhausted",
      approval: null
    };
  }
  if (runState?.status === "failed") {
    const failure = [...(agentState?.events || [])].reverse().find((event) => event.type === "run.failed");
    return {
      content: `Run failed: ${failure?.payload?.error?.message || "unknown error"}`,
      status: "failed",
      approval: null
    };
  }
  return {
    content: `Agent stopped with status: ${runState?.status || "unknown"}`,
    status: runState?.status || "unknown",
    approval: null
  };
}
