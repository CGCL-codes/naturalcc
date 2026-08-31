export function shouldAppendUserMessage(messageAlreadyQueued) {
  return !messageAlreadyQueued;
}


export function modelInputValue(model) {
  return typeof model === "string" ? model : "";
}


const TERMINAL_AGENT_STATUSES = new Set(["completed", "failed", "cancelled", "budget_exhausted"]);


export function canControlAgentRun(status) {
  return Boolean(status) && !TERMINAL_AGENT_STATUSES.has(status);
}


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


export function buildCreateRunPayload(workspace, goal, targetFiles = []) {
  return {
    workspace,
    goal,
    target_files: Array.isArray(targetFiles) ? targetFiles : []
  };
}


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
  return {
    content: `Agent stopped with status: ${runState?.status || "unknown"}`,
    status: runState?.status || "unknown",
    approval: null
  };
}
