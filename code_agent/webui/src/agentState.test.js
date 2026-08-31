import assert from "node:assert/strict";
import test from "node:test";

import { initialAgentState, reduceAgentEvents } from "./agentState.js";


test("agent event reducer tracks approval, changed files, verification and completion", () => {
  const state = reduceAgentEvents(initialAgentState, [
    { sequence: 1, type: "run.created", payload: { goal: "fix" } },
    { sequence: 2, type: "approval.requested", payload: { risk: "write", tool_call: { name: "workspace.apply_patch" } } },
    { sequence: 3, type: "tool.finished", payload: { result: { changed_files: ["x.py"] } } },
    { sequence: 4, type: "verification.required", payload: { changed_files: ["x.py"] } },
    { sequence: 5, type: "run.completed", payload: { final_answer: "done" } }
  ]);
  assert.equal(state.status, "completed");
  assert.equal(state.pendingApproval, null);
  assert.deepEqual(state.changedFiles, ["x.py"]);
  assert.equal(state.verification.required, true);
  assert.equal(state.finalAnswer, "done");
});


test("agent event reducer exposes uncertain interrupted tools as paused", () => {
  const state = reduceAgentEvents(initialAgentState, [
    { sequence: 1, type: "run.created", payload: {} },
    { sequence: 2, type: "run.started", payload: {} },
    { sequence: 3, type: "tool.started", payload: { tool_call: { name: "command.run" } } },
    { sequence: 4, type: "tool.uncertain", payload: { reason: "execution_started_without_persisted_result" } }
  ]);

  assert.equal(state.status, "paused");
  assert.equal(state.uncertainTool.reason, "execution_started_without_persisted_result");
});


test("failed verification remains required", () => {
  const state = reduceAgentEvents(initialAgentState, [
    { sequence: 1, type: "verification.required", payload: {} },
    { sequence: 2, type: "verification.finished", payload: { passed: false, summary: "tests failed" } }
  ]);

  assert.equal(state.verification.required, true);
});


test("budget exhaustion keeps reason and unexecuted tool proposal", () => {
  const state = reduceAgentEvents(initialAgentState, [
    {
      sequence: 1,
      type: "run.budget_exhausted",
      payload: {
        reason: "max_llm_calls",
        unexecuted_tool_calls: [{ name: "workspace.apply_patch" }]
      }
    }
  ]);

  assert.equal(state.status, "budget_exhausted");
  assert.equal(state.budgetExhausted.reason, "max_llm_calls");
  assert.equal(state.budgetExhausted.unexecuted_tool_calls[0].name, "workspace.apply_patch");
});
