import assert from "node:assert/strict";
import test from "node:test";

import { initialAgentState, reduceAgentEvents, hydrateAgentState } from "./agentState.js";

test("terminal and paused events clear stale approval controls", () => {
  for (const type of ["run.failed", "run.cancelled", "run.budget_exhausted", "run.paused"]) {
    const state = reduceAgentEvents(initialAgentState, [
      { type: "approval.requested", payload: { risk: "write" } },
      { type, payload: {} }
    ]);
    assert.equal(state.pendingApproval, null);
  }
});

test("current snapshot overrides old approval events", () => {
  const state = hydrateAgentState("run-1", [
    { type: "approval.requested", payload: { risk: "write" } }
  ], { status: "cancelled", pending_approval: null });
  assert.equal(state.status, "cancelled");
  assert.equal(state.pendingApproval, null);
});

const inputRequest = {
  request_id: "ask-1",
  version: 3,
  question: "Run the release check?",
  answer_type: "confirm",
  options: [],
  tool_call_id: "call-1"
};

test("user input wait is versioned, cleared on resolution and protected by snapshot hydration", () => {
  const requested = { sequence: 3, type: "user_input.requested", payload: inputRequest };
  const waiting = reduceAgentEvents(initialAgentState, [requested]);
  assert.equal(waiting.status, "waiting_input");
  assert.deepEqual(waiting.pendingUserInput, inputRequest);

  const nextRequest = { ...inputRequest, request_id: "ask-2", version: 7 };
  const hydrated = hydrateAgentState("run-1", [requested], {
    status: "waiting_input", pending_user_input: nextRequest
  });
  assert.deepEqual(hydrated.pendingUserInput, nextRequest);
  assert.equal(hydrateAgentState("run-1", [requested], { status: "cancelled" }).pendingUserInput, null);

  const done = reduceAgentEvents(waiting, [
    { sequence: 4, type: "user_input.resolved", payload: { request_id: "ask-1" } }
  ]);
  assert.equal(done.status, "running");
  assert.equal(done.pendingUserInput, null);
  assert.equal(reduceAgentEvents(waiting, [{ type: "run.failed" }]).pendingUserInput, null);
});

test("uncertain tool reconciliation is restored from the snapshot and cleared on resolution", () => {
  const pending = {
    tool_call: { id: "call-1", name: "mcp__notes__write", args: { text: "note" } },
    reason: "mcp_outcome_unknown", version: 8, status: "pending"
  };
  const unresolved = hydrateAgentState("run-1", [], {
    status: "paused", pending_uncertain_tool: pending
  });
  assert.deepEqual(unresolved.uncertainTool, pending);

  const oldEvent = { sequence: 8, type: "tool.uncertain", payload: pending };
  const resolved = reduceAgentEvents(initialAgentState, [
    oldEvent,
    { sequence: 9, type: "tool.uncertain_resolved", payload: { tool_call_id: "call-1" } }
  ]);
  assert.equal(resolved.status, "running");
  assert.equal(resolved.uncertainTool, null);
  assert.equal(hydrateAgentState("run-1", [oldEvent], {
    status: "paused", pending_uncertain_tool: null
  }).uncertainTool, null);
});


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

test("reasoning deltas are grouped by model call and duplicate events are ignored", () => {
  const first = reduceAgentEvents({ ...initialAgentState, runId: "run-1" }, [
    { sequence: 1, type: "model.reasoning.delta", payload: { call_index: 1, text: "Inspect " } },
    { sequence: 2, type: "model.reasoning.delta", payload: { call_index: 1, text: "files." } },
    { sequence: 3, type: "model.content.delta", payload: { call_index: 1, text: "Found" } },
    { sequence: 4, type: "model.reasoning.delta", payload: { call_index: 2, text: "Verify." } }
  ]);
  const state = reduceAgentEvents(first, [
    { sequence: 4, type: "model.reasoning.delta", payload: { call_index: 2, text: "Verify." } },
    { sequence: 5, type: "model.content.delta", payload: { call_index: 2, text: "Done" } }
  ]);
  assert.deepEqual(state.thinking, [
    { callIndex: 1, text: "Inspect files." },
    { callIndex: 2, text: "Verify." }
  ]);
  assert.equal(state.streamedContent, "Done");
  assert.equal(state.events.length, 5);
});
