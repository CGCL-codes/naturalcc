import assert from "node:assert/strict";
import test from "node:test";
import { approvalContext, getRunInteraction, needsInteractionContinuation, resolveRunInteraction, validUserAnswer } from "./agentInteraction.js";

test("typed answers preserve booleans and match choice IDs", () => {
  assert.equal(validUserAnswer({ answer_type: "confirm" }, false), true);
  assert.equal(validUserAnswer({ answer_type: "confirm" }, "false"), false);
  const choice = { answer_type: "choice", options: [{ id: "draft", label: "Draft mode" }] };
  assert.equal(validUserAnswer(choice, "draft"), true);
  assert.equal(validUserAnswer(choice, "Draft mode"), false);
  assert.equal(validUserAnswer({ answer_type: "text" }, " \n"), false);
  assert.equal(validUserAnswer({ answer_type: "text" }, "a".repeat(4001)), false);
});

test("input request uses the refreshed identity and sends only the request-scoped key", async () => {
  const state = {
    runId: "run-1",
    status: "waiting_input",
    pendingUserInput: { request_id: "ask", version: 5, answer_type: "confirm" }
  };
  const interaction = getRunInteraction(state);
  const next = getRunInteraction({ ...state, pendingUserInput: { ...state.pendingUserInput, version: 6 } });
  assert.notEqual(interaction.key, next.key);

  const payload = await resolveRunInteraction((path, options) => {
    assert.equal(path, "/api/agent/runs/run-1/input");
    assert.equal(options.method, "POST");
    return JSON.parse(options.body);
  }, state.runId, interaction, false, "", "request-key");
  assert.deepEqual(payload, {
    request_id: "ask", expected_version: 5, answer: false, api_key: "request-key"
  });
});

test("reconciliation posts the uncertain call ID and keeps an unknown outcome explicit", async () => {
  const state = {
    runId: "run-2",
    status: "paused",
    uncertainTool: { tool_call: { id: "tool-1" }, version: 10, status: "pending" }
  };
  const interaction = getRunInteraction(state);
  const payload = await resolveRunInteraction((path, options) => {
    assert.equal(path, "/api/agent/runs/run-2/uncertain/resolve");
    return JSON.parse(options.body);
  }, state.runId, interaction, "unknown", "  External status unavailable  ", "");
  assert.deepEqual(payload, {
    tool_call_id: "tool-1", expected_version: 10, outcome: "unknown",
    note: "External status unavailable"
  });
  assert.equal(getRunInteraction({ ...state, status: "running" }), null);
  assert.equal(getRunInteraction({ ...state, uncertainTool: { reason: "mcp_outcome_unknown" } }), null);
});

test("approval metadata reflects scope and explicit expiration or revocation", () => {
  assert.equal(approvalContext({ scope: "run_risk", status: "expired" }), "This risk for the current run · Expired");
  assert.equal(approvalContext({ scope: "call", status: "revoked" }), "This call only · Revoked");
  assert.match(approvalContext({ scope: "call", status: "pending", expires_at: "2026-11-24T08:00:00Z" }), /Expires/);
  assert.equal(approvalContext({ risk: "write" }), "");
});

test("a persisted response offers continuation after reload only before execution starts", () => {
  for (const type of ["user_input.resolved", "tool.uncertain_resolved"]) {
    const state = { status: "running", events: [
      { type: "model.requested" }, { type }, { type: "run.budget_updated" }
    ] };
    assert.equal(needsInteractionContinuation(state), true);
    assert.equal(needsInteractionContinuation({ ...state, status: "completed" }), false);
    for (const subsequent of ["model.requested", "tool.started", "context.budget_evaluated", "run.resumed"]) {
      assert.equal(needsInteractionContinuation({
        ...state, events: [...state.events, { type: subsequent }]
      }), false);
    }
  }
});
