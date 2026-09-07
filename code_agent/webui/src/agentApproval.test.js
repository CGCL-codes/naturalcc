import assert from "node:assert/strict";
import test from "node:test";
import { approveAgentAction, approvalErrorMessage } from "./agentApproval.js";

const approval = { risk: "write", tool_call: { id: "mkdir-1", name: "workspace.create_directory" } };

test("missing run stops approval before any mutation request", async () => {
  const calls = [];
  const error = Object.assign(new Error("unknown run"), { status: 404 });
  await assert.rejects(approveAgentAction(async (path, options) => {
    calls.push({ path, options });
    throw error;
  }, "missing", approval), { status: 404 });
  assert.deepEqual(calls, [{ path: "/api/agent/runs/missing", options: undefined }]);
  assert.match(approvalErrorMessage(error), /runtime database/);
});

test("changed approval and terminal runs cannot be approved from old UI state", async () => {
  for (const snapshot of [
    { status: "completed", pending_approval: null },
    { status: "waiting_approval", pending_approval: { ...approval, tool_call: { id: "different" } } },
    { status: "waiting_approval", pending_approval: { ...approval, risk: "execute" } }
  ]) {
    let calls = 0;
    await assert.rejects(approveAgentAction(async () => {
      calls += 1;
      return snapshot;
    }, "run-1", approval), { status: 409 });
    assert.equal(calls, 1);
  }
});

test("approval is bound to the verified pending tool call", async () => {
  const calls = [];
  const result = await approveAgentAction(async (path, options) => {
    calls.push({ path, options });
    return options ? { status: "running" } : { status: "waiting_approval", pending_approval: approval };
  }, "run-1", approval);
  assert.equal(result.status, "running");
  assert.equal(calls[1].path, "/api/agent/runs/run-1/approve");
  assert.deepEqual(JSON.parse(calls[1].options.body), { risk: "write", tool_call_id: "mkdir-1" });
});
