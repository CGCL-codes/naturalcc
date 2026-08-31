import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCreateRunPayload,
  buildAgentRunMessage,
  canControlAgentRun,
  formatToolCall,
  hasAgentRuntimeContent,
  modelInputValue,
  shouldAppendUserMessage
} from "./agentUi.js";


test("approval message includes risk, tool name and command arguments", () => {
  const message = buildAgentRunMessage(
    { status: "waiting_approval" },
    {
      pendingApproval: {
        risk: "execute",
        tool_call: {
          name: "command.run",
          args: { argv: ["wc", "-m", "a.md", "b.md"] }
        }
      }
    }
  );

  assert.equal(message.status, "waiting_approval");
  assert.equal(message.approval.risk, "execute");
  assert.match(message.content, /Approval required: execute/);
  assert.match(message.content, /Tool: command\.run/);
  assert.match(message.content, /Command: wc -m a\.md b\.md/);
});


test("generic tool calls show structured arguments", () => {
  const text = formatToolCall({
    name: "workspace.read",
    args: { path: "README.md" }
  });

  assert.match(text, /Tool: workspace\.read/);
  assert.match(text, /"path": "README\.md"/);
});


test("send-triggered agent runs do not append duplicate user messages", () => {
  assert.equal(shouldAppendUserMessage(false), true);
  assert.equal(shouldAppendUserMessage(true), false);
});


test("agent create run payload includes selected target files", () => {
  assert.deepEqual(
    buildCreateRunPayload("D:/repo/test", "translate this file", ["calculator.c"]),
    {
      workspace: "D:/repo/test",
      goal: "translate this file",
      target_files: ["calculator.c"]
    }
  );
});


test("budget exhausted message explains reason and unexecuted tool", () => {
  const message = buildAgentRunMessage(
    { status: "budget_exhausted" },
    {
      budgetExhausted: {
        reason: "max_llm_calls",
        unexecuted_tool_calls: [
          {
            name: "workspace.apply_patch",
            args: { path: "main.c", old_text: "a", new_text: "b" }
          }
        ]
      }
    }
  );

  assert.equal(message.status, "budget_exhausted");
  assert.match(message.content, /Budget exhausted: max_llm_calls/);
  assert.match(message.content, /Unexecuted proposal/);
  assert.match(message.content, /Tool: workspace\.apply_patch/);
});


test("terminal agent statuses do not expose pause or cancel controls", () => {
  assert.equal(canControlAgentRun("running"), true);
  assert.equal(canControlAgentRun("waiting_approval"), true);
  assert.equal(canControlAgentRun("paused"), true);
  assert.equal(canControlAgentRun("budget_exhausted"), false);
  assert.equal(canControlAgentRun("completed"), false);
  assert.equal(canControlAgentRun("failed"), false);
  assert.equal(canControlAgentRun("cancelled"), false);
});


test("agent runtime stack only occupies space when agent content exists", () => {
  assert.equal(hasAgentRuntimeContent("pipeline", { runId: "r1" }, [], []), false);
  assert.equal(hasAgentRuntimeContent("agent", {}, [], []), false);
  assert.equal(hasAgentRuntimeContent("agent", { runId: "r1" }, [], []), true);
  assert.equal(hasAgentRuntimeContent("agent", {}, [{ id: "r1" }], []), true);
  assert.equal(hasAgentRuntimeContent("agent", {}, [], [{ id: "m1" }]), true);
});


test("model input remains controlled while focused", () => {
  assert.equal(modelInputValue("deepseek/deepseek-chat"), "deepseek/deepseek-chat");
  assert.equal(modelInputValue(null), "");
});
