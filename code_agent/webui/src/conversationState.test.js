import assert from "node:assert/strict";
import test from "node:test";

import {
  buildThreadMessagePayload,
  detectContextCandidate,
  groupThreads,
  hydrateConversationMessages,
  selectThreadAfterDeletion,
  threadNeedsCancellation
} from "./conversationState.js";


test("groupThreads separates today, previous week and earlier conversations", () => {
  const now = new Date("2026-07-30T12:00:00+08:00");
  const groups = groupThreads([
    { id: "today", updated_at: "2026-07-30T08:00:00+08:00" },
    { id: "week", updated_at: "2026-07-26T08:00:00+08:00" },
    { id: "old", updated_at: "2026-06-01T08:00:00+08:00" }
  ], now);

  assert.deepEqual(groups.map((group) => group.label), ["Today", "Previous 7 days", "Earlier"]);
  assert.deepEqual(groups.map((group) => group.items[0].id), ["today", "week", "old"]);
});


test("hydrateConversationMessages restores stable chat cards from API records", () => {
  const messages = hydrateConversationMessages([
    {
      id: "m1",
      role: "user",
      content: "Inspect this file",
      run_id: "run-1",
      kind: "message",
      metadata: {},
      created_at: "2026-07-30T08:00:00+08:00"
    },
    {
      id: "m2",
      role: "assistant",
      content: "Inspection complete",
      run_id: "run-1",
      kind: "final",
      metadata: { status: "completed" },
      created_at: "2026-07-30T08:01:00+08:00"
    }
  ]);

  assert.equal(messages[0].type, "user");
  assert.equal(messages[1].type, "assistant");
  assert.equal(messages[1].status, "complete");
  assert.equal(messages[1].runId, "run-1");
  assert.equal(messages[1].timestamp.toISOString(), "2026-07-30T00:01:00.000Z");
});


test("detectContextCandidate recognizes mentions and Windows absolute paths", () => {
  assert.deepEqual(
    detectContextCandidate("Please inspect @StudentMan", 26),
    { kind: "mention", value: "@StudentMan", query: "StudentMan" }
  );
  assert.deepEqual(
    detectContextCandidate("D:\\shared\\config.py", 19),
    { kind: "absolute", value: "D:\\shared\\config.py", query: "D:\\shared\\config.py" }
  );
});


test("buildThreadMessagePayload includes context paths and external authorization", () => {
  const payload = buildThreadMessagePayload(
    "Refactor it",
    [
      { path: "src/main.py", absolute_path: "D:\\repo\\src\\main.py", external: false },
      { path: "D:\\shared\\config.py", absolute_path: "D:\\shared\\config.py", external: true }
    ],
    { max_llm_calls: 40, max_tool_calls: 60 },
    { codegraph: { enabled: true, auto_sync: true, hide_workspace_search: true } },
    "request-only-key"
  );

  assert.deepEqual(payload, {
    content: "Refactor it",
    target_files: ["src/main.py", "D:\\shared\\config.py"],
    authorized_paths: ["D:\\shared\\config.py"],
    context_items: [
      { path: "src/main.py", absolute_path: "D:\\repo\\src\\main.py", external: false },
      { path: "D:\\shared\\config.py", absolute_path: "D:\\shared\\config.py", external: true }
    ],
    budget: { max_llm_calls: 40, max_tool_calls: 60 },
    capabilities: {
      codegraph: { enabled: true, auto_sync: true, hide_workspace_search: true }
    },
    api_key: "request-only-key"
  });
});


test("selectThreadAfterDeletion moves only when the active thread was deleted", () => {
  const threads = [{ id: "active" }, { id: "next" }, { id: "other" }];

  assert.equal(selectThreadAfterDeletion(threads, "active", "active"), "next");
  assert.equal(selectThreadAfterDeletion(threads, "other", "active"), "active");
});


test("selectThreadAfterDeletion returns an empty selection after deleting the last thread", () => {
  assert.equal(
    selectThreadAfterDeletion([{ id: "active" }], "active", "active"),
    null
  );
});


test("threadNeedsCancellation recognizes every protected active run status", () => {
  for (const status of ["queued", "running", "waiting_approval", "paused"]) {
    assert.equal(threadNeedsCancellation(status), true);
  }
  for (const status of ["completed", "failed", "cancelled", "budget_exhausted", null]) {
    assert.equal(threadNeedsCancellation(status), false);
  }
});
