import assert from "node:assert/strict";
import test from "node:test";
import { readActiveThreadId, rememberActiveThread, threadToRestore, runtimeConfigForEdit } from "./runtimeSettings.js";

test("refresh restores the selected OpenRouter thread even when DeepSeek is first in history", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  rememberActiveThread("openrouter-thread", storage);
  assert.equal(threadToRestore([{ id: "deepseek-thread" }, { id: "openrouter-thread" }], readActiveThreadId(storage)), "openrouter-thread");
});

test("missing or deleted saved conversation leaves launch defaults in place", () => {
  assert.equal(threadToRestore([{ id: "deepseek-thread" }], null), null);
  assert.equal(threadToRestore([{ id: "deepseek-thread" }], "deleted-thread"), null);
  assert.equal(readActiveThreadId({ getItem() { throw new Error("disabled"); } }), null);
});

test("editing thread settings preserves routing fields without mutating launch defaults", () => {
  const defaults = { provider: "openrouter", model: "anthropic/claude-sonnet-4.5", base_url: "https://openrouter.ai/api/v1", fallback_models: ["deepseek/deepseek-chat"] };
  const edited = runtimeConfigForEdit(defaults, "openrouter", "deepseek/deepseek-chat");
  assert.equal(edited.base_url, defaults.base_url);
  assert.deepEqual(edited.fallback_models, defaults.fallback_models);
  assert.equal(defaults.model, "anthropic/claude-sonnet-4.5");
  assert.deepEqual(runtimeConfigForEdit(defaults, "deepseek", "deepseek-chat"), { provider: "deepseek", model: "deepseek-chat" });
});
