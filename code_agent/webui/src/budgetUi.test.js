import assert from "node:assert/strict";
import test from "node:test";

import {
  budgetProgress,
  defaultBudgetDraft,
  normalizeBudgetDraft,
  validateBudgetChange
} from "./budgetUi.js";


test("budgetProgress exposes warning thresholds and clamps visual percentage", () => {
  assert.deepEqual(budgetProgress(6, 10), { used: 6, limit: 10, percent: 60, level: "normal" });
  assert.equal(budgetProgress(7, 10).level, "warning");
  assert.equal(budgetProgress(9, 10).level, "danger");
  assert.deepEqual(budgetProgress(12, 10), { used: 12, limit: 10, percent: 100, level: "exhausted" });
});


test("normalizeBudgetDraft keeps supported non-negative integer limits", () => {
  assert.deepEqual(
    normalizeBudgetDraft(
      { max_llm_calls: "40", max_tool_calls: 60.8, ignored: 2 },
      { max_llm_calls: 24, max_tool_calls: 24 }
    ),
    { max_llm_calls: 40, max_tool_calls: 60 }
  );
});


test("normalizeBudgetDraft round-trips backend maintenance and token limits", () => {
  assert.deepEqual(
    normalizeBudgetDraft(
      { max_llm_calls: 40, max_tool_calls: 60 },
      {
        max_llm_calls: 24,
        max_tool_calls: 24,
        max_compaction_calls: 8,
        max_input_tokens: 120000,
        max_output_tokens: 24000,
        max_seconds: 1800,
        max_cost_usd: 12.5
      }
    ),
    {
      max_llm_calls: 40,
      max_tool_calls: 60,
      max_compaction_calls: 8,
      max_input_tokens: 120000,
      max_output_tokens: 24000,
      max_seconds: 1800,
      max_cost_usd: 12.5
    }
  );
});


test("default web budget exposes the backend input-token limit", () => {
  assert.deepEqual(defaultBudgetDraft, {
    max_llm_calls: 100,
    max_tool_calls: 100,
    max_input_tokens: 120000
  });
});


test("validateBudgetChange rejects limits below current usage", () => {
  assert.deepEqual(
    validateBudgetChange(
      { max_llm_calls: 4, max_tool_calls: 10 },
      { llm_calls: 5, tool_calls: 3 }
    ),
    { max_llm_calls: "Cannot be below the 5 calls already used." }
  );
});


test("validateBudgetChange rejects input-token limit below consumed tokens", () => {
  assert.deepEqual(
    validateBudgetChange(
      { max_llm_calls: 10, max_tool_calls: 10, max_input_tokens: 5000 },
      { llm_calls: 2, tool_calls: 3, input_tokens: 6000 }
    ),
    { max_input_tokens: "Cannot be below the 6000 input tokens already used." }
  );
});
