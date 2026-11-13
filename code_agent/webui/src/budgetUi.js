export const budgetFields = ["max_llm_calls", "max_tool_calls"];
export const defaultBudgetDraft = {
  max_llm_calls: 100,
  max_tool_calls: 100,
  max_input_tokens: 120000
};
const preservedIntegerBudgetFields = [
  "max_compaction_calls",
  "max_input_tokens",
  "max_output_tokens",
  "max_seconds"
];


// [maint-note:875ae9d3221c]
// 维护导航：budgetProgress；下列为本函数源码中的接口与路径索引。
// 输入 usedValue：类型由调用方与函数体约束。
// 输入 limitValue：类型由调用方与函数体约束。
// 返回路径：{ used, limit, percent, level }。
// 直接调用：Math.max。
// 直接调用：Number。
// 直接调用：Math.min。
// 直接调用：Math.round。
// [/maint-note:875ae9d3221c]
export function budgetProgress(usedValue, limitValue) {
  const used = Math.max(0, Number(usedValue) || 0);
  const limit = Math.max(0, Number(limitValue) || 0);
  const rawPercent = limit > 0 ? (used / limit) * 100 : used > 0 ? 100 : 0;
  const percent = Math.min(100, Math.max(0, Math.round(rawPercent)));
  const level = used >= limit && limit >= 0
    ? "exhausted"
    : percent >= 90
      ? "danger"
      : percent >= 70
        ? "warning"
        : "normal";
  return { used, limit, percent, level };
}


// [maint-note:00d4eab5fcbd]
// 维护导航：normalizeBudgetDraft；下列为本函数源码中的接口与路径索引。
// 输入 value：类型由调用方与函数体约束；缺省表达式 {}。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 {}。
// 分支条件：!Object.hasOwn(value, field) && !Object.hasOwn(fallback, field)。
// 分支条件：Number.isFinite(candidate) && candidate >= 0。
// 分支条件：Object.hasOwn(value, "max_cost_usd") || Object.hasOwn(fallback, "max_cost_usd")。
// 分支条件：source === null。
// 遍历边界：const field ← budgetFields。
// 遍历边界：const field ← preservedIntegerBudgetFields。
// 返回路径：normalized。
// 写入目标：normalized[field]。
// 写入目标：normalized.max_cost_usd。
// 直接调用：Number。
// 直接调用：Number.isFinite。
// 直接调用：Math.trunc。
// 直接调用：Math.max。
// 直接调用：Object.hasOwn。
// [/maint-note:00d4eab5fcbd]
export function normalizeBudgetDraft(value = {}, fallback = {}) {
  const normalized = {};
  for (const field of budgetFields) {
    const candidate = Number(value[field]);
    const fallbackValue = Number(fallback[field]);
    normalized[field] = Number.isFinite(candidate) && candidate >= 0
      ? Math.trunc(candidate)
      : Math.max(0, Math.trunc(Number.isFinite(fallbackValue) ? fallbackValue : 0));
  }
  for (const field of preservedIntegerBudgetFields) {
    if (!Object.hasOwn(value, field) && !Object.hasOwn(fallback, field)) continue;
    const candidate = Number(Object.hasOwn(value, field) ? value[field] : fallback[field]);
    if (Number.isFinite(candidate) && candidate >= 0) {
      normalized[field] = Math.trunc(candidate);
    }
  }
  if (Object.hasOwn(value, "max_cost_usd") || Object.hasOwn(fallback, "max_cost_usd")) {
    const source = Object.hasOwn(value, "max_cost_usd") ? value.max_cost_usd : fallback.max_cost_usd;
    if (source === null) {
      normalized.max_cost_usd = null;
    } else {
      const candidate = Number(source);
      if (Number.isFinite(candidate) && candidate >= 0) normalized.max_cost_usd = candidate;
    }
  }
  return normalized;
}


// [maint-note:0ef150b79a6c]
// 维护导航：validateBudgetChange；下列为本函数源码中的接口与路径索引。
// 输入 draft：类型由调用方与函数体约束；缺省表达式 {}。
// 输入 usage：类型由调用方与函数体约束；缺省表达式 {}。
// 分支条件：Number.isFinite(limit) && limit < used。
// 遍历边界：const [limitField, [usageField, unit]] ← Object.entries(usageFields)。
// 返回路径：errors。
// 写入目标：errors[limitField]。
// 直接调用：Object.entries。
// 直接调用：Number。
// 直接调用：Math.max。
// 直接调用：Number.isFinite。
// [/maint-note:0ef150b79a6c]
export function validateBudgetChange(draft = {}, usage = {}) {
  const errors = {};
  const usageFields = {
    max_llm_calls: ["llm_calls", "calls"],
    max_tool_calls: ["tool_calls", "calls"],
    max_input_tokens: ["input_tokens", "input tokens"]
  };
  for (const [limitField, [usageField, unit]] of Object.entries(usageFields)) {
    const limit = Number(draft[limitField]);
    const used = Math.max(0, Number(usage[usageField]) || 0);
    if (Number.isFinite(limit) && limit < used) {
      errors[limitField] = `Cannot be below the ${used} ${unit} already used.`;
    }
  }
  return errors;
}
