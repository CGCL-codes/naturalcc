import type { RuntimeConfig } from './types/types.js'
import {
  MIN_CANONICAL_TOOL_OUTPUT_CHARS,
  MIN_TOOL_OUTPUT_CHARS,
} from '../tools/toolUtils.js'
import { boundedInteger } from './boundedInteger.js'

const DEFAULT_MAX_TURNS = 32
const DEFAULT_TOOL_TIMEOUT_MS = 120_000
const DEFAULT_MAX_TOOL_OUTPUT_CHARS = 8_000
const DEFAULT_MAX_PROJECT_MEMORY_CHARS = 12_000
const DEFAULT_MAX_DYNAMIC_CONTEXT_CHARS = 4_000
const DEFAULT_MAX_SYSTEM_PROMPT_CHARS = 20_000

// [maint-note:d8bfe0db32ba]
// 数据契约导航：CreateRuntimeConfigOptions；字段含义以消费端校验为准。
// 成员 model：string。
// 成员 projectDir：string。
// 成员 baseURL（可省略）：string | null。
// 成员 maxTurns（可省略）：number。
// 成员 toolTimeoutMs（可省略）：number。
// 成员 maxToolOutputChars（可省略）：number。
// 成员 maxProjectMemoryChars（可省略）：number。
// 成员 maxDynamicContextChars（可省略）：number。
// 成员 maxSystemPromptChars（可省略）：number。
// 成员 showToolResults（可省略）：boolean。
// [/maint-note:d8bfe0db32ba]
export interface CreateRuntimeConfigOptions {
  model: string
  projectDir: string
  baseURL?: string | null
  maxTurns?: number
  toolTimeoutMs?: number
  maxToolOutputChars?: number
  maxProjectMemoryChars?: number
  maxDynamicContextChars?: number
  maxSystemPromptChars?: number
  showToolResults?: boolean
}

// [maint-note:2ca24c9d0137]
// 维护导航：createRuntimeConfig；下列为本函数源码中的接口与路径索引。
// 输入 options：CreateRuntimeConfigOptions。
// 返回契约：RuntimeConfig。
// 返回路径：{ model: options.model, baseURL: normalizeOptionalString(options.baseURL), maxTurns: normalizeMaxTurns(options.maxTurns), projectDir: options.projectDir, toolTimeoutMs: normalizePositiveInteger( options.toolTimeoutMs, DEFAULT_TOOL_TIMEOUT_M。
// 直接调用：normalizeOptionalString。
// 直接调用：normalizeMaxTurns。
// 直接调用：normalizePositiveInteger。
// 直接调用：normalizeOutputBudget。
// [/maint-note:2ca24c9d0137]
export function createRuntimeConfig(
  options: CreateRuntimeConfigOptions,
): RuntimeConfig {
  return {
    model: options.model,
    baseURL: normalizeOptionalString(options.baseURL),
    maxTurns: normalizeMaxTurns(options.maxTurns),
    projectDir: options.projectDir,
    toolTimeoutMs: normalizePositiveInteger(
      options.toolTimeoutMs,
      DEFAULT_TOOL_TIMEOUT_MS,
    ),
    maxToolOutputChars: normalizeOutputBudget(options.maxToolOutputChars),
    maxProjectMemoryChars: normalizePositiveInteger(
      options.maxProjectMemoryChars,
      DEFAULT_MAX_PROJECT_MEMORY_CHARS,
    ),
    maxDynamicContextChars: normalizePositiveInteger(
      options.maxDynamicContextChars,
      DEFAULT_MAX_DYNAMIC_CONTEXT_CHARS,
    ),
    maxSystemPromptChars: normalizePositiveInteger(
      options.maxSystemPromptChars,
      DEFAULT_MAX_SYSTEM_PROMPT_CHARS,
    ),
    showToolResults: options.showToolResults ?? true,
  }
}

// [maint-note:4ad67cbd852a]
// 维护导航：normalizeOptionalString；下列为本函数源码中的接口与路径索引。
// 输入 value：string | null（可省略）。
// 返回契约：string | undefined。
// 返回路径：trimmed ? trimmed : undefined。
// 直接调用：value?.trim。
// [/maint-note:4ad67cbd852a]
function normalizeOptionalString(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

// [maint-note:8682ab89b162]
// 维护导航：normalizePositiveInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 fallback：number。
// 输入 minimum：类型由调用方与函数体约束；缺省表达式 1。
// 返回契约：number。
// 分支条件：value === undefined || !Number.isFinite(value) || value <= 0。
// 返回路径：fallback。
// 返回路径：normalized < minimum ? fallback : normalized。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// [/maint-note:8682ab89b162]
function normalizePositiveInteger(
  value: number | undefined,
  fallback: number,
  minimum = 1,
): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return fallback
  }
  const normalized = Math.floor(value)
  return normalized < minimum ? fallback : normalized
}

// [maint-note:003970ca06b2]
// 维护导航：normalizeMaxTurns；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 返回契约：number。
// 返回路径：boundedInteger(value, DEFAULT_MAX_TURNS, 1, 256, 'maxTurns')。
// 直接调用：boundedInteger。
// [/maint-note:003970ca06b2]
function normalizeMaxTurns(value: number | undefined): number {
  return boundedInteger(value, DEFAULT_MAX_TURNS, 1, 256, 'maxTurns')
}

// [maint-note:24b6a24dbdd9]
// 维护导航：normalizeOutputBudget；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 返回契约：number。
// 分支条件：value === undefined || !Number.isFinite(value) || value <= 0。
// 分支条件：normalized < minimum。
// 返回路径：DEFAULT_MAX_TOOL_OUTPUT_CHARS。
// 返回路径：normalized。
// 显式抛出：new RangeError(`maxToolOutputChars must be at least ${minimum}`)。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// 直接调用：Math.max。
// [/maint-note:24b6a24dbdd9]
function normalizeOutputBudget(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_MAX_TOOL_OUTPUT_CHARS
  }
  const normalized = Math.floor(value)
  // Both configuration entry points share one canonical minimum: a budget
  // below it cannot carry the smallest canonical tool result envelope, so it
  // must fail here instead of surfacing as an incidental read_file runtime
  // error inside the Runner. Direct ToolRegistry/context injection keeps the
  // lower generic floor and fails closed per tool.
  const minimum = Math.max(MIN_TOOL_OUTPUT_CHARS, MIN_CANONICAL_TOOL_OUTPUT_CHARS)
  if (normalized < minimum) {
    throw new RangeError(`maxToolOutputChars must be at least ${minimum}`)
  }
  return normalized
}
