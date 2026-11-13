import { realpath, stat } from 'node:fs/promises'

import { AgentError } from '../agent/errors.js'
import type {
  AgentLoopConfig,
  CodingAgentConfig,
  ModelConnectionConfig,
  RunBudgetConfig,
} from '../agent/types/types.js'
import type { ApprovalMode } from '../tools/approval.js'
import {
  MIN_CANONICAL_TOOL_OUTPUT_CHARS,
  MIN_TOOL_OUTPUT_CHARS,
} from '../tools/toolUtils.js'
import { boundedInteger } from '../agent/boundedInteger.js'

export const DEFAULT_MODEL = 'gpt-5.5'
export const DEFAULT_MAX_HISTORY_CHARS = 100_000

// [maint-note:a1004cca038d]
// 数据契约导航：CodingAgentConfigInput；字段含义以消费端校验为准。
// 成员 model（可省略）：string | null。
// 成员 apiKey（可省略）：string | null。
// 成员 baseURL（可省略）：string | null。
// 成员 projectDir（可省略）：string | null。
// 成员 maxTurns（可省略）：number。
// 成员 maxHistoryChars（可省略）：number。
// 成员 toolTimeoutMs（可省略）：number。
// 成员 maxToolOutputChars（可省略）：number。
// 成员 maxProjectMemoryChars（可省略）：number。
// 成员 maxDynamicContextChars（可省略）：number。
// 成员 maxSystemPromptChars（可省略）：number。
// 成员 showToolResults（可省略）：boolean。
// 成员 approvalMode（可省略）：ApprovalMode。
// 成员 runBudget（可省略）：RunBudgetConfig。
// [/maint-note:a1004cca038d]
export interface CodingAgentConfigInput {
  model?: string | null
  apiKey?: string | null
  baseURL?: string | null
  projectDir?: string | null
  maxTurns?: number
  maxHistoryChars?: number
  toolTimeoutMs?: number
  maxToolOutputChars?: number
  maxProjectMemoryChars?: number
  maxDynamicContextChars?: number
  maxSystemPromptChars?: number
  showToolResults?: boolean
  approvalMode?: ApprovalMode
  /** application/test configuration; intentionally not exposed by CLI. */
  runBudget?: RunBudgetConfig
}

// [maint-note:525c0730ebb8]
// 数据契约导航：ResolvedCodingAgentConfig；字段含义以消费端校验为准。
// 成员 model：ModelConnectionConfig。
// 成员 loop：AgentLoopConfig。
// 成员 codingAgent：CodingAgentConfig。
// [/maint-note:525c0730ebb8]
export interface ResolvedCodingAgentConfig {
  model: ModelConnectionConfig
  loop: AgentLoopConfig
  codingAgent: CodingAgentConfig
}

// [maint-note:f42729acba29]
// 维护导航：resolveCodingAgentConfig；下列为本函数源码中的接口与路径索引。
// 输入 input：CodingAgentConfigInput；缺省表达式 {}。
// 返回契约：Promise<ResolvedCodingAgentConfig>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!apiKey。
// 分支条件：approvalMode !== 'prompt' && approvalMode !== 'deny' && approvalMode !== 'allow'。
// 返回路径：{ model: { model, apiKey, baseURL: optionalString(input.baseURL ?? process.env.OPENAI_BASE_URL), }, loop: { maxTurns: resolveMaxTurns(input.maxTurns), maxHistoryChars: positiveInteger( input.maxHistoryChars, DEFAULT_MAX_HISTORY_CHARS, ), ..。
// 显式抛出：new AgentError( 'config_error', 'OPENAI_API_KEY is required to create a coding agent', )。
// 显式抛出：new AgentError('config_error', 'approvalMode must be prompt, deny, or allow')。
// 直接调用：requiredString。
// 直接调用：resolveProjectDir。
// 直接调用：process.cwd。
// 直接调用：optionalString。
// 直接调用：resolveMaxTurns。
// 直接调用：positiveInteger。
// 直接调用：resolveRunBudget。
// 直接调用：outputBudget。
// [/maint-note:f42729acba29]
export async function resolveCodingAgentConfig(
  input: CodingAgentConfigInput = {},
): Promise<ResolvedCodingAgentConfig> {
  const model = requiredString(input.model, process.env.OPENAI_MODEL) || DEFAULT_MODEL
  const apiKey = requiredString(input.apiKey, process.env.OPENAI_API_KEY)
  if (!apiKey) {
    throw new AgentError(
      'config_error',
      'OPENAI_API_KEY is required to create a coding agent',
    )
  }

  const projectDir = await resolveProjectDir(
    requiredString(input.projectDir, process.cwd()) || process.cwd(),
  )

  const approvalMode = input.approvalMode ?? (process.stdin.isTTY ? 'prompt' : 'deny')
  if (approvalMode !== 'prompt' && approvalMode !== 'deny' && approvalMode !== 'allow') {
    throw new AgentError('config_error', 'approvalMode must be prompt, deny, or allow')
  }

  return {
    model: {
      model,
      apiKey,
      baseURL: optionalString(input.baseURL ?? process.env.OPENAI_BASE_URL),
    },
    loop: {
      maxTurns: resolveMaxTurns(input.maxTurns),
      maxHistoryChars: positiveInteger(
        input.maxHistoryChars,
        DEFAULT_MAX_HISTORY_CHARS,
      ),
      ...(input.runBudget ? { runBudget: resolveRunBudget(input.runBudget) } : {}),
    },
    codingAgent: {
      projectDir,
      toolTimeoutMs: positiveInteger(input.toolTimeoutMs, 120_000),
      maxToolOutputChars: outputBudget(input.maxToolOutputChars),
      maxProjectMemoryChars: positiveInteger(
        input.maxProjectMemoryChars,
        12_000,
      ),
      maxDynamicContextChars: positiveInteger(
        input.maxDynamicContextChars,
        4_000,
      ),
      maxSystemPromptChars: positiveInteger(
        input.maxSystemPromptChars,
        20_000,
      ),
      showToolResults: input.showToolResults ?? true,
      approvalMode,
    },
  }
}

// [maint-note:56fcf0d9abc8]
// 维护导航：resolveRunBudget；下列为本函数源码中的接口与路径索引。
// 输入 input：RunBudgetConfig。
// 返回契约：RunBudgetConfig。
// 分支条件：input.segmentMaxModelRequests !== undefined。
// 分支条件：input.maxToolAttempts !== undefined。
// 分支条件：input.maxElapsedMs !== undefined。
// 分支条件：input.maxTotalTokens !== undefined。
// 分支条件：input.maxCostMicrousd !== undefined。
// 分支条件：input.maxOutputTokensPerRequest !== undefined。
// 分支条件：input.safetyMarginTokens !== undefined。
// 分支条件：resolved.maxTotalTokens !== undefined && resolved.maxOutputTokensPerRequest === undefined。
// 返回路径：Object.freeze(resolved)。
// 显式抛出：new AgentError('config_error', 'maxTotalTokens requires maxOutputTokensPerRequest')。
// 写入目标：resolved.segmentMaxModelRequests。
// 写入目标：resolved.maxToolAttempts。
// 写入目标：resolved.maxElapsedMs。
// 写入目标：resolved.maxTotalTokens。
// 写入目标：resolved.maxCostMicrousd。
// 写入目标：resolved.maxOutputTokensPerRequest。
// 写入目标：resolved.safetyMarginTokens。
// 直接调用：strictInteger。
// 直接调用：Object.freeze。
// [/maint-note:56fcf0d9abc8]
function resolveRunBudget(input: RunBudgetConfig): RunBudgetConfig {
  const resolved: { -readonly [Key in keyof RunBudgetConfig]?: RunBudgetConfig[Key] } = {}
  if (input.segmentMaxModelRequests !== undefined) {
    resolved.segmentMaxModelRequests = strictInteger(input.segmentMaxModelRequests, 2, 256, 'segmentMaxModelRequests')
  }
  if (input.maxToolAttempts !== undefined) {
    resolved.maxToolAttempts = strictInteger(input.maxToolAttempts, 0, 1_000_000, 'maxToolAttempts')
  }
  if (input.maxElapsedMs !== undefined) {
    resolved.maxElapsedMs = strictInteger(input.maxElapsedMs, 1, Number.MAX_SAFE_INTEGER, 'maxElapsedMs')
  }
  if (input.maxTotalTokens !== undefined) {
    resolved.maxTotalTokens = strictInteger(input.maxTotalTokens, 1, Number.MAX_SAFE_INTEGER, 'maxTotalTokens')
  }
  if (input.maxCostMicrousd !== undefined) {
    resolved.maxCostMicrousd = strictInteger(input.maxCostMicrousd, 1, Number.MAX_SAFE_INTEGER, 'maxCostMicrousd')
  }
  if (input.maxOutputTokensPerRequest !== undefined) {
    resolved.maxOutputTokensPerRequest = strictInteger(input.maxOutputTokensPerRequest, 1, Number.MAX_SAFE_INTEGER, 'maxOutputTokensPerRequest')
  }
  if (input.safetyMarginTokens !== undefined) {
    resolved.safetyMarginTokens = strictInteger(input.safetyMarginTokens, 0, Number.MAX_SAFE_INTEGER, 'safetyMarginTokens')
  }
  if (resolved.maxTotalTokens !== undefined && resolved.maxOutputTokensPerRequest === undefined) {
    throw new AgentError('config_error', 'maxTotalTokens requires maxOutputTokensPerRequest')
  }
  return Object.freeze(resolved)
}

// [maint-note:8eaf556b6b57]
// 维护导航：strictInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：number。
// 输入 minimum：number。
// 输入 maximum：number。
// 输入 name：string。
// 返回契约：number。
// 分支条件：!Number.isSafeInteger(value) || value < minimum || value > maximum。
// 返回路径：value。
// 显式抛出：new AgentError('config_error', `${name} must be an integer between ${minimum} and ${maximum}`)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:8eaf556b6b57]
function strictInteger(value: number, minimum: number, maximum: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new AgentError('config_error', `${name} must be an integer between ${minimum} and ${maximum}`)
  }
  return value
}

// [maint-note:1dd3b04efea3]
// 维护导航：resolveProjectDir；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info.isDirectory()。
// 返回路径：resolved。
// 显式抛出：new Error('not a directory')。
// 显式抛出：new AgentError('config_error', 'projectDir must be an existing directory')。
// 直接调用：realpath。
// 直接调用：stat。
// 直接调用：info.isDirectory。
// [/maint-note:1dd3b04efea3]
async function resolveProjectDir(value: string): Promise<string> {
  try {
    const resolved = await realpath(value)
    const info = await stat(resolved)
    if (!info.isDirectory()) throw new Error('not a directory')
    return resolved
  } catch {
    throw new AgentError('config_error', 'projectDir must be an existing directory')
  }
}

// [maint-note:6fc0686d90bb]
// 维护导航：requiredString；下列为本函数源码中的接口与路径索引。
// 输入 explicit：string | null | undefined。
// 输入 fallback：string | null | undefined。
// 返回契约：string | undefined。
// 返回路径：optionalString(explicit === undefined ? fallback : explicit)。
// 直接调用：optionalString。
// [/maint-note:6fc0686d90bb]
function requiredString(
  explicit: string | null | undefined,
  fallback: string | null | undefined,
): string | undefined {
  return optionalString(explicit === undefined ? fallback : explicit)
}

// [maint-note:367f71e6bfd5]
// 维护导航：optionalString；下列为本函数源码中的接口与路径索引。
// 输入 value：string | null | undefined。
// 返回契约：string | undefined。
// 返回路径：trimmed || undefined。
// 直接调用：value?.trim。
// [/maint-note:367f71e6bfd5]
function optionalString(value: string | null | undefined): string | undefined {
  const trimmed = value?.trim()
  return trimmed || undefined
}

// [maint-note:95f15ff027e1]
// 维护导航：positiveInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 fallback：number。
// 返回契约：number。
// 分支条件：value === undefined || !Number.isFinite(value) || value <= 0。
// 返回路径：fallback。
// 返回路径：Math.floor(value)。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// [/maint-note:95f15ff027e1]
function positiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return fallback
  }
  return Math.floor(value)
}

// [maint-note:7150789334bc]
// 维护导航：resolveMaxTurns；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 返回契约：number。
// 返回路径：boundedInteger(value, 32, 1, 256, 'maxTurns')。
// 显式抛出：new AgentError('config_error', error instanceof Error ? error.message : String(error))。
// 直接调用：boundedInteger。
// 直接调用：String。
// [/maint-note:7150789334bc]
function resolveMaxTurns(value: number | undefined): number {
  try {
    return boundedInteger(value, 32, 1, 256, 'maxTurns')
  } catch (error) {
    throw new AgentError('config_error', error instanceof Error ? error.message : String(error))
  }
}

// [maint-note:8d65a57ac767]
// 维护导航：outputBudget；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 返回契约：number。
// 分支条件：value === undefined || !Number.isFinite(value)。
// 分支条件：normalized < minimum。
// 返回路径：8_000。
// 返回路径：normalized。
// 显式抛出：new AgentError( 'config_error', `maxToolOutputChars must be at least ${minimum}`, )。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// 直接调用：Math.max。
// [/maint-note:8d65a57ac767]
function outputBudget(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return 8_000
  const normalized = Math.floor(value)
  const minimum = Math.max(MIN_TOOL_OUTPUT_CHARS, MIN_CANONICAL_TOOL_OUTPUT_CHARS)
  if (normalized < minimum) {
    throw new AgentError(
      'config_error',
      `maxToolOutputChars must be at least ${minimum}`,
    )
  }
  return normalized
}
