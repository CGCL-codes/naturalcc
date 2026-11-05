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

export interface ResolvedCodingAgentConfig {
  model: ModelConnectionConfig
  loop: AgentLoopConfig
  codingAgent: CodingAgentConfig
}

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

function strictInteger(value: number, minimum: number, maximum: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new AgentError('config_error', `${name} must be an integer between ${minimum} and ${maximum}`)
  }
  return value
}

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

function requiredString(
  explicit: string | null | undefined,
  fallback: string | null | undefined,
): string | undefined {
  return optionalString(explicit === undefined ? fallback : explicit)
}

function optionalString(value: string | null | undefined): string | undefined {
  const trimmed = value?.trim()
  return trimmed || undefined
}

function positiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value) || value <= 0) {
    return fallback
  }
  return Math.floor(value)
}

function resolveMaxTurns(value: number | undefined): number {
  try {
    return boundedInteger(value, 32, 1, 256, 'maxTurns')
  } catch (error) {
    throw new AgentError('config_error', error instanceof Error ? error.message : String(error))
  }
}

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
