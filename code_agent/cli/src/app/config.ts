import { realpath, stat } from 'node:fs/promises'

import { AgentError } from '../agent/errors.js'
import type {
  AgentLoopConfig,
  CodingAgentConfig,
  ModelConnectionConfig,
} from '../agent/types/types.js'
import type { ApprovalMode } from '../tools/approval.js'
import { MIN_TOOL_OUTPUT_CHARS } from '../tools/toolUtils.js'

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
      maxTurns: positiveInteger(input.maxTurns, 10),
      maxHistoryChars: positiveInteger(
        input.maxHistoryChars,
        DEFAULT_MAX_HISTORY_CHARS,
      ),
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

function outputBudget(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return 8_000
  const normalized = Math.floor(value)
  if (normalized < MIN_TOOL_OUTPUT_CHARS) {
    throw new AgentError(
      'config_error',
      `maxToolOutputChars must be at least ${MIN_TOOL_OUTPUT_CHARS}`,
    )
  }
  return normalized
}
