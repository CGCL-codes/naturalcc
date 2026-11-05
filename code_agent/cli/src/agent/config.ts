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

function normalizeOptionalString(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

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

function normalizeMaxTurns(value: number | undefined): number {
  return boundedInteger(value, DEFAULT_MAX_TURNS, 1, 256, 'maxTurns')
}

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
