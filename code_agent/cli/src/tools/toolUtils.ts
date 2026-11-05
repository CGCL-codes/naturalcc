import { ToolInputError } from './types.js'
import { inputError } from './diagnostics.js'

export const DEFAULT_FILE_LIMIT = 200
export const MAX_FILE_LIMIT = 1000
export const MIN_TOOL_OUTPUT_CHARS = 96
export const MIN_CANONICAL_TOOL_OUTPUT_CHARS = 192
export const MAX_QUERY_LENGTH = 2000
export const MAX_PATCH_LENGTH = 200_000
export const MAX_PATCH_FILES = 32
export const MAX_PATCH_TARGET_BYTES = 2_000_000
export const MAX_PATCH_CHANGE_BYTES = 100_000
export const MAX_PATCH_TOTAL_TARGET_BYTES = 10_000_000

export function assertObject(input: Record<string, unknown>): void {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw inputError('Tool arguments must be a JSON object', 'invalid_argument_shape')
  }
}

export function assertKnownKeys(
  input: Record<string, unknown>,
  keys: readonly string[],
): void {
  const known = new Set(keys)
  for (const key of Object.keys(input)) {
    if (!known.has(key)) throw inputError(`Unknown parameter: ${key}`, 'unknown_field')
  }
}

export function requiredString(
  input: Record<string, unknown>,
  key: string,
  maxLength = MAX_QUERY_LENGTH,
): string {
  const value = input[key]
  if (typeof value !== 'string' || !value.trim()) {
    throw inputError(`${key} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', key, 'non-empty string')
  }
  if (value.includes('\0')) throw inputError(`${key} must not contain NUL`, 'invalid_value', key, 'string without NUL')
  if (value.length > maxLength) throw inputError(`${key} is too long`, 'out_of_range', key, `at most ${maxLength} characters`)
  return value
}

export function optionalString(
  input: Record<string, unknown>,
  key: string,
  fallback: string,
  maxLength = MAX_QUERY_LENGTH,
): string {
  if (input[key] === undefined) return fallback
  const value = input[key]
  if (typeof value !== 'string' || !value.trim()) {
    throw inputError(`${key} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', key, 'non-empty string')
  }
  if (value.includes('\0')) throw inputError(`${key} must not contain NUL`, 'invalid_value', key, 'string without NUL')
  if (value.length > maxLength) throw inputError(`${key} is too long`, 'out_of_range', key, `at most ${maxLength} characters`)
  return value
}

export function optionalBoolean(
  input: Record<string, unknown>,
  key: string,
  fallback: boolean,
): boolean {
  if (input[key] === undefined) return fallback
  if (typeof input[key] !== 'boolean') throw inputError(`${key} must be a boolean`, 'invalid_type', key, 'boolean')
  return input[key] as boolean
}

export function optionalLimit(
  input: Record<string, unknown>,
  key = 'limit',
  fallback = DEFAULT_FILE_LIMIT,
): number {
  if (input[key] === undefined) return fallback
  const value = input[key]
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > MAX_FILE_LIMIT) {
    throw inputError(`${key} must be an integer from 1 to ${MAX_FILE_LIMIT}`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, `integer 1..${MAX_FILE_LIMIT}`)
  }
  return value
}

export function optionalPositiveInteger(
  input: Record<string, unknown>,
  key: string,
  fallback: number,
  max: number,
): number {
  if (input[key] === undefined) return fallback
  const value = input[key]
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > max) {
    throw inputError(`${key} must be an integer from 1 to ${max}`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, `integer 1..${max}`)
  }
  return value
}

export function boundedText(value: string, maxChars: number): { text: string; truncated: boolean } {
  if (value.length <= maxChars) return { text: value, truncated: false }
  if (maxChars <= 20) return { text: value.slice(0, maxChars), truncated: true }
  const marker = `\n…(${value.length - maxChars} chars omitted)…\n`
  const remaining = Math.max(0, maxChars - marker.length)
  const head = Math.ceil(remaining / 2)
  const tail = Math.floor(remaining / 2)
  return {
    text: `${value.slice(0, head)}${marker}${tail ? value.slice(-tail) : ''}`,
    truncated: true,
  }
}

export function boundedJsonResult(
  value: unknown,
  maxChars = 8000,
  fallback?: unknown,
): string {
  const budget = assertOutputBudget(maxChars)
  const full = JSON.stringify(value)
  if (full.length <= budget) return full
  if (fallback === undefined) {
    throw new ToolInputError('Tool output fallback is required when the result exceeds its budget')
  }
  const compact = JSON.stringify(fallback)
  if (compact.length > budget) {
    throw new ToolInputError(`Tool output fallback exceeds the minimum budget of ${MIN_TOOL_OUTPUT_CHARS} characters`)
  }
  return compact
}

export function assertOutputBudget(maxChars?: number): number {
  const budget = maxChars ?? 8000
  if (!Number.isInteger(budget) || budget < MIN_TOOL_OUTPUT_CHARS) {
    throw new ToolInputError(`maxOutputChars must be an integer of at least ${MIN_TOOL_OUTPUT_CHARS}`)
  }
  return budget
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const error = new Error('Tool execution interrupted')
    error.name = 'AbortError'
    throw error
  }
}
