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

// [maint-note:d561acfba954]
// 维护导航：assertObject；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：void。
// 分支条件：!input || typeof input !== 'object' || Array.isArray(input)。
// 显式抛出：inputError('Tool arguments must be a JSON object', 'invalid_argument_shape')。
// 直接调用：Array.isArray。
// 直接调用：inputError。
// [/maint-note:d561acfba954]
export function assertObject(input: Record<string, unknown>): void {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw inputError('Tool arguments must be a JSON object', 'invalid_argument_shape')
  }
}

// [maint-note:e00a4fcd10d1]
// 维护导航：assertKnownKeys；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 keys：readonly string[]。
// 返回契约：void。
// 分支条件：!known.has(key)。
// 遍历边界：const key ← Object.keys(input)。
// 显式抛出：inputError(`Unknown parameter: ${key}`, 'unknown_field')。
// 直接调用：Object.keys。
// 直接调用：known.has。
// 直接调用：inputError。
// [/maint-note:e00a4fcd10d1]
export function assertKnownKeys(
  input: Record<string, unknown>,
  keys: readonly string[],
): void {
  const known = new Set(keys)
  for (const key of Object.keys(input)) {
    if (!known.has(key)) throw inputError(`Unknown parameter: ${key}`, 'unknown_field')
  }
}

// [maint-note:d2e3fde24864]
// 维护导航：requiredString；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 输入 maxLength：类型由调用方与函数体约束；缺省表达式 MAX_QUERY_LENGTH。
// 返回契约：string。
// 分支条件：typeof value !== 'string' || !value.trim()。
// 分支条件：value.includes('\0')。
// 分支条件：value.length > maxLength。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', key, 'non-empty string')。
// 显式抛出：inputError(`${key} must not contain NUL`, 'invalid_value', key, 'string without NUL')。
// 显式抛出：inputError(`${key} is too long`, 'out_of_range', key, `at most ${maxLength} characters`)。
// 直接调用：value.trim。
// 直接调用：inputError。
// 直接调用：value.includes。
// [/maint-note:d2e3fde24864]
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

// [maint-note:f84fa07c1c3c]
// 维护导航：optionalString；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 输入 fallback：string。
// 输入 maxLength：类型由调用方与函数体约束；缺省表达式 MAX_QUERY_LENGTH。
// 返回契约：string。
// 分支条件：input[key] === undefined。
// 分支条件：typeof value !== 'string' || !value.trim()。
// 分支条件：value.includes('\0')。
// 分支条件：value.length > maxLength。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', key, 'non-empty string')。
// 显式抛出：inputError(`${key} must not contain NUL`, 'invalid_value', key, 'string without NUL')。
// 显式抛出：inputError(`${key} is too long`, 'out_of_range', key, `at most ${maxLength} characters`)。
// 直接调用：value.trim。
// 直接调用：inputError。
// 直接调用：value.includes。
// [/maint-note:f84fa07c1c3c]
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

// [maint-note:4be247a601a0]
// 维护导航：optionalBoolean；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 输入 fallback：boolean。
// 返回契约：boolean。
// 分支条件：input[key] === undefined。
// 分支条件：typeof input[key] !== 'boolean'。
// 返回路径：fallback。
// 返回路径：input[key] as boolean。
// 显式抛出：inputError(`${key} must be a boolean`, 'invalid_type', key, 'boolean')。
// 直接调用：inputError。
// [/maint-note:4be247a601a0]
export function optionalBoolean(
  input: Record<string, unknown>,
  key: string,
  fallback: boolean,
): boolean {
  if (input[key] === undefined) return fallback
  if (typeof input[key] !== 'boolean') throw inputError(`${key} must be a boolean`, 'invalid_type', key, 'boolean')
  return input[key] as boolean
}

// [maint-note:8da9668c935a]
// 维护导航：optionalLimit；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：类型由调用方与函数体约束；缺省表达式 'limit'。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 DEFAULT_FILE_LIMIT。
// 返回契约：number。
// 分支条件：input[key] === undefined。
// 分支条件：typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > MAX_FILE_LIMIT。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be an integer from 1 to ${MAX_FILE_LIMIT}`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, `integer 1..${MAX_FILE_LIMIT}`)。
// 直接调用：Number.isInteger。
// 直接调用：inputError。
// [/maint-note:8da9668c935a]
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

// [maint-note:8525943c85fd]
// 维护导航：optionalPositiveInteger；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 输入 fallback：number。
// 输入 max：number。
// 返回契约：number。
// 分支条件：input[key] === undefined。
// 分支条件：typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > max。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be an integer from 1 to ${max}`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, `integer 1..${max}`)。
// 直接调用：Number.isInteger。
// 直接调用：inputError。
// [/maint-note:8525943c85fd]
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

// [maint-note:8f7d45fcb2cb]
// 维护导航：boundedText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxChars：number。
// 返回契约：{ text: string; truncated: boolean }。
// 分支条件：value.length <= maxChars。
// 分支条件：maxChars <= 20。
// 返回路径：{ text: value, truncated: false }。
// 返回路径：{ text: value.slice(0, maxChars), truncated: true }。
// 返回路径：{ text: `${value.slice(0, head)}${marker}${tail ? value.slice(-tail) : ''}`, truncated: true, }。
// 直接调用：value.slice。
// 直接调用：Math.max。
// 直接调用：Math.ceil。
// 直接调用：Math.floor。
// [/maint-note:8f7d45fcb2cb]
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

// [maint-note:7eaf7a1ae7a1]
// 维护导航：boundedJsonResult；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：类型由调用方与函数体约束；缺省表达式 8000。
// 输入 fallback：unknown（可省略）。
// 返回契约：string。
// 分支条件：full.length <= budget。
// 分支条件：fallback === undefined。
// 分支条件：compact.length > budget。
// 返回路径：full。
// 返回路径：compact。
// 显式抛出：new ToolInputError('Tool output fallback is required when the result exceeds its budget')。
// 显式抛出：new ToolInputError(`Tool output fallback exceeds the minimum budget of ${MIN_TOOL_OUTPUT_CHARS} characters`)。
// 直接调用：assertOutputBudget。
// 直接调用：JSON.stringify。
// [/maint-note:7eaf7a1ae7a1]
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

// [maint-note:af8da7a96407]
// 维护导航：assertOutputBudget；下列为本函数源码中的接口与路径索引。
// 输入 maxChars：number（可省略）。
// 返回契约：number。
// 分支条件：!Number.isInteger(budget) || budget < MIN_TOOL_OUTPUT_CHARS。
// 返回路径：budget。
// 显式抛出：new ToolInputError(`maxOutputChars must be an integer of at least ${MIN_TOOL_OUTPUT_CHARS}`)。
// 直接调用：Number.isInteger。
// [/maint-note:af8da7a96407]
export function assertOutputBudget(maxChars?: number): number {
  const budget = maxChars ?? 8000
  if (!Number.isInteger(budget) || budget < MIN_TOOL_OUTPUT_CHARS) {
    throw new ToolInputError(`maxOutputChars must be an integer of at least ${MIN_TOOL_OUTPUT_CHARS}`)
  }
  return budget
}

// [maint-note:6f8c4b51c1ae]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：error。
// 写入目标：error.name。
// [/maint-note:6f8c4b51c1ae]
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const error = new Error('Tool execution interrupted')
    error.name = 'AbortError'
    throw error
  }
}
