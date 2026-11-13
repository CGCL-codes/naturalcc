import type { ToolSchema } from '../types.js'
import { inputError } from '../diagnostics.js'
import {
  assertKnownKeys,
  assertObject,
  MIN_CANONICAL_TOOL_OUTPUT_CHARS,
  optionalPositiveInteger,
  requiredString,
} from '../toolUtils.js'

export const MAX_READ_PAGE_LINES = 300
export const DEFAULT_READ_PAGE_LINES = 200
export const MIN_READ_FILE_OUTPUT_CHARS = MIN_CANONICAL_TOOL_OUTPUT_CHARS
/**
 * the page plan only ever suggests unread follow-up pages. Together
 * with the current page the planning horizon covers at most four pages.
 */
export const MAX_SUGGESTED_CONTINUATION_RANGES = 3

// [maint-note:06a69c93be8f]
// 数据契约导航：ReadFileInput；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 startLine：number。
// 成员 endLine（可省略）：number。
// 成员 maxLines（可省略）：number。
// 成员 includePagePlan（可省略）：boolean。
// [/maint-note:06a69c93be8f]
export interface ReadFileInput {
  path: string
  startLine: number
  endLine?: number
  maxLines?: number
  includePagePlan?: boolean
}

// [maint-note:52c8d20f8028]
// 数据契约导航：ContinuationRange；字段含义以消费端校验为准。
// 成员 startLine：number。
// 成员 endLine：number。
// [/maint-note:52c8d20f8028]
export interface ContinuationRange {
  startLine: number
  endLine: number
}

// [maint-note:119d95ca3beb]
// 数据契约导航：PagePlan；字段含义以消费端校验为准。
// 成员 suggestedContinuationRanges：ContinuationRange[]。
// 成员 pagePlanComplete：boolean。
// 成员 nextPlanStartLine（可省略）：number。
// [/maint-note:119d95ca3beb]
export interface PagePlan {
  suggestedContinuationRanges: ContinuationRange[]
  pagePlanComplete: boolean
  nextPlanStartLine?: number
}

export const readFileSchema: ToolSchema = {
  name: 'read_file',
  description:
    '先用 search_text 定位目标，再读取项目内文本文件的连续行页。需要继续时使用返回的 nextStartLine；不要用无目的的第 1 行逐页扫描。只有明确需要完整读取或跨多页顺序阅读时，才在首个相关调用显式传 includePagePlan:true；结果中的 suggestedContinuationRanges 是尚未读取的后续范围（最多 3 个），可在下一次响应中批量声明其中 1～2 个（显式 startLine/endLine 且 includePagePlan:false），只有用户明确要求完整文件时才考虑一次消费全部；数组不含当前页，不要重复请求当前 startLine/endLine。计划只是建议：工作区文件可能已变化，发生修改后应重新搜索/读取并忽略旧计划。路径必须相对项目根目录。',
  readOnly: true,
  requiresApproval: false,
  repeatKind: 'read',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      startLine: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
      endLine: { type: 'integer', minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
      maxLines: { type: 'integer', minimum: 1, maximum: MAX_READ_PAGE_LINES, default: DEFAULT_READ_PAGE_LINES },
      includePagePlan: { type: 'boolean', default: false },
    },
    required: ['path'],
    additionalProperties: false,
  },
}

// [maint-note:25fc7e6b177c]
// 维护导航：parseReadFileInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：ReadFileInput。
// 分支条件：endLine !== undefined && maxLines !== undefined。
// 分支条件：endLine !== undefined && endLine < startLine。
// 分支条件：endLine !== undefined && endLine - startLine + 1 > MAX_READ_PAGE_LINES。
// 返回路径：{ path: requiredString(input, 'path', 4096), startLine, endLine, maxLines, includePagePlan, }。
// 显式抛出：inputError('endLine and maxLines cannot be used together', 'conflicting_fields', 'endLine', 'endLine or maxLines')。
// 显式抛出：inputError('endLine must be greater than or equal to startLine', 'out_of_range', 'endLine', 'endLine >= startLine')。
// 显式抛出：inputError(`read_file page cannot exceed ${MAX_READ_PAGE_LINES} lines`, 'out_of_range', 'endLine', `at most ${MAX_READ_PAGE_LINES} lines`)。
// 直接调用：assertObject。
// 直接调用：assertKnownKeys。
// 直接调用：safeLineNumber。
// 直接调用：optionalPositiveInteger。
// 直接调用：readBooleanInput。
// 直接调用：inputError。
// 直接调用：requiredString。
// [/maint-note:25fc7e6b177c]
export function parseReadFileInput(input: Record<string, unknown>): ReadFileInput {
  assertObject(input)
  assertKnownKeys(input, ['path', 'startLine', 'endLine', 'maxLines', 'includePagePlan'])
  const startLine = safeLineNumber(input, 'startLine', 1)
  const endLine = input.endLine === undefined
    ? undefined
    : safeLineNumber(input, 'endLine')
  const maxLines = input.maxLines === undefined
    ? undefined
    : optionalPositiveInteger(input, 'maxLines', 1, MAX_READ_PAGE_LINES)
  const includePagePlan = input.includePagePlan === undefined
    ? undefined
    : readBooleanInput(input, 'includePagePlan')
  if (endLine !== undefined && maxLines !== undefined) {
    throw inputError('endLine and maxLines cannot be used together', 'conflicting_fields', 'endLine', 'endLine or maxLines')
  }
  if (endLine !== undefined && endLine < startLine) {
    throw inputError('endLine must be greater than or equal to startLine', 'out_of_range', 'endLine', 'endLine >= startLine')
  }
  if (endLine !== undefined && endLine - startLine + 1 > MAX_READ_PAGE_LINES) {
    throw inputError(`read_file page cannot exceed ${MAX_READ_PAGE_LINES} lines`, 'out_of_range', 'endLine', `at most ${MAX_READ_PAGE_LINES} lines`)
  }
  return {
    path: requiredString(input, 'path', 4096),
    startLine,
    endLine,
    maxLines,
    includePagePlan,
  }
}

// [maint-note:a1fb7f9a223b]
// 维护导航：readBooleanInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 返回契约：boolean。
// 分支条件：typeof value !== 'boolean'。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be a boolean`, 'invalid_type', key, 'boolean')。
// 直接调用：inputError。
// [/maint-note:a1fb7f9a223b]
function readBooleanInput(input: Record<string, unknown>, key: string): boolean {
  const value = input[key]
  if (typeof value !== 'boolean') {
    throw inputError(`${key} must be a boolean`, 'invalid_type', key, 'boolean')
  }
  return value
}

// [maint-note:a2cc8139e685]
// 维护导航：safeLineNumber；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 key：string。
// 输入 fallback：number（可省略）。
// 返回契约：number。
// 分支条件：input[key] === undefined && fallback !== undefined。
// 分支条件：typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：inputError(`${key} must be a safe integer greater than or equal to 1`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, 'safe integer >= 1')。
// 直接调用：Number.isSafeInteger。
// 直接调用：inputError。
// [/maint-note:a2cc8139e685]
function safeLineNumber(
  input: Record<string, unknown>,
  key: string,
  fallback?: number,
): number {
  if (input[key] === undefined && fallback !== undefined) return fallback
  const value = input[key]
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
    throw inputError(`${key} must be a safe integer greater than or equal to 1`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', key, 'safe integer >= 1')
  }
  return value
}
