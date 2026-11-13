import type { ToolSchema } from '../types.js'
import { inputError } from '../diagnostics.js'
import {
  assertKnownKeys,
  assertObject,
  MAX_FILE_LIMIT,
  optionalBoolean,
  optionalLimit,
  optionalString,
  requiredString,
} from '../toolUtils.js'

// [maint-note:d724123d031b]
// 数据契约导航：SearchTextInput；字段含义以消费端校验为准。
// 成员 query：string。
// 成员 mode：'literal' | 'regex'。
// 成员 path：string。
// 成员 pattern（可省略）：string。
// 成员 caseSensitive：boolean。
// 成员 limit：number。
// 成员 includeIgnored：boolean。
// 成员 includeHidden：boolean。
// [/maint-note:d724123d031b]
export interface SearchTextInput {
  query: string
  mode: 'literal' | 'regex'
  path: string
  pattern?: string
  caseSensitive: boolean
  limit: number
  includeIgnored: boolean
  includeHidden: boolean
}

export const MAX_SEARCH_QUERY_UTF8_BYTES = 4_096

export const searchTextSchema: ToolSchema = {
  name: 'search_text',
  description: '通过系统 ripgrep 搜索项目内文本；mode 默认 literal，显式 regex 使用受限的 ripgrep 默认正则引擎。默认遵循 .gitignore 且不含隐藏文件。',
  readOnly: true,
  requiresApproval: false,
  repeatKind: 'read',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', minLength: 1, maxLength: 2000 },
      mode: {
        type: 'string',
        enum: ['literal', 'regex'],
        default: 'literal',
        description: 'literal 按字面字符串搜索；regex 使用单行 ripgrep 默认正则引擎（不支持 PCRE2、lookaround、backreference 或跨行匹配）',
      },
      path: { type: 'string' },
      pattern: { type: 'string', description: '文件路径 glob，由 ripgrep 解释' },
      caseSensitive: { type: 'boolean' },
      limit: { type: 'integer', minimum: 1, maximum: MAX_FILE_LIMIT },
      includeIgnored: { type: 'boolean', description: '包含 VCS ignored 文件' },
      includeHidden: { type: 'boolean', description: '包含隐藏文件' },
    },
    required: ['query'],
    additionalProperties: false,
  },
}

// [maint-note:d78c71a053d8]
// 维护导航：parseSearchTextInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：SearchTextInput。
// 分支条件：Buffer.byteLength(query, 'utf8') > MAX_SEARCH_QUERY_UTF8_BYTES。
// 分支条件：input.regex !== undefined。
// 分支条件：legacyRegex。
// 分支条件：mode === 'regex'。
// 分支条件：mode === 'regex' && (query.includes('\n') || query.includes('\r'))。
// 返回路径：{ query, mode, path: optionalString(input, 'path', '.'), pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''), caseSensitive: optionalBoolean(input, 'caseSensitive', false), limit: optionalLimit(input), in。
// 显式抛出：inputError(`query must not exceed ${MAX_SEARCH_QUERY_UTF8_BYTES} UTF-8 bytes`, 'out_of_range', 'query', `at most ${MAX_SEARCH_QUERY_UTF8_BYTES} UTF-8 bytes`)。
// 显式抛出：inputError('legacy regex:true is not supported; use mode:"regex"', 'invalid_value', 'regex', 'false or omitted', 'Use mode:"regex" instead of regex:true.')。
// 显式抛出：inputError('legacy regex:false conflicts with mode:"regex"; remove regex', 'conflicting_fields', 'regex', 'omit regex when mode is regex')。
// 显式抛出：inputError('regex query must not contain line breaks; multiline search is not supported', 'invalid_value', 'query', 'single-line regex')。
// 直接调用：assertObject。
// 直接调用：assertKnownKeys。
// 直接调用：requiredString。
// 直接调用：Buffer.byteLength。
// 直接调用：inputError。
// 直接调用：parseSearchMode。
// 直接调用：optionalBoolean。
// 直接调用：query.includes。
// 直接调用：optionalString。
// 直接调用：optionalLimit。
// [/maint-note:d78c71a053d8]
export function parseSearchTextInput(input: Record<string, unknown>): SearchTextInput {
  assertObject(input)
  assertKnownKeys(input, [
    'query', 'mode', 'path', 'pattern', 'caseSensitive', 'regex', 'limit', 'includeIgnored', 'includeHidden',
  ])
  const query = requiredString(input, 'query')
  if (Buffer.byteLength(query, 'utf8') > MAX_SEARCH_QUERY_UTF8_BYTES) {
    throw inputError(`query must not exceed ${MAX_SEARCH_QUERY_UTF8_BYTES} UTF-8 bytes`, 'out_of_range', 'query', `at most ${MAX_SEARCH_QUERY_UTF8_BYTES} UTF-8 bytes`)
  }
  const mode = parseSearchMode(input.mode)
  if (input.regex !== undefined) {
    const legacyRegex = optionalBoolean(input, 'regex', false)
    if (legacyRegex) {
      throw inputError('legacy regex:true is not supported; use mode:"regex"', 'invalid_value', 'regex', 'false or omitted', 'Use mode:"regex" instead of regex:true.')
    }
    if (mode === 'regex') {
      throw inputError('legacy regex:false conflicts with mode:"regex"; remove regex', 'conflicting_fields', 'regex', 'omit regex when mode is regex')
    }
  }
  if (mode === 'regex' && (query.includes('\n') || query.includes('\r'))) {
    throw inputError('regex query must not contain line breaks; multiline search is not supported', 'invalid_value', 'query', 'single-line regex')
  }
  return {
    query,
    mode,
    path: optionalString(input, 'path', '.'),
    pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''),
    caseSensitive: optionalBoolean(input, 'caseSensitive', false),
    limit: optionalLimit(input),
    includeIgnored: optionalBoolean(input, 'includeIgnored', false),
    includeHidden: optionalBoolean(input, 'includeHidden', false),
  }
}

// [maint-note:148c9ebcc587]
// 维护导航：parseSearchMode；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：SearchTextInput['mode']。
// 分支条件：value === undefined || value === 'literal'。
// 分支条件：value === 'regex'。
// 返回路径：'literal'。
// 返回路径：'regex'。
// 显式抛出：inputError('mode must be "literal" or "regex"', 'invalid_value', 'mode', 'literal or regex')。
// 直接调用：inputError。
// [/maint-note:148c9ebcc587]
function parseSearchMode(value: unknown): SearchTextInput['mode'] {
  if (value === undefined || value === 'literal') return 'literal'
  if (value === 'regex') return 'regex'
  throw inputError('mode must be "literal" or "regex"', 'invalid_value', 'mode', 'literal or regex')
}
