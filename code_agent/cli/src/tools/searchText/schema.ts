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

function parseSearchMode(value: unknown): SearchTextInput['mode'] {
  if (value === undefined || value === 'literal') return 'literal'
  if (value === 'regex') return 'regex'
  throw inputError('mode must be "literal" or "regex"', 'invalid_value', 'mode', 'literal or regex')
}
