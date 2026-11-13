import type { ToolSchema } from '../types.js'
import { inputError } from '../diagnostics.js'
import {
  assertKnownKeys,
  assertObject,
  MAX_FILE_LIMIT,
  optionalBoolean,
  optionalLimit,
  optionalPositiveInteger,
  optionalString,
} from '../toolUtils.js'

export const DEFAULT_MAX_DEPTH = 64

// [maint-note:f1db2b66858e]
// 数据契约导航：ListFilesMode；字段含义以消费端校验为准。
// 类型组合：'list' | 'count'。
// [/maint-note:f1db2b66858e]
export type ListFilesMode = 'list' | 'count'

// [maint-note:d2a40ae0cb06]
// 数据契约导航：ListFilesInput；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 pattern（可省略）：string。
// 成员 recursive：boolean。
// 成员 maxDepth：number。
// 成员 limit（可省略）：number。
// 成员 includeIgnored：boolean。
// 成员 includeHidden：boolean。
// 成员 mode：ListFilesMode。
// [/maint-note:d2a40ae0cb06]
export interface ListFilesInput {
  path: string
  pattern?: string
  recursive: boolean
  maxDepth: number
  limit?: number
  includeIgnored: boolean
  includeHidden: boolean
  mode: ListFilesMode
}

export const listFilesSchema: ToolSchema = {
  name: 'list_files',
  description: '发现、列举、筛选或统计项目文件时优先使用本工具，不要用 bash/find/rg 等价完成同样的工作。通过系统 ripgrep 递归列出项目内文件；默认遵循 .gitignore 且不含隐藏文件。显式设置 recursive=false 时只列出目标目录的直接文件。mode:"list"（默认）返回 entries，只有 truncated:false 时 entries.length 才是完整匹配数；需要精确数量时改用 mode:"count"，它不返回路径，只返回 { mode:"count", count, complete:true }。本工具不会列出被永久排除的 .git 目录。',
  readOnly: true,
  requiresApproval: false,
  repeatKind: 'read',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: '相对项目根目录的目录或文件路径，默认 .' },
      pattern: { type: 'string', description: 'ripgrep glob，例如 **/*.ts 或 *.ts' },
      recursive: {
        type: 'boolean',
        default: true,
        description: '默认递归；设为 false 时只检查目标目录的直接文件，即使 pattern 含 ** 也不会进入子目录',
      },
      maxDepth: { type: 'integer', minimum: 1, maximum: DEFAULT_MAX_DEPTH },
      limit: { type: 'integer', minimum: 1, maximum: MAX_FILE_LIMIT, description: '仅在 mode:"list" 时有效' },
      includeIgnored: { type: 'boolean', description: '包含 VCS ignored 文件' },
      includeHidden: { type: 'boolean', description: '包含隐藏文件' },
      mode: {
        type: 'string',
        enum: ['list', 'count'],
        default: 'list',
        description: '"list" 返回 entries（默认）；"count" 返回精确文件数，不能与 limit 同时使用',
      },
    },
    additionalProperties: false,
  },
}

// [maint-note:869bae13c4cb]
// 维护导航：parseListFilesInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：ListFilesInput。
// 分支条件：mode === 'count' && input.limit !== undefined。
// 返回路径：{ path: optionalString(input, 'path', '.'), pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''), recursive: optionalBoolean(input, 'recursive', true), maxDepth: optionalPositiveInteger(input, 'maxDepth', 。
// 显式抛出：inputError('limit is not valid when mode is "count"', 'conflicting_fields', 'limit', 'omit limit in count mode')。
// 直接调用：assertObject。
// 直接调用：assertKnownKeys。
// 直接调用：optionalMode。
// 直接调用：inputError。
// 直接调用：optionalString。
// 直接调用：optionalBoolean。
// 直接调用：optionalPositiveInteger。
// 直接调用：optionalLimit。
// [/maint-note:869bae13c4cb]
export function parseListFilesInput(input: Record<string, unknown>): ListFilesInput {
  assertObject(input)
  assertKnownKeys(input, [
    'path', 'pattern', 'recursive', 'maxDepth', 'limit', 'includeIgnored', 'includeHidden', 'mode',
  ])
  const mode = optionalMode(input)
  if (mode === 'count' && input.limit !== undefined) {
    throw inputError('limit is not valid when mode is "count"', 'conflicting_fields', 'limit', 'omit limit in count mode')
  }
  return {
    path: optionalString(input, 'path', '.'),
    pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''),
    recursive: optionalBoolean(input, 'recursive', true),
    maxDepth: optionalPositiveInteger(input, 'maxDepth', DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH),
    limit: input.limit === undefined ? undefined : optionalLimit(input),
    includeIgnored: optionalBoolean(input, 'includeIgnored', false),
    includeHidden: optionalBoolean(input, 'includeHidden', false),
    mode,
  }
}

// [maint-note:fc9f2295acab]
// 维护导航：optionalMode；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：ListFilesMode。
// 分支条件：input.mode === undefined。
// 分支条件：input.mode === 'list' || input.mode === 'count'。
// 返回路径：'list'。
// 返回路径：input.mode。
// 显式抛出：inputError('mode must be "list" or "count"', 'invalid_value', 'mode', 'list or count')。
// 直接调用：inputError。
// [/maint-note:fc9f2295acab]
function optionalMode(input: Record<string, unknown>): ListFilesMode {
  if (input.mode === undefined) return 'list'
  if (input.mode === 'list' || input.mode === 'count') return input.mode
  throw inputError('mode must be "list" or "count"', 'invalid_value', 'mode', 'list or count')
}
