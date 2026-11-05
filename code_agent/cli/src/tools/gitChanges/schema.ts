import type { ToolSchema } from '../types.js'
import { ToolInputError } from '../types.js'
import { assertKnownKeys, assertObject } from '../toolUtils.js'
import type { GitChangeMode, GitChangeScope } from '../backends/git/types.js'

export const MAX_GIT_CHANGES_PATH_BYTES = 4_096
export const MAX_GIT_CHANGES_CURSOR_BYTES = 2_048

export interface GitChangesInput {
  mode: GitChangeMode
  scope: GitChangeScope
  path?: string
  cursor?: string
}

export const gitChangesSchema: ToolSchema = {
  name: 'git_changes',
  description: '按需读取 workspace 的结构化 Git 变更；模式用途见 mode。complete:true 仅表示当前查询完整，不代表全仓内容已读或任务完成。untracked 正文用 read_file。',
  readOnly: true,
  requiresApproval: false,
  repeatKind: 'read',
  parameters: {
    type: 'object',
    properties: {
      mode: { type: 'string', enum: ['summary', 'files', 'patch'], description: 'summary：数量/分类汇总，区分 uniqueFiles 与跨层记录数；files：路径/层/动作/metadata 清单；patch：已知单文件前后差异。全仓内容总结范围未知时先 files，再按需读 patch。' },
      scope: { type: 'string', enum: ['unstaged', 'staged', 'both'], default: 'both', description: '按所需层选择 staged（HEAD→index）、unstaged（index→worktree，含 untracked 元数据）或 both（默认）；只问单层无需查另一层。' },
      path: { type: 'string', description: 'workspace 相对路径；省略时 summary/files 观察全仓。patch 必须指定一个逻辑文件，可直接查询，无需先 summary/files；单 path 完整不等于全仓完整。' },
      cursor: { type: 'string', description: 'files/patch 上页 nextCursor 的不透明 token；保持 mode/scope/path 不变续读，stale/observation changed 后从第一页重查，不混合新旧页。' },
    },
    required: ['mode'],
    additionalProperties: false,
  },
}

export function parseGitChangesInput(input: Record<string, unknown>): GitChangesInput {
  assertObject(input)
  assertKnownKeys(input, ['mode', 'scope', 'path', 'cursor'])
  const mode = parseEnum(input.mode, 'mode', ['summary', 'files', 'patch'] as const)
  const scope = input.scope === undefined
    ? 'both'
    : parseEnum(input.scope, 'scope', ['unstaged', 'staged', 'both'] as const)
  const path = input.path === undefined ? undefined : parsePath(input.path)
  const cursor = input.cursor === undefined
    ? undefined
    : parseBoundedString(input.cursor, 'cursor', MAX_GIT_CHANGES_CURSOR_BYTES)

  if (mode === 'patch' && path === undefined) {
    throw new ToolInputError('path is required when mode is "patch"')
  }
  if (mode === 'summary' && cursor !== undefined) {
    throw new ToolInputError('cursor is only supported for files or patch mode')
  }
  return { mode, scope, ...(path === undefined ? {} : { path }), ...(cursor === undefined ? {} : { cursor }) }
}

function parsePath(value: unknown): string {
  const path = parseBoundedString(value, 'path', MAX_GIT_CHANGES_PATH_BYTES)
  if (path.startsWith('/') || path.startsWith('\\') || /^[A-Za-z]:[\\/]/u.test(path)) {
    throw new ToolInputError('path must be workspace-relative')
  }
  const segments = path.split('/')
  if (segments.some((segment) => segment === '' || segment === '.' || segment === '..')) {
    throw new ToolInputError('path must not contain empty, ".", or ".." segments')
  }
  return path
}

function parseBoundedString(value: unknown, name: string, maxBytes: number): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new ToolInputError(`${name} must be a non-empty string`)
  }
  if (value.includes('\0')) throw new ToolInputError(`${name} must not contain NUL`)
  if (Buffer.byteLength(value, 'utf8') > maxBytes) {
    throw new ToolInputError(`${name} must not exceed ${maxBytes} UTF-8 bytes`)
  }
  return value
}

function parseEnum<const T extends readonly string[]>(
  value: unknown,
  name: string,
  accepted: T,
): T[number] {
  if (typeof value === 'string' && accepted.includes(value)) return value as T[number]
  throw new ToolInputError(`${name} must be one of: ${accepted.join(', ')}`)
}
