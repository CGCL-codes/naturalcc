import type { ToolSchema } from '../types.js'
import { ToolInputError } from '../types.js'
import { assertKnownKeys, assertObject } from '../toolUtils.js'
import type { GitChangeMode, GitChangeScope } from '../backends/git/types.js'

export const MAX_GIT_CHANGES_PATH_BYTES = 4_096
export const MAX_GIT_CHANGES_CURSOR_BYTES = 2_048

// [maint-note:85c577135bcd]
// 数据契约导航：GitChangesInput；字段含义以消费端校验为准。
// 成员 mode：GitChangeMode。
// 成员 scope：GitChangeScope。
// 成员 path（可省略）：string。
// 成员 cursor（可省略）：string。
// [/maint-note:85c577135bcd]
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

// [maint-note:e68a006f021e]
// 维护导航：parseGitChangesInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：GitChangesInput。
// 分支条件：mode === 'patch' && path === undefined。
// 分支条件：mode === 'summary' && cursor !== undefined。
// 返回路径：{ mode, scope, ...(path === undefined ? {} : { path }), ...(cursor === undefined ? {} : { cursor }) }。
// 显式抛出：new ToolInputError('path is required when mode is "patch"')。
// 显式抛出：new ToolInputError('cursor is only supported for files or patch mode')。
// 直接调用：assertObject。
// 直接调用：assertKnownKeys。
// 直接调用：parseEnum。
// 直接调用：parsePath。
// 直接调用：parseBoundedString。
// [/maint-note:e68a006f021e]
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

// [maint-note:234ce480105e]
// 维护导航：parsePath；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：string。
// 分支条件：path.startsWith('/') || path.startsWith('\\') || /^[A-Za-z]:[\\/]/u.test(path)。
// 分支条件：segments.some((segment) => segment === '' || segment === '.' || segment === '..')。
// 返回路径：path。
// 显式抛出：new ToolInputError('path must be workspace-relative')。
// 显式抛出：new ToolInputError('path must not contain empty, ".", or ".." segments')。
// 直接调用：parseBoundedString。
// 直接调用：path.startsWith。
// 直接调用：/^[A-Za-z]:[\\/]/u.test。
// 直接调用：path.split。
// 直接调用：segments.some。
// [/maint-note:234ce480105e]
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

// [maint-note:0b26efef8344]
// 维护导航：parseBoundedString；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 name：string。
// 输入 maxBytes：number。
// 返回契约：string。
// 分支条件：typeof value !== 'string' || value.length === 0。
// 分支条件：value.includes('\0')。
// 分支条件：Buffer.byteLength(value, 'utf8') > maxBytes。
// 返回路径：value。
// 显式抛出：new ToolInputError(`${name} must be a non-empty string`)。
// 显式抛出：new ToolInputError(`${name} must not contain NUL`)。
// 显式抛出：new ToolInputError(`${name} must not exceed ${maxBytes} UTF-8 bytes`)。
// 直接调用：value.includes。
// 直接调用：Buffer.byteLength。
// [/maint-note:0b26efef8344]
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

// [maint-note:fc9a8bbaaca1]
// 维护导航：parseEnum；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 name：string。
// 输入 accepted：T。
// 返回契约：T[number]。
// 分支条件：typeof value === 'string' && accepted.includes(value)。
// 返回路径：value as T[number]。
// 显式抛出：new ToolInputError(`${name} must be one of: ${accepted.join(', ')}`)。
// 直接调用：accepted.includes。
// 直接调用：accepted.join。
// [/maint-note:fc9a8bbaaca1]
function parseEnum<const T extends readonly string[]>(
  value: unknown,
  name: string,
  accepted: T,
): T[number] {
  if (typeof value === 'string' && accepted.includes(value)) return value as T[number]
  throw new ToolInputError(`${name} must be one of: ${accepted.join(', ')}`)
}
