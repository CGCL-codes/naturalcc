import {
  lstat,
  realpath,
} from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve, win32 } from 'node:path'

import { ToolInputError } from './types.js'
import { toolDiagnostic } from './diagnostics.js'

export class WorkspacePathError extends ToolInputError {
  // [maint-note:ddf8eb172f6b]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 input：string。
  // 输入 reason：类型由调用方与函数体约束；缺省表达式 'path is outside the project workspace'。
  // 写入目标：this.name。
  // 直接调用：super。
  // 直接调用：toolDiagnostic。
  // [/maint-note:ddf8eb172f6b]
  constructor(input: string, reason = 'path is outside the project workspace') {
    super(`Invalid workspace path "${input}": ${reason}`, toolDiagnostic('workspace_path_invalid', { field: 'path' }))
    this.name = 'WorkspacePathError'
  }
}

/** Resolve a user path while checking both lexical and symlink boundaries. */
// [maint-note:8a00a2e6031c]
// 路径边界：先做词法约束，再校验 realpath，防止符号链接逃逸。
// 新文件尚无 realpath 时检查最近的已存在父目录。
// 这里只解析用户路径，不创建文件或替模型扩大工作区授权。
// 维护导航：resolveWorkspacePath；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 input：string。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：typeof input !== 'string' || !input.trim()。
// 分支条件：input.includes('\0')。
// 分支条件：isAbsolute(input) || win32.isAbsolute(input)。
// 分支条件：error instanceof WorkspacePathError。
// 遍历边界：parent !== root。
// 返回路径：target。
// 返回路径：candidate。
// 显式抛出：new WorkspacePathError(String(input), 'path is required')。
// 显式抛出：new WorkspacePathError(input, 'path must not contain NUL')。
// 显式抛出：new WorkspacePathError(input, 'absolute paths are not allowed')。
// 显式抛出：error。
// 写入目标：parent。
// 直接调用：input.trim。
// 直接调用：String。
// 直接调用：input.includes。
// 直接调用：isAbsolute。
// 直接调用：win32.isAbsolute。
// 直接调用：realpath(projectDir).catch。
// 直接调用：realpath。
// 直接调用：resolve。
// 直接调用：assertInside。
// 直接调用：lstat。
// 直接调用：dirname。
// 直接调用：realpath(parent).catch。
// [/maint-note:8a00a2e6031c]
export async function resolveWorkspacePath(
  projectDir: string,
  input: string,
): Promise<string> {
  if (typeof input !== 'string' || !input.trim()) {
    throw new WorkspacePathError(String(input), 'path is required')
  }
  if (input.includes('\0')) {
    throw new WorkspacePathError(input, 'path must not contain NUL')
  }
  if (isAbsolute(input) || win32.isAbsolute(input)) {
    throw new WorkspacePathError(input, 'absolute paths are not allowed')
  }

  const root = await realpath(projectDir).catch(() => {
    throw new WorkspacePathError(input, 'project workspace is unavailable')
  })
  const candidate = resolve(root, input)
  assertInside(root, candidate, input)

  try {
    const target = await realpath(candidate)
    assertInside(root, target, input)
    return target
  } catch (error) {
    if (error instanceof WorkspacePathError) throw error
    // New files do not have a realpath yet. Validate the nearest existing
    // parent so a symlinked directory cannot provide an escape hatch.
    let parent = candidate
    while (parent !== root) {
      try {
        await lstat(parent)
      } catch {
        parent = dirname(parent)
        continue
      }
      const resolvedParent = await realpath(parent).catch(() => {
        throw new WorkspacePathError(input, 'existing parent cannot be resolved safely')
      })
      assertInside(root, resolvedParent, input)
      return candidate
    }
    return candidate
  }
}

// [maint-note:20deb6fca328]
// 维护导航：assertInside；下列为本函数源码中的接口与路径索引。
// 输入 root：string。
// 输入 candidate：string。
// 输入 input：string。
// 返回契约：void。
// 分支条件：boundary === '..' || boundary.startsWith(`..${candidateSeparator()}`) || isAbsolute(boundary)。
// 显式抛出：new WorkspacePathError(input)。
// 直接调用：relative。
// 直接调用：boundary.startsWith。
// 直接调用：candidateSeparator。
// 直接调用：isAbsolute。
// [/maint-note:20deb6fca328]
function assertInside(root: string, candidate: string, input: string): void {
  const boundary = relative(root, candidate)
  if (boundary === '..' || boundary.startsWith(`..${candidateSeparator()}`) || isAbsolute(boundary)) {
    throw new WorkspacePathError(input)
  }
}

// [maint-note:628ca34f99e3]
// 维护导航：candidateSeparator；下列为本函数源码中的接口与路径索引。
// 返回契约：string。
// 返回路径：process.platform === 'win32' ? '\\' : '/'。
// [/maint-note:628ca34f99e3]
function candidateSeparator(): string {
  return process.platform === 'win32' ? '\\' : '/'
}
