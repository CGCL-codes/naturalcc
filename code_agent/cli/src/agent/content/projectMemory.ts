import { lstat, open } from 'node:fs/promises'

import { createAbortError } from '../errors.js'
import { resolveWorkspacePath, WorkspacePathError } from '../../tools/workspace.js'
import type { ContextBuildInput, ContextProvider } from '../types/content.js'

export const PROJECT_MEMORY_FILES = ['AGENTS.md', 'CLAUDE.md'] as const
export const MAX_PROJECT_MEMORY_FILE_BYTES = 64 * 1024
export const MAX_PROJECT_MEMORY_TOTAL_BYTES = 96 * 1024
const READ_CHUNK_BYTES = 4 * 1024

export const projectMemoryProvider: ContextProvider = {
  name: 'project-memory',
  // [maint-note:b73b93ccc3f8]
  // 维护导航：build；下列为本函数源码中的接口与路径索引。
  // 输入 input：ContextBuildInput。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 返回路径：{ projectMemory: result.content, warnings: result.warnings, }。
  // 直接调用：readProjectMemory。
  // [/maint-note:b73b93ccc3f8]
  async build(input: ContextBuildInput) {
    const result = await readProjectMemory(input.projectDir, input.signal)
    return {
      projectMemory: result.content,
      warnings: result.warnings,
    }
  },
}

// [maint-note:0ab2d52c4bd1]
// 维护导航：getProjectMemory；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：(await readProjectMemory(projectDir, signal)).content。
// 直接调用：readProjectMemory。
// [/maint-note:0ab2d52c4bd1]
export async function getProjectMemory(
  projectDir: string,
  signal?: AbortSignal,
): Promise<string> {
  return (await readProjectMemory(projectDir, signal)).content
}

// [maint-note:5e1d5935b441]
// 维护导航：getClaudeMdContext；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：类型由调用方与函数体约束；缺省表达式 process.cwd()。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：Promise<string | null>。
// 返回路径：getProjectMemory(projectDir, signal).then((content) => content || null)。
// 直接调用：getProjectMemory(projectDir, signal).then。
// 直接调用：getProjectMemory。
// [/maint-note:5e1d5935b441]
export function getClaudeMdContext(
  projectDir = process.cwd(),
  signal?: AbortSignal,
): Promise<string | null> {
  return getProjectMemory(projectDir, signal).then((content) => content || null)
}

// [maint-note:cf7c23fb12d2]
// 维护导航：readProjectMemoryFile；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 filename：string。
// 输入 signal：AbortSignal | undefined。
// 输入 totalBytes：{ value: number }。
// 返回契约：Promise<{ content: string; warning?: string }>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info.isFile()。
// 分支条件：info.size > MAX_PROJECT_MEMORY_FILE_BYTES。
// 分支条件：isAbortError(error, signal)。
// 分支条件：isMissingFile(error)。
// 返回路径：{ content: '', warning: `${filename} could not be read: not a regular file` }。
// 返回路径：{ content: '', warning: `${filename} could not be read: file exceeds ${MAX_PROJECT_MEMORY_FILE_BYTES} bytes` }。
// 返回路径：{ content: content ? `Contents of ${filename}:\n\n${content}` : '', }。
// 返回路径：{ content: '' }。
// 返回路径：{ content: '', warning: `${filename} could not be read: ${errorCode(error)}`, }。
// 显式抛出：error。
// 直接调用：throwIfAborted。
// 直接调用：resolveWorkspacePath。
// 直接调用：lstat。
// 直接调用：info.isFile。
// 直接调用：readBoundedText。
// 直接调用：isAbortError。
// 直接调用：isMissingFile。
// 直接调用：errorCode。
// [/maint-note:cf7c23fb12d2]
async function readProjectMemoryFile(
  projectDir: string,
  filename: string,
  signal: AbortSignal | undefined,
  totalBytes: { value: number },
): Promise<{ content: string; warning?: string }> {
  try {
    throwIfAborted(signal)
    const path = await resolveWorkspacePath(projectDir, filename)
    const info = await lstat(path)
    if (!info.isFile()) {
      return { content: '', warning: `${filename} could not be read: not a regular file` }
    }
    if (info.size > MAX_PROJECT_MEMORY_FILE_BYTES) {
      return { content: '', warning: `${filename} could not be read: file exceeds ${MAX_PROJECT_MEMORY_FILE_BYTES} bytes` }
    }

    const content = await readBoundedText(path, filename, signal, totalBytes)
    return {
      content: content ? `Contents of ${filename}:\n\n${content}` : '',
    }
  } catch (error) {
    if (isAbortError(error, signal)) throw error
    if (isMissingFile(error)) return { content: '' }
    return {
      content: '',
      warning: `${filename} could not be read: ${errorCode(error)}`,
    }
  }
}

// [maint-note:5a81eb5fdb1b]
// 维护导航：readProjectMemory；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：Promise<{ content: string warnings: string[] }>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 遍历边界：const filename ← PROJECT_MEMORY_FILES。
// 返回路径：{ content: chunks.map((chunk) => chunk.content).filter(Boolean).join('\n\n'), warnings: chunks.flatMap((chunk) => chunk.warning ? [chunk.warning] : []), }。
// 直接调用：throwIfAborted。
// 直接调用：chunks.push。
// 直接调用：readProjectMemoryFile。
// 直接调用：chunks.map((chunk) => chunk.content).filter(Boolean).join。
// 直接调用：chunks.map((chunk) => chunk.content).filter。
// 直接调用：chunks.map。
// 直接调用：chunks.flatMap。
// [/maint-note:5a81eb5fdb1b]
async function readProjectMemory(
  projectDir: string,
  signal?: AbortSignal,
): Promise<{
  content: string
  warnings: string[]
}> {
  const totalBytes = { value: 0 }
  const chunks = []
  for (const filename of PROJECT_MEMORY_FILES) {
    throwIfAborted(signal)
    chunks.push(await readProjectMemoryFile(projectDir, filename, signal, totalBytes))
  }
  return {
    content: chunks.map((chunk) => chunk.content).filter(Boolean).join('\n\n'),
    warnings: chunks.flatMap((chunk) => chunk.warning ? [chunk.warning] : []),
  }
}

// [maint-note:1f08860b2022]
// 维护导航：readBoundedText；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 输入 filename：string。
// 输入 signal：AbortSignal | undefined。
// 输入 totalBytes：{ value: number }。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：remainingFile <= 0 || remainingTotal <= 0。
// 分支条件：extra.bytesRead > 0。
// 分支条件：result.bytesRead === 0。
// 分支条件：data.includes(0)。
// 遍历边界：true。
// 返回路径：new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))。
// 显式抛出：new Error(`${filename} exceeds the project memory byte budget`)。
// 显式抛出：new Error(`${filename} contains NUL bytes`)。
// 显式抛出：new Error(`${filename} is not valid UTF-8`)。
// 写入目标：bytesRead。
// 写入目标：totalBytes.value。
// 直接调用：open。
// 直接调用：throwIfAborted。
// 直接调用：Buffer.allocUnsafe。
// 直接调用：handle.read。
// 直接调用：Math.min。
// 直接调用：chunk.subarray。
// 直接调用：data.includes。
// 直接调用：chunks.push。
// 直接调用：new TextDecoder('utf-8', { fatal: true }).decode。
// 直接调用：Buffer.concat。
// 直接调用：handle.close。
// [/maint-note:1f08860b2022]
async function readBoundedText(
  path: string,
  filename: string,
  signal: AbortSignal | undefined,
  totalBytes: { value: number },
): Promise<string> {
  const handle = await open(path, 'r')
  const chunks: Buffer[] = []
  let bytesRead = 0

  try {
    while (true) {
      throwIfAborted(signal)
      const remainingFile = MAX_PROJECT_MEMORY_FILE_BYTES - bytesRead
      const remainingTotal = MAX_PROJECT_MEMORY_TOTAL_BYTES - totalBytes.value
      if (remainingFile <= 0 || remainingTotal <= 0) {
        const probe = Buffer.allocUnsafe(1)
        const extra = await handle.read(probe, 0, 1, null)
        if (extra.bytesRead > 0) {
          throw new Error(`${filename} exceeds the project memory byte budget`)
        }
        break
      }

      const chunk = Buffer.allocUnsafe(Math.min(READ_CHUNK_BYTES, remainingFile, remainingTotal))
      const result = await handle.read(chunk, 0, chunk.length, null)
      if (result.bytesRead === 0) break
      const data = chunk.subarray(0, result.bytesRead)
      if (data.includes(0)) {
        throw new Error(`${filename} contains NUL bytes`)
      }
      chunks.push(data)
      bytesRead += result.bytesRead
      totalBytes.value += result.bytesRead
    }

    try {
      return new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))
    } catch {
      throw new Error(`${filename} is not valid UTF-8`)
    }
  } finally {
    await handle.close()
  }
}

// [maint-note:7527b253b0bc]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：createAbortError()。
// 直接调用：createAbortError。
// [/maint-note:7527b253b0bc]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

// [maint-note:3a19ffc8e9e9]
// 维护导航：isMissingFile；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：boolean。
// 返回路径：Boolean( error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT', )。
// 直接调用：Boolean。
// [/maint-note:3a19ffc8e9e9]
function isMissingFile(error: unknown): boolean {
  return Boolean(
    error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT',
  )
}

// [maint-note:d7d36a98195d]
// 维护导航：errorCode；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：string。
// 分支条件：error instanceof WorkspacePathError。
// 返回路径：'workspace_boundary'。
// 返回路径：error && typeof error === 'object' && 'code' in error ? String(error.code) : error instanceof Error ? error.message : 'unknown error'。
// 直接调用：String。
// [/maint-note:d7d36a98195d]
function errorCode(error: unknown): string {
  if (error instanceof WorkspacePathError) return 'workspace_boundary'
  return error && typeof error === 'object' && 'code' in error
    ? String(error.code)
    : error instanceof Error ? error.message : 'unknown error'
}

// [maint-note:6a3b5445f6d7]
// 维护导航：isAbortError；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：boolean。
// 返回路径：Boolean( signal?.aborted || (error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError')), )。
// 直接调用：Boolean。
// [/maint-note:6a3b5445f6d7]
function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  return Boolean(
    signal?.aborted ||
      (error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError')),
  )
}
