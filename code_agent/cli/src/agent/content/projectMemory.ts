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
  async build(input: ContextBuildInput) {
    const result = await readProjectMemory(input.projectDir, input.signal)
    return {
      projectMemory: result.content,
      warnings: result.warnings,
    }
  },
}

export async function getProjectMemory(
  projectDir: string,
  signal?: AbortSignal,
): Promise<string> {
  return (await readProjectMemory(projectDir, signal)).content
}

export function getClaudeMdContext(
  projectDir = process.cwd(),
  signal?: AbortSignal,
): Promise<string | null> {
  return getProjectMemory(projectDir, signal).then((content) => content || null)
}

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

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

function isMissingFile(error: unknown): boolean {
  return Boolean(
    error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT',
  )
}

function errorCode(error: unknown): string {
  if (error instanceof WorkspacePathError) return 'workspace_boundary'
  return error && typeof error === 'object' && 'code' in error
    ? String(error.code)
    : error instanceof Error ? error.message : 'unknown error'
}

function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  return Boolean(
    signal?.aborted ||
      (error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError')),
  )
}
