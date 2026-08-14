import {
  lstat,
  realpath,
} from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve, win32 } from 'node:path'

import { ToolInputError } from './types.js'

export class WorkspacePathError extends ToolInputError {
  constructor(input: string, reason = 'path is outside the project workspace') {
    super(`Invalid workspace path "${input}": ${reason}`)
    this.name = 'WorkspacePathError'
  }
}

/** Resolve a user path while checking both lexical and symlink boundaries. */
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

function assertInside(root: string, candidate: string, input: string): void {
  const boundary = relative(root, candidate)
  if (boundary === '..' || boundary.startsWith(`..${candidateSeparator()}`) || isAbsolute(boundary)) {
    throw new WorkspacePathError(input)
  }
}

function candidateSeparator(): string {
  return process.platform === 'win32' ? '\\' : '/'
}
