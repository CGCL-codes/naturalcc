import { chmod, lstat, mkdir, readFile, rmdir, unlink, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve } from 'node:path'

import type { ToolExecutionContext } from '../types.js'
import { resolveWorkspacePath } from '../workspace.js'
import { operationError, withFileDiagnostics } from '../diagnostics.js'
import {
  assertOutputBudget,
  boundedJsonResult,
  MAX_PATCH_CHANGE_BYTES,
  MAX_PATCH_TARGET_BYTES,
  MAX_PATCH_TOTAL_TARGET_BYTES,
  throwIfAborted,
} from '../toolUtils.js'
import { applyHunks, parseApplyPatchInput, parsePatch } from './parser.js'
import type {
  ApplyPatchDependencies,
  ApplyPatchDisplayFile,
  ApplyPatchDisplayPayload,
  ApplyPatchDisplayRecord,
  FileSnapshot,
  PatchChange,
  PatchHunk,
} from './types.js'

/** Upper bounds for the neutral display payload itself. */
const MAX_DISPLAY_RECORDS_PER_FILE = 4_000
const MAX_DISPLAY_CHARS_PER_FILE = 100_000

export async function executeApplyPatch(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  dependencies: ApplyPatchDependencies,
) {
  return withFileDiagnostics(() => applyWorkspacePatch(input, context, dependencies))
}

async function applyWorkspacePatch(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  dependencies: ApplyPatchDependencies,
) {
  const { patch } = parseApplyPatchInput(input)
  assertOutputBudget(context.maxOutputChars)
  const operations = parsePatch(patch)
  const changes: PatchChange[] = []
  const seen = new Set<string>()
  let totalChangeBytes = 0
  let totalTargetBytes = 0

  for (const operation of operations) {
    throwIfAborted(context.signal)
    const path = await resolveWorkspacePath(context.projectDir, operation.path)
    if (seen.has(path)) throw new Error(`Patch contains duplicate file: ${operation.path}`)
    seen.add(path)
    const existingInfo = await lstatIfExists(path)
    const exists = existingInfo !== undefined

    if (operation.kind === 'add') {
      if (exists) throw operationError(`File already exists: ${operation.path}`, 'file_exists', { field: 'path' })
      const content = operation.lines.join('\n') + (operation.lines.length ? '\n' : '')
      changes.push({
        operation,
        path,
        content,
        additions: operation.lines.length,
        deletions: 0,
        before: { exists: false },
        records: operation.lines.map((text, index) => ({
          kind: 'added' as const,
          text,
          newLine: index + 1,
        })),
      })
      totalChangeBytes += Buffer.byteLength(content, 'utf8')
      assertPatchBudgets(totalChangeBytes, totalTargetBytes)
      continue
    }
    if (!exists) throw operationError(`File does not exist: ${operation.path}`, 'file_not_found', { field: 'path' })
    const info = existingInfo
    if (!info) throw operationError(`File does not exist: ${operation.path}`, 'file_not_found', { field: 'path' })
    if (!info.isFile()) throw operationError(`Not a regular file: ${operation.path}`, 'not_regular_file', { field: 'path' })
    if (info.size > MAX_PATCH_TARGET_BYTES) {
      throw new Error(`File is larger than ${MAX_PATCH_TARGET_BYTES} bytes: ${operation.path}`)
    }
    totalTargetBytes += info.size
    assertPatchBudgets(totalChangeBytes, totalTargetBytes)
    const beforeContent = await readFile(path, { signal: context.signal })
    const before: FileSnapshot = {
      exists: true,
      content: beforeContent,
      mode: info.mode,
    }
    if (operation.kind === 'delete') {
      const removedLines = beforeContent
        .toString('utf8')
        .replace(/\r\n/g, '\n')
        .split('\n')
      if (removedLines.at(-1) === '') removedLines.pop()
      changes.push({
        operation,
        path,
        additions: 0,
        deletions: 0,
        before,
        records: removedLines.map((text, index) => ({
          kind: 'removed' as const,
          text,
          oldLine: index + 1,
        })),
      })
      totalChangeBytes += info.size
      assertPatchBudgets(totalChangeBytes, totalTargetBytes)
      continue
    }
    const sourceBuffer = beforeContent
    if (sourceBuffer.length > MAX_PATCH_TARGET_BYTES) {
      throw new Error(`File is larger than ${MAX_PATCH_TARGET_BYTES} bytes: ${operation.path}`)
    }
    if (sourceBuffer.includes(0)) throw operationError(`File is not a text file: ${operation.path}`, 'invalid_encoding', { field: 'path' })
    let source: string
    try {
      source = new TextDecoder('utf-8', { fatal: true }).decode(sourceBuffer)
    } catch {
      throw operationError(`File is not valid UTF-8: ${operation.path}`, 'invalid_encoding', { field: 'path' })
    }
    const applied = applyHunks(source, operation.hunks, operation.path)
    totalChangeBytes += patchChangeBytes(operation.hunks)
    assertPatchBudgets(totalChangeBytes, totalTargetBytes)
    changes.push({
      operation,
      path,
      content: applied.content,
      additions: applied.additions,
      deletions: applied.deletions,
      before,
      records: applied.records,
    })
  }
  const appliedChanges: PatchChange[] = []
  const createdDirectories: string[] = []
  let operationInProgress: PatchChange | undefined
  try {
    for (const change of changes) {
      throwIfAborted(context.signal)
      await resolveWorkspacePath(context.projectDir, change.operation.path)
      throwIfAborted(context.signal)
      const current = await snapshotFile(change.path)
      if (!snapshotsEqual(current, change.before)) {
        throw operationError(`Patch target changed during preflight: ${change.operation.path}; re-check the workspace`, 'patch_target_changed', { field: 'patch' })
      }
      operationInProgress = change
      if (change.operation.kind === 'delete') {
        await unlink(change.path)
      } else {
        if (change.operation.kind === 'add') {
          const missing = await findMissingParentDirectories(context.projectDir, dirname(change.path))
          createdDirectories.push(...missing)
          await mkdir(dirname(change.path), { recursive: true })
        }
        throwIfAborted(context.signal)
        await writeFile(change.path, change.content ?? '', 'utf8')
      }
      change.after = await snapshotFile(change.path)
      appliedChanges.push(change)
      operationInProgress = undefined
      await dependencies.onChangeApplied?.(change.operation.path)
      // Give cancellation and concurrent workspace changes a chance to be
      // observed before the next file operation.
      await new Promise<void>((resolveNext) => setImmediate(resolveNext))
    }
  } catch (error) {
    const restored = await restoreChanges(appliedChanges, createdDirectories)
    if (!restored || operationInProgress) {
      const detail = error instanceof Error ? error.message : String(error)
      throw operationError(`${detail}; workspace restoration was incomplete, re-check the workspace`, 'patch_restoration_incomplete', { field: 'patch' })
    }
    throw error
  }

  const canonical = boundedJsonResult({
    applied: changes.map((change) => ({
      path: change.operation.path,
      action: change.operation.kind,
      additions: change.additions,
      deletions: change.deletions,
    })),
    truncated: false,
  }, context.maxOutputChars, {
    applied: [],
    truncated: true,
  })
  if (canonical.includes('"truncated":true')) {
    // The canonical budget already dropped the summary: attach no display
    // facts rather than letting the UI show a diff the model never saw.
    return { content: canonical }
  }
  return {
    content: canonical,
    display: {
      kind: 'apply_patch_diff',
      facts: buildApplyPatchDisplayPayload(changes) as unknown as Record<string, unknown>,
    },
  }
}

/**
 * Bounded neutral payload assembled ONLY from successfully applied changes.
 * Records/characters beyond the caps are dropped with `displayTruncated`.
 */
function buildApplyPatchDisplayPayload(changes: readonly PatchChange[]): ApplyPatchDisplayPayload {
  let displayTruncated = false
  const files: ApplyPatchDisplayFile[] = []
  for (const change of changes) {
    const raw = change.records ?? []
    let records: ApplyPatchDisplayRecord[] = raw.slice(0, MAX_DISPLAY_RECORDS_PER_FILE)
    if (raw.length > records.length) displayTruncated = true
    let usedChars = 0
    const boundedRecords: ApplyPatchDisplayRecord[] = []
    for (const record of records) {
      const cost = record.text.length
      if (usedChars + cost > MAX_DISPLAY_CHARS_PER_FILE) {
        displayTruncated = true
        break
      }
      usedChars += cost
      boundedRecords.push(record)
    }
    // display statistics come from the retained records, not the
    // canonical per-change counts — the canonical Delete pair keeps its
    // historical 0/0 for the model/one-shot contract, while the UI diff must
    // report the real removals. Records dropped by the display caps keep the
    // canonical counts as the floor so totals never understate the change.
    const added = boundedRecords.filter((record) => record.kind === 'added').length
    const removed = boundedRecords.filter((record) => record.kind === 'removed').length
    files.push({
      path: change.operation.path,
      action: change.operation.kind,
      additions: Math.max(change.additions, added),
      deletions: Math.max(change.deletions, removed),
      records: boundedRecords,
    })
  }
  return { files, displayTruncated }
}

function assertPatchBudgets(totalChangeBytes: number, totalTargetBytes: number): void {
  if (totalChangeBytes > MAX_PATCH_CHANGE_BYTES) {
    throw new Error(`Patch changes exceed ${MAX_PATCH_CHANGE_BYTES} bytes`)
  }
  if (totalTargetBytes > MAX_PATCH_TOTAL_TARGET_BYTES) {
    throw new Error(`Patch targets exceed ${MAX_PATCH_TOTAL_TARGET_BYTES} bytes`)
  }
}

function patchChangeBytes(hunks: PatchHunk[]): number {
  return hunks.reduce((total, hunk) => total + hunk.lines.reduce((bytes, line) => {
    if (line === '\\ No newline at end of file' || line.startsWith(' ')) return bytes
    return bytes + Buffer.byteLength(line.slice(1), 'utf8') + 1
  }, 0), 0)
}

async function restoreChanges(
  changes: PatchChange[],
  createdDirectories: string[],
): Promise<boolean> {
  let restored = true
  for (const change of changes.reverse()) {
    try {
      const current = await snapshotFile(change.path)
      if (!change.after || !snapshotsEqual(current, change.after)) {
        restored = false
        continue
      }
      if (change.before.exists) {
        if (!change.before.content) {
          restored = false
          continue
        }
        await writeFile(change.path, change.before.content)
        if (change.before.mode !== undefined) await chmod(change.path, change.before.mode)
      } else if (current.exists) {
        await unlink(change.path)
      }
    } catch {
      restored = false
    }
  }
  for (const directory of createdDirectories) {
    try {
      await rmdir(directory)
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      if (code !== 'ENOENT') restored = false
    }
  }
  return restored
}

async function snapshotFile(path: string): Promise<FileSnapshot> {
  const info = await lstatIfExists(path)
  if (!info) return { exists: false }
  if (!info.isFile()) return { exists: true, mode: info.mode }
  try {
    return { exists: true, content: await readFile(path), mode: info.mode }
  } catch {
    return { exists: true, mode: info.mode }
  }
}

function snapshotsEqual(left: FileSnapshot, right: FileSnapshot): boolean {
  if (left.exists !== right.exists) return false
  if (!left.exists && !right.exists) return true
  if (left.mode !== right.mode) return false
  if (!left.content && !right.content) return true
  return Boolean(left.content && right.content && left.content.equals(right.content))
}

async function findMissingParentDirectories(projectDir: string, directory: string): Promise<string[]> {
  const root = resolve(projectDir)
  const missing: string[] = []
  let current = resolve(directory)
  while (current !== root) {
    const info = await lstatIfExists(current)
    if (info) {
      if (!info.isDirectory()) throw new Error(`Patch parent is not a directory: ${directory}`)
      break
    }
    missing.push(current)
    const parent = resolve(current, '..')
    if (parent === current || !isInside(root, parent)) {
      throw new Error(`Patch parent is outside the project workspace: ${directory}`)
    }
    current = parent
  }
  return missing
}

function isInside(root: string, candidate: string): boolean {
  const boundary = relative(root, candidate)
  return boundary === '' || (boundary !== '..' && !boundary.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(boundary))
}

async function lstatIfExists(path: string) {
  try {
    return await lstat(path)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}
