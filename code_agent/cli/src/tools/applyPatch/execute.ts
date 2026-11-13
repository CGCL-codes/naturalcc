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

// [maint-note:42a6e8cdf9fc]
// 维护导航：executeApplyPatch；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 dependencies：ApplyPatchDependencies。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：withFileDiagnostics(() => applyWorkspacePatch(input, context, dependencies))。
// 直接调用：withFileDiagnostics。
// [/maint-note:42a6e8cdf9fc]
export async function executeApplyPatch(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  dependencies: ApplyPatchDependencies,
) {
  return withFileDiagnostics(() => applyWorkspacePatch(input, context, dependencies))
}

// [maint-note:d26b59ccfaa8]
// 维护导航：applyWorkspacePatch；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 dependencies：ApplyPatchDependencies。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：seen.has(path)。
// 分支条件：operation.kind === 'add'。
// 分支条件：exists。
// 分支条件：!exists。
// 分支条件：!info。
// 分支条件：!info.isFile()。
// 分支条件：info.size > MAX_PATCH_TARGET_BYTES。
// 分支条件：operation.kind === 'delete'。
// 分支条件：removedLines.at(-1) === ''。
// 分支条件：sourceBuffer.length > MAX_PATCH_TARGET_BYTES。
// 分支条件：sourceBuffer.includes(0)。
// 分支条件：!snapshotsEqual(current, change.before)。
// 分支条件：change.operation.kind === 'delete'。
// 分支条件：change.operation.kind === 'add'。
// 分支条件：!restored || operationInProgress。
// 分支条件：canonical.includes('"truncated":true')。
// 遍历边界：const operation ← operations。
// 遍历边界：const change ← changes。
// 返回路径：{ content: canonical }。
// 返回路径：{ content: canonical, display: { kind: 'apply_patch_diff', facts: buildApplyPatchDisplayPayload(changes) as unknown as Record<string, unknown>, }, }。
// 显式抛出：new Error(`Patch contains duplicate file: ${operation.path}`)。
// 显式抛出：operationError(`File already exists: ${operation.path}`, 'file_exists', { field: 'path' })。
// 显式抛出：operationError(`File does not exist: ${operation.path}`, 'file_not_found', { field: 'path' })。
// 显式抛出：operationError(`Not a regular file: ${operation.path}`, 'not_regular_file', { field: 'path' })。
// 显式抛出：new Error(`File is larger than ${MAX_PATCH_TARGET_BYTES} bytes: ${operation.path}`)。
// 显式抛出：operationError(`File is not a text file: ${operation.path}`, 'invalid_encoding', { field: 'path' })。
// 显式抛出：operationError(`File is not valid UTF-8: ${operation.path}`, 'invalid_encoding', { field: 'path' })。
// 显式抛出：operationError(`Patch target changed during preflight: ${change.operation.path}; re-check the workspace`, 'patch_target_changed', { field: 'patch' })。
// 显式抛出：operationError(`${detail}; workspace restoration was incomplete, re-check the workspace`, 'patch_restoration_incomplete', { field: 'patch' })。
// 显式抛出：error。
// 写入目标：totalChangeBytes。
// 写入目标：totalTargetBytes。
// 写入目标：source。
// 写入目标：operationInProgress。
// 写入目标：change.after。
// 直接调用：parseApplyPatchInput。
// 直接调用：assertOutputBudget。
// 直接调用：parsePatch。
// 直接调用：throwIfAborted。
// 直接调用：resolveWorkspacePath。
// 直接调用：seen.has。
// 直接调用：seen.add。
// 直接调用：lstatIfExists。
// 直接调用：operationError。
// 直接调用：operation.lines.join。
// 直接调用：changes.push。
// 直接调用：operation.lines.map。
// 直接调用：Buffer.byteLength。
// 直接调用：assertPatchBudgets。
// 直接调用：info.isFile。
// 直接调用：readFile。
// 直接调用：beforeContent .toString('utf8') .replace(/\r\n/g, '\n') .split。
// 直接调用：beforeContent .toString('utf8') .replace。
// 直接调用：beforeContent .toString。
// 直接调用：removedLines.at。
// 直接调用：removedLines.pop。
// 直接调用：removedLines.map。
// 直接调用：sourceBuffer.includes。
// 直接调用：new TextDecoder('utf-8', { fatal: true }).decode。
// 直接调用：applyHunks。
// 直接调用：patchChangeBytes。
// 直接调用：snapshotFile。
// 直接调用：snapshotsEqual。
// 直接调用：unlink。
// 直接调用：findMissingParentDirectories。
// 直接调用：dirname。
// 直接调用：createdDirectories.push。
// 直接调用：mkdir。
// 直接调用：writeFile。
// 直接调用：appliedChanges.push。
// 直接调用：dependencies.onChangeApplied。
// 直接调用：restoreChanges。
// 直接调用：String。
// 直接调用：boundedJsonResult。
// 直接调用：changes.map。
// 直接调用：canonical.includes。
// 直接调用：buildApplyPatchDisplayPayload。
// [/maint-note:d26b59ccfaa8]
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
// [maint-note:4953ff34cac8]
// 维护导航：buildApplyPatchDisplayPayload；下列为本函数源码中的接口与路径索引。
// 输入 changes：readonly PatchChange[]。
// 返回契约：ApplyPatchDisplayPayload。
// 分支条件：raw.length > records.length。
// 分支条件：usedChars + cost > MAX_DISPLAY_CHARS_PER_FILE。
// 遍历边界：const change ← changes。
// 遍历边界：const record ← records。
// 返回路径：{ files, displayTruncated }。
// 写入目标：displayTruncated。
// 写入目标：usedChars。
// 直接调用：raw.slice。
// 直接调用：boundedRecords.push。
// 直接调用：boundedRecords.filter。
// 直接调用：files.push。
// 直接调用：Math.max。
// [/maint-note:4953ff34cac8]
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

// [maint-note:b94840276206]
// 维护导航：assertPatchBudgets；下列为本函数源码中的接口与路径索引。
// 输入 totalChangeBytes：number。
// 输入 totalTargetBytes：number。
// 返回契约：void。
// 分支条件：totalChangeBytes > MAX_PATCH_CHANGE_BYTES。
// 分支条件：totalTargetBytes > MAX_PATCH_TOTAL_TARGET_BYTES。
// 显式抛出：new Error(`Patch changes exceed ${MAX_PATCH_CHANGE_BYTES} bytes`)。
// 显式抛出：new Error(`Patch targets exceed ${MAX_PATCH_TOTAL_TARGET_BYTES} bytes`)。
// [/maint-note:b94840276206]
function assertPatchBudgets(totalChangeBytes: number, totalTargetBytes: number): void {
  if (totalChangeBytes > MAX_PATCH_CHANGE_BYTES) {
    throw new Error(`Patch changes exceed ${MAX_PATCH_CHANGE_BYTES} bytes`)
  }
  if (totalTargetBytes > MAX_PATCH_TOTAL_TARGET_BYTES) {
    throw new Error(`Patch targets exceed ${MAX_PATCH_TOTAL_TARGET_BYTES} bytes`)
  }
}

// [maint-note:ed4caae0b6e0]
// 维护导航：patchChangeBytes；下列为本函数源码中的接口与路径索引。
// 输入 hunks：PatchHunk[]。
// 返回契约：number。
// 返回路径：hunks.reduce((total, hunk) => total + hunk.lines.reduce((bytes, line) => { if (line === '\\ No newline at end of file' || line.startsWith(' ')) return bytes return bytes + Buffer.byteLength(line.slice(1), 'utf8') + 1 }, 0), 0)。
// 直接调用：hunks.reduce。
// [/maint-note:ed4caae0b6e0]
function patchChangeBytes(hunks: PatchHunk[]): number {
  return hunks.reduce((total, hunk) => total + hunk.lines.reduce((bytes, line) => {
    if (line === '\\ No newline at end of file' || line.startsWith(' ')) return bytes
    return bytes + Buffer.byteLength(line.slice(1), 'utf8') + 1
  }, 0), 0)
}

// [maint-note:9be5032bd98c]
// 维护导航：restoreChanges；下列为本函数源码中的接口与路径索引。
// 输入 changes：PatchChange[]。
// 输入 createdDirectories：string[]。
// 返回契约：Promise<boolean>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!change.after || !snapshotsEqual(current, change.after)。
// 分支条件：change.before.exists。
// 分支条件：!change.before.content。
// 分支条件：change.before.mode !== undefined。
// 分支条件：current.exists。
// 分支条件：code !== 'ENOENT'。
// 遍历边界：const change ← changes.reverse()。
// 遍历边界：const directory ← createdDirectories。
// 返回路径：restored。
// 写入目标：restored。
// 直接调用：changes.reverse。
// 直接调用：snapshotFile。
// 直接调用：snapshotsEqual。
// 直接调用：writeFile。
// 直接调用：chmod。
// 直接调用：unlink。
// 直接调用：rmdir。
// [/maint-note:9be5032bd98c]
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

// [maint-note:9ee87962fc7b]
// 维护导航：snapshotFile；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 返回契约：Promise<FileSnapshot>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!info。
// 分支条件：!info.isFile()。
// 返回路径：{ exists: false }。
// 返回路径：{ exists: true, mode: info.mode }。
// 返回路径：{ exists: true, content: await readFile(path), mode: info.mode }。
// 直接调用：lstatIfExists。
// 直接调用：info.isFile。
// 直接调用：readFile。
// [/maint-note:9ee87962fc7b]
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

// [maint-note:239e9dc543a5]
// 维护导航：snapshotsEqual；下列为本函数源码中的接口与路径索引。
// 输入 left：FileSnapshot。
// 输入 right：FileSnapshot。
// 返回契约：boolean。
// 分支条件：left.exists !== right.exists。
// 分支条件：!left.exists && !right.exists。
// 分支条件：left.mode !== right.mode。
// 分支条件：!left.content && !right.content。
// 返回路径：false。
// 返回路径：true。
// 返回路径：Boolean(left.content && right.content && left.content.equals(right.content))。
// 直接调用：Boolean。
// 直接调用：left.content.equals。
// [/maint-note:239e9dc543a5]
function snapshotsEqual(left: FileSnapshot, right: FileSnapshot): boolean {
  if (left.exists !== right.exists) return false
  if (!left.exists && !right.exists) return true
  if (left.mode !== right.mode) return false
  if (!left.content && !right.content) return true
  return Boolean(left.content && right.content && left.content.equals(right.content))
}

// [maint-note:c4360880e8d2]
// 维护导航：findMissingParentDirectories；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 directory：string。
// 返回契约：Promise<string[]>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：info。
// 分支条件：!info.isDirectory()。
// 分支条件：parent === current || !isInside(root, parent)。
// 遍历边界：current !== root。
// 返回路径：missing。
// 显式抛出：new Error(`Patch parent is not a directory: ${directory}`)。
// 显式抛出：new Error(`Patch parent is outside the project workspace: ${directory}`)。
// 写入目标：current。
// 直接调用：resolve。
// 直接调用：lstatIfExists。
// 直接调用：info.isDirectory。
// 直接调用：missing.push。
// 直接调用：isInside。
// [/maint-note:c4360880e8d2]
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

// [maint-note:1ac44337e418]
// 维护导航：isInside；下列为本函数源码中的接口与路径索引。
// 输入 root：string。
// 输入 candidate：string。
// 返回契约：boolean。
// 返回路径：boundary === '' || (boundary !== '..' && !boundary.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(boundary))。
// 直接调用：relative。
// 直接调用：boundary.startsWith。
// 直接调用：isAbsolute。
// [/maint-note:1ac44337e418]
function isInside(root: string, candidate: string): boolean {
  const boundary = relative(root, candidate)
  return boundary === '' || (boundary !== '..' && !boundary.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(boundary))
}

// [maint-note:5bd8e78d525e]
// 维护导航：lstatIfExists；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：(error as NodeJS.ErrnoException).code === 'ENOENT'。
// 返回路径：await lstat(path)。
// 返回路径：undefined。
// 显式抛出：error。
// 直接调用：lstat。
// [/maint-note:5bd8e78d525e]
async function lstatIfExists(path: string) {
  try {
    return await lstat(path)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
}
