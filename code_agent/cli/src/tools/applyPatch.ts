import { chmod, lstat, mkdir, readFile, rmdir, unlink, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve } from 'node:path'

import type { Tool, ToolExecutionContext } from './types.js'
import { ToolInputError } from './types.js'
import { MAX_APPROVAL_REASON_CHARS, validateApprovalReason } from './approval.js'
import { resolveWorkspacePath } from './workspace.js'
import {
  MAX_PATCH_LENGTH,
  assertKnownKeys,
  assertObject,
  assertOutputBudget,
  boundedJsonResult,
  MAX_PATCH_CHANGE_BYTES,
  MAX_PATCH_FILES,
  MAX_PATCH_TARGET_BYTES,
  MAX_PATCH_TOTAL_TARGET_BYTES,
  requiredString,
  throwIfAborted,
} from './toolUtils.js'

type PatchOperation =
  | { kind: 'add'; path: string; lines: string[] }
  | { kind: 'delete'; path: string }
  | { kind: 'update'; path: string; hunks: PatchHunk[] }

interface PatchHunk {
  oldStart: number
  lines: string[]
}

interface ApplyPatchInput {
  patch: string
  reason: string
}

export interface ParsedPatchHunkHeader {
  oldStart: number
}

const PATCH_HUNK_EXPECTATION = '"@@" or "@@ -old[,count] +new[,count] @@"'
const MAX_PATCH_DIAGNOSTIC_CHARS = 120
const MAX_PATCH_DIAGNOSTIC_WIDTH = 240

interface FileSnapshot {
  exists: boolean
  content?: Buffer
  mode?: number
}

interface PatchChange {
  operation: PatchOperation
  path: string
  content?: string
  additions: number
  deletions: number
  before: FileSnapshot
  after?: FileSnapshot
}

export const applyPatchTool: Tool = {
  name: 'apply_patch',
  description: 'Apply a precise, reviewable patch inside the workspace; updates and new files require approval. The patch uses a Codex envelope with Add File, Update File, and Delete File directives. Prefer a bare @@ update hunk header; numeric @@ -old[,count] +new[,count] @@ headers are also accepted.',
  readOnly: false,
  requiresApproval: true,
  parameters: {
    type: 'object',
    properties: {
      patch: {
        type: 'string',
        minLength: 1,
        maxLength: MAX_PATCH_LENGTH,
        description: 'Use a Codex envelope with *** Begin Patch and *** End Patch, and Add File, Update File, or Delete File directives. For Update File, prefer a bare @@ hunk header; numeric @@ -old[,count] +new[,count] @@ is also accepted. Every hunk body line must start with a space, +, or -, except the literal \\ No newline at end of file marker. Minimal example: *** Begin Patch\\n*** Update File: file.txt\\n@@\\n-old\\n+new\\n*** End Patch',
      },
      reason: {
        type: 'string',
        minLength: 1,
        maxLength: MAX_APPROVAL_REASON_CHARS,
        description: '用一句简洁、具体、面向用户的话说明为什么必须应用这个 patch，不得编造结果',
      },
    },
    required: ['patch', 'reason'],
    additionalProperties: false,
  },
  parse: parseApplyPatchInput,
  execute: executeApplyPatch,
}

export function parseApplyPatchInput(input: Record<string, unknown>): ApplyPatchInput {
  assertObject(input)
  assertKnownKeys(input, ['patch', 'reason'])
  const patch = requiredString(input, 'patch', MAX_PATCH_LENGTH)
  const operations = parsePatch(patch)
  if (operations.length > MAX_PATCH_FILES) {
    throw new ToolInputError(`patch may contain at most ${MAX_PATCH_FILES} files`)
  }
  return { patch, reason: validateApprovalReason(input.reason) }
}

async function executeApplyPatch(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
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
      if (exists) throw new Error(`File already exists: ${operation.path}`)
      const content = operation.lines.join('\n') + (operation.lines.length ? '\n' : '')
      changes.push({
        operation,
        path,
        content,
        additions: operation.lines.length,
        deletions: 0,
        before: { exists: false },
      })
      totalChangeBytes += Buffer.byteLength(content, 'utf8')
      assertPatchBudgets(totalChangeBytes, totalTargetBytes)
      continue
    }
    if (!exists) throw new Error(`File does not exist: ${operation.path}`)
    const info = existingInfo
    if (!info) throw new Error(`File does not exist: ${operation.path}`)
    if (!info.isFile()) throw new Error(`Not a regular file: ${operation.path}`)
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
      changes.push({ operation, path, additions: 0, deletions: 0, before })
      totalChangeBytes += info.size
      assertPatchBudgets(totalChangeBytes, totalTargetBytes)
      continue
    }
    const sourceBuffer = beforeContent
    if (sourceBuffer.length > MAX_PATCH_TARGET_BYTES) {
      throw new Error(`File is larger than ${MAX_PATCH_TARGET_BYTES} bytes: ${operation.path}`)
    }
    if (sourceBuffer.includes(0)) throw new Error(`File is not a text file: ${operation.path}`)
    let source: string
    try {
      source = new TextDecoder('utf-8', { fatal: true }).decode(sourceBuffer)
    } catch {
      throw new Error(`File is not valid UTF-8: ${operation.path}`)
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
        throw new Error(`Patch target changed during preflight: ${change.operation.path}; re-check the workspace`)
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
      // Give cancellation and concurrent workspace changes a chance to be
      // observed before the next file operation.
      await new Promise<void>((resolveNext) => setImmediate(resolveNext))
    }
  } catch (error) {
    const restored = await restoreChanges(appliedChanges, createdDirectories)
    if (!restored || operationInProgress) {
      const detail = error instanceof Error ? error.message : String(error)
      throw new Error(`${detail}; workspace restoration was incomplete, re-check the workspace`)
    }
    throw error
  }

  return {
    content: boundedJsonResult({
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
    }),
  }
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

function parsePatch(patch: string): PatchOperation[] {
  const lines = patch.replace(/\r\n/g, '\n').split('\n')
  if (lines.at(-1) === '') lines.pop()
  if (lines[0] !== '*** Begin Patch' || lines[lines.length - 1] !== '*** End Patch') {
    throw new ToolInputError('patch must start with *** Begin Patch and end with *** End Patch')
  }
  const operations: PatchOperation[] = []
  let index = 1
  while (index < lines.length - 1) {
    const header = lines[index]
    if (header.startsWith('*** Add File: ')) {
      const path = patchPath(header, '*** Add File: ')
      index += 1
      const content: string[] = []
      while (index < lines.length - 1 && !isFileHeader(lines[index])) {
        if (!lines[index].startsWith('+')) throw new ToolInputError(`Add File patch line must start with +: ${path}`)
        content.push(lines[index].slice(1))
        index += 1
      }
      operations.push({ kind: 'add', path, lines: content })
      continue
    }
    if (header.startsWith('*** Delete File: ')) {
      operations.push({ kind: 'delete', path: patchPath(header, '*** Delete File: ') })
      index += 1
      continue
    }
    if (header.startsWith('*** Update File: ')) {
      const path = patchPath(header, '*** Update File: ')
      index += 1
      const hunks: PatchHunk[] = []
      while (index < lines.length - 1 && !isFileHeader(lines[index])) {
        const hunkHeader = lines[index]
        const parsedHeader = parseHunkHeader(hunkHeader)
        if (!parsedHeader) {
          throw new ToolInputError(formatInvalidHunkHeader(path, index + 1, hunkHeader))
        }
        const { oldStart } = parsedHeader
        index += 1
        const hunkLines: string[] = []
        while (index < lines.length - 1 && !lines[index].startsWith('@@') && !isFileHeader(lines[index])) {
          const line = lines[index]
          if (isUnifiedFileHeader(line)) {
            throw new ToolInputError(`Unsupported unified diff file header for ${patchDiagnostic(path)} at patch line ${index + 1}; use the Codex envelope instead`)
          }
          if (line !== '\\ No newline at end of file' && !/^[ +\-]/.test(line)) {
            throw new ToolInputError(`Invalid patch line for ${path}`)
          }
          hunkLines.push(line)
          index += 1
        }
        if (!hunkLines.length) throw new ToolInputError(`Empty patch hunk for ${path}`)
        hunks.push({ oldStart, lines: hunkLines })
      }
      if (!hunks.length) throw new ToolInputError(`Update patch has no hunks: ${path}`)
      operations.push({ kind: 'update', path, hunks })
      continue
    }
    throw new ToolInputError(`Unknown patch directive: ${header}`)
  }
  if (!operations.length) throw new ToolInputError('patch contains no file operations')
  return operations
}

/**
 * Parse the complete logical line of an Update File hunk header. The bare
 * Codex header is intentionally unambiguous; arbitrary anchors are not
 * accepted because they would silently create a third, fuzzy patch dialect.
 */
export function parseHunkHeader(header: string): ParsedPatchHunkHeader | undefined {
  if (header === '@@') return { oldStart: 1 }
  const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$/.exec(header)
  if (!match) return undefined
  const oldStart = Number(match[1])
  const oldCount = Number(match[2] ?? 0)
  const newStart = Number(match[3])
  const newCount = Number(match[4] ?? 0)
  if (![oldStart, oldCount, newStart, newCount].every((value) => Number.isSafeInteger(value))) return undefined
  return { oldStart }
}

function formatInvalidHunkHeader(path: string, lineNumber: number, header: string): string {
  return `Invalid patch hunk header for ${patchDiagnostic(path)} at patch line ${lineNumber}: ${JSON.stringify(patchDiagnostic(header))}. Expected ${PATCH_HUNK_EXPECTATION}.`
}

function patchDiagnostic(value: string): string {
  const characters = [...value]
  let output = ''
  let consumed = 0
  for (const character of characters) {
    if (consumed >= MAX_PATCH_DIAGNOSTIC_CHARS) break
    const encoded = encodePatchDiagnosticCharacter(character)
    if ([...output].length + [...encoded].length > MAX_PATCH_DIAGNOSTIC_WIDTH) break
    output += encoded
    consumed += 1
  }
  if (consumed < characters.length) output += '…'
  return output
}

function encodePatchDiagnosticCharacter(character: string): string {
  const codePoint = character.codePointAt(0) ?? 0
  if (codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f)) {
    return `\\u{${codePoint.toString(16)}}`
  }
  if (character === '\\') return '\\\\'
  if (character === '"') return '\\"'
  return character
}

function applyHunks(source: string, hunks: PatchHunk[], displayPath: string): {
  content: string
  additions: number
  deletions: number
} {
  const hasFinalNewline = source.endsWith('\n')
  const sourceLines = source.replace(/\r\n/g, '\n').split('\n')
  if (hasFinalNewline) sourceLines.pop()
  let lines = sourceLines
  let offset = 0
  let additions = 0
  let deletions = 0

  for (const hunk of hunks) {
    const oldLines = hunk.lines
      .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('+'))
      .map((line) => line.slice(1))
    const newLines = hunk.lines
      .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('-'))
      .map((line) => line.slice(1))
    const preferred = Math.max(0, hunk.oldStart - 1 + offset)
    const position = findSequence(lines, oldLines, preferred)
    if (position < 0) throw new Error(`Patch context does not match: ${displayPath}`)
    lines = [...lines.slice(0, position), ...newLines, ...lines.slice(position + oldLines.length)]
    offset += newLines.length - oldLines.length
    additions += hunk.lines.filter((line) => line.startsWith('+')).length
    deletions += hunk.lines.filter((line) => line.startsWith('-')).length
  }

  return {
    content: lines.join('\n') + (hasFinalNewline ? '\n' : ''),
    additions,
    deletions,
  }
}

function findSequence(lines: string[], wanted: string[], preferred: number): number {
  if (!wanted.length) return Math.min(preferred, lines.length)
  const candidates = [preferred]
  for (let index = 0; index <= lines.length - wanted.length; index++) {
    if (index !== preferred) candidates.push(index)
  }
  return candidates.find((index) => wanted.every((line, offset) => lines[index + offset] === line)) ?? -1
}

function isFileHeader(line: string): boolean {
  return line.startsWith('*** Add File: ') || line.startsWith('*** Delete File: ') || line.startsWith('*** Update File: ')
}

function isUnifiedFileHeader(line: string): boolean {
  return /^(?:---|\+\+\+)(?:\s|$)/.test(line)
}

function patchPath(line: string, prefix: string): string {
  const path = line.slice(prefix.length).trim()
  if (!path) throw new ToolInputError('patch file path is required')
  return path
}
