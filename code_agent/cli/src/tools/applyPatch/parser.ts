import { inputError, operationError } from '../diagnostics.js'
import { validateApprovalReason } from '../approval.js'
import {
  MAX_PATCH_FILES,
  MAX_PATCH_LENGTH,
  assertKnownKeys,
  assertObject,
  requiredString,
} from '../toolUtils.js'
import type {
  ApplyPatchDisplayRecord,
  ApplyPatchInput,
  ParsedPatchHunkHeader,
  PatchHunk,
  PatchOperation,
} from './types.js'

const PATCH_HUNK_EXPECTATION = '"@@" or "@@ -old[,count] +new[,count] @@"'
const MAX_PATCH_DIAGNOSTIC_CHARS = 120
const MAX_PATCH_DIAGNOSTIC_WIDTH = 240

export function parseApplyPatchInput(input: Record<string, unknown>): ApplyPatchInput {
  assertObject(input)
  assertKnownKeys(input, ['patch', 'reason'])
  const patch = requiredString(input, 'patch', MAX_PATCH_LENGTH)
  const operations = parsePatch(patch)
  if (operations.length > MAX_PATCH_FILES) {
    throw inputError(`patch may contain at most ${MAX_PATCH_FILES} files`, 'out_of_range', 'patch', `at most ${MAX_PATCH_FILES} files`)
  }
  return { patch, reason: validateApprovalReason(input.reason) }
}

export function parsePatch(patch: string): PatchOperation[] {
  const lines = patch.replace(/\r\n/g, '\n').split('\n')
  if (lines.at(-1) === '') lines.pop()
  if (lines[0] !== '*** Begin Patch' || lines[lines.length - 1] !== '*** End Patch') {
    throw inputError('patch must start with *** Begin Patch and end with *** End Patch', 'patch_format_error', 'patch')
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
        if (!lines[index].startsWith('+')) throw inputError(`Add File patch line must start with +: ${path}`, 'patch_format_error', 'patch', 'Add File body lines start with +')
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
          throw inputError(formatInvalidHunkHeader(path, index + 1, hunkHeader), 'patch_hunk_error', 'patch', PATCH_HUNK_EXPECTATION, `At patch line ${index + 1}: use a bare @@ or a numeric hunk header.`)
        }
        const { oldStart } = parsedHeader
        index += 1
        const hunkLines: string[] = []
        while (index < lines.length - 1 && !lines[index].startsWith('@@') && !isFileHeader(lines[index])) {
          const line = lines[index]
          if (isUnifiedFileHeader(line)) {
            throw inputError(`Unsupported unified diff file header for ${patchDiagnostic(path)} at patch line ${index + 1}; use the Codex envelope instead`, 'patch_format_error', 'patch', 'Codex file directives', `At patch line ${index + 1}: use *** Update File instead of unified file headers.`)
          }
          if (line !== '\\ No newline at end of file' && !/^[ +\-]/.test(line)) {
            throw inputError(`Invalid patch line for ${path}`, 'patch_hunk_error', 'patch', 'body lines prefixed by space, + or -')
          }
          hunkLines.push(line)
          index += 1
        }
        if (!hunkLines.length) throw inputError(`Empty patch hunk for ${path}`, 'patch_hunk_error', 'patch', 'non-empty hunk body')
        hunks.push({ oldStart, lines: hunkLines })
      }
      if (!hunks.length) throw inputError(`Update patch has no hunks: ${path}`, 'patch_hunk_error', 'patch', 'at least one hunk')
      operations.push({ kind: 'update', path, hunks })
      continue
    }
    throw inputError(`Unknown patch directive: ${header}`, 'patch_format_error', 'patch', 'Add File, Update File or Delete File')
  }
  if (!operations.length) throw inputError('patch contains no file operations', 'patch_format_error', 'patch', 'at least one file operation')
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

export function applyHunks(source: string, hunks: PatchHunk[], displayPath: string): {
  content: string
  additions: number
  deletions: number
  records: ApplyPatchDisplayRecord[]
} {
  interface TrackedLine {
    text: string
    /** Stable identity from the file before any hunk was applied. */
    originalLine?: number
  }
  interface PendingDisplayRecord {
    kind: ApplyPatchDisplayRecord['kind']
    text: string
    oldLine?: number
    /** Object identity is resolved only after every hunk has been applied. */
    finalLine?: TrackedLine
  }

  const hasFinalNewline = source.endsWith('\n')
  const sourceLines = source.replace(/\r\n/g, '\n').split('\n')
  if (hasFinalNewline) sourceLines.pop()
  let lines: TrackedLine[] = sourceLines.map((text, index) => ({ text, originalLine: index + 1 }))
  let offset = 0
  let additions = 0
  let deletions = 0
  const pendingRecords: PendingDisplayRecord[] = []

  for (const hunk of hunks) {
    const oldLines = hunk.lines
      .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('+'))
      .map((line) => line.slice(1))
    const newLines = hunk.lines
      .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('-'))
      .map((line) => line.slice(1))
    const preferred = Math.max(0, hunk.oldStart - 1 + offset)
    const position = findSequence(lines.map((line) => line.text), oldLines, preferred)
    if (position < 0) throw operationError(`Patch context does not match: ${displayPath}`, 'patch_context_mismatch', { field: 'patch' })
    const replacements: TrackedLine[] = []
    let matchedCursor = position
    for (const line of hunk.lines) {
      if (line === '\\ No newline at end of file') continue
      const text = line.slice(1)
      if (line.startsWith('+')) {
        const tracked = { text }
        replacements.push(tracked)
        pendingRecords.push({ kind: 'added', text, finalLine: tracked })
      } else if (line.startsWith('-')) {
        const tracked = lines[matchedCursor]
        if (tracked === undefined) throw operationError(`Patch context does not match: ${displayPath}`, 'patch_context_mismatch', { field: 'patch' })
        pendingRecords.push({ kind: 'removed', text, oldLine: tracked.originalLine })
        matchedCursor += 1
      } else {
        const tracked = lines[matchedCursor]
        if (tracked === undefined) throw operationError(`Patch context does not match: ${displayPath}`, 'patch_context_mismatch', { field: 'patch' })
        replacements.push(tracked)
        pendingRecords.push({ kind: 'context', text, oldLine: tracked.originalLine, finalLine: tracked })
        matchedCursor += 1
      }
    }
    lines = [...lines.slice(0, position), ...replacements, ...lines.slice(position + oldLines.length)]
    offset += newLines.length - oldLines.length
    additions += hunk.lines.filter((line) => line.startsWith('+')).length
    deletions += hunk.lines.filter((line) => line.startsWith('-')).length
  }

  // New-file positions are deliberately resolved after the last hunk. This
  // keeps an earlier record correct when a later (reverse-order) hunk inserts
  // or removes lines in front of it; an accumulated scalar delta cannot do
  // that because the shift is position-dependent.
  const finalPositions = new Map<TrackedLine, number>()
  lines.forEach((line, index) => finalPositions.set(line, index + 1))
  const records: ApplyPatchDisplayRecord[] = pendingRecords.map((record) => ({
    kind: record.kind,
    text: record.text,
    ...(record.oldLine === undefined ? {} : { oldLine: record.oldLine }),
    ...(record.finalLine === undefined ? {} : { newLine: finalPositions.get(record.finalLine)! }),
  }))

  return {
    content: lines.map((line) => line.text).join('\n') + (hasFinalNewline ? '\n' : ''),
    additions,
    deletions,
    records,
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
  if (!path) throw inputError('patch file path is required', 'patch_format_error', 'patch', 'non-empty workspace-relative file path')
  return path
}
