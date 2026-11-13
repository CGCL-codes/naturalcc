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

// [maint-note:933418d52361]
// 维护导航：parseApplyPatchInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：ApplyPatchInput。
// 分支条件：operations.length > MAX_PATCH_FILES。
// 返回路径：{ patch, reason: validateApprovalReason(input.reason) }。
// 显式抛出：inputError(`patch may contain at most ${MAX_PATCH_FILES} files`, 'out_of_range', 'patch', `at most ${MAX_PATCH_FILES} files`)。
// 直接调用：assertObject。
// 直接调用：assertKnownKeys。
// 直接调用：requiredString。
// 直接调用：parsePatch。
// 直接调用：inputError。
// 直接调用：validateApprovalReason。
// [/maint-note:933418d52361]
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

// [maint-note:252397dae60b]
// 维护导航：parsePatch；下列为本函数源码中的接口与路径索引。
// 输入 patch：string。
// 返回契约：PatchOperation[]。
// 分支条件：lines.at(-1) === ''。
// 分支条件：lines[0] !== '*** Begin Patch' || lines[lines.length - 1] !== '*** End Patch'。
// 分支条件：header.startsWith('*** Add File: ')。
// 分支条件：!lines[index].startsWith('+')。
// 分支条件：header.startsWith('*** Delete File: ')。
// 分支条件：header.startsWith('*** Update File: ')。
// 分支条件：!parsedHeader。
// 分支条件：isUnifiedFileHeader(line)。
// 分支条件：line !== '\\ No newline at end of file' && !/^[ +\-]/.test(line)。
// 分支条件：!hunkLines.length。
// 分支条件：!hunks.length。
// 分支条件：!operations.length。
// 遍历边界：index < lines.length - 1。
// 遍历边界：index < lines.length - 1 && !isFileHeader(lines[index])。
// 遍历边界：index < lines.length - 1 && !lines[index].startsWith('@@') && !isFileHeader(lines[index])。
// 返回路径：operations。
// 显式抛出：inputError('patch must start with *** Begin Patch and end with *** End Patch', 'patch_format_error', 'patch')。
// 显式抛出：inputError(`Add File patch line must start with +: ${path}`, 'patch_format_error', 'patch', 'Add File body lines start with +')。
// 显式抛出：inputError(formatInvalidHunkHeader(path, index + 1, hunkHeader), 'patch_hunk_error', 'patch', PATCH_HUNK_EXPECTATION, `At patch line ${index + 1}: use a bare @@ or a numeric hunk header.`)。
// 显式抛出：inputError(`Unsupported unified diff file header for ${patchDiagnostic(path)} at patch line ${index + 1}; use the Codex envelope instead`, 'patch_format_error', 'patch', 'Codex file directives', `At patch line ${index + 1}: use *** Update F。
// 显式抛出：inputError(`Invalid patch line for ${path}`, 'patch_hunk_error', 'patch', 'body lines prefixed by space, + or -')。
// 显式抛出：inputError(`Empty patch hunk for ${path}`, 'patch_hunk_error', 'patch', 'non-empty hunk body')。
// 显式抛出：inputError(`Update patch has no hunks: ${path}`, 'patch_hunk_error', 'patch', 'at least one hunk')。
// 显式抛出：inputError(`Unknown patch directive: ${header}`, 'patch_format_error', 'patch', 'Add File, Update File or Delete File')。
// 显式抛出：inputError('patch contains no file operations', 'patch_format_error', 'patch', 'at least one file operation')。
// 写入目标：index。
// 直接调用：patch.replace(/\r\n/g, '\n').split。
// 直接调用：patch.replace。
// 直接调用：lines.at。
// 直接调用：lines.pop。
// 直接调用：inputError。
// 直接调用：header.startsWith。
// 直接调用：patchPath。
// 直接调用：isFileHeader。
// 直接调用：lines[index].startsWith。
// 直接调用：content.push。
// 直接调用：lines[index].slice。
// 直接调用：operations.push。
// 直接调用：parseHunkHeader。
// 直接调用：formatInvalidHunkHeader。
// 直接调用：isUnifiedFileHeader。
// 直接调用：patchDiagnostic。
// 直接调用：/^[ +\-]/.test。
// 直接调用：hunkLines.push。
// 直接调用：hunks.push。
// [/maint-note:252397dae60b]
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
// [maint-note:ef2fa52c4b74]
// 维护导航：parseHunkHeader；下列为本函数源码中的接口与路径索引。
// 输入 header：string。
// 返回契约：ParsedPatchHunkHeader | undefined。
// 分支条件：header === '@@'。
// 分支条件：!match。
// 分支条件：![oldStart, oldCount, newStart, newCount].every((value) => Number.isSafeInteger(value))。
// 返回路径：{ oldStart: 1 }。
// 返回路径：undefined。
// 返回路径：{ oldStart }。
// 直接调用：/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$/.exec。
// 直接调用：Number。
// 直接调用：[oldStart, oldCount, newStart, newCount].every。
// [/maint-note:ef2fa52c4b74]
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

// [maint-note:ce4ceb40852d]
// 维护导航：formatInvalidHunkHeader；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 输入 lineNumber：number。
// 输入 header：string。
// 返回契约：string。
// 返回路径：`Invalid patch hunk header for ${patchDiagnostic(path)} at patch line ${lineNumber}: ${JSON.stringify(patchDiagnostic(header))}. Expected ${PATCH_HUNK_EXPECTATION}.`。
// 直接调用：patchDiagnostic。
// 直接调用：JSON.stringify。
// [/maint-note:ce4ceb40852d]
function formatInvalidHunkHeader(path: string, lineNumber: number, header: string): string {
  return `Invalid patch hunk header for ${patchDiagnostic(path)} at patch line ${lineNumber}: ${JSON.stringify(patchDiagnostic(header))}. Expected ${PATCH_HUNK_EXPECTATION}.`
}

// [maint-note:2ddf902f9e5c]
// 维护导航：patchDiagnostic；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 分支条件：consumed >= MAX_PATCH_DIAGNOSTIC_CHARS。
// 分支条件：[...output].length + [...encoded].length > MAX_PATCH_DIAGNOSTIC_WIDTH。
// 分支条件：consumed < characters.length。
// 遍历边界：const character ← characters。
// 返回路径：output。
// 写入目标：output。
// 写入目标：consumed。
// 直接调用：encodePatchDiagnosticCharacter。
// [/maint-note:2ddf902f9e5c]
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

// [maint-note:5d178e8084f5]
// 维护导航：encodePatchDiagnosticCharacter；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：string。
// 分支条件：codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f)。
// 分支条件：character === '\\'。
// 分支条件：character === '"'。
// 返回路径：`\\u{${codePoint.toString(16)}}`。
// 返回路径：'\\\\'。
// 返回路径：'\\"'。
// 返回路径：character。
// 直接调用：character.codePointAt。
// 直接调用：codePoint.toString。
// [/maint-note:5d178e8084f5]
function encodePatchDiagnosticCharacter(character: string): string {
  const codePoint = character.codePointAt(0) ?? 0
  if (codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f)) {
    return `\\u{${codePoint.toString(16)}}`
  }
  if (character === '\\') return '\\\\'
  if (character === '"') return '\\"'
  return character
}

// [maint-note:9e686a6b878c]
// 维护导航：applyHunks；下列为本函数源码中的接口与路径索引。
// 输入 source：string。
// 输入 hunks：PatchHunk[]。
// 输入 displayPath：string。
// 返回契约：{ content: string additions: number deletions: number records: ApplyPatchDisplayRecord[] }。
// 分支条件：hasFinalNewline。
// 分支条件：position < 0。
// 分支条件：line === '\\ No newline at end of file'。
// 分支条件：line.startsWith('+')。
// 分支条件：line.startsWith('-')。
// 分支条件：tracked === undefined。
// 遍历边界：const hunk ← hunks。
// 遍历边界：const line ← hunk.lines。
// 返回路径：{ content: lines.map((line) => line.text).join('\n') + (hasFinalNewline ? '\n' : ''), additions, deletions, records, }。
// 显式抛出：operationError(`Patch context does not match: ${displayPath}`, 'patch_context_mismatch', { field: 'patch' })。
// 写入目标：matchedCursor。
// 写入目标：lines。
// 写入目标：offset。
// 写入目标：additions。
// 写入目标：deletions。
// 直接调用：source.endsWith。
// 直接调用：source.replace(/\r\n/g, '\n').split。
// 直接调用：source.replace。
// 直接调用：sourceLines.pop。
// 直接调用：sourceLines.map。
// 直接调用：hunk.lines .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('+')) .map。
// 直接调用：hunk.lines .filter。
// 直接调用：hunk.lines .filter((line) => line !== '\\ No newline at end of file' && !line.startsWith('-')) .map。
// 直接调用：Math.max。
// 直接调用：findSequence。
// 直接调用：lines.map。
// 直接调用：operationError。
// 直接调用：line.slice。
// 直接调用：line.startsWith。
// 直接调用：replacements.push。
// 直接调用：pendingRecords.push。
// 直接调用：lines.slice。
// 直接调用：hunk.lines.filter。
// 直接调用：lines.forEach。
// 直接调用：pendingRecords.map。
// 直接调用：lines.map((line) => line.text).join。
// [/maint-note:9e686a6b878c]
export function applyHunks(source: string, hunks: PatchHunk[], displayPath: string): {
  content: string
  additions: number
  deletions: number
  records: ApplyPatchDisplayRecord[]
} {
  // [maint-note:c6058f82c484]
  // 数据契约导航：TrackedLine；字段含义以消费端校验为准。
  // 成员 text：string。
  // 成员 originalLine（可省略）：number。
  // [/maint-note:c6058f82c484]
  interface TrackedLine {
    text: string
    /** Stable identity from the file before any hunk was applied. */
    originalLine?: number
  }
  // [maint-note:bfc53164a8ae]
  // 数据契约导航：PendingDisplayRecord；字段含义以消费端校验为准。
  // 成员 kind：ApplyPatchDisplayRecord['kind']。
  // 成员 text：string。
  // 成员 oldLine（可省略）：number。
  // 成员 finalLine（可省略）：TrackedLine。
  // [/maint-note:bfc53164a8ae]
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

// [maint-note:f71a60ca9fda]
// 维护导航：findSequence；下列为本函数源码中的接口与路径索引。
// 输入 lines：string[]。
// 输入 wanted：string[]。
// 输入 preferred：number。
// 返回契约：number。
// 分支条件：!wanted.length。
// 分支条件：index !== preferred。
// 遍历边界：let index = 0; index <= lines.length - wanted.length; index++。
// 返回路径：Math.min(preferred, lines.length)。
// 返回路径：candidates.find((index) => wanted.every((line, offset) => lines[index + offset] === line)) ?? -1。
// 直接调用：Math.min。
// 直接调用：candidates.push。
// 直接调用：candidates.find。
// [/maint-note:f71a60ca9fda]
function findSequence(lines: string[], wanted: string[], preferred: number): number {
  if (!wanted.length) return Math.min(preferred, lines.length)
  const candidates = [preferred]
  for (let index = 0; index <= lines.length - wanted.length; index++) {
    if (index !== preferred) candidates.push(index)
  }
  return candidates.find((index) => wanted.every((line, offset) => lines[index + offset] === line)) ?? -1
}

// [maint-note:c2e2adaf4900]
// 维护导航：isFileHeader；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：boolean。
// 返回路径：line.startsWith('*** Add File: ') || line.startsWith('*** Delete File: ') || line.startsWith('*** Update File: ')。
// 直接调用：line.startsWith。
// [/maint-note:c2e2adaf4900]
function isFileHeader(line: string): boolean {
  return line.startsWith('*** Add File: ') || line.startsWith('*** Delete File: ') || line.startsWith('*** Update File: ')
}

// [maint-note:924c748ccd16]
// 维护导航：isUnifiedFileHeader；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：boolean。
// 返回路径：/^(?:---|\+\+\+)(?:\s|$)/.test(line)。
// 直接调用：/^(?:---|\+\+\+)(?:\s|$)/.test。
// [/maint-note:924c748ccd16]
function isUnifiedFileHeader(line: string): boolean {
  return /^(?:---|\+\+\+)(?:\s|$)/.test(line)
}

// [maint-note:3bf78e345a82]
// 维护导航：patchPath；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 输入 prefix：string。
// 返回契约：string。
// 分支条件：!path。
// 返回路径：path。
// 显式抛出：inputError('patch file path is required', 'patch_format_error', 'patch', 'non-empty workspace-relative file path')。
// 直接调用：line.slice(prefix.length).trim。
// 直接调用：line.slice。
// 直接调用：inputError。
// [/maint-note:3bf78e345a82]
function patchPath(line: string, prefix: string): string {
  const path = line.slice(prefix.length).trim()
  if (!path) throw inputError('patch file path is required', 'patch_format_error', 'patch', 'non-empty workspace-relative file path')
  return path
}
