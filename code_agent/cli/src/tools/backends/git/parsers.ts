import { createHash } from 'node:crypto'

import { GitChangesError, type GitFileRecord, type GitPatchHunk, type GitPatchLayer } from './types.js'

const decoder = new TextDecoder('utf-8', { fatal: true })
export const MAX_GIT_DIFF_LINE_BYTES = 1024 * 1024

export interface GitIndexEntry {
  mode: string
  oid: string
  stage: 0 | 1 | 2 | 3
  path: string
}

export interface GitTreeEntry {
  mode: string
  type: string
  oid: string
  path: string
}

export interface ParsedStatus {
  headOid: string | 'unborn' | 'unknown'
  records: GitFileRecord[]
  facts: Array<{ paths: string[]; value: string }>
}

export interface NumstatEntry {
  path: string
  oldPath?: string
  added: number | 'binary'
  deleted: number | 'binary'
}

export type RetainParsedValue = (value: unknown) => void

export function decodePath(value: Buffer): string {
  try {
    return decoder.decode(value)
  } catch {
    throw new GitChangesError(
      'git_changes_invalid_path_encoding',
      'Git returned a path that is not valid UTF-8',
      'invalid_utf8_path',
    )
  }
}

export function parseIndexEntries(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitIndexEntry[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: GitIndexEntry[] = []
  for (const record of records) {
    const tab = record.indexOf(0x09)
    if (tab < 0) protocol('Invalid ls-files record')
    const header = record.subarray(0, tab).toString('ascii')
    const match = /^([0-7]{6}) ([0-9a-f]+) ([0-3])$/u.exec(header)
    if (!match) protocol('Invalid ls-files stage record')
    const value = {
      mode: match[1]!,
      oid: match[2]!,
      stage: Number(match[3]) as 0 | 1 | 2 | 3,
      path: decodePath(record.subarray(tab + 1)),
    }
    retain?.(value)
    result.push(value)
  }
  return result
}

export function parseTreeEntries(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitTreeEntry[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: GitTreeEntry[] = []
  for (const record of records) {
    const tab = record.indexOf(0x09)
    if (tab < 0) protocol('Invalid ls-tree record')
    const header = record.subarray(0, tab).toString('ascii')
    const match = /^([0-7]{6}) ([a-z]+) ([0-9a-f]+)$/u.exec(header)
    if (!match) protocol('Invalid ls-tree record')
    const value = { mode: match[1]!, type: match[2]!, oid: match[3]!, path: decodePath(record.subarray(tab + 1)) }
    retain?.(value)
    result.push(value)
  }
  return result
}

export function parseUntracked(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): string[] {
  const records = splitNul(output, maxParts(maxEntries, 1)).filter((record) => record.length > 0)
  if (records.length > maxEntries) observationLimit('file_count_exceeded')
  const result: string[] = []
  for (const record of records) {
    const value = decodePath(record)
    retain?.(value)
    result.push(value)
  }
  return result
}

export function parseCheckAttr(
  output: Buffer,
  maxPaths: number,
  retain?: RetainParsedValue,
): Array<{ path: string; attribute: string; value: string }> {
  const attributesPerPath = 3
  const maxRecords = maxParts(maxPaths, attributesPerPath)
  const values = splitNul(output, maxParts(maxRecords, 3, 1))
  if (values.at(-1)?.length === 0) values.pop()
  if (values.length % 3 !== 0) protocol('Invalid check-attr record')
  const result: Array<{ path: string; attribute: string; value: string }> = []
  const paths = new Set<string>()
  const attributesByPath = new Map<string, Set<string>>()
  const requestedAttributes = new Set(['filter', 'diff', 'working-tree-encoding'])
  for (let index = 0; index < values.length; index += 3) {
    if (result.length >= maxRecords) observationLimit('wire_record_count_exceeded')
    const path = decodePath(values[index]!)
    if (!paths.has(path) && paths.size >= maxPaths) observationLimit('file_count_exceeded')
    const attribute = decodePath(values[index + 1]!)
    if (!requestedAttributes.has(attribute)) protocol('Unexpected check-attr attribute')
    const seenAttributes = attributesByPath.get(path) ?? new Set<string>()
    if (seenAttributes.has(attribute)) protocol('Duplicate check-attr attribute')
    const value = {
      path,
      attribute,
      value: decodePath(values[index + 2]!),
    }
    retain?.(value)
    paths.add(path)
    seenAttributes.add(attribute)
    attributesByPath.set(path, seenAttributes)
    result.push(value)
  }
  if (
    paths.size !== maxPaths || result.length !== maxRecords ||
    [...attributesByPath.values()].some((attributes) => attributes.size !== attributesPerPath)
  ) protocol('Incomplete check-attr records')
  return result
}

export function parsePorcelainV2(
  output: Buffer,
  mapPath: (path: string) => string | undefined,
  maxRecords = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): ParsedStatus {
  const records: GitFileRecord[] = []
  const facts: ParsedStatus['facts'] = []
  const parts = splitNul(output, maxParts(maxRecords, 2, 2))
  let headOid: ParsedStatus['headOid'] = 'unknown'
  for (let index = 0; index < parts.length; index += 1) {
    let record = decodePath(parts[index]!)
    if (!record) continue
    while (record.startsWith('# ')) {
      const newline = record.indexOf('\n')
      const header = newline < 0 ? record : record.slice(0, newline)
      if (header.startsWith('# branch.oid ')) {
        const value = header.slice('# branch.oid '.length)
        headOid = value === '(initial)' ? 'unborn' : value
      }
      if (newline < 0) {
        record = ''
        break
      }
      record = record.slice(newline + 1)
    }
    if (!record) continue

    if (record.startsWith('? ')) {
      const path = mapPath(record.slice(2))
      if (path !== undefined) {
        if (records.length >= maxRecords) observationLimit('file_count_exceeded')
        const value: GitFileRecord = {
          layer: 'unstaged', action: 'untracked', path,
          contentChanged: true, binary: 'unknown', modeChanged: false, objectType: 'file',
        }
        retain?.(value)
        records.push(value)
        const fact = { paths: [path], value: record }
        retain?.(fact)
        facts.push(fact)
      }
      continue
    }
    if (record.startsWith('! ')) continue

    if (record.startsWith('1 ')) {
      const match = /^1 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 ordinary record')
      const path = mapPath(match[7]!)
      if (path === undefined) continue
      if (records.length + layerRecordCount(match[1]!) > maxRecords) observationLimit('file_count_exceeded')
      appendLayerRecords(records, match[1]!, path, undefined, match[2]!, match[3]!, match[4]!, match[5]!, match[6]!, false, retain)
      const fact = { paths: [path], value: record }
      retain?.(fact)
      facts.push(fact)
      continue
    }
    if (record.startsWith('2 ')) {
      const match = /^2 (\S{2}) \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([RC][0-9]+) ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 rename record')
      const originalPart = parts[++index]
      if (!originalPart) protocol('Missing porcelain-v2 rename origin')
      const newPath = mapPath(match[8]!)
      const oldRaw = decodePath(originalPart)
      const oldPath = mapPath(oldRaw)
      if (newPath === undefined && oldPath === undefined) continue
      if (match[7]!.startsWith('C')) {
        throw new GitChangesError('git_changes_ambiguous_change', 'Copy-like Git changes are not supported', 'copy_change')
      }
      if (records.length + layerRecordCount(match[1]!) > maxRecords) observationLimit('file_count_exceeded')
      appendLayerRecords(
        records,
        match[1]!,
        newPath ?? oldPath!,
        oldPath,
        match[2]!, match[3]!, match[4]!, match[5]!, match[6]!,
        newPath === undefined || oldPath === undefined,
        retain,
      )
      const fact = { paths: [newPath, oldPath].filter((path): path is string => path !== undefined), value: `${record}\0${oldRaw}` }
      retain?.(fact)
      facts.push(fact)
      continue
    }
    if (record.startsWith('u ')) {
      const match = /^u \S{2} \S+ ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) ([0-7]{6}) [0-9a-f]+ [0-9a-f]+ [0-9a-f]+ ([\s\S]+)$/u.exec(record)
      if (!match) protocol('Invalid porcelain-v2 unmerged record')
      const path = mapPath(match[5]!)
      if (path !== undefined) {
        if (records.length >= maxRecords) observationLimit('file_count_exceeded')
        const value: GitFileRecord = {
          layer: 'unmerged', action: 'unmerged', path,
          contentChanged: 'unknown', binary: 'unknown', modeChanged: true,
          oldMode: match[1], newMode: match[4], objectType: objectType(match[4]!),
        }
        retain?.(value)
        records.push(value)
        const fact = { paths: [path], value: record }
        retain?.(fact)
        facts.push(fact)
      }
      continue
    }
    protocol('Unsupported porcelain-v2 record')
  }
  return { headOid, records, facts }
}

function appendLayerRecords(
  target: GitFileRecord[],
  xy: string,
  path: string,
  oldPath: string | undefined,
  headMode: string,
  indexMode: string,
  workMode: string,
  headOid: string,
  indexOid: string,
  boundaryTruncated = false,
  retain?: RetainParsedValue,
): void {
  const [indexCode, workCode] = xy
  if (indexCode && indexCode !== '.') {
    const value = makeRecord('staged', indexCode, path, oldPath, headMode, indexMode, headOid !== indexOid, boundaryTruncated)
    retain?.(value)
    target.push(value)
  }
  if (workCode && workCode !== '.') {
    const value = makeRecord('unstaged', workCode, path, workCode === 'R' ? oldPath : undefined, indexMode, workMode, true, boundaryTruncated)
    retain?.(value)
    target.push(value)
  }
}

function makeRecord(
  layer: 'staged' | 'unstaged',
  code: string,
  path: string,
  oldPath: string | undefined,
  oldMode: string,
  newMode: string,
  contentChanged: boolean,
  boundaryTruncated: boolean,
): GitFileRecord {
  const action = code === 'A' ? 'added'
    : code === 'D' ? 'deleted'
      : code === 'R' ? 'renamed'
        : code === 'T' ? 'type_changed'
          : 'modified'
  return {
    layer,
    action,
    path,
    ...(action === 'renamed' && oldPath ? { oldPath } : {}),
    ...(boundaryTruncated ? { boundaryTruncated: true } : {}),
    contentChanged,
    binary: 'unknown',
    modeChanged: oldMode !== newMode,
    oldMode,
    newMode,
    objectType: objectType(action === 'deleted' ? oldMode : newMode),
  }
}

export function parseNumstat(
  output: Buffer,
  maxEntries = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): NumstatEntry[] {
  const parts = splitNul(output, maxParts(maxEntries, 3, 1))
  const entries: NumstatEntry[] = []
  for (let index = 0; index < parts.length; index += 1) {
    const record = parts[index]!
    if (record.length === 0) continue
    const firstTab = record.indexOf(0x09)
    const secondTab = firstTab < 0 ? -1 : record.indexOf(0x09, firstTab + 1)
    if (firstTab < 0 || secondTab < 0) protocol('Invalid numstat record')
    const addedText = record.subarray(0, firstTab).toString('ascii')
    const deletedText = record.subarray(firstTab + 1, secondTab).toString('ascii')
    const tail = record.subarray(secondTab + 1)
    const added = addedText === '-' ? 'binary' : parseCount(addedText)
    const deleted = deletedText === '-' ? 'binary' : parseCount(deletedText)
    if (tail.length > 0) {
      if (entries.length >= maxEntries) observationLimit('file_count_exceeded')
      const value: NumstatEntry = { path: decodePath(tail), added, deleted }
      retain?.(value)
      entries.push(value)
      continue
    }
    const oldPart = parts[++index]
    const newPart = parts[++index]
    if (!oldPart || !newPart) protocol('Invalid rename numstat record')
    if (entries.length >= maxEntries) observationLimit('file_count_exceeded')
    const value: NumstatEntry = { oldPath: decodePath(oldPart), path: decodePath(newPart), added, deleted }
    retain?.(value)
    entries.push(value)
  }
  return entries
}

export function parsePatch(
  output: Buffer,
  layer: 'staged' | 'unstaged',
  file: GitFileRecord,
  toWorkspacePath: (repoPath: string) => string | undefined,
  maxHunks = Number.MAX_SAFE_INTEGER,
  retain?: RetainParsedValue,
): GitPatchLayer {
  let text: string
  try {
    text = decoder.decode(output)
  } catch {
    throw new GitChangesError(
      'git_changes_invalid_content_encoding',
      'Git returned a text patch that is not valid UTF-8',
      'invalid_utf8_patch',
    )
  }
  if (!text) {
    if (file.contentChanged === false || file.binary === true) return { layer, file, hunks: [] }
    protocol('Git returned no patch for the selected change')
  }
  const lines = iteratePatchLines(text)
  const first = lines.next()
  if (first.done || !first.value.startsWith('diff --git ')) protocol('Patch must contain exactly one section')
  assertPatchLineBound(first.value)
  const expectedOld = file.oldPath ?? file.path
  const [headerOld, headerNew] = parseDiffGitHeader(
    first.value,
    (path) => toWorkspacePath(path),
    expectedOld,
    file.path,
  )
  if (toWorkspacePath(headerOld) !== expectedOld || toWorkspacePath(headerNew) !== file.path) protocol('Patch endpoints do not match selected Git metadata')

  const projectedFile: GitFileRecord = {
    ...file,
  }
  retain?.(projectedFile)
  const hunks: GitPatchHunk[] = []
  let current: GitPatchHunk | undefined
  let oldLine = 0
  let newLine = 0
  let consumedOld = 0
  let consumedNew = 0
  let previousKind: 'context' | 'add' | 'delete' | undefined
  let sawOldHeader = false
  let sawNewHeader = false
  let sawOldMode = false
  let sawNewMode = false
  let sawNewFileMode = false
  let sawDeletedFileMode = false
  let sawRenameFrom = false
  let sawRenameTo = false
  let sawIndex = false
  let indexMetadata: { oldOid: string; newOid: string; mode?: string } | undefined
  let similarityPercent: number | undefined
  let sawBinary = false

  const finishHunk = (): void => {
    if (!current) return
    if (consumedOld !== current.oldLines || consumedNew !== current.newLines) {
      protocol('Patch hunk line counts do not match its header')
    }
  }

  for (const line of lines) {
    assertPatchLineBound(line)
    if (line.startsWith('diff --git ')) protocol('Patch must contain exactly one section')
    if (line.startsWith('@@ ')) {
      finishHunk()
      const match = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: ?(.*))?$/u.exec(line)
      if (!match) protocol('Invalid patch hunk header')
      const nextHunk: GitPatchHunk = {
        oldStart: parsePatchNumber(match[1]!), oldLines: parsePatchNumber(match[2] ?? '1'),
        newStart: parsePatchNumber(match[3]!), newLines: parsePatchNumber(match[4] ?? '1'),
        ...(match[5] ? { section: match[5] } : {}), lines: [],
      }
      retain?.({ ...nextHunk, lines: undefined })
      current = nextHunk
      if (
        current.oldStart + Math.max(0, current.oldLines - 1) > Number.MAX_SAFE_INTEGER ||
        current.newStart + Math.max(0, current.newLines - 1) > Number.MAX_SAFE_INTEGER
      ) protocol('Patch hunk line range is unsafe')
      oldLine = current.oldStart
      newLine = current.newStart
      consumedOld = 0
      consumedNew = 0
      if (hunks.length >= maxHunks) observationLimit('hunk_count_exceeded')
      hunks.push(current)
      previousKind = undefined
      continue
    }
    if (!current) {
      if (line.startsWith('--- ')) {
        if (sawOldHeader) protocol('Duplicate old patch endpoint')
        validatePatchEndpoint(line.slice(4), 'old', file, toWorkspacePath)
        sawOldHeader = true
        continue
      }
      if (line.startsWith('+++ ')) {
        if (sawNewHeader) protocol('Duplicate new patch endpoint')
        validatePatchEndpoint(line.slice(4), 'new', file, toWorkspacePath)
        sawNewHeader = true
        continue
      }
      if (line.startsWith('rename from ')) {
        if (sawRenameFrom) protocol('Duplicate rename origin')
        validateRenameEndpoint(line.slice('rename from '.length), file.oldPath ?? file.path, toWorkspacePath)
        sawRenameFrom = true
        continue
      }
      if (line.startsWith('rename to ')) {
        if (sawRenameTo) protocol('Duplicate rename destination')
        validateRenameEndpoint(line.slice('rename to '.length), file.path, toWorkspacePath)
        sawRenameTo = true
        continue
      }
      if (line.startsWith('index ')) {
        const match = /^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: ([0-7]{6}))?$/u.exec(line)
        if (sawIndex || !match || match[1]!.length !== match[2]!.length) protocol('Invalid index metadata')
        sawIndex = true
        indexMetadata = {
          oldOid: match[1]!,
          newOid: match[2]!,
          ...(match[3] === undefined ? {} : { mode: match[3] }),
        }
      } else if (line.startsWith('old mode ')) {
        if (sawOldMode || line.slice('old mode '.length) !== file.oldMode) protocol('Invalid old mode metadata')
        sawOldMode = true
      } else if (line.startsWith('new mode ')) {
        if (sawNewMode || line.slice('new mode '.length) !== file.newMode) protocol('Invalid new mode metadata')
        sawNewMode = true
      } else if (line.startsWith('new file mode ')) {
        if (
          sawNewFileMode || file.action !== 'added' ||
          line.slice('new file mode '.length) !== file.newMode
        ) protocol('Invalid new-file mode metadata')
        sawNewFileMode = true
      } else if (line.startsWith('deleted file mode ')) {
        if (
          sawDeletedFileMode || file.action !== 'deleted' ||
          line.slice('deleted file mode '.length) !== file.oldMode
        ) protocol('Invalid deleted-file mode metadata')
        sawDeletedFileMode = true
      } else if (line.startsWith('similarity index ') || line.startsWith('dissimilarity index ')) {
        const match = /^(dis)?similarity index (100|[0-9]{1,2})%$/u.exec(line)
        if (!match || similarityPercent !== undefined) protocol('Invalid similarity metadata')
        const percentage = Number(match[2])
        similarityPercent = match[1] ? 100 - percentage : percentage
      }
      else if (line.startsWith('Binary files ') || line.startsWith('GIT binary patch')) {
        sawBinary = true
        projectedFile.binary = true
      }
      if (isAllowedPatchMetadata(line)) continue
      protocol('Unexpected patch metadata')
    }
    if (line === '\\ No newline at end of file') {
      if (!previousKind) protocol('Orphan no-newline marker')
      const value: GitPatchHunk['lines'][number] = {
        kind: 'no_newline',
        appliesTo: previousKind === 'delete' ? 'old' : 'new',
      }
      retain?.(value)
      current.lines.push(value)
      previousKind = undefined
      continue
    }
    const prefix = line[0]
    const body = line.slice(1)
    if (prefix === ' ') {
      const value: GitPatchHunk['lines'][number] = { kind: 'context', text: body, oldLine, newLine }
      retain?.(value)
      current.lines.push(value)
      oldLine += 1
      newLine += 1
      consumedOld += 1
      consumedNew += 1
      previousKind = 'context'
    } else if (prefix === '-') {
      const value: GitPatchHunk['lines'][number] = { kind: 'delete', text: body, oldLine, newLine: null }
      retain?.(value)
      current.lines.push(value)
      oldLine += 1
      consumedOld += 1
      previousKind = 'delete'
    } else if (prefix === '+') {
      const value: GitPatchHunk['lines'][number] = { kind: 'add', text: body, oldLine: null, newLine }
      retain?.(value)
      current.lines.push(value)
      newLine += 1
      consumedNew += 1
      previousKind = 'add'
    } else {
      protocol('Invalid patch hunk line')
    }
  }
  finishHunk()
  if (sawOldHeader !== sawNewHeader) protocol('Patch endpoint pair is incomplete')
  if (sawRenameFrom !== sawRenameTo) protocol('Rename endpoint pair is incomplete')
  if (sawOldMode !== sawNewMode) protocol('Mode metadata pair is incomplete')
  if (hunks.length > 0 && (!sawOldHeader || !sawNewHeader)) protocol('Text patch omitted old/new file headers')
  if (hunks.length === 0) {
    if (sawOldHeader || sawNewHeader) protocol('Patch endpoints were not followed by a text hunk')
    const unchangedIndex = indexMetadata === undefined || (
      hasFullObjectIds(indexMetadata) && !isZeroOid(indexMetadata.oldOid) &&
      indexMetadata.oldOid === indexMetadata.newOid &&
      (indexMetadata.mode === undefined || (
        !file.modeChanged && indexMetadata.mode === file.oldMode && indexMetadata.mode === file.newMode
      ))
    )
    const supportedMetadataOnly =
      sawBinary ||
      (
        file.action === 'modified' && file.modeChanged && sawOldMode && sawNewMode &&
        unchangedIndex
      ) ||
      (
        file.action === 'renamed' && sawRenameFrom && sawRenameTo &&
        similarityPercent === 100 && unchangedIndex
      ) ||
      (
        file.action === 'added' && sawNewFileMode && sawIndex &&
        indexMetadata !== undefined && isZeroOid(indexMetadata.oldOid) && isEmptyBlobOid(indexMetadata.newOid) &&
        (indexMetadata.mode === undefined || indexMetadata.mode === file.newMode)
      ) ||
      (
        file.action === 'deleted' && sawDeletedFileMode && sawIndex &&
        indexMetadata !== undefined && isEmptyBlobOid(indexMetadata.oldOid) && isZeroOid(indexMetadata.newOid) &&
        (indexMetadata.mode === undefined || indexMetadata.mode === file.oldMode)
      )
    if (!supportedMetadataOnly) protocol('Patch omitted content for a content-changing file')
  }
  projectedFile.contentChanged = sawBinary || hunks.length > 0
  return { layer, file: projectedFile, hunks }
}

function hasFullObjectIds(metadata: { oldOid: string; newOid: string }): boolean {
  return (metadata.oldOid.length === 40 || metadata.oldOid.length === 64) &&
    metadata.newOid.length === metadata.oldOid.length
}

function isZeroOid(value: string): boolean {
  return (value.length === 40 || value.length === 64) && /^0+$/u.test(value)
}

function isEmptyBlobOid(value: string): boolean {
  if (value.length !== 40 && value.length !== 64) return false
  const algorithm = value.length === 40 ? 'sha1' : 'sha256'
  return createHash(algorithm).update('blob 0\0').digest('hex') === value
}

function* iteratePatchLines(text: string): Generator<string> {
  let offset = 0
  while (offset < text.length) {
    const newline = text.indexOf('\n', offset)
    if (newline < 0) {
      yield text.slice(offset)
      return
    }
    yield text.slice(offset, newline)
    offset = newline + 1
  }
}

function assertPatchLineBound(line: string): void {
  if (Buffer.byteLength(line, 'utf8') > MAX_GIT_DIFF_LINE_BYTES) {
    throw new GitChangesError('git_changes_observation_limit', 'Git patch line exceeds limit', 'diff_line_too_large')
  }
}

function parseDiffGitHeader(
  line: string,
  toWorkspacePath: (path: string) => string | undefined,
  expectedOld: string,
  expectedNew: string,
): [string, string] {
  const body = line.slice('diff --git '.length)
  const candidates: Array<[string, string]> = []
  for (const index of unquotedSeparatorOffsets(body)) {
    const left = decodeHeaderPath(body.slice(0, index))
    const right = decodeHeaderPath(body.slice(index + 1))
    if (!left?.startsWith('a/') || !right?.startsWith('b/')) continue
    const oldPath = left.slice(2)
    const newPath = right.slice(2)
    if (toWorkspacePath(oldPath) === expectedOld && toWorkspacePath(newPath) === expectedNew) {
      if (!candidates.some(([old, next]) => old === oldPath && next === newPath)) {
        candidates.push([oldPath, newPath])
      }
    }
  }
  if (candidates.length !== 1) protocol('Invalid or ambiguous diff --git header')
  return candidates[0]!
}

function unquotedSeparatorOffsets(value: string): number[] {
  const result: number[] = []
  let quoted = false
  let escaped = false
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]!
    if (quoted && escaped) {
      escaped = false
      continue
    }
    if (quoted && character === '\\') {
      escaped = true
      continue
    }
    if (character === '"') {
      quoted = !quoted
      continue
    }
    if (!quoted && character === ' ' && index > 0 && index < value.length - 1) result.push(index)
  }
  if (quoted || escaped) protocol('Unterminated quoted Git path')
  return result
}

function validatePatchEndpoint(
  encoded: string,
  side: 'old' | 'new',
  file: GitFileRecord,
  toWorkspacePath: (repoPath: string) => string | undefined,
): void {
  // Git terminates an unquoted ---/+++ pathname with a tab. Pathnames that
  // contain a literal tab are C-quoted, so stripping this delimiter is
  // unambiguous.
  const token = encoded.endsWith('\t') ? encoded.slice(0, -1) : encoded
  if (token === '/dev/null') {
    if ((side === 'old' && file.action === 'added') || (side === 'new' && file.action === 'deleted')) return
    protocol('Unexpected /dev/null patch endpoint')
  }
  const decoded = decodeHeaderPath(token)
  const prefix = side === 'old' ? 'a/' : 'b/'
  if (!decoded?.startsWith(prefix)) protocol('Invalid old/new patch endpoint')
  const expected = side === 'old' ? (file.oldPath ?? file.path) : file.path
  if (toWorkspacePath(decoded.slice(2)) !== expected) protocol('Old/new patch endpoint does not match selected Git metadata')
}

function validateRenameEndpoint(
  encoded: string,
  expected: string,
  toWorkspacePath: (repoPath: string) => string | undefined,
): void {
  const decoded = decodeHeaderPath(encoded)
  if (!decoded || toWorkspacePath(decoded) !== expected) protocol('Rename endpoint does not match selected Git metadata')
}

function decodeHeaderPath(value: string): string | undefined {
  if (!value) return undefined
  if (!value.startsWith('"')) return value.includes('"') ? undefined : value
  const parsed = parseQuotedTokens(value)
  return parsed.length === 1 ? parsed[0] : undefined
}

function parseQuotedTokens(value: string): string[] {
  const result: string[] = []
  let offset = 0
  while (offset < value.length) {
    while (value[offset] === ' ') offset += 1
    if (offset >= value.length) break
    if (value[offset] !== '"') {
      const end = value.indexOf(' ', offset)
      result.push(value.slice(offset, end < 0 ? value.length : end))
      offset = end < 0 ? value.length : end + 1
      continue
    }
    const bytes: number[] = []
    offset += 1
    let closed = false
    while (offset < value.length) {
      const character = value[offset++]!
      if (character === '"') { closed = true; break }
      if (character !== '\\') {
        bytes.push(...Buffer.from(character, 'utf8'))
        continue
      }
      const escaped = value[offset++]
      if (escaped === undefined) protocol('Invalid quoted Git path')
      const simple: Record<string, number> = { a: 7, b: 8, t: 9, n: 10, v: 11, f: 12, r: 13, '\\': 92, '"': 34 }
      if (simple[escaped] !== undefined) {
        bytes.push(simple[escaped]!)
        continue
      }
      if (/[0-7]/u.test(escaped)) {
        const digits = escaped + value.slice(offset, offset + 2)
        if (!/^[0-7]{3}$/u.test(digits)) protocol('Invalid octal Git path escape')
        const byte = Number.parseInt(digits, 8)
        if (byte > 0xff) protocol('Git path octal escape exceeds one byte')
        bytes.push(byte)
        offset += 2
        continue
      }
      protocol('Invalid Git path escape')
    }
    if (!closed) protocol('Unterminated quoted Git path')
    result.push(decodePath(Buffer.from(bytes)))
  }
  return result
}

function isAllowedPatchMetadata(line: string): boolean {
  return line.startsWith('index ') || line.startsWith('old mode ') || line.startsWith('new mode ') ||
    line.startsWith('new file mode ') || line.startsWith('deleted file mode ') ||
    line.startsWith('similarity index ') || line.startsWith('dissimilarity index ') ||
    line.startsWith('rename from ') || line.startsWith('rename to ') ||
    line.startsWith('--- ') || line.startsWith('+++ ') ||
    line.startsWith('Binary files ') || line.startsWith('GIT binary patch') || line === ''
}

function splitNul(value: Buffer, maxRecords = Number.MAX_SAFE_INTEGER): Buffer[] {
  if (value.length > 0 && value[value.length - 1] !== 0) protocol('NUL-delimited Git output is incomplete')
  const result: Buffer[] = []
  let offset = 0
  for (;;) {
    const end = value.indexOf(0, offset)
    if (end < 0) {
      result.push(value.subarray(offset))
      if (result.length > maxRecords) observationLimit('wire_record_count_exceeded')
      return result
    }
    result.push(value.subarray(offset, end))
    if (result.length > maxRecords) observationLimit('wire_record_count_exceeded')
    offset = end + 1
    if (offset === value.length) {
      result.push(Buffer.alloc(0))
      return result
    }
  }
}

function parseCount(value: string): number {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(value)) protocol('Invalid numstat count')
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed)) protocol('Unsafe numstat count')
  return parsed
}

function parsePatchNumber(value: string): number {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(value)) protocol('Invalid patch line number')
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed)) protocol('Unsafe patch line number')
  return parsed
}

function objectType(mode: string): GitFileRecord['objectType'] {
  if (mode === '120000') return 'symlink'
  if (mode === '160000') return 'gitlink'
  if (mode.startsWith('100')) return 'file'
  return 'other'
}

function protocol(message: string): never {
  throw new GitChangesError('git_changes_patch_protocol_error', message, 'invalid_git_wire')
}

function observationLimit(reason: string): never {
  throw new GitChangesError('git_changes_observation_limit', 'Git observation exceeded a bounded limit', reason)
}

function maxParts(entries: number, multiplier: number, extra = 0): number {
  const value = entries * multiplier + extra
  return Number.isSafeInteger(value) ? value : Number.MAX_SAFE_INTEGER
}

function layerRecordCount(xy: string): number {
  return [...xy].filter((code) => code !== '.').length
}
