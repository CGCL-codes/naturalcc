import { createHash } from 'node:crypto'
import { posix } from 'node:path'

import { resolveWorkspacePath } from '../workspace.js'
import { assertOutputBudget, throwIfAborted } from '../toolUtils.js'
import type { ToolExecutionContext, ToolExecutionResult } from '../types.js'
import {
  GitChangesError,
  type GitChangesBackend,
  type GitChangesErrorCode,
  type GitChangesObservation,
  type GitFileRecord,
  type GitPatchHunk,
  type GitPatchLayer,
  type GitPatchLine,
} from '../backends/git/types.js'
import { parseGitChangesInput, type GitChangesInput } from './schema.js'

interface CursorPayload {
  v: 1
  w: string
  q: string
  r: string
  o: number
}

interface FlatPatchLine {
  patchIndex: number
  hunkIndex: number
  lineIndex: number
  line: GitPatchLine
}

const RESULT_VERSION = 1
const DEFAULT_TIMEOUT_MS = 5_000
const MAX_TIMEOUT_MS = 5_000

export async function executeGitChanges(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  backend: GitChangesBackend,
): Promise<ToolExecutionResult> {
  const options = parseGitChangesInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const deadline = Date.now() + Math.min(
    MAX_TIMEOUT_MS,
    Math.max(1, Math.floor(context.timeoutMs ?? DEFAULT_TIMEOUT_MS)),
  )
  const remainingCallMs = (): number => {
    throwIfAborted(context.signal)
    const remaining = deadline - Date.now()
    if (remaining <= 0) {
      throw new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)
    }
    return remaining
  }
  const assertCallActive = (): void => { remainingCallMs() }
  assertCallActive()

  const query = queryDigest(options)
  let cursor: CursorPayload | undefined
  if (options.cursor !== undefined) {
    try {
      cursor = decodeCursor(options.cursor, backend)
    } catch (error) {
      if (error instanceof GitChangesError) {
        return errorResult(error.code, error.reason, error.recoverable, budget)
      }
      throw error
    }
    if (cursor.q !== query) {
      return errorResult('git_changes_invalid_cursor', 'cursor_query_mismatch', false, budget)
    }
  }

  // Authorize the lexical parent rather than following a final symlink: Git
  // identities are repository paths, while resolveWorkspacePath protects the
  // containing workspace boundary.
  try {
    await resolveWorkspacePath(context.projectDir, options.path ? posix.dirname(options.path) : '.')
    assertCallActive()
    const backendTimeoutMs = remainingCallMs()
    const observation = await backend.observe({
      projectDir: context.projectDir,
      mode: options.mode,
      scope: options.scope,
      ...(options.path === undefined ? {} : { path: options.path }),
      signal: context.signal,
      timeoutMs: backendTimeoutMs,
      outputReserveBytes: Math.min(32 * 1024 * 1024, budget * 4),
    })
    assertCallActive()

    if (!observation.applicable) {
      if (cursor) return errorResult('git_changes_stale_cursor', 'repository_no_longer_applicable', true, budget)
      const result = serializeSuccess({
        version: RESULT_VERSION,
        applicable: false,
        mode: options.mode,
        scope: options.scope,
        reason: observation.reason ?? 'not_git_repository',
        complete: true,
      }, budget)
      assertCallActive()
      return result
    }

    const workspaceId = requiredObservationString(observation.workspaceId, 'workspace identity')
    const revision = requiredObservationString(observation.revision, 'revision')
    const coverage = observation.coverage
    if (!coverage) {
      throw new GitChangesError('git_changes_git_failed', 'Git observation omitted coverage', 'invalid_backend_observation')
    }
    if (cursor && cursor.w !== workspaceId) {
      return errorResult('git_changes_invalid_cursor', 'cursor_workspace_mismatch', false, budget)
    }
    if (cursor && cursor.r !== revision) {
      return errorResult('git_changes_stale_cursor', 'observation_changed', true, budget)
    }

    const records = [...(observation.records ?? [])].sort(compareRecords)
    if (options.mode === 'summary') {
      const result = serializeSuccess({
        version: RESULT_VERSION,
        applicable: true,
        mode: 'summary',
        scope: options.scope,
        revision,
        coverage,
        summary: summarize(records),
        complete: true,
      }, budget)
      assertCallActive()
      return result
    }

    if (options.mode === 'files') {
      const result = paginateFiles({ options, observation, records, workspaceId, revision, query, cursor, backend, budget })
      assertCallActive()
      return result
    }
    const result = paginatePatch({ options, observation, workspaceId, revision, query, cursor, backend, budget })
    assertCallActive()
    return result
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    // Preserve the Git observation's first terminal cause. A later caller
    // abort during bounded cleanup must not relabel timeout/output/protocol
    // failures as interruption.
    if (error instanceof GitChangesError) {
      return errorResult(error.code, error.reason, error.recoverable, budget)
    }
    if (context.signal?.aborted) {
      const interrupted = new Error('Tool execution interrupted')
      interrupted.name = 'AbortError'
      throw interrupted
    }
    throw error
  }
}

function paginateFiles(input: {
  options: GitChangesInput
  observation: GitChangesObservation
  records: GitFileRecord[]
  workspaceId: string
  revision: string
  query: string
  cursor?: CursorPayload
  backend: GitChangesBackend
  budget: number
}): ToolExecutionResult {
  const offset = input.cursor?.o ?? 0
  if (offset > input.records.length) {
    return errorResult('git_changes_stale_cursor', 'cursor_offset_out_of_range', true, input.budget)
  }
  const make = (count: number) => {
    const nextOffset = offset + count
    const complete = nextOffset >= input.records.length
    return {
      version: RESULT_VERSION,
      applicable: true,
      mode: 'files',
      scope: input.options.scope,
      ...(input.options.path === undefined ? {} : { path: input.options.path }),
      revision: input.revision,
      coverage: input.observation.coverage,
      offset,
      items: input.records.slice(offset, nextOffset),
      complete,
      ...(complete ? {} : {
        nextCursor: sealCursor(input.backend, {
          v: 1, w: input.workspaceId, q: input.query, r: input.revision, o: nextOffset,
        }),
      }),
    }
  }
  if (offset === input.records.length) return serializeSuccess(make(0), input.budget)
  return fitPage(input.records.length - offset, make, input.budget, 'git_changes_observation_limit')
}

function paginatePatch(input: {
  options: GitChangesInput
  observation: GitChangesObservation
  workspaceId: string
  revision: string
  query: string
  cursor?: CursorPayload
  backend: GitChangesBackend
  budget: number
}): ToolExecutionResult {
  const patches = input.observation.patches ?? []
  const flat = flattenPatch(patches)
  const offset = input.cursor?.o ?? 0
  if (offset > flat.length) {
    return errorResult('git_changes_stale_cursor', 'cursor_offset_out_of_range', true, input.budget)
  }
  const make = (count: number) => {
    const nextOffset = offset + count
    const complete = nextOffset >= flat.length
    return {
      version: RESULT_VERSION,
      applicable: true,
      mode: 'patch',
      scope: input.options.scope,
      path: input.options.path,
      revision: input.revision,
      coverage: input.observation.coverage,
      offset,
      patches: projectPatchPage(patches, flat.slice(offset, nextOffset), flat.length === 0),
      ...(input.observation.guidance ? { guidance: input.observation.guidance } : {}),
      complete,
      ...(complete ? {} : {
        nextCursor: sealCursor(input.backend, {
          v: 1, w: input.workspaceId, q: input.query, r: input.revision, o: nextOffset,
        }),
      }),
    }
  }
  if (flat.length === 0) return serializeSuccess(make(0), input.budget)
  return fitPage(flat.length - offset, make, input.budget, 'git_changes_line_too_large')
}

function fitPage(
  available: number,
  make: (count: number) => Record<string, unknown>,
  budget: number,
  tooLargeCode: GitChangesErrorCode,
): ToolExecutionResult {
  // A complete page has no cursor and can therefore be smaller than every
  // continuation page. Measure it as an independent candidate. The cheap
  // unit lower bound avoids constructing an unbounded full projection; the
  // serializer itself stops as soon as the character budget is exceeded.
  const complete = available <= budget
    ? stringifyWithinBudget(make(available), budget)
    : undefined
  if (complete !== undefined) return { content: complete }
  if (available === 1) {
    return errorResult(tooLargeCode, 'single_page_unit_exceeds_output_budget', false, budget)
  }

  // Continuation envelopes all contain a cursor, so their prefix sizes remain
  // monotonic. Grow from one unit without serializing the complete candidate.
  let acceptedCount = 0
  let accepted: string | undefined
  let probe = 1
  const continuationLimit = available - 1
  while (probe <= continuationLimit) {
    const serialized = stringifyWithinBudget(make(probe), budget)
    if (serialized === undefined) break
    accepted = serialized
    acceptedCount = probe
    probe = Math.min(continuationLimit, probe * 2)
    if (probe === acceptedCount) break
  }
  if (!accepted) return errorResult(tooLargeCode, 'single_page_unit_exceeds_output_budget', false, budget)

  let low = acceptedCount + 1
  let high = Math.min(continuationLimit, probe) - 1
  while (low <= high) {
    const count = Math.floor((low + high) / 2)
    const serialized = stringifyWithinBudget(make(count), budget)
    if (serialized !== undefined) {
      accepted = serialized
      low = count + 1
    } else {
      high = count - 1
    }
  }
  return { content: accepted }
}

type JsonWriteState = 'written' | 'omitted' | 'overflow'

/** JSON.stringify for the plain result envelopes, with an exact UTF-16
 * character budget and no over-budget intermediate string. */
function stringifyWithinBudget(value: unknown, budget: number): string | undefined {
  const chunks: string[] = []
  const active = new Set<object>()
  let length = 0

  const append = (chunk: string): boolean => {
    if (length + chunk.length > budget) return false
    chunks.push(chunk)
    length += chunk.length
    return true
  }

  const appendString = (value: string): boolean => {
    if (!append('"')) return false
    for (let index = 0; index < value.length; index += 1) {
      const code = value.charCodeAt(index)
      let encoded: string
      if (code === 0x22) encoded = '\\"'
      else if (code === 0x5c) encoded = '\\\\'
      else if (code === 0x08) encoded = '\\b'
      else if (code === 0x0c) encoded = '\\f'
      else if (code === 0x0a) encoded = '\\n'
      else if (code === 0x0d) encoded = '\\r'
      else if (code === 0x09) encoded = '\\t'
      else if (code < 0x20 || (code >= 0xd800 && code <= 0xdfff)) {
        if (
          code >= 0xd800 && code <= 0xdbff && index + 1 < value.length &&
          value.charCodeAt(index + 1) >= 0xdc00 && value.charCodeAt(index + 1) <= 0xdfff
        ) {
          encoded = value.slice(index, index + 2)
          index += 1
        } else {
          encoded = `\\u${code.toString(16).padStart(4, '0')}`
        }
      } else encoded = value[index]!
      if (!append(encoded)) return false
    }
    return append('"')
  }

  const normalize = (input: unknown, key: string): unknown => {
    if (input !== null && typeof input === 'object') {
      const toJSON = (input as { toJSON?: unknown }).toJSON
      if (typeof toJSON === 'function') input = toJSON.call(input, key)
    }
    if (input instanceof Number || input instanceof String || input instanceof Boolean) {
      return input.valueOf()
    }
    return input
  }

  const isOmitted = (input: unknown): boolean => (
    input === undefined || typeof input === 'function' || typeof input === 'symbol'
  )

  const writeValue = (input: unknown, key: string, arrayItem: boolean): JsonWriteState => {
    const normalized = normalize(input, key)
    if (isOmitted(normalized)) return arrayItem
      ? (append('null') ? 'written' : 'overflow')
      : 'omitted'
    if (normalized === null) return append('null') ? 'written' : 'overflow'
    if (typeof normalized === 'string') return appendString(normalized) ? 'written' : 'overflow'
    if (typeof normalized === 'boolean') return append(normalized ? 'true' : 'false') ? 'written' : 'overflow'
    if (typeof normalized === 'number') {
      const encoded = JSON.stringify(normalized)
      return append(encoded) ? 'written' : 'overflow'
    }
    if (typeof normalized === 'bigint') throw new TypeError('Do not know how to serialize a BigInt')
    if (typeof normalized !== 'object') return 'omitted'
    if (active.has(normalized)) throw new TypeError('Converting circular structure to JSON')
    active.add(normalized)
    try {
      if (Array.isArray(normalized)) {
        if (!append('[')) return 'overflow'
        for (let index = 0; index < normalized.length; index += 1) {
          if (index > 0 && !append(',')) return 'overflow'
          if (writeValue(normalized[index], String(index), true) === 'overflow') return 'overflow'
        }
        return append(']') ? 'written' : 'overflow'
      }

      if (!append('{')) return 'overflow'
      let wroteProperty = false
      for (const property of Object.keys(normalized)) {
        const propertyValue = normalize((normalized as Record<string, unknown>)[property], property)
        if (isOmitted(propertyValue)) continue
        if (wroteProperty && !append(',')) return 'overflow'
        if (!appendString(property) || !append(':')) return 'overflow'
        if (writeNormalizedValue(propertyValue, property) === 'overflow') return 'overflow'
        wroteProperty = true
      }
      return append('}') ? 'written' : 'overflow'
    } finally {
      active.delete(normalized)
    }
  }

  const writeNormalizedValue = (input: unknown, key: string): JsonWriteState => {
    if (input === null) return append('null') ? 'written' : 'overflow'
    if (typeof input === 'string') return appendString(input) ? 'written' : 'overflow'
    if (typeof input === 'boolean') return append(input ? 'true' : 'false') ? 'written' : 'overflow'
    if (typeof input === 'number') return append(JSON.stringify(input)) ? 'written' : 'overflow'
    if (typeof input === 'bigint') throw new TypeError('Do not know how to serialize a BigInt')
    return writeValue(input, key, false)
  }

  return writeValue(value, '', false) === 'written' ? chunks.join('') : undefined
}

function serializeSuccess(value: Record<string, unknown>, budget: number): ToolExecutionResult {
  const content = JSON.stringify(value)
  if (content.length <= budget) return { content }
  return errorResult('git_changes_observation_limit', 'result_envelope_exceeds_output_budget', false, budget)
}

function errorResult(
  code: GitChangesErrorCode,
  reason: string,
  recoverable: boolean,
  budget: number,
): ToolExecutionResult {
  const hint = code === 'git_changes_invalid_cursor'
    ? 'Start a fresh query without cursor; copy nextCursor exactly when continuing.'
    : code === 'git_changes_stale_cursor'
      ? 'Restart the same query without cursor; do not mix observations.'
      : code === 'git_changes_timeout' || code === 'git_changes_observation_limit'
        ? 'Narrow the query scope and inspect the current workspace before retrying.'
        : 'Inspect the repository configuration and current workspace; do not bypass its restrictions.'
  const detailed = JSON.stringify({ version: RESULT_VERSION, error: { code, reason, recoverable, hint } })
  const full = JSON.stringify({ version: RESULT_VERSION, error: { code, reason, recoverable } })
  const compact = JSON.stringify({ version: RESULT_VERSION, error: { code } })
  return {
    content: detailed.length <= budget ? detailed : full.length <= budget ? full : compact.slice(0, budget),
    isError: true,
    errorCode: code,
  }
}

function queryDigest(input: GitChangesInput): string {
  return createHash('sha256')
    .update(JSON.stringify({ mode: input.mode, scope: input.scope, path: input.path ?? null }))
    .digest('hex')
}

function sealCursor(backend: GitChangesBackend, payload: CursorPayload): string {
  return backend.sealCursor(JSON.stringify(payload))
}

function decodeCursor(token: string, backend: GitChangesBackend): CursorPayload {
  const opened = backend.openCursor(token)
  if (!opened) throw new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_authentication_failed')
  let value: unknown
  try {
    value = JSON.parse(opened)
  } catch {
    throw new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_payload_invalid')
  }
  if (!isRecord(value) || !hasExactKeys(value, ['v', 'w', 'q', 'r', 'o'])) {
    throw new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_payload_invalid')
  }
  if (
    value.v !== 1 || typeof value.w !== 'string' || typeof value.q !== 'string' ||
    typeof value.r !== 'string' || typeof value.o !== 'number' || !Number.isSafeInteger(value.o) || value.o < 0
  ) {
    throw new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_payload_invalid')
  }
  return value as unknown as CursorPayload
}

function summarize(records: readonly GitFileRecord[]) {
  const byLayer = { staged: 0, unstaged: 0, unmerged: 0 }
  const byAction = {
    added: 0,
    modified: 0,
    deleted: 0,
    renamed: 0,
    untracked: 0,
    type_changed: 0,
    unmerged: 0,
  }
  const unique = new Set<string>()
  let recordsWithBinary = 0
  let recordsWithModeChange = 0
  let recordsWithTypeChange = 0
  let recordsUnmerged = 0
  for (const record of records) {
    byLayer[record.layer] += 1
    byAction[record.action] += 1
    unique.add(record.path)
    if (record.binary === true) recordsWithBinary += 1
    if (record.modeChanged) recordsWithModeChange += 1
    if (record.action === 'type_changed') recordsWithTypeChange += 1
    if (record.action === 'unmerged') recordsUnmerged += 1
  }
  return {
    uniqueFiles: unique.size,
    layerRecords: records.length,
    byLayer,
    byAction,
    recordsWithBinary,
    recordsWithModeChange,
    recordsWithTypeChange,
    recordsUnmerged,
  }
}

function flattenPatch(patches: readonly GitPatchLayer[]): FlatPatchLine[] {
  const result: FlatPatchLine[] = []
  patches.forEach((patch, patchIndex) => {
    patch.hunks.forEach((hunk, hunkIndex) => {
      hunk.lines.forEach((line, lineIndex) => result.push({ patchIndex, hunkIndex, lineIndex, line }))
    })
  })
  return result
}

function projectPatchPage(
  patches: readonly GitPatchLayer[],
  selected: readonly FlatPatchLine[],
  includeMetadataOnly: boolean,
): GitPatchLayer[] {
  if (includeMetadataOnly) return patches.map((patch) => ({ ...patch, hunks: [] }))
  const projected: GitPatchLayer[] = []
  for (let patchIndex = 0; patchIndex < patches.length; patchIndex += 1) {
    const sourcePatch = patches[patchIndex]!
    const selectedForPatch = selected.filter((line) => line.patchIndex === patchIndex)
    if (selectedForPatch.length === 0) {
      if (sourcePatch.hunks.length === 0) projected.push({ ...sourcePatch, hunks: [] })
      continue
    }
    const hunks: GitPatchHunk[] = []
    for (let hunkIndex = 0; hunkIndex < sourcePatch.hunks.length; hunkIndex += 1) {
      const sourceHunk = sourcePatch.hunks[hunkIndex]!
      const selectedForHunk = selectedForPatch.filter((line) => line.hunkIndex === hunkIndex)
      if (selectedForHunk.length === 0) continue
      const first = selectedForHunk[0]!
      const last = selectedForHunk[selectedForHunk.length - 1]!
      hunks.push({
        oldStart: sourceHunk.oldStart,
        oldLines: sourceHunk.oldLines,
        newStart: sourceHunk.newStart,
        newLines: sourceHunk.newLines,
        ...(sourceHunk.section === undefined ? {} : { section: sourceHunk.section }),
        continuedFromPrevious: first.lineIndex > 0,
        continuesNext: last.lineIndex < sourceHunk.lines.length - 1,
        lines: selectedForHunk.map((entry) => entry.line),
      } as GitPatchHunk)
    }
    projected.push({ ...sourcePatch, hunks })
  }
  return projected
}

function compareRecords(left: GitFileRecord, right: GitFileRecord): number {
  return layerRank(left.layer) - layerRank(right.layer) ||
    left.path.localeCompare(right.path) ||
    (left.oldPath ?? '').localeCompare(right.oldPath ?? '') ||
    left.action.localeCompare(right.action)
}

function layerRank(layer: GitFileRecord['layer']): number {
  return layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2
}

function requiredObservationString(value: string | undefined, label: string): string {
  if (value) return value
  throw new GitChangesError('git_changes_git_failed', `Git observation omitted ${label}`, 'invalid_backend_observation')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  return actual.length === expected.length && actual.every((key, index) => key === expected[index])
}
