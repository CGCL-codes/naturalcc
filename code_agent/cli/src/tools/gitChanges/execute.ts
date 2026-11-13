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

// [maint-note:f4df510c9e1d]
// 数据契约导航：CursorPayload；字段含义以消费端校验为准。
// 成员 v：1。
// 成员 w：string。
// 成员 q：string。
// 成员 r：string。
// 成员 o：number。
// [/maint-note:f4df510c9e1d]
interface CursorPayload {
  v: 1
  w: string
  q: string
  r: string
  o: number
}

// [maint-note:3667933d5ae6]
// 数据契约导航：FlatPatchLine；字段含义以消费端校验为准。
// 成员 patchIndex：number。
// 成员 hunkIndex：number。
// 成员 lineIndex：number。
// 成员 line：GitPatchLine。
// [/maint-note:3667933d5ae6]
interface FlatPatchLine {
  patchIndex: number
  hunkIndex: number
  lineIndex: number
  line: GitPatchLine
}

const RESULT_VERSION = 1
const DEFAULT_TIMEOUT_MS = 5_000
const MAX_TIMEOUT_MS = 5_000

// [maint-note:8ec2faa514dc]
// 维护导航：executeGitChanges；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 backend：GitChangesBackend。
// 返回契约：Promise<ToolExecutionResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：options.cursor !== undefined。
// 分支条件：error instanceof GitChangesError。
// 分支条件：cursor.q !== query。
// 分支条件：!observation.applicable。
// 分支条件：cursor。
// 分支条件：!coverage。
// 分支条件：cursor && cursor.w !== workspaceId。
// 分支条件：cursor && cursor.r !== revision。
// 分支条件：options.mode === 'summary'。
// 分支条件：options.mode === 'files'。
// 分支条件：error instanceof Error && error.name === 'AbortError'。
// 分支条件：context.signal?.aborted。
// 返回路径：errorResult(error.code, error.reason, error.recoverable, budget)。
// 返回路径：errorResult('git_changes_invalid_cursor', 'cursor_query_mismatch', false, budget)。
// 返回路径：errorResult('git_changes_stale_cursor', 'repository_no_longer_applicable', true, budget)。
// 返回路径：result。
// 返回路径：errorResult('git_changes_invalid_cursor', 'cursor_workspace_mismatch', false, budget)。
// 返回路径：errorResult('git_changes_stale_cursor', 'observation_changed', true, budget)。
// 显式抛出：error。
// 显式抛出：new GitChangesError('git_changes_git_failed', 'Git observation omitted coverage', 'invalid_backend_observation')。
// 显式抛出：interrupted。
// 写入目标：cursor。
// 写入目标：interrupted.name。
// 直接调用：parseGitChangesInput。
// 直接调用：assertOutputBudget。
// 直接调用：Date.now。
// 直接调用：Math.min。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：assertCallActive。
// 直接调用：queryDigest。
// 直接调用：decodeCursor。
// 直接调用：errorResult。
// 直接调用：resolveWorkspacePath。
// 直接调用：posix.dirname。
// 直接调用：remainingCallMs。
// 直接调用：backend.observe。
// 直接调用：serializeSuccess。
// 直接调用：requiredObservationString。
// 直接调用：[...(observation.records ?? [])].sort。
// 直接调用：summarize。
// 直接调用：paginateFiles。
// 直接调用：paginatePatch。
// [/maint-note:8ec2faa514dc]
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
  // [maint-note:bda60d827e40]
  // 维护导航：remainingCallMs；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 分支条件：remaining <= 0。
  // 返回路径：remaining。
  // 显式抛出：new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)。
  // 直接调用：throwIfAborted。
  // 直接调用：Date.now。
  // [/maint-note:bda60d827e40]
  const remainingCallMs = (): number => {
    throwIfAborted(context.signal)
    const remaining = deadline - Date.now()
    if (remaining <= 0) {
      throw new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)
    }
    return remaining
  }
  // [maint-note:f02c944214f2]
  // 维护导航：assertCallActive；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 直接调用：remainingCallMs。
  // [/maint-note:f02c944214f2]
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

// [maint-note:237c4a81859c]
// 维护导航：paginateFiles；下列为本函数源码中的接口与路径索引。
// 输入 input：{ options: GitChangesInput observation: GitChangesObservation records: GitFileRecord[] workspaceId: string revision: string query: string cursor?: CursorPayload backend: GitChangesBackend budget: number }。
// 返回契约：ToolExecutionResult。
// 分支条件：offset > input.records.length。
// 分支条件：offset === input.records.length。
// 返回路径：errorResult('git_changes_stale_cursor', 'cursor_offset_out_of_range', true, input.budget)。
// 返回路径：serializeSuccess(make(0), input.budget)。
// 返回路径：fitPage(input.records.length - offset, make, input.budget, 'git_changes_observation_limit')。
// 直接调用：errorResult。
// 直接调用：serializeSuccess。
// 直接调用：make。
// 直接调用：fitPage。
// [/maint-note:237c4a81859c]
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
  // [maint-note:5913c3886d4a]
  // 维护导航：make；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 返回路径：{ version: RESULT_VERSION, applicable: true, mode: 'files', scope: input.options.scope, ...(input.options.path === undefined ? {} : { path: input.options.path }), revision: input.revision, coverage: input.observation.coverage, offset, items。
  // 直接调用：input.records.slice。
  // 直接调用：sealCursor。
  // [/maint-note:5913c3886d4a]
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

// [maint-note:934787a40df2]
// 维护导航：paginatePatch；下列为本函数源码中的接口与路径索引。
// 输入 input：{ options: GitChangesInput observation: GitChangesObservation workspaceId: string revision: string query: string cursor?: CursorPayload backend: GitChangesBackend budget: number }。
// 返回契约：ToolExecutionResult。
// 分支条件：offset > flat.length。
// 分支条件：flat.length === 0。
// 返回路径：errorResult('git_changes_stale_cursor', 'cursor_offset_out_of_range', true, input.budget)。
// 返回路径：serializeSuccess(make(0), input.budget)。
// 返回路径：fitPage(flat.length - offset, make, input.budget, 'git_changes_line_too_large')。
// 直接调用：flattenPatch。
// 直接调用：errorResult。
// 直接调用：serializeSuccess。
// 直接调用：make。
// 直接调用：fitPage。
// [/maint-note:934787a40df2]
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
  // [maint-note:1749978750e7]
  // 维护导航：make；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 返回路径：{ version: RESULT_VERSION, applicable: true, mode: 'patch', scope: input.options.scope, path: input.options.path, revision: input.revision, coverage: input.observation.coverage, offset, patches: projectPatchPage(patches, flat.slice(offset, 。
  // 直接调用：projectPatchPage。
  // 直接调用：flat.slice。
  // 直接调用：sealCursor。
  // [/maint-note:1749978750e7]
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

// [maint-note:e2e1ee3c06b4]
// 维护导航：fitPage；下列为本函数源码中的接口与路径索引。
// 输入 available：number。
// 输入 make：(count: number) => Record<string, unknown>。
// 输入 budget：number。
// 输入 tooLargeCode：GitChangesErrorCode。
// 返回契约：ToolExecutionResult。
// 分支条件：complete !== undefined。
// 分支条件：available === 1。
// 分支条件：serialized === undefined。
// 分支条件：probe === acceptedCount。
// 分支条件：!accepted。
// 分支条件：serialized !== undefined。
// 遍历边界：probe <= continuationLimit。
// 遍历边界：low <= high。
// 返回路径：{ content: complete }。
// 返回路径：errorResult(tooLargeCode, 'single_page_unit_exceeds_output_budget', false, budget)。
// 返回路径：{ content: accepted }。
// 写入目标：accepted。
// 写入目标：acceptedCount。
// 写入目标：probe。
// 写入目标：low。
// 写入目标：high。
// 直接调用：stringifyWithinBudget。
// 直接调用：make。
// 直接调用：errorResult。
// 直接调用：Math.min。
// 直接调用：Math.floor。
// [/maint-note:e2e1ee3c06b4]
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

// [maint-note:5a53b51ff227]
// 数据契约导航：JsonWriteState；字段含义以消费端校验为准。
// 类型组合：'written' | 'omitted' | 'overflow'。
// [/maint-note:5a53b51ff227]
type JsonWriteState = 'written' | 'omitted' | 'overflow'

/** JSON.stringify for the plain result envelopes, with an exact UTF-16
 * character budget and no over-budget intermediate string. */
// [maint-note:abb7c0f51d51]
// 维护导航：stringifyWithinBudget；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 budget：number。
// 返回契约：string | undefined。
// 返回路径：writeValue(value, '', false) === 'written' ? chunks.join('') : undefined。
// 直接调用：writeValue。
// 直接调用：chunks.join。
// [/maint-note:abb7c0f51d51]
function stringifyWithinBudget(value: unknown, budget: number): string | undefined {
  const chunks: string[] = []
  const active = new Set<object>()
  let length = 0

  // [maint-note:3bea1744cd9d]
  // 维护导航：append；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string。
  // 返回契约：boolean。
  // 分支条件：length + chunk.length > budget。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：length。
  // 直接调用：chunks.push。
  // [/maint-note:3bea1744cd9d]
  const append = (chunk: string): boolean => {
    if (length + chunk.length > budget) return false
    chunks.push(chunk)
    length += chunk.length
    return true
  }

  // [maint-note:7f675a8fa692]
  // 维护导航：appendString；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 返回契约：boolean。
  // 分支条件：!append('"')。
  // 分支条件：code === 0x22。
  // 分支条件：code === 0x5c。
  // 分支条件：code === 0x08。
  // 分支条件：code === 0x0c。
  // 分支条件：code === 0x0a。
  // 分支条件：code === 0x0d。
  // 分支条件：code === 0x09。
  // 分支条件：code < 0x20 || (code >= 0xd800 && code <= 0xdfff)。
  // 分支条件：code >= 0xd800 && code <= 0xdbff && index + 1 < value.length && value.charCodeAt(index + 1) >= 0xdc00 && value.charCodeAt(index + 1) <= 0xdfff。
  // 分支条件：!append(encoded)。
  // 遍历边界：let index = 0; index < value.length; index += 1。
  // 返回路径：false。
  // 返回路径：append('"')。
  // 写入目标：index。
  // 写入目标：encoded。
  // 直接调用：append。
  // 直接调用：value.charCodeAt。
  // 直接调用：value.slice。
  // 直接调用：code.toString(16).padStart。
  // 直接调用：code.toString。
  // [/maint-note:7f675a8fa692]
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

  // [maint-note:7b04a2afe516]
  // 维护导航：normalize；下列为本函数源码中的接口与路径索引。
  // 输入 input：unknown。
  // 输入 key：string。
  // 返回契约：unknown。
  // 分支条件：input !== null && typeof input === 'object'。
  // 分支条件：typeof toJSON === 'function'。
  // 分支条件：input instanceof Number || input instanceof String || input instanceof Boolean。
  // 返回路径：input.valueOf()。
  // 返回路径：input。
  // 写入目标：input。
  // 直接调用：toJSON.call。
  // 直接调用：input.valueOf。
  // [/maint-note:7b04a2afe516]
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

  // [maint-note:84371ae813d9]
  // 维护导航：isOmitted；下列为本函数源码中的接口与路径索引。
  // 输入 input：unknown。
  // 返回契约：boolean。
  // [/maint-note:84371ae813d9]
  const isOmitted = (input: unknown): boolean => (
    input === undefined || typeof input === 'function' || typeof input === 'symbol'
  )

  // [maint-note:0a2c513b2841]
  // 维护导航：writeValue；下列为本函数源码中的接口与路径索引。
  // 输入 input：unknown。
  // 输入 key：string。
  // 输入 arrayItem：boolean。
  // 返回契约：JsonWriteState。
  // 分支条件：isOmitted(normalized)。
  // 分支条件：normalized === null。
  // 分支条件：typeof normalized === 'string'。
  // 分支条件：typeof normalized === 'boolean'。
  // 分支条件：typeof normalized === 'number'。
  // 分支条件：typeof normalized === 'bigint'。
  // 分支条件：typeof normalized !== 'object'。
  // 分支条件：active.has(normalized)。
  // 分支条件：Array.isArray(normalized)。
  // 分支条件：!append('[')。
  // 分支条件：index > 0 && !append(',')。
  // 分支条件：writeValue(normalized[index], String(index), true) === 'overflow'。
  // 分支条件：!append('{')。
  // 分支条件：isOmitted(propertyValue)。
  // 分支条件：wroteProperty && !append(',')。
  // 分支条件：!appendString(property) || !append(':')。
  // 分支条件：writeNormalizedValue(propertyValue, property) === 'overflow'。
  // 遍历边界：let index = 0; index < normalized.length; index += 1。
  // 遍历边界：const property ← Object.keys(normalized)。
  // 返回路径：arrayItem ? (append('null') ? 'written' : 'overflow') : 'omitted'。
  // 返回路径：append('null') ? 'written' : 'overflow'。
  // 返回路径：appendString(normalized) ? 'written' : 'overflow'。
  // 返回路径：append(normalized ? 'true' : 'false') ? 'written' : 'overflow'。
  // 返回路径：append(encoded) ? 'written' : 'overflow'。
  // 返回路径：'omitted'。
  // 返回路径：'overflow'。
  // 返回路径：append(']') ? 'written' : 'overflow'。
  // 返回路径：append('}') ? 'written' : 'overflow'。
  // 显式抛出：new TypeError('Do not know how to serialize a BigInt')。
  // 显式抛出：new TypeError('Converting circular structure to JSON')。
  // 写入目标：index。
  // 写入目标：wroteProperty。
  // 直接调用：normalize。
  // 直接调用：isOmitted。
  // 直接调用：append。
  // 直接调用：appendString。
  // 直接调用：JSON.stringify。
  // 直接调用：active.has。
  // 直接调用：active.add。
  // 直接调用：Array.isArray。
  // 直接调用：writeValue。
  // 直接调用：String。
  // 直接调用：Object.keys。
  // 直接调用：writeNormalizedValue。
  // 直接调用：active.delete。
  // [/maint-note:0a2c513b2841]
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

  // [maint-note:227f346d6c46]
  // 维护导航：writeNormalizedValue；下列为本函数源码中的接口与路径索引。
  // 输入 input：unknown。
  // 输入 key：string。
  // 返回契约：JsonWriteState。
  // 分支条件：input === null。
  // 分支条件：typeof input === 'string'。
  // 分支条件：typeof input === 'boolean'。
  // 分支条件：typeof input === 'number'。
  // 分支条件：typeof input === 'bigint'。
  // 返回路径：append('null') ? 'written' : 'overflow'。
  // 返回路径：appendString(input) ? 'written' : 'overflow'。
  // 返回路径：append(input ? 'true' : 'false') ? 'written' : 'overflow'。
  // 返回路径：append(JSON.stringify(input)) ? 'written' : 'overflow'。
  // 返回路径：writeValue(input, key, false)。
  // 显式抛出：new TypeError('Do not know how to serialize a BigInt')。
  // 直接调用：append。
  // 直接调用：appendString。
  // 直接调用：JSON.stringify。
  // 直接调用：writeValue。
  // [/maint-note:227f346d6c46]
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

// [maint-note:056fa13968c7]
// 维护导航：serializeSuccess；下列为本函数源码中的接口与路径索引。
// 输入 value：Record<string, unknown>。
// 输入 budget：number。
// 返回契约：ToolExecutionResult。
// 分支条件：content.length <= budget。
// 返回路径：{ content }。
// 返回路径：errorResult('git_changes_observation_limit', 'result_envelope_exceeds_output_budget', false, budget)。
// 直接调用：JSON.stringify。
// 直接调用：errorResult。
// [/maint-note:056fa13968c7]
function serializeSuccess(value: Record<string, unknown>, budget: number): ToolExecutionResult {
  const content = JSON.stringify(value)
  if (content.length <= budget) return { content }
  return errorResult('git_changes_observation_limit', 'result_envelope_exceeds_output_budget', false, budget)
}

// [maint-note:ff86db83b466]
// 维护导航：errorResult；下列为本函数源码中的接口与路径索引。
// 输入 code：GitChangesErrorCode。
// 输入 reason：string。
// 输入 recoverable：boolean。
// 输入 budget：number。
// 返回契约：ToolExecutionResult。
// 返回路径：{ content: detailed.length <= budget ? detailed : full.length <= budget ? full : compact.slice(0, budget), isError: true, errorCode: code, }。
// 直接调用：JSON.stringify。
// 直接调用：compact.slice。
// [/maint-note:ff86db83b466]
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

// [maint-note:28fb2f3f595e]
// 维护导航：queryDigest；下列为本函数源码中的接口与路径索引。
// 输入 input：GitChangesInput。
// 返回契约：string。
// 返回路径：createHash('sha256') .update(JSON.stringify({ mode: input.mode, scope: input.scope, path: input.path ?? null })) .digest('hex')。
// 直接调用：createHash('sha256') .update(JSON.stringify({ mode: input.mode, scope: input.scope, path: input.path ?? null })) .digest。
// 直接调用：createHash('sha256') .update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:28fb2f3f595e]
function queryDigest(input: GitChangesInput): string {
  return createHash('sha256')
    .update(JSON.stringify({ mode: input.mode, scope: input.scope, path: input.path ?? null }))
    .digest('hex')
}

// [maint-note:20aa33060e75]
// 维护导航：sealCursor；下列为本函数源码中的接口与路径索引。
// 输入 backend：GitChangesBackend。
// 输入 payload：CursorPayload。
// 返回契约：string。
// 返回路径：backend.sealCursor(JSON.stringify(payload))。
// 直接调用：backend.sealCursor。
// 直接调用：JSON.stringify。
// [/maint-note:20aa33060e75]
function sealCursor(backend: GitChangesBackend, payload: CursorPayload): string {
  return backend.sealCursor(JSON.stringify(payload))
}

// [maint-note:caaf148cce18]
// 维护导航：decodeCursor；下列为本函数源码中的接口与路径索引。
// 输入 token：string。
// 输入 backend：GitChangesBackend。
// 返回契约：CursorPayload。
// 分支条件：!opened。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['v', 'w', 'q', 'r', 'o'])。
// 分支条件：value.v !== 1 || typeof value.w !== 'string' || typeof value.q !== 'string' || typeof value.r !== 'string' || typeof value.o !== 'number' || !Number.isSafeInteger(value.o) || value.o < 0。
// 返回路径：value as unknown as CursorPayload。
// 显式抛出：new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_authentication_failed')。
// 显式抛出：new GitChangesError('git_changes_invalid_cursor', 'Invalid Git changes cursor', 'cursor_payload_invalid')。
// 写入目标：value。
// 直接调用：backend.openCursor。
// 直接调用：JSON.parse。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Number.isSafeInteger。
// [/maint-note:caaf148cce18]
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

// [maint-note:7746425fe432]
// 维护导航：summarize；下列为本函数源码中的接口与路径索引。
// 输入 records：readonly GitFileRecord[]。
// 分支条件：record.binary === true。
// 分支条件：record.modeChanged。
// 分支条件：record.action === 'type_changed'。
// 分支条件：record.action === 'unmerged'。
// 遍历边界：const record ← records。
// 返回路径：{ uniqueFiles: unique.size, layerRecords: records.length, byLayer, byAction, recordsWithBinary, recordsWithModeChange, recordsWithTypeChange, recordsUnmerged, }。
// 写入目标：byLayer[record.layer]。
// 写入目标：byAction[record.action]。
// 写入目标：recordsWithBinary。
// 写入目标：recordsWithModeChange。
// 写入目标：recordsWithTypeChange。
// 写入目标：recordsUnmerged。
// 直接调用：unique.add。
// [/maint-note:7746425fe432]
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

// [maint-note:a7b9960d6319]
// 维护导航：flattenPatch；下列为本函数源码中的接口与路径索引。
// 输入 patches：readonly GitPatchLayer[]。
// 返回契约：FlatPatchLine[]。
// 返回路径：result。
// 直接调用：patches.forEach。
// [/maint-note:a7b9960d6319]
function flattenPatch(patches: readonly GitPatchLayer[]): FlatPatchLine[] {
  const result: FlatPatchLine[] = []
  patches.forEach((patch, patchIndex) => {
    patch.hunks.forEach((hunk, hunkIndex) => {
      hunk.lines.forEach((line, lineIndex) => result.push({ patchIndex, hunkIndex, lineIndex, line }))
    })
  })
  return result
}

// [maint-note:5a29fe5b0f5b]
// 维护导航：projectPatchPage；下列为本函数源码中的接口与路径索引。
// 输入 patches：readonly GitPatchLayer[]。
// 输入 selected：readonly FlatPatchLine[]。
// 输入 includeMetadataOnly：boolean。
// 返回契约：GitPatchLayer[]。
// 分支条件：includeMetadataOnly。
// 分支条件：selectedForPatch.length === 0。
// 分支条件：sourcePatch.hunks.length === 0。
// 分支条件：selectedForHunk.length === 0。
// 遍历边界：let patchIndex = 0; patchIndex < patches.length; patchIndex += 1。
// 遍历边界：let hunkIndex = 0; hunkIndex < sourcePatch.hunks.length; hunkIndex += 1。
// 返回路径：patches.map((patch) => ({ ...patch, hunks: [] }))。
// 返回路径：projected。
// 写入目标：patchIndex。
// 写入目标：hunkIndex。
// 直接调用：patches.map。
// 直接调用：selected.filter。
// 直接调用：projected.push。
// 直接调用：selectedForPatch.filter。
// 直接调用：hunks.push。
// 直接调用：selectedForHunk.map。
// [/maint-note:5a29fe5b0f5b]
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

// [maint-note:1ec87a7cf0d0]
// 维护导航：compareRecords；下列为本函数源码中的接口与路径索引。
// 输入 left：GitFileRecord。
// 输入 right：GitFileRecord。
// 返回契约：number。
// 返回路径：layerRank(left.layer) - layerRank(right.layer) || left.path.localeCompare(right.path) || (left.oldPath ?? '').localeCompare(right.oldPath ?? '') || left.action.localeCompare(right.action)。
// 直接调用：layerRank。
// 直接调用：left.path.localeCompare。
// 直接调用：(left.oldPath ?? '').localeCompare。
// 直接调用：left.action.localeCompare。
// [/maint-note:1ec87a7cf0d0]
function compareRecords(left: GitFileRecord, right: GitFileRecord): number {
  return layerRank(left.layer) - layerRank(right.layer) ||
    left.path.localeCompare(right.path) ||
    (left.oldPath ?? '').localeCompare(right.oldPath ?? '') ||
    left.action.localeCompare(right.action)
}

// [maint-note:62e10a2b0a9c]
// 维护导航：layerRank；下列为本函数源码中的接口与路径索引。
// 输入 layer：GitFileRecord['layer']。
// 返回契约：number。
// 返回路径：layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2。
// [/maint-note:62e10a2b0a9c]
function layerRank(layer: GitFileRecord['layer']): number {
  return layer === 'staged' ? 0 : layer === 'unstaged' ? 1 : 2
}

// [maint-note:a18c52e3f5dd]
// 维护导航：requiredObservationString；下列为本函数源码中的接口与路径索引。
// 输入 value：string | undefined。
// 输入 label：string。
// 返回契约：string。
// 分支条件：value。
// 返回路径：value。
// 显式抛出：new GitChangesError('git_changes_git_failed', `Git observation omitted ${label}`, 'invalid_backend_observation')。
// [/maint-note:a18c52e3f5dd]
function requiredObservationString(value: string | undefined, label: string): string {
  if (value) return value
  throw new GitChangesError('git_changes_git_failed', `Git observation omitted ${label}`, 'invalid_backend_observation')
}

// [maint-note:4caa9e77eb36]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value && typeof value === 'object' && !Array.isArray(value))。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:4caa9e77eb36]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

// [maint-note:8905f3024997]
// 维护导航：hasExactKeys；下列为本函数源码中的接口与路径索引。
// 输入 value：Record<string, unknown>。
// 输入 keys：readonly string[]。
// 返回契约：boolean。
// 返回路径：actual.length === expected.length && actual.every((key, index) => key === expected[index])。
// 直接调用：Object.keys(value).sort。
// 直接调用：Object.keys。
// 直接调用：[...keys].sort。
// 直接调用：actual.every。
// [/maint-note:8905f3024997]
function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort()
  const expected = [...keys].sort()
  return actual.length === expected.length && actual.every((key, index) => key === expected[index])
}
