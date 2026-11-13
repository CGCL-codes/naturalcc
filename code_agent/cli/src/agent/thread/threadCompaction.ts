import { createHash, randomUUID } from 'node:crypto'

import type { ModelMessage, ModelUserMessage } from '../../model/types/types.js'
import {
  COMPACTION_BODY_LIMITS,
  COMPACTION_LIMITS,
  type CompactionBody,
  type CompactionBodyEntry,
  type CompactionSource,
  type CompactionSourceUnit,
  type CompactionSummary,
  type CompactionUsage,
} from '../compaction/contracts.js'
import { toolCallsOf } from './AgentThread.js'
import { serializedRequestChars } from '../compaction/viewSerializer.js'

/**
 * thread-side compaction core. The Thread is the ONLY owner of raw
 * history and of the derived summary:
 *
 * - freeze produces a DEEP-CLONED, recursively frozen source: mutating any
 *   nested ToolResult block of the source cannot touch Thread raw;
 * - the provider returns an UNTRUSTED JSON body; parseCompactionBody
 *   validates structure/source allow-lists/caps and the core rebuilds
 *   identity, coverage, verification and the envelope from its own frozen
 *   source — the provider never grants itself verification;
 * - a freeze receipt (lease) binds thread/generation/run and pending cursor;
 *   install re-checks the lease synchronously at publish time;
 * - unit resolution uses immutable run/exchange/range identity, so a
 *   saved-only summary stays valid after its run closes (classification
 *   change, not identity change) and pending units invalidate as a whole.
 */

// [maint-note:68aaa74bff0b]
// 数据契约导航：FreezeReceipt；字段含义以消费端校验为准。
// 成员 leaseId：string。
// 成员 sourceVersion：number。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 pendingCursor：number。
// 成员 source：CompactionSource。
// 成员 unitMessageCounts：readonly number[]。
// 成员 protectedUnitIds：readonly string[]。
// 成员 newUnitIds：readonly string[]。
// [/maint-note:68aaa74bff0b]
export interface FreezeReceipt {
  readonly leaseId: string
  /** Monotonic within one lease; a successful re-sign invalidates old candidates. */
  readonly sourceVersion: number
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly pendingCursor: number
  readonly source: CompactionSource
  /** Message count per unit, parallel to source.units (engine trimming). */
  readonly unitMessageCounts: readonly number[]
  /** Current request and newest complete source range kept raw on purpose. */
  readonly protectedUnitIds: readonly string[]
  /**
   * only the NEWLY compressible units (not previously covered) —
   * the growth-trigger set. Empty means nothing new is summarizable and no
   * maintenance request may be dispatched.
   */
  readonly newUnitIds: readonly string[]
}

// [maint-note:5a3e7a9e9daf]
// 数据契约导航：ThreadCompactionView；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 generation：string。
// 成员 compactionLease（可省略）：string。
// 成员 receiptRegistry（可省略）：OwnerReceiptRegistry。
// 成员 failedFrontiers（可省略）：readonly string[]。
// 成员 messages：readonly ModelMessage[]。
// 成员 completedRuns：ReadonlyArray<{ readonly runKey: string readonly start?: number readonly end?: number readonly revision?: number readonly source?: { readonly sourceId: string readonly start: number readonly end: number readonly revision: number } readonly 。
// 成员 activeRun（可省略）：{ readonly runKey: string readonly savedCursor: number readonly prompt: string readonly nextExchangeOrdinal: number readonly exchangeRanges: ReadonlyArray<{ readonly exchangeId: string readonly start: number readonly end: number readonly re。
// 成员 activeSummary（可省略）：CompactionSummary。
// [/maint-note:5a3e7a9e9daf]
export interface ThreadCompactionView {
  readonly id: string
  readonly generation: string
  /** The one live install lease (core-owned); undefined after run close. */
  readonly compactionLease?: string
  /** per-thread receipt registry (created lazily by the real owner). */
  readonly receiptRegistry?: OwnerReceiptRegistry
  /** Bounded cross-run latch of deterministically failed source frontiers. */
  readonly failedFrontiers?: readonly string[]
  readonly messages: readonly ModelMessage[]
  readonly completedRuns: ReadonlyArray<{
    readonly runKey: string
    readonly start?: number
    readonly end?: number
    readonly revision?: number
    readonly source?: {
      readonly sourceId: string
      readonly start: number
      readonly end: number
      readonly revision: number
    }
    readonly exchangeRanges: ReadonlyArray<{
      readonly exchangeId: string
      readonly start: number
      readonly end: number
      readonly revision: number
    }>
  }>
  readonly activeRun?: {
    readonly runKey: string
    readonly savedCursor: number
    readonly prompt: string
    readonly nextExchangeOrdinal: number
    readonly exchangeRanges: ReadonlyArray<{
      readonly exchangeId: string
      readonly start: number
      readonly end: number
      readonly revision: number
    }>
  }
  readonly activeSummary?: CompactionSummary
}

// [maint-note:901c534bbf58]
// 数据契约导航：ThreadCompactionMutableState；字段含义以消费端校验为准。
// 成员 activeSummary（可省略）：CompactionSummary。
// 成员 failedFrontiers（可省略）：readonly string[]。
// [/maint-note:901c534bbf58]
export interface ThreadCompactionMutableState extends ThreadCompactionView {
  activeSummary?: CompactionSummary
  failedFrontiers?: readonly string[]
}

// [maint-note:306720b1bdc8]
// 维护导航：freezeCompactionSource；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
// 返回契约：FreezeReceipt。
// 分支条件：activeRun。
// 分支条件：pendingUnit。
// 分支条件：run.start !== undefined && run.end !== undefined && run.revision !== undefined。
// 分支条件：run.source。
// 分支条件：occupied.has(index)。
// 分支条件：message?.role === 'assistant' && toolCallsOf(message).length === 0。
// 分支条件：picked.length >= COMPACTION_LIMITS.maxInputUnits。
// 分支条件：pendingEntries.length > 0 && pending.length > 0。
// 遍历边界：const range ← activeRun.exchangeRanges。
// 遍历边界：const run ← [...state.completedRuns].reverse()。
// 遍历边界：let index = run.source.start; index < run.source.end; index += 1。
// 遍历边界：const range ← run.exchangeRanges。
// 遍历边界：let index = range.start; index < range.end; index += 1。
// 遍历边界：let index = run.end - 1; index >= run.start; index -= 1。
// 遍历边界：const range ← [...run.exchangeRanges].reverse()。
// 遍历边界：const entry ← priorityOrder。
// 遍历边界：const entry ← ordered。
// 返回路径：receipt。
// 写入目标：index。
// 直接调用：previousCoverage.map。
// 直接调用：tryPush。
// 直接调用：isClosedPendingExchange。
// 直接调用：candidates.push。
// 直接调用：newUnitIds.push。
// 直接调用：[...state.completedRuns].reverse。
// 直接调用：occupied.add。
// 直接调用：occupied.has。
// 直接调用：toolCallsOf。
// 直接调用：[...run.exchangeRanges].reverse。
// 直接调用：candidates.filter。
// 直接调用：exchangeCandidates.reduce。
// 直接调用：eligible .filter((entry) => entry.unit.kind === 'current_saved_exchange') .map(toCoord).sort。
// 直接调用：eligible .filter((entry) => entry.unit.kind === 'current_saved_exchange') .map。
// 直接调用：eligible .filter。
// 直接调用：eligible .filter((entry) => entry.unit.kind === 'historical_exchange' || entry.unit.kind === 'historical_user' || entry.unit.kind === 'historical_terminal') .map(toCoord).sort。
// 直接调用：eligible .filter((entry) => entry.unit.kind === 'historical_exchange' || entry.unit.kind === 'historical_user' || entry.unit.kind === 'historical_terminal') .map。
// 直接调用：[...reReadRanges].sort。
// 直接调用：[...reReadForPriority].reverse。
// 直接调用：eligible.filter。
// 直接调用：emit。
// 直接调用：state.messages.slice。
// 直接调用：[...picked].sort。
// 直接调用：ordered.map。
// 直接调用：canonicalCounts.push。
// 直接调用：canonicalMessages.push。
// 直接调用：deepFreeze。
// 直接调用：randomUUID。
// 直接调用：state.completedRuns.at。
// 直接调用：Date.now。
// 直接调用：Object.freeze。
// 直接调用：newUnitIds.filter。
// 直接调用：registerIssuedReceipt。
// 直接调用：ownerReceiptRegistry。
// [/maint-note:306720b1bdc8]
export function freezeCompactionSource(
  state: ThreadCompactionView,
  options: {
    readonly pendingClosed?: readonly ModelMessage[]
  } = {},
): FreezeReceipt {
  // Candidates in priority order: CURRENT run's saved exchanges first (they
  // are what a first long run needs compacted), then historical runs
  // newest-first. Selection is bounded BEFORE hashing: we stop walking once
  // the unit/char caps are reached, never cloning the full history first.
  // candidates are LIGHTWEIGHT coordinates (identity + range) —
  // no message cloning happens during enumeration. Only units entering the
  // final bounded selection are cloned, so freeze cost does not grow with
  // the full history beyond the caps.
  const candidates: Array<{ unit: CompactionSourceUnit; range: { start: number; end: number } }> = []
  // when an active summary exists, the replacement source must be the
  // strict-superset RAW set — OLD coverage is re-read from the original
  // owner and merged with the new units — never only the new units.
  const previousCoverage = state.activeSummary?.coveredUnits ?? []
  const alreadyCovered = new Set(previousCoverage.map((unit) => unit.exchangeId))
  const newUnitIds: string[] = []
  const reReadRanges: Array<{ unit: CompactionSourceUnit; start: number; end: number }> = []
  // [maint-note:862b59a44779]
  // 维护导航：tryPush；下列为本函数源码中的接口与路径索引。
  // 输入 kind：CompactionSourceUnit['kind']。
  // 输入 runKey：string。
  // 输入 range：{ start: number; end: number; revision: number; exchangeId: string }。
  // 返回契约：void。
  // 分支条件：alreadyCovered.has(range.exchangeId)。
  // 分支条件：covered && covered.kind !== 'current_closed_pending_exchange' && range.end > range.start。
  // 返回路径：无返回值。
  // 直接调用：alreadyCovered.has。
  // 直接调用：previousCoverage.find。
  // 直接调用：reReadRanges.push。
  // 直接调用：candidates.push。
  // 直接调用：newUnitIds.push。
  // [/maint-note:862b59a44779]
  const tryPush = (
    kind: CompactionSourceUnit['kind'],
    runKey: string,
    range: { start: number; end: number; revision: number; exchangeId: string },
  ): void => {
    if (alreadyCovered.has(range.exchangeId)) {
      // Re-read the OLD covered range from raw so the replacement summary
      // covers old+new together (strict superset of the previous coverage).
      const covered = previousCoverage.find((unit) => unit.exchangeId === range.exchangeId)
      if (covered && covered.kind !== 'current_closed_pending_exchange' && range.end > range.start) {
        reReadRanges.push({ unit: { ...covered }, start: range.start, end: range.end })
      }
      return
    }
    candidates.push({
      unit: {
        kind,
        exchangeId: range.exchangeId,
        threadId: state.id,
        generation: state.generation,
        runKey,
        start: range.start,
        end: range.end,
        revision: range.revision,
        contentHash: '',
      },
      range: { start: range.start, end: range.end },
    })
    newUnitIds.push(range.exchangeId)
  }

  const activeRun = state.activeRun
  if (activeRun) {
    for (const range of activeRun.exchangeRanges) {
      tryPush('current_saved_exchange', activeRun.runKey, range)
    }
    const pending = options.pendingClosed ?? []
    const pendingUnit: CompactionSourceUnit | undefined = pending.length > 0 && isClosedPendingExchange(activeRun, pending)
      ? {
        kind: 'current_closed_pending_exchange',
        exchangeId: `${activeRun.runKey}:pending-exchange-${activeRun.nextExchangeOrdinal}`,
        threadId: state.id,
        generation: state.generation,
        runKey: activeRun.runKey,
        start: -1,
        end: -1,
        revision: -1,
        contentHash: '',
      }
      : undefined
    if (pendingUnit) {
      candidates.push({ unit: pendingUnit, range: { start: -1, end: -1 } })
      newUnitIds.push(pendingUnit.exchangeId)
    }
  }
  for (const run of [...state.completedRuns].reverse()) {
    // Historical terminal Assistant is an independent trusted source range,
    // not an exchange and not UI prose. Derive it only from messages outside
    // the run's user anchor and closed exchange ranges.
    if (run.start !== undefined && run.end !== undefined && run.revision !== undefined) {
      const occupied = new Set<number>()
      if (run.source) {
        for (let index = run.source.start; index < run.source.end; index += 1) occupied.add(index)
      }
      for (const range of run.exchangeRanges) {
        for (let index = range.start; index < range.end; index += 1) occupied.add(index)
      }
      for (let index = run.end - 1; index >= run.start; index -= 1) {
        if (occupied.has(index)) continue
        const message = state.messages[index]
        if (message?.role === 'assistant' && toolCallsOf(message).length === 0) {
          tryPush('historical_terminal', run.runKey, {
            exchangeId: `${run.runKey}:terminal`,
            start: index,
            end: index + 1,
            revision: run.revision,
          })
          break
        }
      }
    }
    for (const range of [...run.exchangeRanges].reverse()) {
      tryPush('historical_exchange', run.runKey, range)
    }
    if (run.source) {
      tryPush('historical_user', run.runKey, {
        exchangeId: run.source.sourceId,
        start: run.source.start,
        end: run.source.end,
        revision: run.source.revision,
      })
    }
  }

  // protection is decided by CANONICAL EXCHANGE identity — source
  // expansion added historical user/terminal ranges, but those ranges do not
  // become tool exchanges and cannot displace the latest complete tool
  // evidence. A legal closed pending exchange is newest by definition; saved
  // exchanges otherwise compare by their canonical raw position.
  const exchangeCandidates = candidates.filter((entry) => (
    entry.unit.kind === 'historical_exchange' ||
    entry.unit.kind === 'current_saved_exchange' ||
    entry.unit.kind === 'current_closed_pending_exchange'
  ))
  const newestExchange = exchangeCandidates.reduce<(typeof candidates)[number] | undefined>((best, entry) => {
    if (!best) return entry
    // Pending units are always newest (they follow every saved range).
    if (entry.unit.kind === 'current_closed_pending_exchange') return entry
    if (best.unit.kind === 'current_closed_pending_exchange') return best
    return entry.unit.end > best.unit.end ? entry : best
  }, undefined)
  const protectedUnitIds = [
    ...(activeRun ? [`${activeRun.runKey}:request`] : []),
    ...(newestExchange ? [newestExchange.unit.exchangeId] : []),
  ]
  const eligible = newestExchange
    ? candidates.filter((entry) => entry.unit.exchangeId !== newestExchange.unit.exchangeId)
    : candidates

  // the FINAL selection order is canonical raw-chronological —
  // oldest re-read coverage first, then new units by ascending raw position
  // (priority governs WHICH units enter under the caps, never the output
  // sequence). Only entries that survive the caps are cloned now.
  // Three separate duties: (1) protection by identity —
  // done above; (2) PRIORITY selection under the caps — the current run's
  // new eligible units enter FIRST, then new history newest-first, then
  // re-read old coverage fills the remainder; (3) only the SELECTED set is
  // serialized in canonical raw-chronological order.
  // [maint-note:09c5cab31f0f]
  // 维护导航：toCoord；下列为本函数源码中的接口与路径索引。
  // 输入 entry：{ unit: CompactionSourceUnit; range: { start: number; end: number } }。
  // [/maint-note:09c5cab31f0f]
  const toCoord = (entry: { unit: CompactionSourceUnit; range: { start: number; end: number } }) =>
    ({ unit: entry.unit, start: entry.range.start, end: entry.range.end })
  const newCurrent = eligible
    .filter((entry) => entry.unit.kind === 'current_saved_exchange')
    .map(toCoord).sort((l, r) => r.start - l.start)
  const newHistory = eligible
    .filter((entry) => entry.unit.kind === 'historical_exchange' ||
      entry.unit.kind === 'historical_user' ||
      entry.unit.kind === 'historical_terminal')
    .map(toCoord).sort((l, r) => r.start - l.start)
  const reReadForPriority = [...reReadRanges].sort((l, r) => r.start - l.start)
  // C is the replacement's mandatory base — it enters the caps FIRST
  // (oldest coverage first so trimming the tail drops the newest re-read, and
  // C only ever shrinks when truly uncontainable). N fills the remaining
  // capacity by current/history priority. Canonical serialization below is
  // unaffected: it re-orders only the selected set.
  const priorityOrder = [...[...reReadForPriority].reverse(), ...newCurrent, ...newHistory]
  const pendingEntries = eligible.filter((entry) => entry.unit.kind === 'current_closed_pending_exchange')
  const picked: Array<{ unit: CompactionSourceUnit; raw: readonly ModelMessage[] }> = []
  let charBudget = COMPACTION_LIMITS.maxInputChars
  // [maint-note:2febe5d2e916]
  // 维护导航：emit；下列为本函数源码中的接口与路径索引。
  // 输入 unit：CompactionSourceUnit。
  // 输入 raw：readonly ModelMessage[]。
  // 返回契约：boolean。
  // 分支条件：picked.length >= COMPACTION_LIMITS.maxInputUnits || chars > charBudget。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：charBudget。
  // 直接调用：JSON.stringify。
  // 直接调用：picked.push。
  // 直接调用：deepFreeze。
  // 直接调用：createHash('sha256').update(JSON.stringify(raw)).digest。
  // 直接调用：createHash('sha256').update。
  // 直接调用：createHash。
  // 直接调用：deepFreezeMessages。
  // [/maint-note:2febe5d2e916]
  const emit = (unit: CompactionSourceUnit, raw: readonly ModelMessage[]): boolean => {
    const chars = JSON.stringify(raw).length
    if (picked.length >= COMPACTION_LIMITS.maxInputUnits || chars > charBudget) return false
    charBudget -= chars
    picked.push({ unit: deepFreeze({ ...unit, contentHash: createHash('sha256').update(JSON.stringify(raw)).digest('hex') }), raw: deepFreezeMessages(raw) })
    return true
  }
  for (const entry of priorityOrder) {
    if (picked.length >= COMPACTION_LIMITS.maxInputUnits) break
    emit(entry.unit, state.messages.slice(entry.start, entry.end))
  }
  const pending = options.pendingClosed ?? []
  if (pendingEntries.length > 0 && pending.length > 0) {
    emit(pendingEntries[0]!.unit, pending)
  }
  // Canonical serialization: the SELECTED units are re-ordered by raw start
  // (ascending); pending units (-1) sort last as the chronologically newest.
  const ordered = [...picked].sort((l, r) => {
    const lStart = l.unit.start
    const rStart = r.unit.start
    if (lStart === rStart) return 0
    if (lStart < 0) return 1
    if (rStart < 0) return -1
    return lStart - rStart
  })
  const canonicalUnits = ordered.map((entry) => entry.unit)
  const canonicalMessages: ModelMessage[] = []
  const canonicalCounts: number[] = []
  for (const entry of ordered) {
    canonicalCounts.push(entry.raw.length)
    canonicalMessages.push(...entry.raw)
  }
  const units = canonicalUnits
  const messages = canonicalMessages
  const unitMessageCounts = canonicalCounts

  const receipt = deepFreeze({
    leaseId: randomUUID(),
    sourceVersion: 1,
    threadId: state.id,
    generation: state.generation,
    runKey: activeRun?.runKey ?? state.completedRuns.at(-1)?.runKey ?? '',
    pendingCursor: activeRun?.savedCursor ?? 0,
    source: deepFreeze({
      schemaVersion: 1 as const,
      threadId: state.id,
      generation: state.generation,
      frozenAt: Date.now(),
      units: Object.freeze(canonicalUnits),
      messages: Object.freeze(canonicalMessages),
    }),
    unitMessageCounts: Object.freeze(canonicalCounts),
    protectedUnitIds,
    newUnitIds: Object.freeze(newUnitIds.filter((id) => canonicalUnits.some((unit) => unit.exchangeId === id))),
  })
  // register the CORE's own isolated copy; caller mutation of the
  // returned object can never rewrite what prepare/install validate against.
  registerIssuedReceipt(ownerReceiptRegistry(state), receipt)
  return receipt
}

/**
 * Parse and validate the UNTRUSTED provider body against the frozen source:
 * exact JSON shape, sourceIds limited to the frozen unit ids, entry/char/
 * byte caps, and never all-empty.
 */
// [maint-note:3162a4eb3dbb]
// 维护导航：parseCompactionBody；下列为本函数源码中的接口与路径索引。
// 输入 raw：string。
// 输入 source：CompactionSource。
// 返回契约：{ ok: true; body: CompactionBody } | { ok: false; reason: string }。
// 分支条件：!parsed || typeof parsed !== 'object' || Array.isArray(parsed)。
// 分支条件：!Array.isArray(value)。
// 分支条件：value.length > COMPACTION_BODY_LIMITS.maxEntriesPerSection。
// 分支条件：!item || typeof item !== 'object' || Array.isArray(item)。
// 分支条件：typeof record.text !== 'string' || !record.text.trim()。
// 分支条件：record.text.length > COMPACTION_BODY_LIMITS.maxEntryChars。
// 分支条件：!Array.isArray(record.sourceIds)。
// 分支条件：typeof id !== 'string' || !allowList.has(id)。
// 分支条件：record.sourceIds.length === 0 && section !== 'suggestions' && section !== 'unknowns'。
// 分支条件：entries === 0。
// 分支条件：Buffer.byteLength(JSON.stringify(body), 'utf8') > COMPACTION_BODY_LIMITS.maxBodyBytes。
// 遍历边界：const section ← ['facts', 'failures', 'unknowns', 'unfinished', 'suggestions'] as const。
// 遍历边界：const item ← value。
// 遍历边界：const id ← record.sourceIds。
// 返回路径：{ ok: false, reason: 'The summary body is not valid JSON' }。
// 返回路径：{ ok: false, reason: 'The summary body must be a JSON object' }。
// 返回路径：{ ok: false, reason: `The summary body section "${section}" must be an array` }。
// 返回路径：{ ok: false, reason: `The summary body section "${section}" exceeds the entry cap` }。
// 返回路径：{ ok: false, reason: `Each "${section}" entry must be an object` }。
// 返回路径：{ ok: false, reason: `Each "${section}" entry needs non-empty text` }。
// 返回路径：{ ok: false, reason: `A "${section}" entry exceeds the character cap` }。
// 返回路径：{ ok: false, reason: `Each "${section}" entry needs a sourceIds array` }。
// 返回路径：{ ok: false, reason: `A "${section}" entry cites an unknown sourceId` }。
// 返回路径：{ ok: false, reason: `A "${section}" entry must cite its evidence` }。
// 返回路径：{ ok: false, reason: 'The summary body is entirely empty' }。
// 返回路径：{ ok: false, reason: 'The summary body exceeds the byte cap' }。
// 返回路径：{ ok: true, body }。
// 写入目标：parsed。
// 写入目标：entries。
// 写入目标：sections[section]。
// 直接调用：JSON.parse。
// 直接调用：Array.isArray。
// 直接调用：source.units.map。
// 直接调用：record.text.trim。
// 直接调用：allowList.has。
// 直接调用：sectionEntries.push。
// 直接调用：deepFreeze。
// 直接调用：Object.freeze。
// 直接调用：Buffer.byteLength。
// 直接调用：JSON.stringify。
// [/maint-note:3162a4eb3dbb]
export function parseCompactionBody(
  raw: string,
  source: CompactionSource,
): { ok: true; body: CompactionBody } | { ok: false; reason: string } {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return { ok: false, reason: 'The summary body is not valid JSON' }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, reason: 'The summary body must be a JSON object' }
  }
  const allowList = new Set(source.units.map((unit) => unit.exchangeId))
  const sections: Record<string, readonly CompactionBodyEntry[]> = {}
  let entries = 0
  for (const section of ['facts', 'failures', 'unknowns', 'unfinished', 'suggestions'] as const) {
    const value = (parsed as Record<string, unknown>)[section]
    if (!Array.isArray(value)) {
      return { ok: false, reason: `The summary body section "${section}" must be an array` }
    }
    if (value.length > COMPACTION_BODY_LIMITS.maxEntriesPerSection) {
      return { ok: false, reason: `The summary body section "${section}" exceeds the entry cap` }
    }
    const sectionEntries: CompactionBodyEntry[] = []
    for (const item of value) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        return { ok: false, reason: `Each "${section}" entry must be an object` }
      }
      const record = item as Record<string, unknown>
      if (typeof record.text !== 'string' || !record.text.trim()) {
        return { ok: false, reason: `Each "${section}" entry needs non-empty text` }
      }
      if (record.text.length > COMPACTION_BODY_LIMITS.maxEntryChars) {
        return { ok: false, reason: `A "${section}" entry exceeds the character cap` }
      }
      if (!Array.isArray(record.sourceIds)) {
        return { ok: false, reason: `Each "${section}" entry needs a sourceIds array` }
      }
      for (const id of record.sourceIds) {
        if (typeof id !== 'string' || !allowList.has(id)) {
          return { ok: false, reason: `A "${section}" entry cites an unknown sourceId` }
        }
      }
      if (record.sourceIds.length === 0 && section !== 'suggestions' && section !== 'unknowns') {
        return { ok: false, reason: `A "${section}" entry must cite its evidence` }
      }
      sectionEntries.push(deepFreeze({ text: record.text, sourceIds: Object.freeze([...record.sourceIds]) as string[] }))
      entries += 1
    }
    sections[section] = Object.freeze([...sectionEntries])
  }
  if (entries === 0) {
    return { ok: false, reason: 'The summary body is entirely empty' }
  }
  const body: CompactionBody = deepFreeze({
    facts: sections.facts!,
    failures: sections.failures!,
    unknowns: sections.unknowns!,
    unfinished: sections.unfinished!,
    suggestions: sections.suggestions!,
  })
  if (Buffer.byteLength(JSON.stringify(body), 'utf8') > COMPACTION_BODY_LIMITS.maxBodyBytes) {
    return { ok: false, reason: 'The summary body exceeds the byte cap' }
  }
  return { ok: true, body }
}

/**
 * an opaque, frozen handle to a core-registered candidate. The
 * handle carries NO trustworthy payload — the core keeps the real record in
 * a module-private WeakMap keyed by object identity, so a copied or forged
 * handle either misses the registry entirely or resolves to the ORIGINAL
 * unmodified record; caller-supplied receipt/summary fields are never read.
 */
// [maint-note:cc4d8ab05df5]
// 数据契约导航：SealedCompactionCandidate；字段含义以消费端校验为准。
// 成员 kind：'core-compaction-candidate'。
// 成员 handleId：string。
// [/maint-note:cc4d8ab05df5]
export interface SealedCompactionCandidate {
  readonly kind: 'core-compaction-candidate'
  readonly handleId: string
}

// [maint-note:cd01a0e36c70]
// 数据契约导航：CoreCandidateRecord；字段含义以消费端校验为准。
// 成员 receipt：FreezeReceipt。
// 成员 summary：CompactionSummary。
// [/maint-note:cd01a0e36c70]
interface CoreCandidateRecord {
  readonly receipt: FreezeReceipt
  readonly summary: CompactionSummary
}

/** Private registry: object identity IS the trust. */
const candidateRegistry = new WeakMap<SealedCompactionCandidate, CoreCandidateRecord>()

/**
 * registry of RECEIPTS the core itself issued, keyed by leaseId
 * and scoped to ONE owner (thread view). The stored copy is deep-cloned at
 * freeze time, so later caller-side mutation of the returned receipt object
 * (including its leaseId, units or messages) can never rewrite what the core
 * validates against. A receipt whose leaseId is not registered here was never
 * issued by this core and prepare refuses it outright. Because the registry
 * is per-owner, another Thread's freezes can never evict this Thread's
 * in-flight source records.
 */
// [maint-note:40652649d871]
// 数据契约导航：OwnerReceiptRegistry；字段含义以消费端校验为准。
// 成员 entries：Map<string, FreezeReceipt>。
// [/maint-note:40652649d871]
export interface OwnerReceiptRegistry {
  readonly entries: Map<string, FreezeReceipt>
}

// [maint-note:51a32e19bc0d]
// 维护导航：createOwnerReceiptRegistry；下列为本函数源码中的接口与路径索引。
// 返回契约：OwnerReceiptRegistry。
// 返回路径：{ entries: new Map() }。
// [/maint-note:51a32e19bc0d]
export function createOwnerReceiptRegistry(): OwnerReceiptRegistry {
  return { entries: new Map() }
}

/** Core-internal: register the isolated copy of a freshly frozen receipt. */
// [maint-note:b9c48983e948]
// 维护导航：registerIssuedReceipt；下列为本函数源码中的接口与路径索引。
// 输入 owner：OwnerReceiptRegistry。
// 输入 receipt：FreezeReceipt。
// 返回契约：FreezeReceipt。
// 分支条件：owner.entries.size > 16。
// 分支条件：oldest !== undefined。
// 返回路径：isolated。
// 直接调用：deepFreeze。
// 直接调用：structuredClone。
// 直接调用：owner.entries.set。
// 直接调用：owner.entries.keys().next。
// 直接调用：owner.entries.keys。
// 直接调用：owner.entries.delete。
// [/maint-note:b9c48983e948]
function registerIssuedReceipt(owner: OwnerReceiptRegistry, receipt: FreezeReceipt): FreezeReceipt {
  const isolated = deepFreeze(structuredClone(receipt))
  owner.entries.set(receipt.leaseId, isolated)
  // Bound per-owner: keep the most recent 16 leases of THIS thread.
  if (owner.entries.size > 16) {
    const oldest = owner.entries.keys().next().value
    if (oldest !== undefined) owner.entries.delete(oldest)
  }
  return isolated
}

/** Lazy per-view registry: keyed by the VIEW object, not the whole process. */
const viewRegistries = new WeakMap<ThreadCompactionView, OwnerReceiptRegistry>()

// [maint-note:c63595a0ed9f]
// 维护导航：ownerReceiptRegistry；下列为本函数源码中的接口与路径索引。
// 输入 view：ThreadCompactionView。
// 返回契约：OwnerReceiptRegistry。
// 分支条件：existing。
// 返回路径：existing。
// 返回路径：created。
// 直接调用：viewRegistries.get。
// 直接调用：createOwnerReceiptRegistry。
// 直接调用：viewRegistries.set。
// [/maint-note:c63595a0ed9f]
function ownerReceiptRegistry(view: ThreadCompactionView): OwnerReceiptRegistry {
  const existing = view.receiptRegistry ?? viewRegistries.get(view)
  if (existing) return existing
  const created = createOwnerReceiptRegistry()
  viewRegistries.set(view, created)
  return created
}

// [maint-note:67f5991fec85]
// 维护导航：resolveIssuedReceipt；下列为本函数源码中的接口与路径索引。
// 输入 owner：OwnerReceiptRegistry。
// 输入 receipt：FreezeReceipt。
// 返回契约：FreezeReceipt | undefined。
// 分支条件：core === undefined。
// 分支条件：JSON.stringify(core) !== JSON.stringify(receipt)。
// 返回路径：undefined。
// 返回路径：core。
// 直接调用：owner.entries.get。
// 直接调用：JSON.stringify。
// [/maint-note:67f5991fec85]
function resolveIssuedReceipt(owner: OwnerReceiptRegistry, receipt: FreezeReceipt): FreezeReceipt | undefined {
  // Only the CORE's own frozen copy is trusted; identity of the caller's
  // object is irrelevant and its payload is never re-read.
  const core = owner.entries.get(receipt.leaseId)
  if (core === undefined) return undefined
  // The caller's copy must still MATCH the issued record field-for-field —
  // a tampered copy of a real lease is rejected, not silently trusted.
  if (JSON.stringify(core) !== JSON.stringify(receipt)) return undefined
  return core
}

/**
 * core-side finalization of a shell trim. The engine submits only
 * the SELECTED unit IDs of a receipt the core issued; the core rebuilds
 * messages/counts/order from ITS OWN frozen record, verifies the selection is
 * a closed contiguous span of the issued canonical order, and atomically
 * replaces the issued record with the FINAL receipt that dispatch, prepare
 * and install all share. Callers can never hand-write messages/counts/hash
 * and ask the core to trust them.
 */
// [maint-note:871a24e4157e]
// 维护导航：finalizeIssuedReceiptTrim；下列为本函数源码中的接口与路径索引。
// 输入 registry：OwnerReceiptRegistry。
// 输入 original：FreezeReceipt。
// 输入 selectedUnitIds：readonly string[]。
// 返回契约：FreezeReceipt | undefined。
// 分支条件：core === undefined。
// 分支条件：selectedSet.has(issuedOrder[index]!)。
// 分支条件：first === -1。
// 分支条件：!selectedSet.has(issuedOrder[index]!)。
// 分支条件：index >= first && index <= last。
// 遍历边界：let index = 0; index < issuedOrder.length; index += 1。
// 遍历边界：let index = first; index <= last; index += 1。
// 遍历边界：let index = 0; index < core.source.units.length; index += 1。
// 返回路径：undefined。
// 返回路径：trimmed。
// 写入目标：index。
// 写入目标：first。
// 写入目标：last。
// 写入目标：cursor。
// 直接调用：resolveIssuedReceipt。
// 直接调用：core.source.units.map。
// 直接调用：selectedSet.has。
// 直接调用：core.source.units.slice。
// 直接调用：counts.push。
// 直接调用：messages.push。
// 直接调用：core.source.messages.slice。
// 直接调用：deepFreeze。
// 直接调用：Object.freeze。
// 直接调用：registry.entries.set。
// [/maint-note:871a24e4157e]
export function finalizeIssuedReceiptTrim(
  registry: OwnerReceiptRegistry,
  original: FreezeReceipt,
  selectedUnitIds: readonly string[],
): FreezeReceipt | undefined {
  const core = resolveIssuedReceipt(registry, original)
  if (core === undefined) return undefined
  const selectedSet = new Set(selectedUnitIds)
  const issuedOrder = core.source.units.map((unit) => unit.exchangeId)
  let first = -1
  let last = -1
  for (let index = 0; index < issuedOrder.length; index += 1) {
    if (selectedSet.has(issuedOrder[index]!)) {
      if (first === -1) first = index
      last = index
    }
  }
  if (first === -1) return undefined
  for (let index = first; index <= last; index += 1) {
    if (!selectedSet.has(issuedOrder[index]!)) return undefined
  }
  // Rebuild from the CORE record.
  let cursor = 0
  const messages: ModelMessage[] = []
  const counts: number[] = []
  const units = core.source.units.slice(first, last + 1)
  for (let index = 0; index < core.source.units.length; index += 1) {
    const count = core.unitMessageCounts[index] ?? 0
    if (index >= first && index <= last) {
      counts.push(count)
      messages.push(...core.source.messages.slice(cursor, cursor + count))
    }
    cursor += count
  }
  const trimmed: FreezeReceipt = deepFreeze({
    ...core,
    sourceVersion: core.sourceVersion + 1,
    source: deepFreeze({
      ...core.source,
      units: Object.freeze(units),
      messages: Object.freeze(messages),
    }),
    unitMessageCounts: Object.freeze(counts),
    // The trim keeps the SAME lease: the issued record is REPLACED by the
    // final version the dispatch actually uses, so prepare/install bind to
    // exactly what the provider saw; the original full record is gone.
  })
  registry.entries.set(trimmed.leaseId, trimmed)
  return trimmed
}

/**
 * Owner-facing wrapper: the engine submits only the selected unit
 * IDs; this resolves the owner's own registry, finalizes the trim against the
 * core record and returns the FINAL core-issued receipt.
 */
// [maint-note:4958337aa636]
// 维护导航：finalizeCompactionReceiptTrim；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 original：FreezeReceipt。
// 输入 selectedUnitIds：readonly string[]。
// 返回契约：FreezeReceipt | undefined。
// 返回路径：finalizeIssuedReceiptTrim(ownerReceiptRegistry(state), original, selectedUnitIds)。
// 直接调用：finalizeIssuedReceiptTrim。
// 直接调用：ownerReceiptRegistry。
// [/maint-note:4958337aa636]
export function finalizeCompactionReceiptTrim(
  state: ThreadCompactionView,
  original: FreezeReceipt,
  selectedUnitIds: readonly string[],
): FreezeReceipt | undefined {
  return finalizeIssuedReceiptTrim(ownerReceiptRegistry(state), original, selectedUnitIds)
}

/** Bounded, core-computed measurement bound to the real candidate. */
// [maint-note:01c3eac05cfe]
// 数据契约导航：CandidateMeasurement；字段含义以消费端校验为准。
// 成员 rawChars：number。
// 成员 summaryEnvelopeChars：number。
// 成员 currentRequestChars（可省略）：number。
// 成员 candidateRequestChars（可省略）：number。
// [/maint-note:01c3eac05cfe]
export interface CandidateMeasurement {
  readonly rawChars: number
  readonly summaryEnvelopeChars: number
  /** Complete current request after real request-only section construction. */
  readonly currentRequestChars?: number
  /** Complete candidate request after the same construction. */
  readonly candidateRequestChars?: number
}

// [maint-note:b7f4da9d349f]
// 维护导航：mintCandidate；下列为本函数源码中的接口与路径索引。
// 输入 receipt：FreezeReceipt。
// 输入 summary：CompactionSummary。
// 返回契约：SealedCompactionCandidate。
// 返回路径：handle。
// 直接调用：Object.freeze。
// 直接调用：candidateRegistry.set。
// 直接调用：deepFreeze。
// 直接调用：structuredClone。
// [/maint-note:b7f4da9d349f]
function mintCandidate(receipt: FreezeReceipt, summary: CompactionSummary): SealedCompactionCandidate {
  const handle: SealedCompactionCandidate = Object.freeze({
    kind: 'core-compaction-candidate',
    handleId: `candidate-${receipt.leaseId}`,
  })
  candidateRegistry.set(handle, {
    receipt,
    // Private deep clone: the registered (and later published) value is
    // isolated from every caller-held object.
    summary: deepFreeze(structuredClone(summary)),
  })
  return handle
}

/**
 * the ONLY path from an untrusted provider body to an installable
 * candidate. The core parses the body against the receipt's own source,
 * rebuilds the envelope, re-verifies units and the CURRENT owner state, and
 * returns a registry-backed opaque handle. The record is kept privately —
 * prepare runs INSIDE the core controller, so no live state escapes.
 */
// [maint-note:432d3dbc88a7]
// 维护导航：prepareCompactionCandidate；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 receipt：FreezeReceipt。
// 输入 rawBody：string。
// 输入 usage：CompactionUsage（可省略）。
// 输入 requestOnlySections：readonly string[]；缺省表达式 []。
// 返回契约：{ ok: true; candidate: SealedCompactionCandidate; measurement: CandidateMeasurement } | { ok: false; reason: string }。
// 分支条件：issued === undefined。
// 分支条件：!parsed.ok。
// 分支条件：!validateAgainstView(state, issued, summary)。
// 返回路径：{ ok: false, reason: 'The receipt was not issued by this thread core' }。
// 返回路径：{ ok: false, reason: parsed.reason }。
// 返回路径：{ ok: false, reason: error instanceof Error ? error.message : 'The summary envelope is not installable' }。
// 返回路径：{ ok: false, reason: 'The receipt is no longer installable on this thread' }。
// 返回路径：{ ok: true, candidate: mintCandidate(issued, summary), measurement }。
// 写入目标：summary。
// 直接调用：resolveIssuedReceipt。
// 直接调用：ownerReceiptRegistry。
// 直接调用：parseCompactionBody。
// 直接调用：buildCompactionSummary。
// 直接调用：validateAgainstView。
// 直接调用：serializedRequestChars。
// 直接调用：summaryViewMessage。
// 直接调用：mintCandidate。
// [/maint-note:432d3dbc88a7]
export function prepareCompactionCandidate(
  state: ThreadCompactionView,
  receipt: FreezeReceipt,
  rawBody: string,
  usage?: CompactionUsage,
  requestOnlySections: readonly string[] = [],
): { ok: true; candidate: SealedCompactionCandidate; measurement: CandidateMeasurement } | { ok: false; reason: string } {
  // only a receipt the CORE issued (and still matches field-for-
  // field) can produce a candidate. Forged or tampered copies fail here.
  const issued = resolveIssuedReceipt(ownerReceiptRegistry(state), receipt)
  if (issued === undefined) {
    return { ok: false, reason: 'The receipt was not issued by this thread core' }
  }
  // Body validation uses the CORE's frozen source, never the caller's copy.
  const parsed = parseCompactionBody(rawBody, issued.source)
  if (!parsed.ok) return { ok: false, reason: parsed.reason }
  let summary: CompactionSummary
  try {
    summary = buildCompactionSummary(issued, parsed.body, usage)
  } catch (error) {
    // an expected envelope overflow is a bounded maintenance failure,
    // never an escape into a run error.
    return { ok: false, reason: error instanceof Error ? error.message : 'The summary envelope is not installable' }
  }
  // Pre-validate against the CURRENT state so install cannot be deferred past
  // an owner change and then succeed.
  if (!validateAgainstView(state, issued, summary)) {
    return { ok: false, reason: 'The receipt is no longer installable on this thread' }
  }
  // the core computes the REAL net-gain measurement through the
  // COMMON final serializer — the summary's actual request-view projection
  // (framework note + body, JSON-escaped exactly as it travels) plus the
  // request-only envelope, versus the covered raw in the same view. The
  // request-only sections are identical on both sides (the boundary's
  // read-only peek), so the comparison isolates the substitution while the
  // absolute numbers stay the true outbound costs. No constants stand in.
  const measurement: CandidateMeasurement = {
    rawChars: serializedRequestChars(issued.source.messages, requestOnlySections),
    summaryEnvelopeChars: serializedRequestChars([summaryViewMessage(summary)], requestOnlySections),
  }
  return { ok: true, candidate: mintCandidate(issued, summary), measurement }
}

/** Registry lookup — the only way install obtains the trusted record. */
// [maint-note:718c7ea3527c]
// 维护导航：resolveCandidate；下列为本函数源码中的接口与路径索引。
// 输入 handle：SealedCompactionCandidate。
// 返回契约：CoreCandidateRecord | undefined。
// 分支条件：handle === null || typeof handle !== 'object'。
// 返回路径：undefined。
// 返回路径：candidateRegistry.get(handle)。
// 直接调用：candidateRegistry.get。
// [/maint-note:718c7ea3527c]
function resolveCandidate(handle: SealedCompactionCandidate): CoreCandidateRecord | undefined {
  if (handle === null || typeof handle !== 'object') return undefined
  return candidateRegistry.get(handle)
}

/** @internal Isolated candidate projection used by the owning AgentThread. */
// [maint-note:2132ad09eb80]
// 维护导航：compactionCandidateSummary；下列为本函数源码中的接口与路径索引。
// 输入 handle：SealedCompactionCandidate。
// 返回契约：CompactionSummary | undefined。
// 返回路径：resolved ? deepFreeze(structuredClone(resolved.summary)) : undefined。
// 直接调用：resolveCandidate。
// 直接调用：deepFreeze。
// 直接调用：structuredClone。
// [/maint-note:2132ad09eb80]
export function compactionCandidateSummary(handle: SealedCompactionCandidate): CompactionSummary | undefined {
  const resolved = resolveCandidate(handle)
  return resolved ? deepFreeze(structuredClone(resolved.summary)) : undefined
}

// [maint-note:7457bb7a6c5e]
// 维护导航：validateAgainstView；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 receipt：FreezeReceipt。
// 输入 summary：CompactionSummary。
// 返回契约：boolean。
// 分支条件：state.compactionLease !== receipt.leaseId。
// 分支条件：resolveIssuedReceipt(ownerReceiptRegistry(state), receipt) === undefined。
// 分支条件：receipt.threadId !== state.id || receipt.generation !== state.generation。
// 分支条件：!state.activeRun || receipt.runKey !== state.activeRun.runKey。
// 分支条件：receipt.pendingCursor !== state.activeRun.savedCursor。
// 分支条件：summary.threadId !== state.id || summary.generation !== state.generation。
// 分支条件：summary.verification !== 'schema_and_source_valid'。
// 分支条件：Buffer.byteLength(JSON.stringify(summary), 'utf8') > COMPACTION_LIMITS.maxSummaryBytes。
// 返回路径：false。
// 返回路径：true。
// 直接调用：resolveIssuedReceipt。
// 直接调用：ownerReceiptRegistry。
// 直接调用：Buffer.byteLength。
// 直接调用：JSON.stringify。
// [/maint-note:7457bb7a6c5e]
function validateAgainstView(
  state: ThreadCompactionView,
  receipt: FreezeReceipt,
  summary: CompactionSummary,
): boolean {
  if (state.compactionLease !== receipt.leaseId) return false
  // W03: a lease identifies the run owner, not an eternally reusable source
  // version. A same-lease trim atomically replaces the issued record; any
  // candidate prepared from the older version immediately loses publish
  // authority even though the owning run itself is still active.
  if (resolveIssuedReceipt(ownerReceiptRegistry(state), receipt) === undefined) return false
  if (receipt.threadId !== state.id || receipt.generation !== state.generation) return false
  if (!state.activeRun || receipt.runKey !== state.activeRun.runKey) return false
  if (receipt.pendingCursor !== state.activeRun.savedCursor) return false
  if (summary.threadId !== state.id || summary.generation !== state.generation) return false
  if (summary.verification !== 'schema_and_source_valid') return false
  if (Buffer.byteLength(JSON.stringify(summary), 'utf8') > COMPACTION_LIMITS.maxSummaryBytes) return false
  return true
}

/** The core-rebuilt envelope: identity/coverage/verification are ours. */
// [maint-note:fef98604c7de]
// 维护导航：buildCompactionSummary；下列为本函数源码中的接口与路径索引。
// 输入 receipt：FreezeReceipt。
// 输入 body：CompactionBody。
// 输入 usage：CompactionUsage（可省略）。
// 返回契约：CompactionSummary。
// 分支条件：Buffer.byteLength(JSON.stringify(summary), 'utf8') > COMPACTION_LIMITS.maxSummaryBytes。
// 返回路径：deepFreeze(summary)。
// 显式抛出：new Error('The rebuilt summary exceeds the envelope byte cap')。
// 直接调用：Date.now。
// 直接调用：JSON.stringify。
// 直接调用：Buffer.byteLength。
// 直接调用：deepFreeze。
// [/maint-note:fef98604c7de]
export function buildCompactionSummary(
  receipt: FreezeReceipt,
  body: CompactionBody,
  usage?: CompactionUsage,
): CompactionSummary {
  const summary: CompactionSummary = {
    schemaVersion: 1,
    threadId: receipt.threadId,
    generation: receipt.generation,
    createdAt: Date.now(),
    verification: 'schema_and_source_valid',
    coveredUnits: receipt.source.units,
    text: JSON.stringify(body),
    ...(usage ? { usage } : {}),
  }
  if (Buffer.byteLength(JSON.stringify(summary), 'utf8') > COMPACTION_LIMITS.maxSummaryBytes) {
    throw new Error('The rebuilt summary exceeds the envelope byte cap')
  }
  return deepFreeze(summary)
}

// [maint-note:14dc369cac92]
// 维护导航：installCompactionSummary；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionMutableState & { readonly compactionLease?: string }。
// 输入 receiptOrCandidate：FreezeReceipt | SealedCompactionCandidate。
// 输入 maybeSummary：CompactionSummary（可省略）。
// 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
// 返回契约：boolean。
// 分支条件：resolved === undefined。
// 分支条件：!validateAgainstView(state, sealed.receipt, sealed.summary)。
// 分支条件：!validateSummaryUnits(state, sealed.summary, options.pendingClosed ?? [])。
// 分支条件：previous && !isStrictSuperset(sealed.summary.coveredUnits, previous.coveredUnits)。
// 返回路径：false。
// 返回路径：true。
// 写入目标：state.activeSummary。
// 直接调用：resolveCandidate。
// 直接调用：validateAgainstView。
// 直接调用：validateSummaryUnits。
// 直接调用：isStrictSuperset。
// 直接调用：deepFreeze。
// 直接调用：structuredClone。
// [/maint-note:14dc369cac92]
export function installCompactionSummary(
  state: ThreadCompactionMutableState & { readonly compactionLease?: string },
  receiptOrCandidate: FreezeReceipt | SealedCompactionCandidate,
  maybeSummary?: CompactionSummary,
  options: {
    readonly pendingClosed?: readonly ModelMessage[]
  } = {},
): boolean {
  // only core-sealed candidates install. The legacy (receipt, summary)
  // shape remains for the engine's internal call path which already used
  // buildCompactionSummary — but the engine now goes through
  // prepareCompactionCandidate, so the legacy branch is removed entirely.
  const resolved = resolveCandidate(receiptOrCandidate as SealedCompactionCandidate)
  if (resolved === undefined) return false
  const { receipt, summary } = resolved
  const sealed = resolved
  // the lease must be the CURRENT registered one AND its issuing run
  // must still OWN the thread — release() clears the lease, so a closed run
  // leaves no install path (no completedRuns fallback). An already PUBLISHED
  // summary stays reusable (that is a view read, not an install).
  if (!validateAgainstView(state, sealed.receipt, sealed.summary)) return false
  if (!validateSummaryUnits(state, sealed.summary, options.pendingClosed ?? [])) return false
  const previous = state.activeSummary
  if (previous && !isStrictSuperset(sealed.summary.coveredUnits, previous.coveredUnits)) return false
  // store another isolated copy — the published value is immutable and
  // unreachable for caller-side mutation.
  state.activeSummary = deepFreeze(structuredClone(sealed.summary))
  return true
}

// [maint-note:d145b8139aaf]
// 维护导航：summaryStillValid；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
// 返回契约：boolean。
// 分支条件：!summary。
// 返回路径：true。
// 返回路径：validateSummaryUnits(state, summary, options.pendingClosed ?? [])。
// 直接调用：validateSummaryUnits。
// [/maint-note:d145b8139aaf]
export function summaryStillValid(
  state: ThreadCompactionView,
  options: {
    readonly pendingClosed?: readonly ModelMessage[]
  } = {},
): boolean {
  const summary = state.activeSummary
  if (!summary) return true
  return validateSummaryUnits(state, summary, options.pendingClosed ?? [])
}

// [maint-note:369c6be029bd]
// 维护导航：frontierKey；下列为本函数源码中的接口与路径索引。
// 输入 units：readonly CompactionSourceUnit[]。
// 返回契约：string。
// 返回路径：units.map((unit) => unit.exchangeId).join('\u0000')。
// 直接调用：units.map((unit) => unit.exchangeId).join。
// 直接调用：units.map。
// [/maint-note:369c6be029bd]
export function frontierKey(units: readonly CompactionSourceUnit[]): string {
  // Delimiter-joined stable identity; bounded by the unit cap.
  return units.map((unit) => unit.exchangeId).join('\u0000')
}

// [maint-note:5ebca995ad9a]
// 维护导航：rememberFailedFrontier；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionMutableState。
// 输入 key：string。
// 返回契约：void。
// 分支条件：existing.includes(key)。
// 返回路径：无返回值。
// 写入目标：state.failedFrontiers。
// 直接调用：existing.includes。
// 直接调用：existing.slice。
// [/maint-note:5ebca995ad9a]
export function rememberFailedFrontier(state: ThreadCompactionMutableState, key: string): void {
  const existing = state.failedFrontiers ?? []
  if (existing.includes(key)) return
  // Bounded: keep the most recent 8 failed frontiers.
  state.failedFrontiers = [...existing.slice(-7), key]
}

// [maint-note:76845ada9a1f]
// 维护导航：hasFailedFrontier；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 key：string。
// 返回契约：boolean。
// 返回路径：(state.failedFrontiers ?? []).includes(key)。
// 直接调用：(state.failedFrontiers ?? []).includes。
// [/maint-note:76845ada9a1f]
export function hasFailedFrontier(state: ThreadCompactionView, key: string): boolean {
  return (state.failedFrontiers ?? []).includes(key)
}

// [maint-note:ebd3804031cc]
// 维护导航：isStrictSuperset；下列为本函数源码中的接口与路径索引。
// 输入 next：readonly CompactionSourceUnit[]。
// 输入 previous：readonly CompactionSourceUnit[]。
// 返回契约：boolean。
// 分支条件：nextIds.size <= previousIds.size。
// 分支条件：!nextIds.has(id)。
// 遍历边界：const id ← previousIds。
// 返回路径：false。
// 返回路径：true。
// 直接调用：previous.map。
// 直接调用：next.map。
// 直接调用：nextIds.has。
// [/maint-note:ebd3804031cc]
export function isStrictSuperset(
  next: readonly CompactionSourceUnit[],
  previous: readonly CompactionSourceUnit[],
): boolean {
  const previousIds = new Set(previous.map((unit) => unit.exchangeId))
  const nextIds = new Set(next.map((unit) => unit.exchangeId))
  if (nextIds.size <= previousIds.size) return false
  for (const id of previousIds) {
    if (!nextIds.has(id)) return false
  }
  return true
}

// [maint-note:ee77dd3bfe69]
// 维护导航：validateSummaryUnits；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadCompactionView。
// 输入 summary：CompactionSummary。
// 输入 pendingClosed：readonly ModelMessage[]。
// 返回契约：boolean。
// 分支条件：unit.threadId !== state.id || unit.generation !== state.generation。
// 分支条件：unit.kind === 'current_closed_pending_exchange'。
// 分支条件：unit.start !== -1 || unit.end !== -1。
// 分支条件：!run || run.runKey !== unit.runKey。
// 分支条件：usedPending || pendingClosed.length === 0。
// 分支条件：!isClosedPendingExchange(run, pendingClosed)。
// 分支条件：hash !== unit.contentHash。
// 分支条件：unit.kind === 'historical_user'。
// 分支条件：!source || source.sourceId !== unit.exchangeId || source.start !== unit.start || source.end !== unit.end || source.revision !== unit.revision。
// 分支条件：slice.length !== 1 || slice[0]?.role !== 'user'。
// 分支条件：unit.kind === 'historical_terminal'。
// 分支条件：!run || run.revision !== unit.revision || unit.exchangeId !== `${unit.runKey}:terminal`。
// 分支条件：unit.start < (run.start ?? 0) || unit.end !== unit.start + 1 || unit.end > (run.end ?? 0)。
// 分支条件：message?.role !== 'assistant' || toolCallsOf(message).length > 0。
// 分支条件：unit.start < 0 || unit.end <= unit.start || unit.end > state.messages.length。
// 分支条件：!activeRange && !completedRange。
// 遍历边界：const unit ← summary.coveredUnits。
// 返回路径：false。
// 返回路径：true。
// 写入目标：usedPending。
// 直接调用：isClosedPendingExchange。
// 直接调用：createHash('sha256').update(JSON.stringify(pendingClosed)).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// 直接调用：state.completedRuns.find。
// 直接调用：state.messages.slice。
// 直接调用：createHash('sha256').update(JSON.stringify(slice)).digest。
// 直接调用：toolCallsOf。
// 直接调用：createHash('sha256').update(JSON.stringify([message])).digest。
// 直接调用：active.exchangeRanges.find。
// 直接调用：state.completedRuns .filter((run) => run.runKey === unit.runKey) .flatMap((run) => run.exchangeRanges) .find。
// 直接调用：state.completedRuns .filter((run) => run.runKey === unit.runKey) .flatMap。
// 直接调用：state.completedRuns .filter。
// [/maint-note:ee77dd3bfe69]
function validateSummaryUnits(
  state: ThreadCompactionView,
  summary: CompactionSummary,
  pendingClosed: readonly ModelMessage[],
): boolean {
  let usedPending = false
  for (const unit of summary.coveredUnits) {
    if (unit.threadId !== state.id || unit.generation !== state.generation) return false
    if (unit.kind === 'current_closed_pending_exchange') {
      if (unit.start !== -1 || unit.end !== -1) return false
      const run = state.activeRun
      if (!run || run.runKey !== unit.runKey) return false
      if (usedPending || pendingClosed.length === 0) return false
      if (!isClosedPendingExchange(run, pendingClosed)) return false
      const hash = createHash('sha256').update(JSON.stringify(pendingClosed)).digest('hex')
      if (hash !== unit.contentHash) return false
      usedPending = true
      continue
    }
    if (unit.kind === 'historical_user') {
      const run = state.completedRuns.find((candidate) => candidate.runKey === unit.runKey)
      const source = run?.source
      if (!source || source.sourceId !== unit.exchangeId ||
        source.start !== unit.start || source.end !== unit.end || source.revision !== unit.revision) return false
      const slice = state.messages.slice(unit.start, unit.end)
      if (slice.length !== 1 || slice[0]?.role !== 'user') return false
      const hash = createHash('sha256').update(JSON.stringify(slice)).digest('hex')
      if (hash !== unit.contentHash) return false
      continue
    }
    if (unit.kind === 'historical_terminal') {
      const run = state.completedRuns.find((candidate) => candidate.runKey === unit.runKey)
      if (!run || run.revision !== unit.revision || unit.exchangeId !== `${unit.runKey}:terminal`) return false
      if (unit.start < (run.start ?? 0) || unit.end !== unit.start + 1 || unit.end > (run.end ?? 0)) return false
      const message = state.messages[unit.start]
      if (message?.role !== 'assistant' || toolCallsOf(message).length > 0) return false
      const hash = createHash('sha256').update(JSON.stringify([message])).digest('hex')
      if (hash !== unit.contentHash) return false
      continue
    }
    if (unit.start < 0 || unit.end <= unit.start || unit.end > state.messages.length) return false
    // Classification, not identity: a saved exchange resolves against the
    // active run OR any completed run with the same immutable range.
    const active = state.activeRun
    const activeRange = active && active.runKey === unit.runKey
      ? active.exchangeRanges.find((candidate) => matchesRange(candidate, unit))
      : undefined
    const completedRange = state.completedRuns
      .filter((run) => run.runKey === unit.runKey)
      .flatMap((run) => run.exchangeRanges)
      .find((candidate) => matchesRange(candidate, unit))
    if (!activeRange && !completedRange) return false
    const slice = state.messages.slice(unit.start, unit.end)
    const hash = createHash('sha256').update(JSON.stringify(slice)).digest('hex')
    if (hash !== unit.contentHash) return false
  }
  return true
}

// [maint-note:fb69d990ef16]
// 维护导航：matchesRange；下列为本函数源码中的接口与路径索引。
// 输入 candidate：{ exchangeId: string; start: number; end: number; revision: number }。
// 输入 unit：CompactionSourceUnit。
// 返回契约：boolean。
// 返回路径：candidate.exchangeId === unit.exchangeId && candidate.start === unit.start && candidate.end === unit.end && candidate.revision === unit.revision。
// [/maint-note:fb69d990ef16]
function matchesRange(
  candidate: { exchangeId: string; start: number; end: number; revision: number },
  unit: CompactionSourceUnit,
): boolean {
  return candidate.exchangeId === unit.exchangeId &&
    candidate.start === unit.start &&
    candidate.end === unit.end &&
    candidate.revision === unit.revision
}

// [maint-note:8b08c429af8f]
// 维护导航：isClosedPendingExchange；下列为本函数源码中的接口与路径索引。
// 输入 run：{ savedCursor: number; prompt: string }。
// 输入 pending：readonly ModelMessage[]。
// 返回契约：boolean。
// 分支条件：run.savedCursor === 0。
// 分支条件：!user || user.role !== 'user' || userTextOf(user) !== run.prompt。
// 分支条件：exchange.length < 2 || exchange[0]?.role !== 'assistant'。
// 返回路径：false。
// 返回路径：calls.length > 0 && results.length === calls.length && results.every((message, index) => ( message.role === 'toolResult' && message.toolCallId === calls[index]?.id && message.toolName === calls[index]?.name ))。
// 直接调用：pending.slice。
// 直接调用：userTextOf。
// 直接调用：toolCallsOf。
// 直接调用：exchange.slice。
// 直接调用：results.every。
// [/maint-note:8b08c429af8f]
export function isClosedPendingExchange(
  run: { savedCursor: number; prompt: string },
  pending: readonly ModelMessage[],
): boolean {
  const exchange = run.savedCursor === 0 ? pending.slice(1) : pending
  if (run.savedCursor === 0) {
    const user = pending[0]
    if (!user || user.role !== 'user' || userTextOf(user) !== run.prompt) return false
  }
  if (exchange.length < 2 || exchange[0]?.role !== 'assistant') return false
  const calls = toolCallsOf(exchange[0])
  const results = exchange.slice(1)
  return calls.length > 0 && results.length === calls.length && results.every((message, index) => (
    message.role === 'toolResult' &&
    message.toolCallId === calls[index]?.id &&
    message.toolName === calls[index]?.name
  ))
}

/** The derived summary message injected into the request view. */
// [maint-note:53d35d8cec96]
// 维护导航：summaryViewMessage；下列为本函数源码中的接口与路径索引。
// 输入 summary：CompactionSummary。
// 返回契约：ModelUserMessage。
// 返回路径：{ role: 'user', content: [{ type: 'text', text: [ '[Framework note: the following DERIVED summary replaces earlier complete exchanges that were compacted. Raw evidence remains retained by the framework; entries cite their sources and sugges。
// 直接调用：[ '[Framework note: the following DERIVED summary replaces earlier complete exchanges that were compacted. Raw evidence remains retained by the framework; entries cite their sources and suggestions are marked inference.]', summary.text, ].j。
// [/maint-note:53d35d8cec96]
export function summaryViewMessage(summary: CompactionSummary): ModelUserMessage {
  return {
    role: 'user',
    content: [{
      type: 'text',
      text: [
        '[Framework note: the following DERIVED summary replaces earlier complete exchanges that were compacted. Raw evidence remains retained by the framework; entries cite their sources and suggestions are marked inference.]',
        summary.text,
      ].join('\n'),
    }],
  }
}

// [maint-note:c30df4b17090]
// 维护导航：deepFreezeMessages；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：ModelMessage[]。
// 返回路径：deepFreeze(structuredClone(messages)) as ModelMessage[]。
// 直接调用：deepFreeze。
// 直接调用：structuredClone。
// [/maint-note:c30df4b17090]
function deepFreezeMessages(messages: readonly ModelMessage[]): ModelMessage[] {
  return deepFreeze(structuredClone(messages)) as ModelMessage[]
}

// [maint-note:97f52502c7f9]
// 维护导航：deepFreeze；下列为本函数源码中的接口与路径索引。
// 输入 value：T。
// 返回契约：T。
// 分支条件：!value || typeof value !== 'object' || Object.isFrozen(value)。
// 遍历边界：const child ← Object.values(value as Record<string, unknown>)。
// 返回路径：value。
// 返回路径：Object.freeze(value)。
// 直接调用：Object.isFrozen。
// 直接调用：Object.values。
// 直接调用：deepFreeze。
// 直接调用：Object.freeze。
// [/maint-note:97f52502c7f9]
function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
  return Object.freeze(value)
}

// [maint-note:98effe837155]
// 维护导航：userTextOf；下列为本函数源码中的接口与路径索引。
// 输入 message：ModelMessage。
// 返回契约：string。
// 分支条件：message.role !== 'user'。
// 返回路径：''。
// 返回路径：message.content .filter((block): block is { type: 'text'; text: string } => block.type === 'text') .map((block) => block.text) .join('')。
// 直接调用：message.content .filter((block): block is { type: 'text'; text: string } => block.type === 'text') .map((block) => block.text) .join。
// 直接调用：message.content .filter((block): block is { type: 'text'; text: string } => block.type === 'text') .map。
// 直接调用：message.content .filter。
// [/maint-note:98effe837155]
function userTextOf(message: ModelMessage): string {
  if (message.role !== 'user') return ''
  return message.content
    .filter((block): block is { type: 'text'; text: string } => block.type === 'text')
    .map((block) => block.text)
    .join('')
}
