import { UiPresentationLimits } from '../presentation/limits.js'
import { boundRetainedLines } from '../presentation/boundedLines.js'

/**
 * bounded detail store — the SINGLE holder of retained detail lines.
 * Stable messages keep only a light reference (record ID + counters + the
 * dual truncation state), so LRU eviction actually frees memory and no
 * second copy of the history can outlive the budget.
 *
 * The store is pure state evolved deterministically by the message
 * controller's reducer: one commit per (generation, recordId); repeated or
 * abandoned React renders never change cache facts. `/clear` starts a new
 * generation and invalidates every entry, so a late old-run event cannot
 * resurrect detail.
 */

export interface DetailStoreEntry {
  recordId: string
  /** Bounded viewer label (tool name etc.); never callId/requestId. */
  label: string
  lines: readonly string[]
  sourceTruncated: boolean
  viewCollapsed: boolean
  /** Σ (line.length + 1): separators are real cost; empty lines are not free. */
  chars: number
}

export interface DetailStoreState {
  readonly generation: number
  /** Insertion order = LRU order (oldest first). */
  readonly entries: ReadonlyMap<string, DetailStoreEntry>
  readonly retainedChars: number
}

export interface DetailCommit {
  readonly state: DetailStoreState
  /** recordIds evicted by this commit, oldest first. */
  readonly evicted: readonly string[]
  /** True when the commit itself clipped the incoming record. */
  readonly clipped: boolean
}

export function createDetailStoreState(generation = 1): DetailStoreState {
  return { generation, entries: new Map(), retainedChars: 0 }
}

/**
 * Clip one record against the dual per-record cap: characters (separator
 * included) and the record-count cap. A clip is an upstream-of-cache loss:
 * the entry is marked `sourceTruncated` so the viewer never promises a
 * completeness the cache no longer has.
 */
/**
 * Commit one record under the stable record identity. Re-committing the
 * same recordId replaces and refreshes its LRU position without evicting;
 * every commit then evicts oldest-first until the whole store is back under
 * the global character budget (always keeping at least one entry).
 */
export function commitDetailRecord(
  state: DetailStoreState,
  input: {
    recordId: string
    label: string
    lines: readonly string[]
    sourceTruncated: boolean
    viewCollapsed: boolean
  },
): DetailCommit {
  const clippedRecord = boundRetainedLines(
    input.lines,
    UiPresentationLimits.detailRecordMaxChars,
    UiPresentationLimits.detailRecordMaxRecords,
  )
  const entry: DetailStoreEntry = {
    recordId: input.recordId,
    label: input.label,
    lines: clippedRecord.lines,
    sourceTruncated: input.sourceTruncated || clippedRecord.clipped,
    viewCollapsed: input.viewCollapsed,
    chars: clippedRecord.chars,
  }
  const entries = new Map(state.entries)
  const previous = entries.get(input.recordId)
  let retainedChars = state.retainedChars + entry.chars - (previous?.chars ?? 0)
  entries.delete(input.recordId)
  entries.set(input.recordId, entry)

  const evicted: string[] = []
  while (retainedChars > UiPresentationLimits.detailCacheMaxChars && entries.size > 1) {
    const oldest = entries.keys().next().value
    if (oldest === undefined) break
    const evictedEntry = entries.get(oldest)
    entries.delete(oldest)
    if (evictedEntry) retainedChars -= evictedEntry.chars
    evicted.push(oldest)
  }
  return {
    state: { generation: state.generation, entries, retainedChars },
    evicted,
    clipped: clippedRecord.clipped,
  }
}

export function detailStoreHas(state: DetailStoreState, recordId: string): boolean {
  return state.entries.has(recordId)
}

export function detailStoreGet(state: DetailStoreState, recordId: string): DetailStoreEntry | undefined {
  return state.entries.get(recordId)
}

/** recordIds from oldest to newest (LRU order). */
export function detailStoreOrder(state: DetailStoreState): readonly string[] {
  return [...state.entries.keys()]
}

export function detailStoreCount(state: DetailStoreState): number {
  return state.entries.size
}
