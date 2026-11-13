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

// [maint-note:d5295bb78946]
// 数据契约导航：DetailStoreEntry；字段含义以消费端校验为准。
// 成员 recordId：string。
// 成员 label：string。
// 成员 lines：readonly string[]。
// 成员 sourceTruncated：boolean。
// 成员 viewCollapsed：boolean。
// 成员 chars：number。
// [/maint-note:d5295bb78946]
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

// [maint-note:53d4643a97f5]
// 数据契约导航：DetailStoreState；字段含义以消费端校验为准。
// 成员 generation：number。
// 成员 entries：ReadonlyMap<string, DetailStoreEntry>。
// 成员 retainedChars：number。
// [/maint-note:53d4643a97f5]
export interface DetailStoreState {
  readonly generation: number
  /** Insertion order = LRU order (oldest first). */
  readonly entries: ReadonlyMap<string, DetailStoreEntry>
  readonly retainedChars: number
}

// [maint-note:ff321c8f316d]
// 数据契约导航：DetailCommit；字段含义以消费端校验为准。
// 成员 state：DetailStoreState。
// 成员 evicted：readonly string[]。
// 成员 clipped：boolean。
// [/maint-note:ff321c8f316d]
export interface DetailCommit {
  readonly state: DetailStoreState
  /** recordIds evicted by this commit, oldest first. */
  readonly evicted: readonly string[]
  /** True when the commit itself clipped the incoming record. */
  readonly clipped: boolean
}

// [maint-note:f00a93f623e8]
// 维护导航：createDetailStoreState；下列为本函数源码中的接口与路径索引。
// 输入 generation：类型由调用方与函数体约束；缺省表达式 1。
// 返回契约：DetailStoreState。
// 返回路径：{ generation, entries: new Map(), retainedChars: 0 }。
// [/maint-note:f00a93f623e8]
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
// [maint-note:e9d066e3b17d]
// 维护导航：commitDetailRecord；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailStoreState。
// 输入 input：{ recordId: string label: string lines: readonly string[] sourceTruncated: boolean viewCollapsed: boolean }。
// 返回契约：DetailCommit。
// 分支条件：oldest === undefined。
// 分支条件：evictedEntry。
// 遍历边界：retainedChars > UiPresentationLimits.detailCacheMaxChars && entries.size > 1。
// 返回路径：{ state: { generation: state.generation, entries, retainedChars }, evicted, clipped: clippedRecord.clipped, }。
// 写入目标：retainedChars。
// 直接调用：boundRetainedLines。
// 直接调用：entries.get。
// 直接调用：entries.delete。
// 直接调用：entries.set。
// 直接调用：entries.keys().next。
// 直接调用：entries.keys。
// 直接调用：evicted.push。
// [/maint-note:e9d066e3b17d]
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

// [maint-note:95b2fd1f6565]
// 维护导航：detailStoreHas；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailStoreState。
// 输入 recordId：string。
// 返回契约：boolean。
// 返回路径：state.entries.has(recordId)。
// 直接调用：state.entries.has。
// [/maint-note:95b2fd1f6565]
export function detailStoreHas(state: DetailStoreState, recordId: string): boolean {
  return state.entries.has(recordId)
}

// [maint-note:0c1a3c4f9f30]
// 维护导航：detailStoreGet；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailStoreState。
// 输入 recordId：string。
// 返回契约：DetailStoreEntry | undefined。
// 返回路径：state.entries.get(recordId)。
// 直接调用：state.entries.get。
// [/maint-note:0c1a3c4f9f30]
export function detailStoreGet(state: DetailStoreState, recordId: string): DetailStoreEntry | undefined {
  return state.entries.get(recordId)
}

/** recordIds from oldest to newest (LRU order). */
// [maint-note:8bb93014de79]
// 维护导航：detailStoreOrder；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailStoreState。
// 返回契约：readonly string[]。
// 返回路径：[...state.entries.keys()]。
// 直接调用：state.entries.keys。
// [/maint-note:8bb93014de79]
export function detailStoreOrder(state: DetailStoreState): readonly string[] {
  return [...state.entries.keys()]
}

// [maint-note:e4134cddf0eb]
// 维护导航：detailStoreCount；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailStoreState。
// 返回契约：number。
// 返回路径：state.entries.size。
// [/maint-note:e4134cddf0eb]
export function detailStoreCount(state: DetailStoreState): number {
  return state.entries.size
}
