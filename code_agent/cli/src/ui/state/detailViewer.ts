/**
 * detail-viewer controller (pure state machine). The viewer is a dynamic
 * overlay in the interactive region: opening, paging, switching and closing
 * never rewrite the committed stable history. Every transition is a bounded,
 * synchronous reducer step — no timers, no filesystem, no tool re-runs.
 */

import type { StableBatch } from './messageController.js'
import { detailStoreGet, type DetailStoreState } from './detailStore.js'
import { wrapDisplayLines } from '../terminal/index.js'

// [maint-note:df5e98a3c27b]
// 数据契约导航：DetailViewerRecord；字段含义以消费端校验为准。
// 成员 recordId：string。
// 成员 totalLines：number。
// 成员 sourceTruncated：boolean。
// 成员 label：string。
// 成员 evicted（可省略）：boolean。
// 成员 totalLogicalLines（可省略）：number。
// [/maint-note:df5e98a3c27b]
export interface DetailViewerRecord {
  /** Stable message-record identity used as cache key and switch target. */
  recordId: string
  totalLines: number
  sourceTruncated: boolean
  /** Bounded label shown in the viewer header; never callId/requestId. */
  label: string
  /** Lightweight history candidate remains reachable after heavy LRU loss. */
  evicted?: boolean
  totalLogicalLines?: number
}

// [maint-note:796c2484ab75]
// 数据契约导航：DetailViewerGeometry；字段含义以消费端校验为准。
// 成员 outerWidth：number。
// 成员 innerWidth：number。
// 成员 viewportRows：number。
// 成员 panelRows：number。
// 成员 compact：boolean。
// [/maint-note:796c2484ab75]
export interface DetailViewerGeometry {
  outerWidth: number
  innerWidth: number
  viewportRows: number
  panelRows: number
  compact: boolean
}

/**
 * One geometry owner for wrapping, paging and frame reservation. Full mode
 * accounts for the two border columns and five chrome rows; tiny terminals
 * fall back to a borderless panel. Two rows are conservatively retained for
 * the idle footer/status region owned outside the viewer.
 */
// [maint-note:3976f5bc4f5e]
// 维护导航：detailViewerGeometry；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 输入 rows：number。
// 返回契约：DetailViewerGeometry。
// 分支条件：outerWidth >= 12 && availablePanelRows >= 6。
// 返回路径：{ outerWidth, innerWidth: Math.max(1, outerWidth - 2), viewportRows, panelRows: viewportRows + 5, compact: false, }。
// 返回路径：{ outerWidth, innerWidth: outerWidth, viewportRows: Math.max(1, availablePanelRows), panelRows: availablePanelRows, compact: true, }。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：Math.min。
// [/maint-note:3976f5bc4f5e]
export function detailViewerGeometry(width: number, rows: number): DetailViewerGeometry {
  const outerWidth = Math.max(1, Math.floor(width))
  const availablePanelRows = Math.max(0, Math.floor(rows) - 2)
  if (outerWidth >= 12 && availablePanelRows >= 6) {
    const viewportRows = Math.max(1, Math.min(12, availablePanelRows - 5))
    return {
      outerWidth,
      innerWidth: Math.max(1, outerWidth - 2),
      viewportRows,
      panelRows: viewportRows + 5,
      compact: false,
    }
  }
  return {
    outerWidth,
    innerWidth: outerWidth,
    viewportRows: Math.max(1, availablePanelRows),
    panelRows: availablePanelRows,
    compact: true,
  }
}

// [maint-note:92e2654257c7]
// 数据契约导航：ProjectedDetailViewerRecord；字段含义以消费端校验为准。
// 成员 logicalLines：number。
// 成员 visualLines：number。
// [/maint-note:92e2654257c7]
export interface ProjectedDetailViewerRecord extends DetailViewerRecord {
  logicalLines: number
  visualLines: number
}

/**
 * Build viewer candidates from light stable-message metadata plus the one
 * bounded detail store. Missing entries are intentionally retained as
 * explicit eviction records; projection never recreates their heavy body.
 */
// [maint-note:7245fb65fdaa]
// 维护导航：projectDetailViewerRecords；下列为本函数源码中的接口与路径索引。
// 输入 batches：readonly StableBatch[]。
// 输入 detailStore：DetailStoreState。
// 输入 contentWidth：number。
// 返回契约：readonly ProjectedDetailViewerRecord[]。
// 分支条件：!folded || seen.has(message.id)。
// 遍历边界：const batch ← batches。
// 遍历边界：const message ← batch.messages。
// 遍历边界：const line ← lines。
// 返回路径：records。
// 写入目标：visualLines。
// 直接调用：seen.has。
// 直接调用：seen.add。
// 直接调用：detailStoreGet。
// 直接调用：wrapDisplayLines。
// 直接调用：records.push。
// [/maint-note:7245fb65fdaa]
export function projectDetailViewerRecords(
  batches: readonly StableBatch[],
  detailStore: DetailStoreState,
  contentWidth: number,
): readonly ProjectedDetailViewerRecord[] {
  const seen = new Set<string>()
  const records: ProjectedDetailViewerRecord[] = []
  for (const batch of batches) {
    for (const message of batch.messages) {
      const detail = message.toolPresentation?.detail
      const folded = message.viewCollapsed === true || detail?.viewCollapsed === true || detail?.sourceTruncated === true
      if (!folded || seen.has(message.id)) continue
      seen.add(message.id)
      const entry = detailStoreGet(detailStore, message.id)
      const lines = entry?.lines ?? []
      let visualLines = 0
      for (const line of lines) {
        visualLines += line === ''
          ? 1
          : wrapDisplayLines(line, contentWidth, { hard: true, trim: false, wordWrap: false }).length
      }
      const logicalLines = entry?.lines.length ?? detail?.totalRecords ?? 0
      records.push({
        recordId: message.id,
        label: entry?.label ?? message.toolName ?? (message.role === 'assistant' ? 'assistant' : 'detail'),
        sourceTruncated: entry?.sourceTruncated ?? detail?.sourceTruncated ?? false,
        totalLines: visualLines,
        totalLogicalLines: logicalLines,
        logicalLines,
        visualLines,
        evicted: entry === undefined,
      })
    }
  }
  return records
}

// [maint-note:1acfce5b00ba]
// 数据契约导航：DetailViewerState；字段含义以消费端校验为准。
// 成员 open：boolean。
// 成员 records：readonly DetailViewerRecord[]。
// 成员 index：number。
// 成员 firstLine：number。
// 成员 generation：number。
// [/maint-note:1acfce5b00ba]
export interface DetailViewerState {
  open: boolean
  /** Records with retained detail, newest-last; Tab walks this list. */
  records: readonly DetailViewerRecord[]
  index: number
  firstLine: number
  /** Bumped on open: the viewer re-clamps pages when records change. */
  generation: number
}

// [maint-note:2564a1e2de6c]
// 数据契约导航：DetailViewerSnapshotOptions；字段含义以消费端校验为准。
// 成员 viewportRows：number。
// 成员 totalLines：number。
// [/maint-note:2564a1e2de6c]
export interface DetailViewerSnapshotOptions {
  viewportRows: number
  totalLines: number
}

// [maint-note:3788e6fd65df]
// 数据契约导航：DetailViewerAction；字段含义以消费端校验为准。
// 类型组合：| { type: 'open'; records: readonly DetailViewerRecord[]; index?: number } | { type: 'close' } | { type: 'pageUp'; viewportRows: number } | { type: 'pageDown'; viewportRows: number } | { type: 'home' } | { type: 'end'; viewportRows: number 。
// [/maint-note:3788e6fd65df]
export type DetailViewerAction =
  | { type: 'open'; records: readonly DetailViewerRecord[]; index?: number }
  | { type: 'close' }
  | { type: 'pageUp'; viewportRows: number }
  | { type: 'pageDown'; viewportRows: number }
  | { type: 'home' }
  | { type: 'end'; viewportRows: number }
  | { type: 'nextRecord' }
  | { type: 'previousRecord' }
  | { type: 'recordsChanged'; records: readonly DetailViewerRecord[] }

export const initialDetailViewerState: DetailViewerState = {
  open: false,
  records: [],
  index: 0,
  firstLine: 1,
  generation: 0,
}

// [maint-note:d1847be7ceb2]
// 维护导航：clampIndex；下列为本函数源码中的接口与路径索引。
// 输入 index：number。
// 输入 count：number。
// 返回契约：number。
// 分支条件：count === 0。
// 返回路径：0。
// 返回路径：Math.min(Math.max(0, index), count - 1)。
// 直接调用：Math.min。
// 直接调用：Math.max。
// [/maint-note:d1847be7ceb2]
function clampIndex(index: number, count: number): number {
  if (count === 0) return 0
  return Math.min(Math.max(0, index), count - 1)
}

// [maint-note:34ffc44bd3e0]
// 维护导航：detailViewerReducer；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailViewerState。
// 输入 action：DetailViewerAction。
// 返回契约：DetailViewerState。
// 分支条件：switch action.type。
// 分支条件：case 'open'。
// 分支条件：action.records.length === 0。
// 分支条件：case 'close'。
// 分支条件：case 'pageUp'。
// 分支条件：!state.open。
// 分支条件：case 'pageDown'。
// 分支条件：!record。
// 分支条件：case 'home'。
// 分支条件：case 'end'。
// 分支条件：case 'nextRecord'。
// 分支条件：!state.open || state.records.length === 0。
// 分支条件：case 'previousRecord'。
// 分支条件：case 'recordsChanged'。
// 返回路径：state.open ? { ...state, open: false } : state。
// 返回路径：{ open: true, records: action.records, index, firstLine: 1, generation: state.generation + 1, }。
// 返回路径：state。
// 返回路径：{ ...state, firstLine: Math.max(1, state.firstLine - rows) }。
// 返回路径：{ ...state, firstLine: Math.min(maxFirst, state.firstLine + rows) }。
// 返回路径：state.open ? { ...state, firstLine: 1 } : state。
// 返回路径：{ ...state, firstLine: Math.max(1, record.totalLines - rows + 1) }。
// 返回路径：{ ...state, index: next, firstLine: 1 }。
// 返回路径：{ ...state, index: previous, firstLine: 1 }。
// 返回路径：{ ...state, open: false, records: [], index: 0, firstLine: 1 }。
// 返回路径：{ ...state, records: action.records, index: 0, firstLine: 1 }。
// 返回路径：{ ...state, records: action.records, index, firstLine: 1 }。
// 直接调用：clampIndex。
// 直接调用：Math.max。
// 直接调用：Math.min。
// [/maint-note:34ffc44bd3e0]
export function detailViewerReducer(
  state: DetailViewerState,
  action: DetailViewerAction,
): DetailViewerState {
  switch (action.type) {
    case 'open': {
      if (action.records.length === 0) {
        // No viewable record: opening is a no-op (callers surface a hint).
        return state.open ? { ...state, open: false } : state
      }
      const index = clampIndex(action.index ?? action.records.length - 1, action.records.length)
      return {
        open: true,
        records: action.records,
        index,
        firstLine: 1,
        generation: state.generation + 1,
      }
    }
    case 'close':
      return state.open ? { ...state, open: false } : state
    case 'pageUp': {
      if (!state.open) return state
      const rows = Math.max(1, action.viewportRows)
      return { ...state, firstLine: Math.max(1, state.firstLine - rows) }
    }
    case 'pageDown': {
      if (!state.open) return state
      const record = state.records[state.index]
      if (!record) return state
      const rows = Math.max(1, action.viewportRows)
      const maxFirst = Math.max(1, record.totalLines - rows + 1)
      return { ...state, firstLine: Math.min(maxFirst, state.firstLine + rows) }
    }
    case 'home':
      return state.open ? { ...state, firstLine: 1 } : state
    case 'end': {
      if (!state.open) return state
      const record = state.records[state.index]
      if (!record) return state
      const rows = Math.max(1, action.viewportRows)
      return { ...state, firstLine: Math.max(1, record.totalLines - rows + 1) }
    }
    case 'nextRecord': {
      if (!state.open || state.records.length === 0) return state
      const next = (state.index + 1) % state.records.length
      return { ...state, index: next, firstLine: 1 }
    }
    case 'previousRecord': {
      if (!state.open || state.records.length === 0) return state
      const previous = (state.index - 1 + state.records.length) % state.records.length
      return { ...state, index: previous, firstLine: 1 }
    }
    case 'recordsChanged': {
      // A cleared generation has no retained record left: close instead of
      // rendering an empty overlay. A late old-generation list cannot revive
      // a closed viewer.
      if (action.records.length === 0) {
        return { ...state, open: false, records: [], index: 0, firstLine: 1 }
      }
      if (!state.open) return { ...state, records: action.records, index: 0, firstLine: 1 }
      const index = clampIndex(state.index, action.records.length)
      return { ...state, records: action.records, index, firstLine: 1 }
    }
    default:
      return state
  }
}

/** Lines of the current record a viewport of `viewportRows` should render. */
// [maint-note:8bb0a7ee6982]
// 维护导航：detailViewerPage；下列为本函数源码中的接口与路径索引。
// 输入 state：DetailViewerState。
// 输入 lines：readonly string[]。
// 输入 viewportRows：number。
// 返回契约：{ start: number; end: number; total: number }。
// 分支条件：total === 0。
// 返回路径：{ start: 1, end: 0, total }。
// 返回路径：{ start: first, end: Math.min(total, first + rows - 1), total }。
// 直接调用：Math.max。
// 直接调用：Math.min。
// [/maint-note:8bb0a7ee6982]
export function detailViewerPage(
  state: DetailViewerState,
  lines: readonly string[],
  viewportRows: number,
): { start: number; end: number; total: number } {
  const total = lines.length
  const rows = Math.max(1, viewportRows)
  if (total === 0) return { start: 1, end: 0, total }
  const maxFirst = Math.max(1, total - rows + 1)
  const first = Math.min(Math.max(1, state.firstLine), maxFirst)
  return { start: first, end: Math.min(total, first + rows - 1), total }
}
