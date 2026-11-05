/**
 * detail-viewer controller (pure state machine). The viewer is a dynamic
 * overlay in the interactive region: opening, paging, switching and closing
 * never rewrite the committed stable history. Every transition is a bounded,
 * synchronous reducer step — no timers, no filesystem, no tool re-runs.
 */

import type { StableBatch } from './messageController.js'
import { detailStoreGet, type DetailStoreState } from './detailStore.js'
import { wrapDisplayLines } from '../terminal/index.js'

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

export interface ProjectedDetailViewerRecord extends DetailViewerRecord {
  logicalLines: number
  visualLines: number
}

/**
 * Build viewer candidates from light stable-message metadata plus the one
 * bounded detail store. Missing entries are intentionally retained as
 * explicit eviction records; projection never recreates their heavy body.
 */
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

export interface DetailViewerState {
  open: boolean
  /** Records with retained detail, newest-last; Tab walks this list. */
  records: readonly DetailViewerRecord[]
  index: number
  firstLine: number
  /** Bumped on open: the viewer re-clamps pages when records change. */
  generation: number
}

export interface DetailViewerSnapshotOptions {
  viewportRows: number
  totalLines: number
}

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

function clampIndex(index: number, count: number): number {
  if (count === 0) return 0
  return Math.min(Math.max(0, index), count - 1)
}

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
