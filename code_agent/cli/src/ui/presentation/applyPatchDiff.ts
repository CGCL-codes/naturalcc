import { sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'
import { UiPresentationLimits } from './limits.js'

/**
 * structured diff projection. The pipeline has two stages that share one
 * row ledger:
 *
 * 1. `projectDiffRows` — width-independent LOGICAL projection: context
 *    collapse (one line before/after each modification group) and the single
 *    hidden-record set. Nothing here looks at terminal width.
 * 2. `finalizeDiffView` — the FINAL visual ledger at a concrete content
 *    width: grapheme-safe soft wrap with blank continuation gutters, and the
 *    first-reached-wins physical-row / character budgets with the collapse
 *    notice reserved BEFORE filling.
 *
 * Renderers, the frame ledger and plainText all consume stage-2 output;
 * the detail source keeps the full stage-1 rows. Nothing is parsed back out
 * of formatted text, and the canonical patch/ToolResult is never rewritten.
 */

export type DiffRecordKind = 'context' | 'added' | 'removed'

export interface DiffRecord {
  kind: DiffRecordKind
  text: string
  oldLine?: number
  newLine?: number
}

export interface DiffFile {
  path: string
  action: 'add' | 'update' | 'delete'
  additions: number
  deletions: number
  records: readonly DiffRecord[]
}

export interface DiffPayload {
  files: readonly DiffFile[]
  displayTruncated?: boolean
  /** Canonical ToolResult already reported truncation upstream. */
  sourceTruncated?: boolean
}

/** One logical (unwrapped) row of the projected body. */
export interface DiffRow {
  kind: DiffRecordKind | 'header' | 'omitted' | 'notice'
  /** Rendered line-number column (right-aligned) or spaces. */
  lineNumber: string
  /** Exactly one column: '+', '-', ' ', '⋮', or a space for headers. */
  marker: string
  /** Visible source/header text after the single UI separator space. */
  text: string
}

/** One physical visible line after wrap + budget folding. */
export interface DiffVisualLine {
  kind: DiffRow['kind']
  lineNumber: string
  marker: string
  text: string
  /** Soft-wrap continuation: blank gutter, no repeated line number. */
  continuation: boolean
}

export interface DiffView {
  /** Final visible physical lines (≤ budgets). */
  lines: readonly DiffVisualLine[]
  physicalRows: number
  characters: number
  viewCollapsed: boolean
  /** Total logical source records hidden by collapse or budget, counted once. */
  hiddenRecords: number
  /** A single logical line was cut mid-wrap by the row budget. */
  continuationHidden: boolean
  sourceTruncated: boolean
  totals: { files: number; additions: number; deletions: number }
}

export interface DiffLogicalProjection {
  /** Context-collapsed logical rows; the full detail source renders these. */
  rows: readonly DiffRow[]
  /** Digits of the widest applicable line number; stable per projection. */
  numberWidth: number
  /** Logical source records retained in the payload. */
  totalRecords: number
  /** Records already hidden by context collapse (subset of hiddenRecords). */
  collapsedRecords: number
  sourceTruncated: boolean
  totals: { files: number; additions: number; deletions: number }
}

const VERTICAL_ELLIPSIS = '⋮'
const COLLAPSE_NOTICE = (hidden: number): string => `${hidden} diff rows hidden · Ctrl+O view details`
const CONTINUATION_NOTICE = 'continuation hidden'

/**
 * Collapse consecutive context runs to one record before and one after each
 * modification group. Overlapping/neighbouring windows merge; exactly one ⋮
 * row separates hidden regions; the elision marker always precedes the
 * retained leading context and follows the retained trailing context; no
 * elision is emitted when nothing is hidden at a file edge.
 */
export function collapseContext(records: readonly DiffRecord[]): Array<DiffRecord | { kind: 'elided'; count: number }> {
  const output: Array<DiffRecord | { kind: 'elided'; count: number }> = []
  const total = records.length
  if (total === 0) return output
  const isContext = (record: DiffRecord) => record.kind === 'context'

  const pushElided = (count: number) => {
    if (count <= 0) return
    output.push({ kind: 'elided', count })
  }

  // Leading context before the first modification group: hide the earlier
  // lines FIRST, then keep the single record adjacent to the group.
  let leadEnd = 0
  while (leadEnd < total && isContext(records[leadEnd]!)) leadEnd += 1
  if (leadEnd === total) {
    // Pure-context file: keep the single first record, elide the rest.
    output.push(records[0]!)
    pushElided(total - 1)
    return output
  }
  if (leadEnd > 1) {
    pushElided(leadEnd - 1)
    output.push(records[leadEnd - 1]!)
  } else if (leadEnd === 1) {
    output.push(records[0]!)
  }
  let index = leadEnd

  while (index < total) {
    // A modification group: consecutive added/removed records, kept intact.
    const groupStart = index
    while (index < total && !isContext(records[index]!)) index += 1
    for (let cursor = groupStart; cursor < index; cursor += 1) output.push(records[cursor]!)

    // Trailing context of this group / leading context of the next.
    const runStart = index
    while (index < total && isContext(records[index]!)) index += 1
    const runLength = index - runStart
    if (index >= total) {
      // File tail: keep the record adjacent to the group, then elide the rest.
      if (runLength === 1) {
        output.push(records[runStart]!)
      } else if (runLength > 1) {
        output.push(records[runStart]!)
        pushElided(runLength - 1)
      }
      return output
    }
    if (runLength <= 2) {
      for (let cursor = runStart; cursor < index; cursor += 1) output.push(records[cursor]!)
    } else {
      // Keep the first (this group's after-context) and the last (the next
      // group's before-context); elide the middle in one marker.
      output.push(records[runStart]!)
      pushElided(runLength - 2)
      output.push(records[index - 1]!)
    }
  }
  return output
}

function maxLineNumber(payload: DiffPayload): number {
  let max = 1
  for (const file of payload.files) {
    for (const record of file.records) {
      if (record.newLine !== undefined && record.newLine > max) max = record.newLine
      if (record.oldLine !== undefined && record.oldLine > max) max = record.oldLine
    }
  }
  return max
}

/**
 * Stage 1: width-independent logical projection with context collapse.
 * `hidden` is the SINGLE set of hidden records — the difference between
 * retained source records and visible logical rows.
 */
export function projectDiffRows(payload: DiffPayload, options: { collapse?: boolean } = {}): DiffLogicalProjection {
  const collapse = options.collapse !== false
  const numberWidth = Math.max(1, String(maxLineNumber(payload)).length)
  const rows: DiffRow[] = []
  let visibleRecords = 0
  let collapsedRecords = 0

  const numberColumn = (value: number | undefined): string =>
    value === undefined
      ? ' '.repeat(numberWidth)
      : String(value).padStart(numberWidth, ' ')

  for (const [fileIndex, file] of payload.files.entries()) {
    rows.push({
      kind: 'header',
      lineNumber: ' '.repeat(numberWidth),
      marker: ' ',
      text: sanitizeTerminalText(`${file.action} ${file.path} (+${file.additions}/-${file.deletions})`),
    })
    const entries: Array<DiffRecord | { kind: 'elided'; count: number }> = collapse
      ? collapseContext(file.records)
      : file.records.map((record) => record as DiffRecord)
    for (const entry of entries) {
      if (entry.kind === 'elided') {
        collapsedRecords += entry.count
        rows.push({
          kind: 'omitted',
          lineNumber: ' '.repeat(numberWidth),
          marker: VERTICAL_ELLIPSIS,
          text: `${entry.count} unchanged lines`,
        })
        continue
      }
      visibleRecords += 1
      rows.push({
        kind: entry.kind,
        lineNumber: numberColumn(entry.newLine ?? entry.oldLine),
        marker: entry.kind === 'added' ? '+' : entry.kind === 'removed' ? '-' : ' ',
        text: sanitizeTerminalText(entry.text),
      })
    }
    if (fileIndex < payload.files.length - 1) {
      rows.push({ kind: 'notice', lineNumber: ' '.repeat(numberWidth), marker: ' ', text: '' })
    }
  }

  const totalRecords = payload.files.reduce((sum, file) => sum + file.records.length, 0)
  return {
    rows,
    numberWidth,
    totalRecords,
    collapsedRecords,
    sourceTruncated: payload.sourceTruncated === true || payload.displayTruncated === true,
    totals: {
      files: payload.files.length,
      additions: payload.files.reduce((sum, file) => sum + file.additions, 0),
      deletions: payload.files.reduce((sum, file) => sum + file.deletions, 0),
    },
  }
}

const GUTTER_SEPARATOR = ' │ '

/** Exact serialized cost of one physical line, newline included. */
function lineCost(line: { lineNumber: string; marker: string; text: string }): number {
  return line.lineNumber.length + GUTTER_SEPARATOR.length + line.marker.length + 1 + line.text.length + 1
}

function serializeLine(line: { lineNumber: string; marker: string; text: string }): string {
  return `${line.lineNumber}${GUTTER_SEPARATOR}${line.marker} ${line.text}`
}

/**
 * Wrap one logical row into physical text segments at `textWidth` using the
 * display-width helper (grapheme-safe, control-safe). The first segment
 * keeps the row's line number and marker; continuations get a blank gutter.
 */
export function wrapDiffRow(row: DiffRow, contentWidth: number): readonly DiffVisualLine[] {
  const numberWidth = row.lineNumber.length
  const gutter = numberWidth + GUTTER_SEPARATOR.length + 2
  const textWidth = Math.max(1, contentWidth - gutter)
  const segments = row.text === ''
    ? ['']
    : wrapDisplayLines(row.text, textWidth, { hard: true, trim: false, wordWrap: false })
  return segments.map((text, index) => ({
    kind: row.kind,
    lineNumber: index === 0 ? row.lineNumber : ' '.repeat(numberWidth),
    marker: index === 0 ? row.marker : ' ',
    text,
    continuation: index > 0,
  }))
}

/**
 * Stage 2: the final visual ledger. Wraps every logical row, then enforces
 * the 40-physical-row / 6,000-character budgets with the collapse notice
 * reserved up front: the notice line and its characters are deducted from
 * the budget BEFORE source rows are admitted, so `exact 40` really means
 * the visible body (notice included) fits.
 */
export function finalizeDiffView(
  projection: DiffLogicalProjection,
  contentWidth: number,
  options: { maxRows?: number; maxChars?: number } = {},
): DiffView {
  const maxRows = options.maxRows ?? UiPresentationLimits.applyPatchDefaultPhysicalRows
  const maxChars = options.maxChars ?? UiPresentationLimits.applyPatchMaxChars
  const totalRecords = projection.totalRecords

  if (maxRows <= 0 || maxChars <= 0) {
    return {
      lines: [],
      physicalRows: 0,
      characters: 0,
      viewCollapsed: totalRecords > 0 || projection.rows.length > 0,
      hiddenRecords: totalRecords,
      continuationHidden: false,
      sourceTruncated: projection.sourceTruncated,
      totals: projection.totals,
    }
  }

  interface BuildResult {
    lines: DiffVisualLine[]
    hidden: number
    continuationHidden: boolean
    chars: number
  }

  const noticeRow = (text: string): DiffRow => ({
    kind: 'notice',
    lineNumber: ' '.repeat(projection.numberWidth),
    marker: VERTICAL_ELLIPSIS,
    text,
  })

  const wrapNotice = (text: string): readonly DiffVisualLine[] => wrapDiffRow(noticeRow(text), contentWidth)

  const boundedNotices = (hidden: number, continuation: boolean): readonly DiffVisualLine[] => {
    const full = [
      ...(hidden > 0 ? [COLLAPSE_NOTICE(hidden)] : []),
      ...(continuation ? [CONTINUATION_NOTICE] : []),
    ]
    const candidates: readonly (readonly string[])[] = [
      full,
      [
        ...(hidden > 0 ? [`${hidden} hidden · Ctrl+O`] : []),
        ...(continuation ? ['continuation hidden'] : []),
      ],
      [
        ...(hidden > 0 ? [`${hidden} hidden`] : []),
        ...(continuation ? ['more hidden'] : []),
      ],
      hidden > 0 ? [String(hidden)] : continuation ? ['more'] : [],
    ]
    for (const candidate of candidates) {
      const wrapped = candidate.flatMap((part) => wrapNotice(part))
      const chars = wrapped.reduce((sum, line) => sum + lineCost(line), 0)
      if (wrapped.length <= maxRows && chars <= maxChars) return wrapped
    }
    // If even the shortest truthful marker cannot fit, render no marker.
    // Metadata still records the loss; appending an oversized marker would
    // violate the hard ledger and let Ink invent uncounted physical rows.
    return []
  }

  /**
   * `notices` is the exact set of trailing marker lines whose rows/characters
   * are deducted BEFORE source lines are admitted, so `exact 40` means the
   * whole visible body — notices included — fits the budgets.
   */
  const build = (notices: readonly DiffVisualLine[]): BuildResult => {
    const lines: DiffVisualLine[] = []
    let visibleRecords = 0
    let continuationHidden = false
    let chars = 0
    const reserveRows = notices.length
    const reserveChars = notices.reduce((sum, notice) => sum + lineCost(notice), 0)

    for (const row of projection.rows) {
      const wrapped = wrapDiffRow(row, contentWidth)
      const isRecord = row.kind === 'context' || row.kind === 'added' || row.kind === 'removed'
      for (const [index, line] of wrapped.entries()) {
        const cost = lineCost(line)
        if (lines.length + reserveRows >= maxRows || chars + cost + reserveChars > maxChars) {
          // A record cut after its first physical line still counts as
          // visible once; the mid-line cut surfaces as continuationHidden.
          if (index > 0) continuationHidden = true
          return { lines, hidden: Math.max(0, totalRecords - visibleRecords), continuationHidden, chars }
        }
        if (index === 0 && isRecord) visibleRecords += 1
        lines.push(line)
        chars += cost
      }
    }
    return { lines, hidden: Math.max(0, totalRecords - visibleRecords), continuationHidden, chars }
  }

  // First attempt with nothing reserved: maybe the whole body fits.
  let attempt = build([])
  if (attempt.hidden === 0 && !attempt.continuationHidden) {
    return {
      lines: attempt.lines,
      physicalRows: attempt.lines.length,
      characters: attempt.chars,
      viewCollapsed: false,
      hiddenRecords: 0,
      continuationHidden: false,
      sourceTruncated: projection.sourceTruncated,
      totals: projection.totals,
    }
  }

  // Fold something. Notices are physical rows too, so wrap and reserve them
  // through the same ledger. Rebuild to a fixed point because reserving a
  // wrapped notice can hide more records and therefore change its exact
  // count (and occasionally its wrapped height).
  let notices: readonly DiffVisualLine[] = []
  let continuationSeen = attempt.continuationHidden
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const expectedHidden = attempt.hidden
    const expectedContinuation = continuationSeen
    const nextNotices = boundedNotices(expectedHidden, expectedContinuation)
    const nextAttempt = build(nextNotices)
    continuationSeen ||= nextAttempt.continuationHidden
    notices = nextNotices
    attempt = nextAttempt
    if (nextAttempt.hidden === expectedHidden && continuationSeen === expectedContinuation) break
  }
  const lines = [...attempt.lines, ...notices]
  const characters = lines.reduce((sum, line) => sum + lineCost(line), 0)
  return {
    lines,
    physicalRows: lines.length,
    characters,
    viewCollapsed: true,
    hiddenRecords: attempt.hidden,
    continuationHidden: attempt.continuationHidden,
    sourceTruncated: projection.sourceTruncated,
    totals: projection.totals,
  }
}

/**
 * NO_COLOR / screen-reader projection of FINAL visual lines: the colorless
 * render is exactly these serialized lines — colors are an enhancement, the
 * gutter carries the semantics.
 */
export function renderDiffLinesAsText(lines: readonly DiffVisualLine[]): string {
  return lines.map((line) => serializeLine(line)).join('\n')
}

/**
 * Full unfolded NO_COLOR projection of the LOGICAL rows — the bounded detail
 * source retained for the viewer. Callers pass the UNCANCELED projection
 * (collapse:false) so every retained record — including context hidden by
 * the default fold — is reachable from the viewer. One line per logical
 * row, never wrapped: the viewer re-wraps at its own width.
 */
export function renderDiffRowsAsText(rows: readonly DiffRow[]): string {
  return rows.map((row) => serializeLine(row)).join('\n')
}

/** Physical row count of committed visual lines at a (possibly new) width. */
export function countDiffVisualLines(lines: readonly DiffVisualLine[], contentWidth: number): number {
  let count = 0
  for (const line of lines) {
    const gutter = line.lineNumber.length + GUTTER_SEPARATOR.length + 2
    const textWidth = Math.max(1, contentWidth - gutter)
    count += line.text === ''
      ? 1
      : wrapDisplayLines(line.text, textWidth, { hard: true, trim: false, wordWrap: false }).length
  }
  return count
}
