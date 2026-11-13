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

// [maint-note:697a65ac28cf]
// 数据契约导航：DiffRecordKind；字段含义以消费端校验为准。
// 类型组合：'context' | 'added' | 'removed'。
// [/maint-note:697a65ac28cf]
export type DiffRecordKind = 'context' | 'added' | 'removed'

// [maint-note:82fc816bc1d2]
// 数据契约导航：DiffRecord；字段含义以消费端校验为准。
// 成员 kind：DiffRecordKind。
// 成员 text：string。
// 成员 oldLine（可省略）：number。
// 成员 newLine（可省略）：number。
// [/maint-note:82fc816bc1d2]
export interface DiffRecord {
  kind: DiffRecordKind
  text: string
  oldLine?: number
  newLine?: number
}

// [maint-note:b3cffae37661]
// 数据契约导航：DiffFile；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 action：'add' | 'update' | 'delete'。
// 成员 additions：number。
// 成员 deletions：number。
// 成员 records：readonly DiffRecord[]。
// [/maint-note:b3cffae37661]
export interface DiffFile {
  path: string
  action: 'add' | 'update' | 'delete'
  additions: number
  deletions: number
  records: readonly DiffRecord[]
}

// [maint-note:27578e22ca5d]
// 数据契约导航：DiffPayload；字段含义以消费端校验为准。
// 成员 files：readonly DiffFile[]。
// 成员 displayTruncated（可省略）：boolean。
// 成员 sourceTruncated（可省略）：boolean。
// [/maint-note:27578e22ca5d]
export interface DiffPayload {
  files: readonly DiffFile[]
  displayTruncated?: boolean
  /** Canonical ToolResult already reported truncation upstream. */
  sourceTruncated?: boolean
}

/** One logical (unwrapped) row of the projected body. */
// [maint-note:2bf3abc93d23]
// 数据契约导航：DiffRow；字段含义以消费端校验为准。
// 成员 kind：DiffRecordKind | 'header' | 'omitted' | 'notice'。
// 成员 lineNumber：string。
// 成员 marker：string。
// 成员 text：string。
// [/maint-note:2bf3abc93d23]
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
// [maint-note:ee254f7412ea]
// 数据契约导航：DiffVisualLine；字段含义以消费端校验为准。
// 成员 kind：DiffRow['kind']。
// 成员 lineNumber：string。
// 成员 marker：string。
// 成员 text：string。
// 成员 continuation：boolean。
// [/maint-note:ee254f7412ea]
export interface DiffVisualLine {
  kind: DiffRow['kind']
  lineNumber: string
  marker: string
  text: string
  /** Soft-wrap continuation: blank gutter, no repeated line number. */
  continuation: boolean
}

// [maint-note:74b74853fb6c]
// 数据契约导航：DiffView；字段含义以消费端校验为准。
// 成员 lines：readonly DiffVisualLine[]。
// 成员 physicalRows：number。
// 成员 characters：number。
// 成员 viewCollapsed：boolean。
// 成员 hiddenRecords：number。
// 成员 continuationHidden：boolean。
// 成员 sourceTruncated：boolean。
// 成员 totals：{ files: number; additions: number; deletions: number }。
// [/maint-note:74b74853fb6c]
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

// [maint-note:05d74fe56dc1]
// 数据契约导航：DiffLogicalProjection；字段含义以消费端校验为准。
// 成员 rows：readonly DiffRow[]。
// 成员 numberWidth：number。
// 成员 totalRecords：number。
// 成员 collapsedRecords：number。
// 成员 sourceTruncated：boolean。
// 成员 totals：{ files: number; additions: number; deletions: number }。
// [/maint-note:05d74fe56dc1]
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
// [maint-note:bd9371f02f0b]
// 维护导航：COLLAPSE_NOTICE；下列为本函数源码中的接口与路径索引。
// 输入 hidden：number。
// 返回契约：string。
// [/maint-note:bd9371f02f0b]
const COLLAPSE_NOTICE = (hidden: number): string => `${hidden} diff rows hidden · Ctrl+O view details`
const CONTINUATION_NOTICE = 'continuation hidden'

/**
 * Collapse consecutive context runs to one record before and one after each
 * modification group. Overlapping/neighbouring windows merge; exactly one ⋮
 * row separates hidden regions; the elision marker always precedes the
 * retained leading context and follows the retained trailing context; no
 * elision is emitted when nothing is hidden at a file edge.
 */
// [maint-note:444b65d40b67]
// 维护导航：collapseContext；下列为本函数源码中的接口与路径索引。
// 输入 records：readonly DiffRecord[]。
// 返回契约：Array<DiffRecord | { kind: 'elided'; count: number }>。
// 分支条件：total === 0。
// 分支条件：leadEnd === total。
// 分支条件：leadEnd > 1。
// 分支条件：leadEnd === 1。
// 分支条件：index >= total。
// 分支条件：runLength === 1。
// 分支条件：runLength > 1。
// 分支条件：runLength <= 2。
// 遍历边界：leadEnd < total && isContext(records[leadEnd]!)。
// 遍历边界：index < total。
// 遍历边界：index < total && !isContext(records[index]!)。
// 遍历边界：let cursor = groupStart; cursor < index; cursor += 1。
// 遍历边界：index < total && isContext(records[index]!)。
// 遍历边界：let cursor = runStart; cursor < index; cursor += 1。
// 返回路径：output。
// 写入目标：leadEnd。
// 写入目标：index。
// 写入目标：cursor。
// 直接调用：isContext。
// 直接调用：output.push。
// 直接调用：pushElided。
// [/maint-note:444b65d40b67]
export function collapseContext(records: readonly DiffRecord[]): Array<DiffRecord | { kind: 'elided'; count: number }> {
  const output: Array<DiffRecord | { kind: 'elided'; count: number }> = []
  const total = records.length
  if (total === 0) return output
  // [maint-note:769ef31f6209]
  // 维护导航：isContext；下列为本函数源码中的接口与路径索引。
  // 输入 record：DiffRecord。
  // [/maint-note:769ef31f6209]
  const isContext = (record: DiffRecord) => record.kind === 'context'

  // [maint-note:1efe7bf4a95e]
  // 维护导航：pushElided；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 分支条件：count <= 0。
  // 返回路径：无返回值。
  // 直接调用：output.push。
  // [/maint-note:1efe7bf4a95e]
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

// [maint-note:2177fdc294b2]
// 维护导航：maxLineNumber；下列为本函数源码中的接口与路径索引。
// 输入 payload：DiffPayload。
// 返回契约：number。
// 分支条件：record.newLine !== undefined && record.newLine > max。
// 分支条件：record.oldLine !== undefined && record.oldLine > max。
// 遍历边界：const file ← payload.files。
// 遍历边界：const record ← file.records。
// 返回路径：max。
// 写入目标：max。
// [/maint-note:2177fdc294b2]
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
// [maint-note:f333aa11c42c]
// 维护导航：projectDiffRows；下列为本函数源码中的接口与路径索引。
// 输入 payload：DiffPayload。
// 输入 options：{ collapse?: boolean }；缺省表达式 {}。
// 返回契约：DiffLogicalProjection。
// 分支条件：entry.kind === 'elided'。
// 分支条件：fileIndex < payload.files.length - 1。
// 遍历边界：const [fileIndex, file] ← payload.files.entries()。
// 遍历边界：const entry ← entries。
// 返回路径：{ rows, numberWidth, totalRecords, collapsedRecords, sourceTruncated: payload.sourceTruncated === true || payload.displayTruncated === true, totals: { files: payload.files.length, additions: payload.files.reduce((sum, file) => sum + file.ad。
// 写入目标：collapsedRecords。
// 写入目标：visibleRecords。
// 直接调用：Math.max。
// 直接调用：String。
// 直接调用：maxLineNumber。
// 直接调用：payload.files.entries。
// 直接调用：rows.push。
// 直接调用：' '.repeat。
// 直接调用：sanitizeTerminalText。
// 直接调用：collapseContext。
// 直接调用：file.records.map。
// 直接调用：numberColumn。
// 直接调用：payload.files.reduce。
// [/maint-note:f333aa11c42c]
export function projectDiffRows(payload: DiffPayload, options: { collapse?: boolean } = {}): DiffLogicalProjection {
  const collapse = options.collapse !== false
  const numberWidth = Math.max(1, String(maxLineNumber(payload)).length)
  const rows: DiffRow[] = []
  let visibleRecords = 0
  let collapsedRecords = 0

  // [maint-note:530adea4a0a4]
  // 维护导航：numberColumn；下列为本函数源码中的接口与路径索引。
  // 输入 value：number | undefined。
  // 返回契约：string。
  // 直接调用：' '.repeat。
  // 直接调用：String(value).padStart。
  // 直接调用：String。
  // [/maint-note:530adea4a0a4]
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
// [maint-note:1eaabf5838a6]
// 维护导航：lineCost；下列为本函数源码中的接口与路径索引。
// 输入 line：{ lineNumber: string; marker: string; text: string }。
// 返回契约：number。
// 返回路径：line.lineNumber.length + GUTTER_SEPARATOR.length + line.marker.length + 1 + line.text.length + 1。
// [/maint-note:1eaabf5838a6]
function lineCost(line: { lineNumber: string; marker: string; text: string }): number {
  return line.lineNumber.length + GUTTER_SEPARATOR.length + line.marker.length + 1 + line.text.length + 1
}

// [maint-note:019058296126]
// 维护导航：serializeLine；下列为本函数源码中的接口与路径索引。
// 输入 line：{ lineNumber: string; marker: string; text: string }。
// 返回契约：string。
// 返回路径：`${line.lineNumber}${GUTTER_SEPARATOR}${line.marker} ${line.text}`。
// [/maint-note:019058296126]
function serializeLine(line: { lineNumber: string; marker: string; text: string }): string {
  return `${line.lineNumber}${GUTTER_SEPARATOR}${line.marker} ${line.text}`
}

/**
 * Wrap one logical row into physical text segments at `textWidth` using the
 * display-width helper (grapheme-safe, control-safe). The first segment
 * keeps the row's line number and marker; continuations get a blank gutter.
 */
// [maint-note:d02e7387f5fb]
// 维护导航：wrapDiffRow；下列为本函数源码中的接口与路径索引。
// 输入 row：DiffRow。
// 输入 contentWidth：number。
// 返回契约：readonly DiffVisualLine[]。
// 返回路径：segments.map((text, index) => ({ kind: row.kind, lineNumber: index === 0 ? row.lineNumber : ' '.repeat(numberWidth), marker: index === 0 ? row.marker : ' ', text, continuation: index > 0, }))。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines。
// 直接调用：segments.map。
// [/maint-note:d02e7387f5fb]
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
// [maint-note:a678f0e7715a]
// 维护导航：finalizeDiffView；下列为本函数源码中的接口与路径索引。
// 输入 projection：DiffLogicalProjection。
// 输入 contentWidth：number。
// 输入 options：{ maxRows?: number; maxChars?: number }；缺省表达式 {}。
// 返回契约：DiffView。
// 分支条件：maxRows <= 0 || maxChars <= 0。
// 分支条件：attempt.hidden === 0 && !attempt.continuationHidden。
// 分支条件：nextAttempt.hidden === expectedHidden && continuationSeen === expectedContinuation。
// 遍历边界：let iteration = 0; iteration < 64; iteration += 1。
// 返回路径：{ lines: [], physicalRows: 0, characters: 0, viewCollapsed: totalRecords > 0 || projection.rows.length > 0, hiddenRecords: totalRecords, continuationHidden: false, sourceTruncated: projection.sourceTruncated, totals: projection.totals, }。
// 返回路径：{ lines: attempt.lines, physicalRows: attempt.lines.length, characters: attempt.chars, viewCollapsed: false, hiddenRecords: 0, continuationHidden: false, sourceTruncated: projection.sourceTruncated, totals: projection.totals, }。
// 返回路径：{ lines, physicalRows: lines.length, characters, viewCollapsed: true, hiddenRecords: attempt.hidden, continuationHidden: attempt.continuationHidden, sourceTruncated: projection.sourceTruncated, totals: projection.totals, }。
// 写入目标：iteration。
// 写入目标：continuationSeen。
// 写入目标：notices。
// 写入目标：attempt。
// 直接调用：build。
// 直接调用：boundedNotices。
// 直接调用：lines.reduce。
// [/maint-note:a678f0e7715a]
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

  // [maint-note:75e7eea5cd6c]
  // 数据契约导航：BuildResult；字段含义以消费端校验为准。
  // 成员 lines：DiffVisualLine[]。
  // 成员 hidden：number。
  // 成员 continuationHidden：boolean。
  // 成员 chars：number。
  // [/maint-note:75e7eea5cd6c]
  interface BuildResult {
    lines: DiffVisualLine[]
    hidden: number
    continuationHidden: boolean
    chars: number
  }

  // [maint-note:75bb3a91a518]
  // 维护导航：noticeRow；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：DiffRow。
  // 直接调用：' '.repeat。
  // [/maint-note:75bb3a91a518]
  const noticeRow = (text: string): DiffRow => ({
    kind: 'notice',
    lineNumber: ' '.repeat(projection.numberWidth),
    marker: VERTICAL_ELLIPSIS,
    text,
  })

  // [maint-note:6cdd7af5c4cc]
  // 维护导航：wrapNotice；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：readonly DiffVisualLine[]。
  // 直接调用：wrapDiffRow。
  // 直接调用：noticeRow。
  // [/maint-note:6cdd7af5c4cc]
  const wrapNotice = (text: string): readonly DiffVisualLine[] => wrapDiffRow(noticeRow(text), contentWidth)

  // [maint-note:e69b3692765b]
  // 维护导航：boundedNotices；下列为本函数源码中的接口与路径索引。
  // 输入 hidden：number。
  // 输入 continuation：boolean。
  // 返回契约：readonly DiffVisualLine[]。
  // 分支条件：wrapped.length <= maxRows && chars <= maxChars。
  // 遍历边界：const candidate ← candidates。
  // 返回路径：wrapped。
  // 返回路径：[]。
  // 直接调用：COLLAPSE_NOTICE。
  // 直接调用：String。
  // 直接调用：candidate.flatMap。
  // 直接调用：wrapped.reduce。
  // [/maint-note:e69b3692765b]
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
  // [maint-note:30ee0cf50371]
  // 维护导航：build；下列为本函数源码中的接口与路径索引。
  // 输入 notices：readonly DiffVisualLine[]。
  // 返回契约：BuildResult。
  // 分支条件：lines.length + reserveRows >= maxRows || chars + cost + reserveChars > maxChars。
  // 分支条件：index > 0。
  // 分支条件：index === 0 && isRecord。
  // 遍历边界：const row ← projection.rows。
  // 遍历边界：const [index, line] ← wrapped.entries()。
  // 返回路径：{ lines, hidden: Math.max(0, totalRecords - visibleRecords), continuationHidden, chars }。
  // 写入目标：continuationHidden。
  // 写入目标：visibleRecords。
  // 写入目标：chars。
  // 直接调用：notices.reduce。
  // 直接调用：wrapDiffRow。
  // 直接调用：wrapped.entries。
  // 直接调用：lineCost。
  // 直接调用：Math.max。
  // 直接调用：lines.push。
  // [/maint-note:30ee0cf50371]
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
// [maint-note:858ad70cf35f]
// 维护导航：renderDiffLinesAsText；下列为本函数源码中的接口与路径索引。
// 输入 lines：readonly DiffVisualLine[]。
// 返回契约：string。
// 返回路径：lines.map((line) => serializeLine(line)).join('\n')。
// 直接调用：lines.map((line) => serializeLine(line)).join。
// 直接调用：lines.map。
// [/maint-note:858ad70cf35f]
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
// [maint-note:deb46cb883ef]
// 维护导航：renderDiffRowsAsText；下列为本函数源码中的接口与路径索引。
// 输入 rows：readonly DiffRow[]。
// 返回契约：string。
// 返回路径：rows.map((row) => serializeLine(row)).join('\n')。
// 直接调用：rows.map((row) => serializeLine(row)).join。
// 直接调用：rows.map。
// [/maint-note:deb46cb883ef]
export function renderDiffRowsAsText(rows: readonly DiffRow[]): string {
  return rows.map((row) => serializeLine(row)).join('\n')
}

/** Physical row count of committed visual lines at a (possibly new) width. */
// [maint-note:30ab2748a5c0]
// 维护导航：countDiffVisualLines；下列为本函数源码中的接口与路径索引。
// 输入 lines：readonly DiffVisualLine[]。
// 输入 contentWidth：number。
// 返回契约：number。
// 遍历边界：const line ← lines。
// 返回路径：count。
// 写入目标：count。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines。
// [/maint-note:30ab2748a5c0]
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
