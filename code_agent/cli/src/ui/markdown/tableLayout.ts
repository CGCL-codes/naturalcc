import type {
  InlineSegment,
  MarkdownTableAlignment,
  MarkdownTableBlock,
} from './renderer.js'
import {
  displayLineWidth,
  displayWidth,
  splitDisplayGraphemes,
  truncateDisplayText,
} from '../terminal/index.js'

// [maint-note:a88df9723eac]
// 数据契约导航：MarkdownTableVisualRow；字段含义以消费端校验为准。
// 成员 segments：readonly MarkdownTableVisualSegment[]。
// 成员 bold（可省略）：boolean。
// 成员 role（可省略）：MarkdownTableRowRole。
// [/maint-note:a88df9723eac]
export interface MarkdownTableVisualRow {
  readonly segments: readonly MarkdownTableVisualSegment[]
  readonly bold?: boolean
  readonly role?: MarkdownTableRowRole
}

// [maint-note:4584b6ff25ee]
// 数据契约导航：MarkdownTableRowRole；字段含义以消费端校验为准。
// 类型组合：'border' | 'header' | 'body' | 'field'。
// [/maint-note:4584b6ff25ee]
export type MarkdownTableRowRole = 'border' | 'header' | 'body' | 'field'
// [maint-note:aad09b4a1d95]
// 数据契约导航：MarkdownTableSegmentRole；字段含义以消费端校验为准。
// 类型组合：'border' | 'header' | 'auxiliary'。
// [/maint-note:aad09b4a1d95]
export type MarkdownTableSegmentRole = 'border' | 'header' | 'auxiliary'

// [maint-note:89fff8ea5f3f]
// 数据契约导航：MarkdownTableVisualSegment；字段含义以消费端校验为准。
// 成员 visualRole（可省略）：MarkdownTableSegmentRole。
// [/maint-note:89fff8ea5f3f]
export interface MarkdownTableVisualSegment extends InlineSegment {
  readonly visualRole?: MarkdownTableSegmentRole
}

const MIN_READABLE_COLUMN_WIDTH = 3
const TERMINAL_TAB_SIZE = 8

/**
 * Project one parser-backed Markdown table into final terminal rows.
 *
 * This is the sole table layout ledger: completed Ink output, Markdown
 * plain text and frame row counting consume the same padded/wrapped rows.
 * It never parses source Markdown and never turns a prior terminal frame
 * back into table input.
 */
// [maint-note:8e50dce4516c]
// 维护导航：buildMarkdownTableRows；下列为本函数源码中的接口与路径索引。
// 输入 block：MarkdownTableBlock。
// 输入 width：number。
// 返回契约：readonly MarkdownTableVisualRow[]。
// 分支条件：availableCellWidth < minimums.reduce((sum, value) => sum + value, 0)。
// 遍历边界：let rowIndex = 0; rowIndex < logicalRows.length; rowIndex += 1。
// 返回路径：buildFieldRows(block, columnCount, safeWidth)。
// 返回路径：rows。
// 写入目标：rowIndex。
// 直接调用：Number.isFinite。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：logicalRows.map。
// 直接调用：Array.from。
// 直接调用：minimums.reduce。
// 直接调用：buildFieldRows。
// 直接调用：allocateColumnWidths。
// 直接调用：buildHorizontalRule。
// 直接调用：rows.push。
// 直接调用：buildLogicalRow。
// [/maint-note:8e50dce4516c]
export function buildMarkdownTableRows(
  block: MarkdownTableBlock,
  width: number,
): readonly MarkdownTableVisualRow[] {
  const safeWidth = Number.isFinite(width) ? Math.max(1, Math.floor(width)) : 1
  const logicalRows = [block.header, ...block.rows]
  const columnCount = Math.max(
    1,
    block.alignments.length,
    ...logicalRows.map(row => row.length),
  )
  // Every cell owns one padding cell on each side; N+1 vertical rules close
  // the outer frame and divide the columns. All decoration is charged before
  // content widths are allocated.
  const decorationWidth = 3 * columnCount + 1
  const availableCellWidth = safeWidth - decorationWidth
  const minimums = Array.from({ length: columnCount }, (_, column) => Math.max(
    MIN_READABLE_COLUMN_WIDTH,
    ...logicalRows.map(row => widestCellGrapheme(row[column] ?? [])),
  ))

  if (availableCellWidth < minimums.reduce((sum, value) => sum + value, 0)) {
    return buildFieldRows(block, columnCount, safeWidth)
  }

  const desired = Array.from({ length: columnCount }, (_, column) => Math.max(
    minimums[column] ?? MIN_READABLE_COLUMN_WIDTH,
    ...logicalRows.map(row => cellDisplayWidth(row[column] ?? [])),
  ))
  const widths = allocateColumnWidths(minimums, desired, availableCellWidth)
  const rows: MarkdownTableVisualRow[] = [buildHorizontalRule(widths, 'top')]
  for (let rowIndex = 0; rowIndex < logicalRows.length; rowIndex += 1) {
    rows.push(...buildLogicalRow(
      logicalRows[rowIndex] ?? [],
      widths,
      block.alignments,
      rowIndex === 0,
    ))
    rows.push(buildHorizontalRule(widths, rowIndex === logicalRows.length - 1 ? 'bottom' : 'middle'))
  }
  return rows
}

// [maint-note:3e3b3e06fc83]
// 维护导航：buildLogicalRow；下列为本函数源码中的接口与路径索引。
// 输入 cells：readonly InlineSegment[][]。
// 输入 widths：readonly number[]。
// 输入 alignments：readonly MarkdownTableAlignment[]。
// 输入 header：boolean。
// 返回契约：MarkdownTableVisualRow[]。
// 遍历边界：let physicalRow = 0; physicalRow < physicalRowCount; physicalRow += 1。
// 遍历边界：let column = 0; column < widths.length; column += 1。
// 返回路径：rows。
// 写入目标：physicalRow。
// 写入目标：column。
// 直接调用：widths.map。
// 直接调用：Math.max。
// 直接调用：wrappedCells.map。
// 直接调用：borderSegment。
// 直接调用：appendSegment。
// 直接调用：appendSegments。
// 直接调用：padCell。
// 直接调用：rows.push。
// [/maint-note:3e3b3e06fc83]
function buildLogicalRow(
  cells: readonly InlineSegment[][],
  widths: readonly number[],
  alignments: readonly MarkdownTableAlignment[],
  header: boolean,
): MarkdownTableVisualRow[] {
  const wrappedCells = widths.map((width, column) => wrapInlineSegments(cells[column] ?? [], width))
  const physicalRowCount = Math.max(1, ...wrappedCells.map(rows => rows.length))
  const rows: MarkdownTableVisualRow[] = []

  for (let physicalRow = 0; physicalRow < physicalRowCount; physicalRow += 1) {
    const segments: MarkdownTableVisualSegment[] = [borderSegment('│')]
    for (let column = 0; column < widths.length; column += 1) {
      const cell = wrappedCells[column]?.[physicalRow]?.segments ?? []
      appendSegment(segments, { kind: 'text', text: ' ' })
      appendSegments(segments, padCell(cell, widths[column] ?? 1, alignments[column] ?? 'left'))
      appendSegment(segments, { kind: 'text', text: ' ' })
      appendSegment(segments, borderSegment('│'))
    }
    // Header emphasis is applied to content segments by the Ink consumer.
    // Keeping the row itself non-bold prevents generated border segments
    // from inheriting that emphasis.
    rows.push({ segments, role: header ? 'header' : 'body' })
  }
  return rows
}

// [maint-note:dc887e57795d]
// 维护导航：buildHorizontalRule；下列为本函数源码中的接口与路径索引。
// 输入 widths：readonly number[]。
// 输入 position：'top' | 'middle' | 'bottom'。
// 返回契约：MarkdownTableVisualRow。
// 返回路径：{ segments: [borderSegment(text)], role: 'border' }。
// 直接调用：widths.map(width => '─'.repeat(width + 2)).join。
// 直接调用：widths.map。
// 直接调用：borderSegment。
// [/maint-note:dc887e57795d]
function buildHorizontalRule(
  widths: readonly number[],
  position: 'top' | 'middle' | 'bottom',
): MarkdownTableVisualRow {
  const [left, junction, right] = position === 'top'
    ? ['┌', '┬', '┐']
    : position === 'bottom' ? ['└', '┴', '┘'] : ['├', '┼', '┤']
  const text = left + widths.map(width => '─'.repeat(width + 2)).join(junction) + right
  return { segments: [borderSegment(text)], role: 'border' }
}

// [maint-note:916f9b77a5f9]
// 维护导航：borderSegment；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：MarkdownTableVisualSegment。
// 返回路径：{ kind: 'text', text, visualRole: 'border' }。
// [/maint-note:916f9b77a5f9]
function borderSegment(text: string): MarkdownTableVisualSegment {
  return { kind: 'text', text, visualRole: 'border' }
}

// [maint-note:7a90fed45381]
// 维护导航：buildFieldRows；下列为本函数源码中的接口与路径索引。
// 输入 block：MarkdownTableBlock。
// 输入 columnCount：number。
// 输入 width：number。
// 返回契约：MarkdownTableVisualRow[]。
// 遍历边界：let column = 0; column < columnCount; column += 1。
// 遍历边界：let rowIndex = 0; rowIndex < block.rows.length; rowIndex += 1。
// 返回路径：rows。
// 写入目标：column。
// 写入目标：rowIndex。
// 直接调用：wrapInlineSegments([ { kind: 'text', text: 'columns:', visualRole: 'auxiliary' }, ], width).map。
// 直接调用：wrapInlineSegments。
// 直接调用：inlineText(header).trim。
// 直接调用：inlineText。
// 直接调用：rows.push。
// 直接调用：wrapInlineSegments([ { kind: 'text', text: `column ${column + 1}: `, visualRole: 'auxiliary' }, ...visibleHeader.map(segment => ({ ...segment, visualRole: 'header' as const })), ], width).map。
// 直接调用：visibleHeader.map。
// 直接调用：wrapInlineSegments([ { kind: 'text', text: `row ${rowIndex + 1}:`, visualRole: 'auxiliary' }, ], width).map。
// 直接调用：wrapInlineSegments([ { kind: 'text', text: `column ${column + 1}: `, visualRole: 'auxiliary' }, ...(inlineText(value) ? value : [{ kind: 'text' as const, text: '(empty)' }]), ], width).map。
// [/maint-note:7a90fed45381]
function buildFieldRows(
  block: MarkdownTableBlock,
  columnCount: number,
  width: number,
): MarkdownTableVisualRow[] {
  const rows: MarkdownTableVisualRow[] = wrapInlineSegments([
    { kind: 'text', text: 'columns:', visualRole: 'auxiliary' },
  ], width).map(row => ({ ...row, bold: true, role: 'field' }))
  for (let column = 0; column < columnCount; column += 1) {
    const header = block.header[column] ?? []
    const visibleHeader = inlineText(header).trim()
      ? header
      : [{ kind: 'text' as const, text: '(empty header)' }]
    rows.push(...wrapInlineSegments([
      { kind: 'text', text: `column ${column + 1}: `, visualRole: 'auxiliary' },
      ...visibleHeader.map(segment => ({ ...segment, visualRole: 'header' as const })),
    ], width).map(row => ({ ...row, role: 'field' as const })))
  }

  for (let rowIndex = 0; rowIndex < block.rows.length; rowIndex += 1) {
    rows.push(...wrapInlineSegments([
      { kind: 'text', text: `row ${rowIndex + 1}:`, visualRole: 'auxiliary' },
    ], width).map(row => ({ ...row, bold: true, role: 'field' as const })))
    const row = block.rows[rowIndex] ?? []
    for (let column = 0; column < columnCount; column += 1) {
      const value = row[column] ?? []
      rows.push(...wrapInlineSegments([
        { kind: 'text', text: `column ${column + 1}: `, visualRole: 'auxiliary' },
        ...(inlineText(value) ? value : [{ kind: 'text' as const, text: '(empty)' }]),
      ], width).map(row => ({ ...row, role: 'field' as const })))
    }
  }
  return rows
}

// [maint-note:cde21290587f]
// 维护导航：allocateColumnWidths；下列为本函数源码中的接口与路径索引。
// 输入 minimums：readonly number[]。
// 输入 desired：readonly number[]。
// 输入 available：number。
// 返回契约：number[]。
// 分支条件：(widths[column] ?? 1) >= (desired[column] ?? 1)。
// 分支条件：!changed。
// 遍历边界：remaining > 0。
// 遍历边界：let column = 0; column < widths.length && remaining > 0; column += 1。
// 返回路径：widths。
// 写入目标：column。
// 写入目标：widths[column]。
// 写入目标：remaining。
// 写入目标：changed。
// 直接调用：Math.max。
// 直接调用：widths.reduce。
// [/maint-note:cde21290587f]
function allocateColumnWidths(
  minimums: readonly number[],
  desired: readonly number[],
  available: number,
): number[] {
  const widths = [...minimums]
  let remaining = Math.max(0, available - widths.reduce((sum, value) => sum + value, 0))
  while (remaining > 0) {
    let changed = false
    for (let column = 0; column < widths.length && remaining > 0; column += 1) {
      if ((widths[column] ?? 1) >= (desired[column] ?? 1)) continue
      widths[column] = (widths[column] ?? 1) + 1
      remaining -= 1
      changed = true
    }
    if (!changed) break
  }
  return widths
}

// [maint-note:c1ffda3f9b22]
// 维护导航：padCell；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 输入 width：number。
// 输入 alignment：MarkdownTableAlignment。
// 返回契约：MarkdownTableVisualSegment[]。
// 返回路径：[ ...(left > 0 ? [{ kind: 'text' as const, text: ' '.repeat(left) }] : []), ...segments, ...(right > 0 ? [{ kind: 'text' as const, text: ' '.repeat(right) }] : []), ]。
// 直接调用：Math.max。
// 直接调用：displayLineWidth。
// 直接调用：inlineText。
// 直接调用：Math.floor。
// 直接调用：' '.repeat。
// [/maint-note:c1ffda3f9b22]
function padCell(
  segments: readonly MarkdownTableVisualSegment[],
  width: number,
  alignment: MarkdownTableAlignment,
): MarkdownTableVisualSegment[] {
  const padding = Math.max(0, width - displayLineWidth(inlineText(segments)))
  const left = alignment === 'right'
    ? padding
    : alignment === 'center' ? Math.floor(padding / 2) : 0
  const right = padding - left
  return [
    ...(left > 0 ? [{ kind: 'text' as const, text: ' '.repeat(left) }] : []),
    ...segments,
    ...(right > 0 ? [{ kind: 'text' as const, text: ' '.repeat(right) }] : []),
  ]
}

// [maint-note:0bfe139294f8]
// 维护导航：wrapInlineSegments；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 输入 width：number。
// 返回契约：MarkdownTableVisualRow[]。
// 分支条件：grapheme.text === '\n'。
// 分支条件：grapheme.text === '\t'。
// 分支条件：currentWidth > 0 && currentWidth + tabWidth > safeWidth。
// 分支条件：spaces > 0。
// 分支条件：currentWidth >= safeWidth。
// 分支条件：graphemeWidth > safeWidth。
// 分支条件：currentWidth > 0。
// 分支条件：currentWidth > 0 && currentWidth + graphemeWidth > safeWidth。
// 分支条件：current.length > 0 || rows.length === 0。
// 遍历边界：const grapheme ← styledGraphemes(segments)。
// 返回路径：rows。
// 写入目标：currentWidth。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：styledGraphemes。
// 直接调用：finish。
// 直接调用：Math.min。
// 直接调用：appendSegment。
// 直接调用：withText。
// 直接调用：' '.repeat。
// 直接调用：displayLineWidth。
// 直接调用：truncateDisplayText。
// 直接调用：appendSegments。
// [/maint-note:0bfe139294f8]
function wrapInlineSegments(segments: readonly MarkdownTableVisualSegment[], width: number): MarkdownTableVisualRow[] {
  const safeWidth = Math.max(1, Math.floor(width))
  const rows: MarkdownTableVisualRow[] = []
  let current: MarkdownTableVisualSegment[] = []
  let currentWidth = 0

  // [maint-note:0242c5e5b5b0]
  // 维护导航：finish；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：current。
  // 写入目标：currentWidth。
  // 直接调用：rows.push。
  // [/maint-note:0242c5e5b5b0]
  const finish = (): void => {
    rows.push({ segments: current })
    current = []
    currentWidth = 0
  }

  for (const grapheme of styledGraphemes(segments)) {
    if (grapheme.text === '\n') {
      finish()
      continue
    }
    if (grapheme.text === '\t') {
      const tabWidth = TERMINAL_TAB_SIZE - (currentWidth % TERMINAL_TAB_SIZE)
      if (currentWidth > 0 && currentWidth + tabWidth > safeWidth) finish()
      const spaces = Math.min(tabWidth, safeWidth - currentWidth)
      if (spaces > 0) {
        appendSegment(current, withText(grapheme.segments[0], ' '.repeat(spaces)))
        currentWidth += spaces
      }
      if (currentWidth >= safeWidth) finish()
      continue
    }

    const graphemeWidth = Math.max(1, displayLineWidth(grapheme.text))
    if (graphemeWidth > safeWidth) {
      if (currentWidth > 0) finish()
      appendSegment(current, withText(
        grapheme.segments[0],
        truncateDisplayText(grapheme.text, safeWidth),
      ))
      finish()
      continue
    }
    if (currentWidth > 0 && currentWidth + graphemeWidth > safeWidth) finish()
    appendSegments(current, grapheme.segments)
    currentWidth += graphemeWidth
  }
  if (current.length > 0 || rows.length === 0) finish()
  return rows
}

// [maint-note:7b5867444a7c]
// 数据契约导航：StyledGrapheme；字段含义以消费端校验为准。
// 成员 text：string。
// 成员 segments：readonly MarkdownTableVisualSegment[]。
// [/maint-note:7b5867444a7c]
interface StyledGrapheme {
  readonly text: string
  readonly segments: readonly MarkdownTableVisualSegment[]
}

/**
 * Segment the complete visible cell before mapping style spans back onto
 * each cluster. Markdown may place a style boundary between code points that
 * still form one terminal grapheme (for example emoji + ZWJ + emoji).
 */
// [maint-note:6a6f21749885]
// 维护导航：styledGraphemes；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 返回契约：StyledGrapheme[]。
// 分支条件：segment.text。
// 遍历边界：const segment ← segments。
// 遍历边界：const grapheme ← splitDisplayGraphemes(text)。
// 遍历边界：spans[spanIndex] && spans[spanIndex]!.end <= graphemeStart。
// 遍历边界：let index = spanIndex; spans[index] && spans[index]!.start < graphemeEnd; index += 1。
// 返回路径：result。
// 写入目标：text。
// 写入目标：spanIndex。
// 写入目标：index。
// 写入目标：graphemeStart。
// 直接调用：spans.push。
// 直接调用：splitDisplayGraphemes。
// 直接调用：Math.max。
// 直接调用：Math.min。
// 直接调用：appendSegment。
// 直接调用：withText。
// 直接调用：text.slice。
// 直接调用：result.push。
// [/maint-note:6a6f21749885]
function styledGraphemes(segments: readonly MarkdownTableVisualSegment[]): StyledGrapheme[] {
  const spans: Array<{ start: number; end: number; segment: MarkdownTableVisualSegment }> = []
  let text = ''
  for (const segment of segments) {
    const start = text.length
    text += segment.text
    if (segment.text) spans.push({ start, end: text.length, segment })
  }

  const result: StyledGrapheme[] = []
  let graphemeStart = 0
  let spanIndex = 0
  for (const grapheme of splitDisplayGraphemes(text)) {
    const graphemeEnd = graphemeStart + grapheme.length
    while (spans[spanIndex] && spans[spanIndex]!.end <= graphemeStart) spanIndex += 1
    const styled: MarkdownTableVisualSegment[] = []
    for (let index = spanIndex; spans[index] && spans[index]!.start < graphemeEnd; index += 1) {
      const span = spans[index]!
      const overlapStart = Math.max(graphemeStart, span.start)
      const overlapEnd = Math.min(graphemeEnd, span.end)
      appendSegment(styled, withText(
        span.segment,
        text.slice(overlapStart, overlapEnd),
      ))
    }
    result.push({
      text: grapheme,
      segments: styled.length > 0 ? styled : [{ kind: 'text', text: grapheme }],
    })
    graphemeStart = graphemeEnd
  }
  return result
}

// [maint-note:ba4083b8e3a6]
// 维护导航：withText；下列为本函数源码中的接口与路径索引。
// 输入 segment：MarkdownTableVisualSegment | undefined。
// 输入 text：string。
// 返回契约：MarkdownTableVisualSegment。
// 返回路径：segment ? { ...segment, text } : { kind: 'text', text }。
// [/maint-note:ba4083b8e3a6]
function withText(segment: MarkdownTableVisualSegment | undefined, text: string): MarkdownTableVisualSegment {
  return segment ? { ...segment, text } : { kind: 'text', text }
}

// [maint-note:175bec1f1abe]
// 维护导航：cellDisplayWidth；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 返回契约：number。
// 返回路径：displayWidth(expandDisplayTabs(inlineText(segments)))。
// 直接调用：displayWidth。
// 直接调用：expandDisplayTabs。
// 直接调用：inlineText。
// [/maint-note:175bec1f1abe]
function cellDisplayWidth(segments: readonly MarkdownTableVisualSegment[]): number {
  return displayWidth(expandDisplayTabs(inlineText(segments)))
}

// [maint-note:68c5e5fdc89d]
// 维护导航：widestCellGrapheme；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 返回契约：number。
// 返回路径：Math.max(1, ...splitDisplayGraphemes(expandDisplayTabs(inlineText(segments))) .filter(grapheme => grapheme !== '\n') .map(grapheme => displayLineWidth(grapheme)))。
// 直接调用：Math.max。
// 直接调用：splitDisplayGraphemes(expandDisplayTabs(inlineText(segments))) .filter(grapheme => grapheme !== '\n') .map。
// 直接调用：splitDisplayGraphemes(expandDisplayTabs(inlineText(segments))) .filter。
// 直接调用：splitDisplayGraphemes。
// 直接调用：expandDisplayTabs。
// 直接调用：inlineText。
// [/maint-note:68c5e5fdc89d]
function widestCellGrapheme(segments: readonly MarkdownTableVisualSegment[]): number {
  return Math.max(1, ...splitDisplayGraphemes(expandDisplayTabs(inlineText(segments)))
    .filter(grapheme => grapheme !== '\n')
    .map(grapheme => displayLineWidth(grapheme)))
}

// [maint-note:552346b309eb]
// 维护导航：expandDisplayTabs；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 分支条件：grapheme === '\n'。
// 分支条件：grapheme === '\t'。
// 遍历边界：const grapheme ← splitDisplayGraphemes(value)。
// 返回路径：result。
// 写入目标：result。
// 写入目标：column。
// 直接调用：splitDisplayGraphemes。
// 直接调用：' '.repeat。
// 直接调用：Math.max。
// 直接调用：displayLineWidth。
// [/maint-note:552346b309eb]
function expandDisplayTabs(value: string): string {
  let column = 0
  let result = ''
  for (const grapheme of splitDisplayGraphemes(value)) {
    if (grapheme === '\n') {
      result += grapheme
      column = 0
      continue
    }
    if (grapheme === '\t') {
      const spaces = TERMINAL_TAB_SIZE - (column % TERMINAL_TAB_SIZE)
      result += ' '.repeat(spaces)
      column += spaces
      continue
    }
    result += grapheme
    column += Math.max(1, displayLineWidth(grapheme))
  }
  return result
}

// [maint-note:25ebbb839715]
// 维护导航：inlineText；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 返回契约：string。
// 返回路径：segments.map(segment => segment.text).join('')。
// 直接调用：segments.map(segment => segment.text).join。
// 直接调用：segments.map。
// [/maint-note:25ebbb839715]
function inlineText(segments: readonly MarkdownTableVisualSegment[]): string {
  return segments.map(segment => segment.text).join('')
}

// [maint-note:4dc4b7bde8dc]
// 维护导航：appendSegments；下列为本函数源码中的接口与路径索引。
// 输入 target：MarkdownTableVisualSegment[]。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 返回契约：void。
// 遍历边界：const segment ← segments。
// 直接调用：appendSegment。
// [/maint-note:4dc4b7bde8dc]
function appendSegments(target: MarkdownTableVisualSegment[], segments: readonly MarkdownTableVisualSegment[]): void {
  for (const segment of segments) appendSegment(target, segment)
}

// [maint-note:23733db3fe67]
// 维护导航：appendSegment；下列为本函数源码中的接口与路径索引。
// 输入 target：MarkdownTableVisualSegment[]。
// 输入 segment：MarkdownTableVisualSegment。
// 返回契约：void。
// 分支条件：!segment.text。
// 分支条件：previous && previous.kind === segment.kind && previous.href === segment.href && previous.visualRole === segment.visualRole。
// 返回路径：无返回值。
// 写入目标：target[target.length - 1]。
// 直接调用：target.at。
// 直接调用：target.push。
// [/maint-note:23733db3fe67]
function appendSegment(target: MarkdownTableVisualSegment[], segment: MarkdownTableVisualSegment): void {
  if (!segment.text) return
  const previous = target.at(-1)
  if (previous && previous.kind === segment.kind && previous.href === segment.href && previous.visualRole === segment.visualRole) {
    target[target.length - 1] = { ...previous, text: previous.text + segment.text }
  } else {
    target.push(segment)
  }
}
