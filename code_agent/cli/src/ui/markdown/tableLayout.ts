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

export interface MarkdownTableVisualRow {
  readonly segments: readonly MarkdownTableVisualSegment[]
  readonly bold?: boolean
  readonly role?: MarkdownTableRowRole
}

export type MarkdownTableRowRole = 'border' | 'header' | 'body' | 'field'
export type MarkdownTableSegmentRole = 'border' | 'header' | 'auxiliary'

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

function borderSegment(text: string): MarkdownTableVisualSegment {
  return { kind: 'text', text, visualRole: 'border' }
}

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

function wrapInlineSegments(segments: readonly MarkdownTableVisualSegment[], width: number): MarkdownTableVisualRow[] {
  const safeWidth = Math.max(1, Math.floor(width))
  const rows: MarkdownTableVisualRow[] = []
  let current: MarkdownTableVisualSegment[] = []
  let currentWidth = 0

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

interface StyledGrapheme {
  readonly text: string
  readonly segments: readonly MarkdownTableVisualSegment[]
}

/**
 * Segment the complete visible cell before mapping style spans back onto
 * each cluster. Markdown may place a style boundary between code points that
 * still form one terminal grapheme (for example emoji + ZWJ + emoji).
 */
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

function withText(segment: MarkdownTableVisualSegment | undefined, text: string): MarkdownTableVisualSegment {
  return segment ? { ...segment, text } : { kind: 'text', text }
}

function cellDisplayWidth(segments: readonly MarkdownTableVisualSegment[]): number {
  return displayWidth(expandDisplayTabs(inlineText(segments)))
}

function widestCellGrapheme(segments: readonly MarkdownTableVisualSegment[]): number {
  return Math.max(1, ...splitDisplayGraphemes(expandDisplayTabs(inlineText(segments)))
    .filter(grapheme => grapheme !== '\n')
    .map(grapheme => displayLineWidth(grapheme)))
}

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

function inlineText(segments: readonly MarkdownTableVisualSegment[]): string {
  return segments.map(segment => segment.text).join('')
}

function appendSegments(target: MarkdownTableVisualSegment[], segments: readonly MarkdownTableVisualSegment[]): void {
  for (const segment of segments) appendSegment(target, segment)
}

function appendSegment(target: MarkdownTableVisualSegment[], segment: MarkdownTableVisualSegment): void {
  if (!segment.text) return
  const previous = target.at(-1)
  if (previous && previous.kind === segment.kind && previous.href === segment.href && previous.visualRole === segment.visualRole) {
    target[target.length - 1] = { ...previous, text: previous.text + segment.text }
  } else {
    target.push(segment)
  }
}
