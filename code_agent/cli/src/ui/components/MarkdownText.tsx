import { Box, Text } from 'ink'
import { useRef } from 'react'
import type { InlineSegment, MarkdownBlock } from '../markdown/renderer.js'
import { renderMarkdown, StreamingMarkdownRenderer, type MarkdownProjection } from '../markdown/renderer.js'
import {
  buildMarkdownTableRows,
  type MarkdownTableRowRole,
  type MarkdownTableVisualSegment,
  type MarkdownTableVisualRow,
} from '../markdown/tableLayout.js'
import { calculateContentWidth, displayWidth, fillDisplayWidth, splitDisplayGraphemes, truncateDisplayText } from '../terminal/index.js'
import { UI_PALETTE, type UiPaletteColor } from '../theme.js'

// [maint-note:3767f12fd917]
// 数据契约导航：MarkdownTextProps；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 width（可省略）：number。
// 成员 streaming（可省略）：boolean。
// [/maint-note:3767f12fd917]
interface MarkdownTextProps {
  content: string
  width?: number
  streaming?: boolean
}

// [maint-note:b4c36a862095]
// 维护导航：MarkdownText；下列为本函数源码中的接口与路径索引。
// 输入 { content, width = 80, streaming = false }：MarkdownTextProps。
// 分支条件：!streaming。
// 分支条件：delta === content && existing && existing.width === width && !content.startsWith(existing.source)。
// 返回路径：<MarkdownBlocks blocks={projection.blocks} width={width} />。
// 写入目标：projection。
// 写入目标：rendererRef.current。
// 直接调用：useRef。
// 直接调用：renderMarkdown。
// 直接调用：content.startsWith。
// 直接调用：content.slice。
// 直接调用：renderer.reset。
// 直接调用：renderer.push。
// [/maint-note:b4c36a862095]
export function MarkdownText({ content, width = 80, streaming = false }: MarkdownTextProps) {
  const rendererRef = useRef<{ width: number; renderer: StreamingMarkdownRenderer; source: string } | null>(null)
  let projection: MarkdownProjection
  if (!streaming) {
    projection = renderMarkdown(content, { maxWidth: width })
    rendererRef.current = null
  } else {
    const existing = rendererRef.current
    const renderer = existing && existing.width === width
      ? existing.renderer
      : new StreamingMarkdownRenderer({ maxWidth: width })
    const delta = existing && existing.width === width && content.startsWith(existing.source)
      ? content.slice(existing.source.length)
      : content
    if (delta === content && existing && existing.width === width && !content.startsWith(existing.source)) renderer.reset()
    projection = renderer.push(delta)
    rendererRef.current = { width, renderer, source: content }
  }
  return <MarkdownBlocks blocks={projection.blocks} width={width} />
}

// [maint-note:ccff91ff5b12]
// 维护导航：MarkdownBlocks；下列为本函数源码中的接口与路径索引。
// 输入 { blocks, width }：{ blocks: readonly MarkdownBlock[]; width: number }。
// 返回路径：<MarkdownVisualRows rows={rows} width={width} />。
// 直接调用：buildMarkdownVisualRows。
// [/maint-note:ccff91ff5b12]
export function MarkdownBlocks({ blocks, width }: { blocks: readonly MarkdownBlock[]; width: number }) {
  const rows = buildMarkdownVisualRows(blocks, width)
  return <MarkdownVisualRows rows={rows} width={width} />
}

/** Render the exact styled physical rows used by the active-frame ledger. */
// [maint-note:5d65b922c8ad]
// 维护导航：MarkdownVisualRows；下列为本函数源码中的接口与路径索引。
// 输入 { rows, width }：{ rows: readonly MarkdownVisualRow[]; width: number }。
// 返回路径：( <Box flexDirection="column" width={safeWidth}> {rows.map((row, index) => ( <Text key={index} bold={row.bold} color={row.color ?? rowColor(row.role)}> {inlineElements(row.segments, row.role)} </Text> ))} </Box> )。
// 直接调用：calculateContentWidth。
// 直接调用：rows.map。
// [/maint-note:5d65b922c8ad]
export function MarkdownVisualRows({ rows, width }: { rows: readonly MarkdownVisualRow[]; width: number }) {
  const safeWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  return (
    <Box flexDirection="column" width={safeWidth}>
      {rows.map((row, index) => (
        <Text key={index} bold={row.bold} color={row.color ?? rowColor(row.role)}>
          {inlineElements(row.segments, row.role)}
        </Text>
      ))}
    </Box>
  )
}

/**
 * A final visual-row primitive shared by the renderer and frame ledger.
 * Every returned row is already hard-wrapped to its available width, so the
 * Ink tree renders one physical row per item without relying on a second
 * word-wrapping estimate.
 */
// [maint-note:a22d0665f05b]
// 维护导航：countMarkdownBlockRows；下列为本函数源码中的接口与路径索引。
// 输入 blocks：readonly MarkdownBlock[]。
// 输入 width：number。
// 返回契约：number。
// 返回路径：buildMarkdownVisualRows(blocks, width).length。
// 直接调用：buildMarkdownVisualRows。
// [/maint-note:a22d0665f05b]
export function countMarkdownBlockRows(blocks: readonly MarkdownBlock[], width: number): number {
  return buildMarkdownVisualRows(blocks, width).length
}

// [maint-note:50472191015f]
// 数据契约导航：MarkdownVisualRow；字段含义以消费端校验为准。
// 成员 color（可省略）：UiPaletteColor。
// [/maint-note:50472191015f]
export interface MarkdownVisualRow extends MarkdownTableVisualRow {
  readonly color?: UiPaletteColor
}

// [maint-note:8537194d597f]
// 维护导航：buildMarkdownVisualRows；下列为本函数源码中的接口与路径索引。
// 输入 blocks：readonly MarkdownBlock[]。
// 输入 width：number。
// 返回契约：readonly MarkdownVisualRow[]。
// 返回路径：rows.length > 0 ? rows : [{ segments: [] }]。
// 直接调用：blocks.flatMap。
// [/maint-note:8537194d597f]
export function buildMarkdownVisualRows(blocks: readonly MarkdownBlock[], width: number): readonly MarkdownVisualRow[] {
  const rows = blocks.flatMap(block => buildMarkdownBlockRows(block, width))
  return rows.length > 0 ? rows : [{ segments: [] }]
}

// [maint-note:37a7d687f26b]
// 维护导航：buildMarkdownBlockRows；下列为本函数源码中的接口与路径索引。
// 输入 block：MarkdownBlock。
// 输入 width：number。
// 返回契约：MarkdownVisualRow[]。
// 分支条件：switch block.kind。
// 分支条件：case 'paragraph'。
// 分支条件：case 'heading'。
// 分支条件：case 'blockquote'。
// 分支条件：case 'list'。
// 分支条件：item.nested.length > 0。
// 分支条件：case 'code'。
// 分支条件：case 'table'。
// 分支条件：case 'hr'。
// 遍历边界：const item ← block.items。
// 返回路径：wrapInlineSegments(block.content, safeWidth)。
// 返回路径：wrapInlineSegments(block.content, safeWidth).map(row => ({ ...row, bold: true, color: UI_PALETTE.markdownAccent, role: 'header' as const, }))。
// 返回路径：nested.map((row) => prefix ? { ...row, segments: [{ kind: 'text' as const, text: prefix }, ...row.segments] } : row)。
// 返回路径：rows.length > 0 ? rows : [{ segments: [] }]。
// 返回路径：[ ...(block.language ? wrapPlainSegments(`[${block.language}]`, safeWidth, 'gray') : []), ...block.lines.flatMap(line => wrapPlainSegments(line, safeWidth)), ]。
// 返回路径：[...buildMarkdownTableRows(block, safeWidth)]。
// 返回路径：[{ segments: [{ kind: 'text', text: fillDisplayWidth('─', Math.min(safeWidth, 40)) }], color: 'gray' }]。
// 直接调用：safeContentWidth。
// 直接调用：wrapInlineSegments。
// 直接调用：wrapInlineSegments(block.content, safeWidth).map。
// 直接调用：calculateContentWidth。
// 直接调用：buildMarkdownVisualRows。
// 直接调用：nested.map。
// 直接调用：rows.push。
// 直接调用：buildMarkdownVisualRows(item.nested, nestedWidth).map。
// 直接调用：wrapPlainSegments。
// 直接调用：block.lines.flatMap。
// 直接调用：buildMarkdownTableRows。
// 直接调用：fillDisplayWidth。
// 直接调用：Math.min。
// [/maint-note:37a7d687f26b]
function buildMarkdownBlockRows(block: MarkdownBlock, width: number): MarkdownVisualRow[] {
  const safeWidth = safeContentWidth(width)
  switch (block.kind) {
    case 'paragraph':
      return wrapInlineSegments(block.content, safeWidth)
    case 'heading':
      return wrapInlineSegments(block.content, safeWidth).map(row => ({
        ...row,
        bold: true,
        color: UI_PALETTE.markdownAccent,
        role: 'header' as const,
      }))
    case 'blockquote': {
      const prefix = safeWidth >= 3 ? '│ ' : safeWidth === 2 ? '│' : ''
      const nestedWidth = calculateContentWidth({ terminalWidth: safeWidth, prefixWidth: prefix, minWidth: 1 })
      const nested = buildMarkdownVisualRows(block.blocks, nestedWidth)
      return nested.map((row) => prefix
        ? { ...row, segments: [{ kind: 'text' as const, text: prefix }, ...row.segments] }
        : row)
    }
    case 'list': {
      const rows: MarkdownVisualRow[] = []
      for (const item of block.items) {
        rows.push(...wrapInlineSegments(item.content, safeWidth, `${item.marker} `))
        if (item.nested.length > 0) {
          const indent = safeWidth >= 3 ? 2 : 0
          const nestedWidth = calculateContentWidth({ terminalWidth: safeWidth, paddingLeft: indent, minWidth: 1 })
          rows.push(...buildMarkdownVisualRows(item.nested, nestedWidth).map(row => ({
            ...row,
            segments: indent > 0 ? [{ kind: 'text' as const, text: ' '.repeat(indent) }, ...row.segments] : row.segments,
          })))
        }
      }
      return rows.length > 0 ? rows : [{ segments: [] }]
    }
    case 'code':
      return [
        ...(block.language ? wrapPlainSegments(`[${block.language}]`, safeWidth, 'gray') : []),
        ...block.lines.flatMap(line => wrapPlainSegments(line, safeWidth)),
      ]
    case 'table': {
      return [...buildMarkdownTableRows(block, safeWidth)]
    }
    case 'hr':
      return [{ segments: [{ kind: 'text', text: fillDisplayWidth('─', Math.min(safeWidth, 40)) }], color: 'gray' }]
  }
}

// [maint-note:7ce2459ade3c]
// 维护导航：wrapPlainSegments；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 输入 color：MarkdownVisualRow['color']（可省略）。
// 返回契约：MarkdownVisualRow[]。
// 返回路径：wrapInlineSegments([{ kind: 'text', text: value }], width).map(row => ({ ...row, color }))。
// 直接调用：wrapInlineSegments([{ kind: 'text', text: value }], width).map。
// 直接调用：wrapInlineSegments。
// [/maint-note:7ce2459ade3c]
function wrapPlainSegments(value: string, width: number, color?: MarkdownVisualRow['color']): MarkdownVisualRow[] {
  return wrapInlineSegments([{ kind: 'text', text: value }], width).map(row => ({ ...row, color }))
}

// [maint-note:00380ef916f6]
// 维护导航：safeContentWidth；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：number。
// 返回路径：calculateContentWidth({ terminalWidth: width, minWidth: 1 })。
// 直接调用：calculateContentWidth。
// [/maint-note:00380ef916f6]
function safeContentWidth(width: number): number {
  return calculateContentWidth({ terminalWidth: width, minWidth: 1 })
}

// [maint-note:f876d8f2b7b2]
// 维护导航：wrapInlineSegments；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly InlineSegment[]。
// 输入 width：number。
// 输入 prefix：类型由调用方与函数体约束；缺省表达式 ''。
// 返回契约：MarkdownVisualRow[]。
// 分支条件：character === '\n'。
// 分支条件：character === '\t'。
// 分支条件：currentWidth > 0 && currentWidth + tabWidth > safeWidth。
// 分支条件：spaces > 0。
// 分支条件：currentWidth >= safeWidth。
// 分支条件：displayWidth(character) > safeWidth。
// 分支条件：currentWidth > 0。
// 分支条件：currentWidth > 0 && currentWidth + characterWidth > safeWidth。
// 分支条件：current.length > 0 || rows.length === 0。
// 遍历边界：const segment ← all。
// 遍历边界：const character ← splitDisplayGraphemes(segment.text)。
// 返回路径：rows。
// 写入目标：currentWidth。
// 直接调用：Math.max。
// 直接调用：splitDisplayGraphemes。
// 直接调用：finish。
// 直接调用：Math.min。
// 直接调用：pushSegment。
// 直接调用：' '.repeat。
// 直接调用：displayWidth。
// 直接调用：truncateDisplayText。
// [/maint-note:f876d8f2b7b2]
function wrapInlineSegments(segments: readonly InlineSegment[], width: number, prefix = ''): MarkdownVisualRow[] {
  const safeWidth = Math.max(1, width)
  const rows: MarkdownVisualRow[] = []
  let current: InlineSegment[] = []
  let currentWidth = 0

  // [maint-note:52c9d291612b]
  // 维护导航：pushSegment；下列为本函数源码中的接口与路径索引。
  // 输入 kind：InlineSegment['kind']。
  // 输入 text：string。
  // 输入 href：string（可省略）。
  // 返回契约：void。
  // 分支条件：!text。
  // 分支条件：previous && previous.kind === kind && previous.href === href。
  // 返回路径：无返回值。
  // 写入目标：current[current.length - 1]。
  // 直接调用：current.at。
  // 直接调用：current.push。
  // [/maint-note:52c9d291612b]
  const pushSegment = (kind: InlineSegment['kind'], text: string, href?: string): void => {
    if (!text) return
    const previous = current.at(-1)
    if (previous && previous.kind === kind && previous.href === href) {
      current[current.length - 1] = { ...previous, text: previous.text + text }
    } else {
      current.push({ kind, text, ...(href ? { href } : {}) })
    }
  }
  // [maint-note:5981499ddf2a]
  // 维护导航：finish；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：current。
  // 写入目标：currentWidth。
  // 直接调用：rows.push。
  // [/maint-note:5981499ddf2a]
  const finish = (): void => {
    rows.push({ segments: current })
    current = []
    currentWidth = 0
  }

  const all = prefix ? [{ kind: 'text' as const, text: prefix }, ...segments] : segments
  for (const segment of all) {
    for (const character of splitDisplayGraphemes(segment.text)) {
      if (character === '\n') {
        finish()
        continue
      }
      if (character === '\t') {
        // Terminals advance a tab to the next 8-column stop. Materialize the
        // spaces in the shared row projection so Ink and the ledger charge
        // exactly the same width. A narrow viewport consumes the remaining
        // cells and starts the following content on the next row.
        const tabWidth = 8 - (currentWidth % 8)
        if (currentWidth > 0 && currentWidth + tabWidth > safeWidth) finish()
        const spaces = Math.min(tabWidth, safeWidth - currentWidth)
        if (spaces > 0) {
          pushSegment(segment.kind, ' '.repeat(spaces), segment.href)
          currentWidth += spaces
        }
        if (currentWidth >= safeWidth) finish()
        continue
      }
      if (displayWidth(character) > safeWidth) {
        // A double-cell grapheme cannot fit in a one-cell terminal. Use the
        // same single-cell ellipsis policy as the rest of the terminal
        // projection instead of allowing Ink to overflow the row.
        if (currentWidth > 0) finish()
        pushSegment(segment.kind, truncateDisplayText(character, safeWidth), segment.href)
        finish()
        continue
      }
      const characterWidth = Math.max(1, displayWidth(character))
      if (currentWidth > 0 && currentWidth + characterWidth > safeWidth) finish()
      pushSegment(segment.kind, character, segment.href)
      currentWidth += characterWidth
    }
  }
  if (current.length > 0 || rows.length === 0) finish()
  return rows
}

// [maint-note:258119766fdb]
// 维护导航：inlineElements；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly MarkdownTableVisualSegment[]。
// 输入 rowRole：MarkdownTableRowRole（可省略）。
// 返回路径：segments.map((segment, index) => { if (segment.visualRole === 'border') { return <Text key={index} color={UI_PALETTE.neutral} bold={false}>{segment.text}</Text> } if (segment.visualRole === 'auxiliary') { return <Text key={index} color={UI_。
// 直接调用：segments.map。
// [/maint-note:258119766fdb]
function inlineElements(segments: readonly MarkdownTableVisualSegment[], rowRole?: MarkdownTableRowRole) {
  return segments.map((segment, index) => {
    if (segment.visualRole === 'border') {
      return <Text key={index} color={UI_PALETTE.neutral} bold={false}>{segment.text}</Text>
    }
    if (segment.visualRole === 'auxiliary') {
      return <Text key={index} color={UI_PALETTE.neutral}>{segment.text}</Text>
    }
    const header = rowRole === 'header' || segment.visualRole === 'header'
    if (segment.kind === 'strong') {
      return <Text key={index} color={header ? UI_PALETTE.markdownAccent : UI_PALETTE.markdownStrong} bold>{segment.text}</Text>
    }
    if (segment.kind === 'emphasis') return <Text key={index} color={header ? UI_PALETTE.markdownAccent : undefined} bold={header} italic>{segment.text}</Text>
    if (segment.kind === 'code') return <Text key={index} color={UI_PALETTE.markdownAccent} bold={header}>{segment.text}</Text>
    if (segment.kind === 'link') return <Text key={index} color={UI_PALETTE.markdownAccent} bold={header} underline>{segment.text}</Text>
    if (segment.kind === 'deleted') return <Text key={index} color={header ? UI_PALETTE.markdownAccent : undefined} bold={header} strikethrough>{segment.text}</Text>
    return <Text key={index} color={header ? UI_PALETTE.markdownAccent : undefined} bold={header}>{segment.text}</Text>
  })
}

// [maint-note:5944f5e8434c]
// 维护导航：rowColor；下列为本函数源码中的接口与路径索引。
// 输入 role：MarkdownTableRowRole | undefined。
// 返回契约：UiPaletteColor | undefined。
// 分支条件：role === 'border'。
// 分支条件：role === 'header'。
// 返回路径：UI_PALETTE.neutral。
// 返回路径：UI_PALETTE.markdownAccent。
// 返回路径：undefined。
// [/maint-note:5944f5e8434c]
function rowColor(role: MarkdownTableRowRole | undefined): UiPaletteColor | undefined {
  if (role === 'border') return UI_PALETTE.neutral
  if (role === 'header') return UI_PALETTE.markdownAccent
  return undefined
}
