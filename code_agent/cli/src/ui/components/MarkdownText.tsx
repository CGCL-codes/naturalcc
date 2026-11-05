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

interface MarkdownTextProps {
  content: string
  width?: number
  streaming?: boolean
}

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

export function MarkdownBlocks({ blocks, width }: { blocks: readonly MarkdownBlock[]; width: number }) {
  const rows = buildMarkdownVisualRows(blocks, width)
  return <MarkdownVisualRows rows={rows} width={width} />
}

/** Render the exact styled physical rows used by the active-frame ledger. */
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
export function countMarkdownBlockRows(blocks: readonly MarkdownBlock[], width: number): number {
  return buildMarkdownVisualRows(blocks, width).length
}

export interface MarkdownVisualRow extends MarkdownTableVisualRow {
  readonly color?: UiPaletteColor
}

export function buildMarkdownVisualRows(blocks: readonly MarkdownBlock[], width: number): readonly MarkdownVisualRow[] {
  const rows = blocks.flatMap(block => buildMarkdownBlockRows(block, width))
  return rows.length > 0 ? rows : [{ segments: [] }]
}

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

function wrapPlainSegments(value: string, width: number, color?: MarkdownVisualRow['color']): MarkdownVisualRow[] {
  return wrapInlineSegments([{ kind: 'text', text: value }], width).map(row => ({ ...row, color }))
}

function safeContentWidth(width: number): number {
  return calculateContentWidth({ terminalWidth: width, minWidth: 1 })
}

function wrapInlineSegments(segments: readonly InlineSegment[], width: number, prefix = ''): MarkdownVisualRow[] {
  const safeWidth = Math.max(1, width)
  const rows: MarkdownVisualRow[] = []
  let current: InlineSegment[] = []
  let currentWidth = 0

  const pushSegment = (kind: InlineSegment['kind'], text: string, href?: string): void => {
    if (!text) return
    const previous = current.at(-1)
    if (previous && previous.kind === kind && previous.href === href) {
      current[current.length - 1] = { ...previous, text: previous.text + text }
    } else {
      current.push({ kind, text, ...(href ? { href } : {}) })
    }
  }
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

function rowColor(role: MarkdownTableRowRole | undefined): UiPaletteColor | undefined {
  if (role === 'border') return UI_PALETTE.neutral
  if (role === 'header') return UI_PALETTE.markdownAccent
  return undefined
}
