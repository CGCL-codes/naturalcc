import { Box, Text } from 'ink'
import { useRef, type ReactElement } from 'react'
import type { InlineSegment, MarkdownBlock } from '../markdown/renderer.js'
import { renderMarkdown, StreamingMarkdownRenderer, type MarkdownProjection } from '../markdown/renderer.js'
import { calculateContentWidth, displayWidth, fillDisplayWidth, truncateDisplayText, wrapDisplayLines } from '../terminal/index.js'

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
  const safeWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  return (
    <Box flexDirection="column" width={safeWidth}>
      {blocks.map((block, index) => <MarkdownBlockView key={`${block.kind}-${index}`} block={block} width={width} />)}
    </Box>
  )
}

function MarkdownBlockView({ block, width }: { block: MarkdownBlock; width: number }) {
  switch (block.kind) {
    case 'paragraph':
      return safeInlineText(block.content, width, <Text wrap="wrap">{inlineElements(block.content)}</Text>)
    case 'heading':
      return safeInlineText(block.content, width, <Text wrap="wrap" bold color="cyan">{inlineElements(block.content)}</Text>)
    case 'blockquote':
      return <MarkdownBlockquote block={block} width={width} />
    case 'list':
      return (
        <Box flexDirection="column" width={safeContentWidth(width)}>
          {block.items.map((item, index) => (
            <Box key={`${item.marker}-${index}`} flexDirection="column" width={safeContentWidth(width)}>
              {safeInlineText(item.content, width, <Text wrap="wrap">{item.marker} {inlineElements(item.content)}</Text>, `${item.marker} `)}
              {item.nested.length > 0 && <NestedListBlocks blocks={item.nested} width={width} />}
            </Box>
          ))}
        </Box>
      )
    case 'code':
      return (
        <Box flexDirection="column" width={safeContentWidth(width)}>
          {block.language && <Text color="gray">[{block.language}]</Text>}
          {block.lines.flatMap((line, index) => wrapDisplayLines(line, safeContentWidth(width), { hard: true, wordWrap: false }).map((wrapped, lineIndex) => (
            <Text key={`${index}-${lineIndex}`} color="yellow" wrap="wrap">{wrapped}</Text>
          )))}
        </Box>
      )
    case 'table':
      return <MarkdownTable block={block} width={width} />
    case 'hr':
      return <Text color="gray">{fillDisplayWidth('─', Math.min(safeContentWidth(width), 40))}</Text>
  }
}

function safeContentWidth(width: number): number {
  return calculateContentWidth({ terminalWidth: width, minWidth: 1 })
}

function safeInlineText(
  content: readonly InlineSegment[],
  width: number,
  styled: ReactElement,
  prefix = '',
): ReactElement {
  const safeWidth = safeContentWidth(width)
  if (safeWidth > 1) return styled
  return <Text wrap="wrap">{truncateDisplayText(`${prefix}${inlineText(content)}`, safeWidth)}</Text>
}

function MarkdownBlockquote({ block, width }: { block: Extract<MarkdownBlock, { kind: 'blockquote' }>; width: number }) {
  const safeWidth = safeContentWidth(width)
  const prefix = safeWidth >= 3 ? '│ ' : safeWidth === 2 ? '│' : ''
  const nestedWidth = calculateContentWidth({
    terminalWidth: safeWidth,
    prefixWidth: prefix,
    minWidth: 1,
  })
  return (
    <Box flexDirection="row" width={safeWidth}>
      {prefix && <Text color="gray">{prefix}</Text>}
      <MarkdownBlocks blocks={block.blocks} width={nestedWidth} />
    </Box>
  )
}

function NestedListBlocks({ blocks, width }: { blocks: readonly MarkdownBlock[]; width: number }) {
  const safeWidth = safeContentWidth(width)
  const indent = safeWidth >= 3 ? 2 : 0
  const nestedWidth = calculateContentWidth({ terminalWidth: safeWidth, paddingLeft: indent, minWidth: 1 })
  return (
    <Box marginLeft={indent} width={nestedWidth}>
      <MarkdownBlocks blocks={blocks} width={nestedWidth} />
    </Box>
  )
}

function tableRowElements(cells: readonly InlineSegment[][]) {
  return cells.flatMap((cell, index) => [
    ...(index > 0 ? [<Text key={`separator-${index}`}> │ </Text>] : []),
    ...inlineElements(cell),
  ])
}

function inlineText(segments: readonly InlineSegment[]): string {
  return segments.map((segment) => segment.text).join('')
}

function MarkdownTable({ block, width }: { block: Extract<MarkdownBlock, { kind: 'table' }>; width: number }) {
  const safeWidth = safeContentWidth(width)
  const rows = [block.header, ...block.rows]
  const columnCount = Math.max(1, ...rows.map((row) => row.length))
  const separatorWidth = Math.max(0, (columnCount - 1) * 3)
  const minimumWidth = columnCount + separatorWidth
  if (safeWidth < minimumWidth) {
    return (
      <Box flexDirection="column" width={safeWidth}>
        {rows.flatMap((row, rowIndex) => row.flatMap((cell, cellIndex) => {
          const label = rowIndex === 0 ? `column ${cellIndex + 1}` : `row ${rowIndex} column ${cellIndex + 1}`
          return wrapDisplayLines(`${label}: ${inlineText(cell)}`, safeWidth, { hard: true, wordWrap: false })
            .map((line, lineIndex) => <Text wrap="wrap" key={`${rowIndex}-${cellIndex}-${lineIndex}`}>{line}</Text>)
        }))}
      </Box>
    )
  }
  const widths = tableColumnWidths(rows, safeWidth - separatorWidth, columnCount)
  return (
    <Box flexDirection="column" width={safeWidth}>
      {rows.flatMap((row, rowIndex) => {
        const text = row.map((cell, index) => truncateDisplayText(inlineText(cell), widths[index] ?? 1)).join(' │ ')
        return wrapDisplayLines(text, safeWidth, { hard: true, wordWrap: false, trim: false }).map((line, lineIndex) => (
          <Text wrap="wrap" bold={rowIndex === 0} key={`${rowIndex}-${lineIndex}`}>{line}</Text>
        ))
      })}
    </Box>
  )
}

function tableColumnWidths(rows: readonly (readonly InlineSegment[][])[], totalWidth: number, count: number): number[] {
  const desired = Array.from({ length: count }, (_, column) => Math.max(1, ...rows.map((row) => displayWidth(inlineText(row[column] ?? [])))))
  const widths = desired.map(() => 1)
  let remaining = Math.max(0, totalWidth - count)
  while (remaining > 0) {
    let changed = false
    for (let index = 0; index < widths.length && remaining > 0; index += 1) {
      if ((widths[index] ?? 1) < (desired[index] ?? 1)) {
        widths[index] = (widths[index] ?? 1) + 1
        remaining -= 1
        changed = true
      }
    }
    if (!changed) break
  }
  return widths
}

function inlineElements(segments: readonly InlineSegment[]) {
  return segments.map((segment, index) => {
    if (segment.kind === 'strong') return <Text key={index} bold>{segment.text}</Text>
    if (segment.kind === 'emphasis') return <Text key={index} italic>{segment.text}</Text>
    if (segment.kind === 'code') return <Text key={index} color="yellow">{segment.text}</Text>
    if (segment.kind === 'link') return <Text key={index} color="cyan" underline>{segment.text}</Text>
    if (segment.kind === 'deleted') return <Text key={index} strikethrough>{segment.text}</Text>
    return <Text key={index}>{segment.text}</Text>
  })
}
