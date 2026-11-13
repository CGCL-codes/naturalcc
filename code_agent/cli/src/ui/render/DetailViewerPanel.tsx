import { Box, Text } from 'ink'
import { sanitizeTerminalText, truncateDisplayText } from '../terminal/index.js'

// [maint-note:4e4cbcc4aafd]
// 数据契约导航：DetailViewerPanelProps；字段含义以消费端校验为准。
// 成员 recordOrdinal：number。
// 成员 recordCount：number。
// 成员 label：string。
// 成员 lines：readonly string[]。
// 成员 start：number。
// 成员 end：number。
// 成员 totalVisual：number。
// 成员 totalLogical：number。
// 成员 sourceTruncated：boolean。
// 成员 evicted：boolean。
// 成员 width：number。
// 成员 height：number。
// 成员 compact（可省略）：boolean。
// [/maint-note:4e4cbcc4aafd]
export interface DetailViewerPanelProps {
  /** 1-based record ordinal (1..n) among viewable records. */
  recordOrdinal: number
  recordCount: number
  label: string
  /** Current page: already wrapped VISUAL lines. */
  lines: readonly string[]
  /** 1-based inclusive visual-row range. */
  start: number
  end: number
  /** Independent full counts: the page slice is never the denominator. */
  totalVisual: number
  totalLogical: number
  sourceTruncated: boolean
  evicted: boolean
  width: number
  height: number
  /** Borderless bounded fallback for terminals too small for full chrome. */
  compact?: boolean
}

/**
 * detail viewer panel — a dynamic overlay in the interactive region. It
 * renders only the current page of bounded cached lines; opening, paging or
 * closing it never touches the committed stable history. The range line
 * reports the full retained totals independently of the slice: `rows a-b of
 * V · N lines retained`, so paging state can never masquerade as the record
 * size. Long logical lines arrive pre-wrapped, so every retained line is
 * reachable by paging instead of being sideways-truncated forever.
 */
// [maint-note:99e2d59277de]
// 维护导航：DetailViewerPanel；下列为本函数源码中的接口与路径索引。
// 输入 { recordOrdinal, recordCount, label, lines, start, end, totalVisual, totalLogical, sourceTruncated, evicted, width, height, compact = false, }：DetailViewerPanelProps。
// 分支条件：compact。
// 返回路径：( <Box flexDirection="column" width={safeWidth} height={rows}> {compactLines.map((line, index) => ( <Text key={index} color={evicted ? 'yellow' : undefined} wrap="wrap"> {sanitizeTerminalText(line)} </Text> ))} </Box> )。
// 返回路径：( <Box flexDirection="column" width={safeWidth} borderStyle="round" borderColor="gray"> <Text color="gray" wrap="truncate">{header}</Text> {evicted ? ( <Text color="yellow">details no longer retained</Text> ) : ( <Box flexDirection="column"。
// 直接调用：Math.max。
// 直接调用：truncateDisplayText。
// 直接调用：`record ${recordOrdinal}/${recordCount} · ${label}`.replace。
// 直接调用：lines.slice。
// 直接调用：compactLines.map。
// 直接调用：lines.map。
// [/maint-note:99e2d59277de]
export function DetailViewerPanel({
  recordOrdinal, recordCount, label, lines, start, end, totalVisual, totalLogical,
  sourceTruncated, evicted, width, height, compact = false,
}: DetailViewerPanelProps) {
  const safeWidth = Math.max(1, width)
  const rows = Math.max(1, height)
  const innerWidth = compact ? safeWidth : Math.max(1, safeWidth - 2)
  const header = truncateDisplayText(
    `record ${recordOrdinal}/${recordCount} · ${label}`.replace(/\s+/gu, ' '),
    innerWidth,
  )
  const range = totalVisual === 0
    ? 'no retained lines'
    : `rows ${start}-${end} of ${totalVisual} · ${totalLogical} lines retained`
  const footer = evicted
    ? 'details no longer retained'
    : 'Esc/Ctrl+O close · PgUp/PgDn scroll · Tab next · Shift+Tab previous'
  if (compact) {
    const compactLines = evicted
      ? ['details no longer retained']
      : lines.slice(0, rows)
    return (
      <Box flexDirection="column" width={safeWidth} height={rows}>
        {compactLines.map((line, index) => (
          <Text key={index} color={evicted ? 'yellow' : undefined} wrap="wrap">
            {sanitizeTerminalText(line)}
          </Text>
        ))}
      </Box>
    )
  }
  return (
    <Box flexDirection="column" width={safeWidth} borderStyle="round" borderColor="gray">
      <Text color="gray" wrap="truncate">{header}</Text>
      {evicted ? (
        <Text color="yellow">details no longer retained</Text>
      ) : (
        <Box flexDirection="column" height={rows}>
          {lines.map((line, index) => (
            <Text key={index} wrap="wrap">{sanitizeTerminalText(line)}</Text>
          ))}
        </Box>
      )}
      <Text color="gray" wrap="truncate">{truncateDisplayText(`${range}${sourceTruncated ? ' · source truncated' : ''}`, innerWidth)}</Text>
      <Text color="gray" wrap="truncate">{truncateDisplayText(footer, innerWidth)}</Text>
    </Box>
  )
}
