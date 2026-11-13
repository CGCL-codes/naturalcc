import { Box, Text } from 'ink'
import { serializeToolTableRow, type ToolDisplayBlock, type ToolResultPresentation } from '../toolPresenters/index.js'
import { calculateContentWidth, wrapDisplayLines } from '../terminal/index.js'

// [maint-note:14b9024ed831]
// 维护导航：ToolResultView；下列为本函数源码中的接口与路径索引。
// 输入 { presentation, width = 80 }：{ presentation: ToolResultPresentation; width?: number }。
// 返回路径：( <Box flexDirection="column" width={safeWidth}> {presentation.blocks.map((block, index) => <ToolBlock key={`${block.kind}-${index}`} block={block} width={safeWidth} />)} </Box> )。
// 直接调用：calculateContentWidth。
// 直接调用：presentation.blocks.map。
// [/maint-note:14b9024ed831]
export function ToolResultView({ presentation, width = 80 }: { presentation: ToolResultPresentation; width?: number }) {
  const safeWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  return (
    <Box flexDirection="column" width={safeWidth}>
      {presentation.blocks.map((block, index) => <ToolBlock key={`${block.kind}-${index}`} block={block} width={safeWidth} />)}
    </Box>
  )
}

// [maint-note:8847e9d81a0a]
// 维护导航：ToolBlock；下列为本函数源码中的接口与路径索引。
// 输入 { block, width }：{ block: ToolDisplayBlock; width: number }。
// 分支条件：switch block.kind。
// 分支条件：case 'summary'。
// 分支条件：case 'spacer'。
// 分支条件：case 'field'。
// 分支条件：case 'list'。
// 分支条件：case 'table'。
// 分支条件：case 'notice'。
// 分支条件：case 'diff'。
// 返回路径：<Text wrap="wrap">{block.text}</Text>。
// 返回路径：<Box height={1} />。
// 返回路径：( <Box flexDirection="column" width={width}> {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)} {wrapDisplayLines(block.tex。
// 返回路径：( <Box flexDirection="column" width={width}> {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)} {block.items.flatMap((item,。
// 返回路径：( <Box flexDirection="column" width={width}> {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)} {block.rows.flatMap((row, i。
// 返回路径：<>{wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={index}>{line}</Text>)}</>。
// 返回路径：<DiffLinesView lines={block.lines} width={width} />。
// 直接调用：wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map。
// 直接调用：wrapDisplayLines。
// 直接调用：wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map。
// 直接调用：block.items.flatMap。
// 直接调用：block.rows.flatMap。
// [/maint-note:8847e9d81a0a]
function ToolBlock({ block, width }: { block: ToolDisplayBlock; width: number }) {
  switch (block.kind) {
    case 'summary':
      return <Text wrap="wrap">{block.text}</Text>
    case 'spacer':
      return <Box height={1} />
    case 'field':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map((line, index) => <Text wrap="wrap" key={index}>{line}</Text>)}
        </Box>
      )
    case 'list':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {block.items.flatMap((item, index) => wrapDisplayLines(item, width, { hard: true, wordWrap: false }).map((line, lineIndex) => <Text wrap="wrap" key={`${index}-${lineIndex}`}>{line}</Text>))}
        </Box>
      )
    case 'table':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(`${block.label}:`, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="blue" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {block.rows.flatMap((row, index) => wrapDisplayLines(serializeToolTableRow(row), width, { hard: true, wordWrap: false }).map((line, lineIndex) => <Text wrap="wrap" key={`${index}-${lineIndex}`}>{line}</Text>))}
        </Box>
      )
    case 'notice':
      return <>{wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={index}>{line}</Text>)}</>
    case 'diff':
      return <DiffLinesView lines={block.lines} width={width} />
  }
}

/**
 * color projection over the FINAL visual lines. Line numbers, the
 * separator and elisions stay neutral gray; only the marker column, the one
 * UI separator space and the visible source text carry the bright
 * green/red foreground with the dark background — including soft-wrap
 * continuations, which keep the same colors on a blank gutter and never
 * repeat the line number. SGR closes at every physical line: each Ink
 * `Text` is one line, so no escape state leaks past the row. Colors are
 * enhancement only — the serialized ledger carries the semantics, so
 * NO_COLOR terminals show the identical +/-/space/⋮ columns.
 */
// [maint-note:8579deb9b767]
// 维护导航：DiffLinesView；下列为本函数源码中的接口与路径索引。
// 输入 { lines, width }：{ lines: readonly import('../presentation/applyPatchDiff.js').DiffVisualLine[]; width: number }。
// 返回路径：( <Box flexDirection="column" width={width}> {lines.flatMap((line, index) => { const gutter = line.lineNumber.length + ' │ '.length + 2 const textWidth = Math.max(1, width - gutter) const segments = line.text === '' ? [''] : wrapDisplayLine。
// 直接调用：lines.flatMap。
// [/maint-note:8579deb9b767]
function DiffLinesView({ lines, width }: { lines: readonly import('../presentation/applyPatchDiff.js').DiffVisualLine[]; width: number }) {
  return (
    <Box flexDirection="column" width={width}>
      {lines.flatMap((line, index) => {
        const gutter = line.lineNumber.length + ' │ '.length + 2
        const textWidth = Math.max(1, width - gutter)
        const segments = line.text === ''
          ? ['']
          : wrapDisplayLines(line.text, textWidth, { hard: true, trim: false, wordWrap: false })
        return segments.map((segment, segmentIndex) => (
          <DiffLineView
            key={`${index}-${segmentIndex}`}
            lineNumber={segmentIndex === 0 ? line.lineNumber : ' '.repeat(line.lineNumber.length)}
            marker={segmentIndex === 0 ? line.marker : ' '}
            text={segment}
            kind={line.kind}
            width={width}
          />
        ))
      })}
    </Box>
  )
}

// [maint-note:5a4ad6c8a6bd]
// 维护导航：DiffLineView；下列为本函数源码中的接口与路径索引。
// 输入 { lineNumber, marker, text, kind, width }：{ lineNumber: string marker: string text: string kind: import('../presentation/applyPatchDiff.js').DiffVisualLine['kind'] width: number }。
// 分支条件：kind === 'added'。
// 分支条件：kind === 'removed'。
// 返回路径：( <Box width={width}> {gutter} <Text color="green" backgroundColor="#10361c" wrap="wrap">{`${marker} ${text}`}</Text> </Box> )。
// 返回路径：( <Box width={width}> {gutter} <Text color="red" backgroundColor="#3c1418" wrap="wrap">{`${marker} ${text}`}</Text> </Box> )。
// 返回路径：( <Box width={width}> {gutter} <Text color="gray" wrap="wrap">{`${marker} ${text}`}</Text> </Box> )。
// [/maint-note:5a4ad6c8a6bd]
function DiffLineView({ lineNumber, marker, text, kind, width }: {
  lineNumber: string
  marker: string
  text: string
  kind: import('../presentation/applyPatchDiff.js').DiffVisualLine['kind']
  width: number
}) {
  const gutter = (
    <Text color="gray" wrap="truncate">{`${lineNumber} │ `}</Text>
  )
  if (kind === 'added') {
    return (
      <Box width={width}>
        {gutter}
        <Text color="green" backgroundColor="#10361c" wrap="wrap">{`${marker} ${text}`}</Text>
      </Box>
    )
  }
  if (kind === 'removed') {
    return (
      <Box width={width}>
        {gutter}
        <Text color="red" backgroundColor="#3c1418" wrap="wrap">{`${marker} ${text}`}</Text>
      </Box>
    )
  }
  return (
    <Box width={width}>
      {gutter}
      <Text color="gray" wrap="wrap">{`${marker} ${text}`}</Text>
    </Box>
  )
}
