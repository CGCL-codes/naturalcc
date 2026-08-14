import { Box, Text } from 'ink'
import { serializeToolTableRow, type ToolDisplayBlock, type ToolResultPresentation } from '../toolPresenters/index.js'
import { calculateContentWidth, wrapDisplayLines } from '../terminal/index.js'

export function ToolResultView({ presentation, width = 80 }: { presentation: ToolResultPresentation; width?: number }) {
  const safeWidth = calculateContentWidth({ terminalWidth: width, minWidth: 1 })
  return (
    <Box flexDirection="column" width={safeWidth}>
      {presentation.blocks.map((block, index) => <ToolBlock key={`${block.kind}-${index}`} block={block} width={safeWidth} />)}
    </Box>
  )
}

function ToolBlock({ block, width }: { block: ToolDisplayBlock; width: number }) {
  switch (block.kind) {
    case 'summary':
      return <Text wrap="wrap">{block.text}</Text>
    case 'field':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(block.label, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map((line, index) => <Text wrap="wrap" key={index}>{line}</Text>)}
        </Box>
      )
    case 'list':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(block.label, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {block.items.flatMap((item, index) => wrapDisplayLines(item, width, { hard: true, wordWrap: false }).map((line, lineIndex) => <Text wrap="wrap" key={`${index}-${lineIndex}`}>{line}</Text>))}
        </Box>
      )
    case 'table':
      return (
        <Box flexDirection="column" width={width}>
          {wrapDisplayLines(block.label, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={`label-${index}`}>{line}</Text>)}
          {block.rows.flatMap((row, index) => wrapDisplayLines(serializeToolTableRow(row), width, { hard: true, wordWrap: false }).map((line, lineIndex) => <Text wrap="wrap" key={`${index}-${lineIndex}`}>{line}</Text>))}
        </Box>
      )
    case 'notice':
      return <>{wrapDisplayLines(block.text, width, { hard: true, wordWrap: false }).map((line, index) => <Text color="gray" wrap="wrap" key={index}>{line}</Text>)}</>
  }
}
