import { Box, Text } from 'ink'
import { calculateContentWidth, fillDisplayWidth, truncateDisplayText } from '../terminal/index.js'

interface ThinkingBarProps {
  loading: boolean
  isStreaming: boolean
  streamingContent: string
  thinkTime: number | null
  elapsed: number
  dots: string
  width: number
}

export function ThinkingBar({ loading, isStreaming, streamingContent, thinkTime, elapsed, dots, width }: ThinkingBarProps) {
  if (!loading) return null
  if (width <= 4) {
    return <Box width={Math.max(1, width)}><Text>{truncateDisplayText(isStreaming ? '…' : 'thinking', Math.max(1, width))}</Text></Box>
  }
  const dividerWidth = calculateContentWidth({ terminalWidth: width, paddingLeft: 2, minWidth: 0 })

  return (
    <>
      {!isStreaming && (
        <Box flexDirection="column" marginBottom={1}>
          <Text>thinking for <Text color='yellow'>{(thinkTime ?? elapsed).toFixed(1)}</Text> s {dots}</Text>
        </Box>
      )}
      {isStreaming && (
        <Box flexDirection="column" marginBottom={1}>
          {streamingContent && <Text color="gray">streaming…</Text>}
          <Text>{"\n"}work for <Text color='yellow'>{(thinkTime ?? elapsed).toFixed(1)}</Text> s</Text>
        </Box>
      )}
      <Box flexDirection="column">
        <Text>{fillDisplayWidth('─', dividerWidth)}</Text>
        <Text color='gray'>Esc to interrupted</Text>
      </Box>
    </>
  )
}
