import { Box, Text } from 'ink'
import { normalizeTerminalWidth, truncateDisplayText } from '../terminal/index.js'
import { type SessionTimingSnapshot } from '../state/sessionTiming.js'
import { renderWorkingStatusLine } from './TimingProjection.js'

interface ThinkingBarProps {
  loading: boolean
  isStreaming?: boolean
  showingLatest?: boolean
  /** Legacy props remain optional for existing embedding fixtures. */
  thinkTime?: number | null
  elapsed?: number
  dots?: string
  durationMs?: number
  approvalPaused?: boolean
  width: number
}

export function ThinkingBar({ loading, isStreaming = false, showingLatest = false, durationMs = 0, approvalPaused = false, width }: ThinkingBarProps) {
  if (!loading || approvalPaused) return null
  const safeWidth = normalizeTerminalWidth(width, 1)

  return (
    <Box flexDirection="column" width={safeWidth} marginBottom={1}>
      {isStreaming && <Text color="gray">{truncateDisplayText(showingLatest ? 'Streaming… · showing latest output' : 'Streaming…', safeWidth)}</Text>}
      <Text>{renderWorkingStatusLine(durationMs, safeWidth)}</Text>
      <Text color="gray">{truncateDisplayText('Esc to interrupt', safeWidth)}</Text>
    </Box>
  )
}

export type ThinkingTimingSnapshot = Pick<SessionTimingSnapshot, 'currentTurnMs' | 'approvalPaused'>
