import { Box, Text } from 'ink'
import { normalizeTerminalWidth, truncateDisplayText } from '../terminal/index.js'
import { type SessionTimingSnapshot } from '../state/sessionTiming.js'
import { renderWorkingStatusLine } from './TimingProjection.js'

// [maint-note:941ef8b14b0f]
// 数据契约导航：ThinkingBarProps；字段含义以消费端校验为准。
// 成员 loading：boolean。
// 成员 isStreaming（可省略）：boolean。
// 成员 showingLatest（可省略）：boolean。
// 成员 thinkTime（可省略）：number | null。
// 成员 elapsed（可省略）：number。
// 成员 dots（可省略）：string。
// 成员 durationMs（可省略）：number。
// 成员 approvalPaused（可省略）：boolean。
// 成员 width：number。
// [/maint-note:941ef8b14b0f]
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

// [maint-note:b46d4c6e5b80]
// 维护导航：ThinkingBar；下列为本函数源码中的接口与路径索引。
// 输入 { loading, isStreaming = false, showingLatest = false, durationMs = 0, approvalPaused = false, width }：ThinkingBarProps。
// 分支条件：!loading || approvalPaused。
// 返回路径：null。
// 返回路径：( <Box flexDirection="column" width={safeWidth} marginBottom={1}> {isStreaming && <Text color="gray">{truncateDisplayText(showingLatest ? 'Streaming… · showing latest output' : 'Streaming…', safeWidth)}</Text>} <Text>{renderWorkingStatusLin。
// 直接调用：normalizeTerminalWidth。
// 直接调用：truncateDisplayText。
// 直接调用：renderWorkingStatusLine。
// [/maint-note:b46d4c6e5b80]
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

// [maint-note:5a281ecd22d3]
// 数据契约导航：ThinkingTimingSnapshot；字段含义以消费端校验为准。
// 类型组合：Pick<SessionTimingSnapshot, 'currentTurnMs' | 'approvalPaused'>。
// [/maint-note:5a281ecd22d3]
export type ThinkingTimingSnapshot = Pick<SessionTimingSnapshot, 'currentTurnMs' | 'approvalPaused'>
