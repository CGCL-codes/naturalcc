import { Box, Text } from 'ink'
import { displayWidth, fillDisplayWidth, normalizeTerminalWidth, truncateDisplayText } from '../terminal/index.js'
import { formatDuration, type SessionFooterDescriptor } from '../state/sessionTiming.js'
import { UI_PALETTE } from '../theme.js'

// [maint-note:53ea9cb0328e]
// 维护导航：renderWorkingStatus；下列为本函数源码中的接口与路径索引。
// 输入 durationMs：number。
// 返回契约：string。
// 返回路径：`Working for ${formatDuration(durationMs)}`。
// 直接调用：formatDuration。
// [/maint-note:53ea9cb0328e]
export function renderWorkingStatus(durationMs: number): string {
  return `Working for ${formatDuration(durationMs)}`
}

// [maint-note:7eb1c051a5fe]
// 维护导航：renderWorkingStatusLine；下列为本函数源码中的接口与路径索引。
// 输入 durationMs：number。
// 输入 width：number。
// 返回契约：string。
// 分支条件：displayWidth(status) <= safeWidth。
// 返回路径：status。
// 返回路径：displayWidth(duration) <= safeWidth ? duration : truncateDisplayText(duration, safeWidth)。
// 直接调用：normalizeTerminalWidth。
// 直接调用：renderWorkingStatus。
// 直接调用：displayWidth。
// 直接调用：formatDuration。
// 直接调用：truncateDisplayText。
// [/maint-note:7eb1c051a5fe]
export function renderWorkingStatusLine(durationMs: number, width: number): string {
  const safeWidth = normalizeTerminalWidth(width, 1)
  const status = renderWorkingStatus(durationMs)
  if (displayWidth(status) <= safeWidth) return status

  // Keep the useful numeric fact when the descriptive label cannot fit. Only
  // truncate the duration itself as a last resort; this prevents Ink from
  // wrapping either the status or its units into multiple logical rows.
  const duration = formatDuration(durationMs)
  return displayWidth(duration) <= safeWidth
    ? duration
    : truncateDisplayText(duration, safeWidth)
}

// [maint-note:67755ec98d66]
// 维护导航：renderTurnSeparator；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：string。
// 返回路径：fillDisplayWidth('─', normalizeTerminalWidth(width, 1))。
// 直接调用：fillDisplayWidth。
// 直接调用：normalizeTerminalWidth。
// [/maint-note:67755ec98d66]
export function renderTurnSeparator(width: number): string {
  return fillDisplayWidth('─', normalizeTerminalWidth(width, 1))
}

// [maint-note:99428cc345da]
// 维护导航：renderSessionFooter；下列为本函数源码中的接口与路径索引。
// 输入 durationMs：number。
// 输入 width：number。
// 返回契约：string。
// 分支条件：displayWidth(full) <= safeWidth。
// 分支条件：displayWidth(compact) <= safeWidth。
// 返回路径：full + fillDisplayWidth('─', safeWidth - displayWidth(full))。
// 返回路径：compact + fillDisplayWidth('─', safeWidth - displayWidth(compact))。
// 返回路径：fillDisplayWidth('─', safeWidth)。
// 直接调用：normalizeTerminalWidth。
// 直接调用：formatDuration。
// 直接调用：displayWidth。
// 直接调用：fillDisplayWidth。
// [/maint-note:99428cc345da]
export function renderSessionFooter(durationMs: number, width: number): string {
  const safeWidth = normalizeTerminalWidth(width, 1)
  const duration = formatDuration(durationMs)
  const full = `─ Worked for ${duration} `
  if (displayWidth(full) <= safeWidth) return full + fillDisplayWidth('─', safeWidth - displayWidth(full))

  const compact = `─ ${duration} `
  if (displayWidth(compact) <= safeWidth) return compact + fillDisplayWidth('─', safeWidth - displayWidth(compact))

  // Once even the compact label cannot fit, reserve the whole row for a
  // safe horizontal rule rather than emitting a clipped, misleading label.
  return fillDisplayWidth('─', safeWidth)
}

// [maint-note:733f7900472a]
// 维护导航：TurnSeparator；下列为本函数源码中的接口与路径索引。
// 输入 { width }：{ width: number }。
// 返回路径：<Text color={UI_PALETTE.boundary}>{renderTurnSeparator(width)}</Text>。
// 直接调用：renderTurnSeparator。
// [/maint-note:733f7900472a]
export function TurnSeparator({ width }: { width: number }) {
  return <Text color={UI_PALETTE.boundary}>{renderTurnSeparator(width)}</Text>
}

// [maint-note:185fb849ad07]
// 维护导航：SessionFooter；下列为本函数源码中的接口与路径索引。
// 输入 { footer, width }：{ footer: SessionFooterDescriptor; width: number }。
// 返回路径：( <Box flexDirection="column"> <Text color={UI_PALETTE.boundary}>{renderSessionFooter(footer.durationMs, width)}</Text> </Box> )。
// 直接调用：renderSessionFooter。
// [/maint-note:185fb849ad07]
export function SessionFooter({ footer, width }: { footer: SessionFooterDescriptor; width: number }) {
  return (
    <Box flexDirection="column">
      <Text color={UI_PALETTE.boundary}>{renderSessionFooter(footer.durationMs, width)}</Text>
    </Box>
  )
}
