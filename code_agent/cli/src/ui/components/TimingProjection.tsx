import { Box, Text } from 'ink'
import { displayWidth, fillDisplayWidth, normalizeTerminalWidth, truncateDisplayText } from '../terminal/index.js'
import { formatDuration, type SessionFooterDescriptor } from '../state/sessionTiming.js'
import { UI_PALETTE } from '../theme.js'

export function renderWorkingStatus(durationMs: number): string {
  return `Working for ${formatDuration(durationMs)}`
}

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

export function renderTurnSeparator(width: number): string {
  return fillDisplayWidth('─', normalizeTerminalWidth(width, 1))
}

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

export function TurnSeparator({ width }: { width: number }) {
  return <Text color={UI_PALETTE.boundary}>{renderTurnSeparator(width)}</Text>
}

export function SessionFooter({ footer, width }: { footer: SessionFooterDescriptor; width: number }) {
  return (
    <Box flexDirection="column">
      <Text color={UI_PALETTE.boundary}>{renderSessionFooter(footer.durationMs, width)}</Text>
    </Box>
  )
}
