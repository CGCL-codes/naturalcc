import { Box, Text } from 'ink'
import { UserBodyLines } from './UserSurface.js'
import type { Message, MessageTerminalReason } from '../types/message.js'
import { MarkdownText } from './MarkdownText.js'
import { ToolResultView } from './ToolResultView.js'
import { SessionFooter, TurnSeparator } from './TimingProjection.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'
import type { SessionTimingSnapshot } from '../state/sessionTiming.js'

interface MessageListProps {
  messages: readonly Message[]
  width: number
  layoutGeneration?: number
  timing?: Pick<SessionTimingSnapshot, 'separators' | 'footer' | 'activeRunId'>
}

export function MessageList({ messages, width, layoutGeneration: _layoutGeneration, timing }: MessageListProps) {
  const messageWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  const separators = timing?.separators ?? []
  const separatorsByRunId = new Map(separators.map((separator) => [separator.runId, separator]))
  const renderedSeparators = new Set<string>()
  // When idle, the newest completed turn owns the footer slot. While another
  // turn is running the footer is hidden and every completed turn remains a
  // pure separator. This keeps one completed turn from producing two adjacent
  // boundaries.
  const footerRunId = timing?.activeRunId === null && timing.footer
    ? separators.at(-1)?.runId ?? null
    : null
  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {messages.map((msg) => {
        const runId = terminalRunId(msg.correlationKey)
        const candidate = runId === null ? undefined : separatorsByRunId.get(runId)
        const boundary = candidate && !renderedSeparators.has(candidate.id) ? candidate : undefined
        if (boundary) renderedSeparators.add(boundary.id)
        const footer = boundary && boundary.runId === footerRunId ? timing?.footer : undefined
        const separator = footer ? undefined : boundary
        return [
          <Box key={msg.id} flexDirection="column" width={Math.max(1, width)} marginBottom={1}>
            <MessageHeader message={msg} width={messageWidth} />
            {msg.role === 'user' && <UserBodyLines content={msg.content} width={messageWidth} />}
            {msg.role === 'assistant' && <MarkdownText content={msg.content} width={messageWidth} streaming={msg.isDraft === true} />}
            {msg.role === 'tool' && msg.toolPresentation && <ToolResultView presentation={msg.toolPresentation} width={messageWidth} />}
            {msg.role === 'tool' && !msg.toolPresentation && wrapDisplayLines(sanitizeTerminalText(msg.content), messageWidth, { hard: true, wordWrap: false }).map((line, index) => <Text key={index}>{line}</Text>)}
          </Box>,
          separator && <TurnSeparator key={separator.id} width={width} />,
          footer && <SessionFooter key={footer.id} footer={footer} width={width} />,
        ]
      })}
      {separators.filter((separator) => !renderedSeparators.has(separator.id) && separator.runId !== footerRunId).map((separator) => (
        <TurnSeparator key={separator.id} width={width} />
      ))}
      {timing?.footer && !separators.some((separator) => separator.runId === footerRunId && renderedSeparators.has(separator.id)) && (
        <SessionFooter footer={timing.footer} width={width} />
      )}
    </Box>
  )
}

/**
 * Stable, human-readable diagnostics for the paused reason enum. The UI never
 * guesses from checkpoint text, and no protocol id or raw provider payload is
 * rendered here.
 */
function pausedHeading(reason: MessageTerminalReason | undefined): string {
  switch (reason) {
    case 'budget_exhausted':
      return 'Paused: model request budget exhausted'
    case 'no_progress':
      return 'Paused: no progress'
    default:
      return 'Paused'
  }
}

function terminalRunId(correlationKey: string | undefined): number | null {
  const match = correlationKey?.match(/^run:(\d+):terminal$/u)
  return match ? Number(match[1]) : null
}

function MessageHeader({ message, width }: { message: Message; width: number }) {
  const render = (value: string, color?: string) => wrapDisplayLines(sanitizeTerminalText(value), width, { hard: true, wordWrap: false }).map((line, index) => <Text color={color} key={index}>{line}</Text>)
  if (message.role === 'tool') {
    const color = message.toolPhase === 'denied'
      ? 'red'
      : message.tone === 'error' ? 'red' : 'green'
    const label = message.toolName ?? 'tool'
    return <Box flexDirection="column">{render(`● ${label}`, color)}</Box>
  }
  if (message.terminalKind === 'interrupted') {
    return <Box flexDirection="column">{render(`[Interrupted] ${message.time}`, 'red')}</Box>
  }
  if (message.terminalKind === 'paused') {
    return <Box flexDirection="column">{render(`${pausedHeading(message.terminalReason)} ${message.time}`, 'yellow')}</Box>
  }
  if (message.tone === 'error') {
    return <Box flexDirection="column">{render(`[Error] ${message.time}`, 'red')}</Box>
  }
  return <Box flexDirection="column">{render(message.time, 'gray')}</Box>
}
