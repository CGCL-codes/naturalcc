import { Box, Text } from 'ink'
import type { Message, MessageTerminalReason } from '../types/message.js'
import { MarkdownText, MarkdownVisualRows, type MarkdownVisualRow } from './MarkdownText.js'
import { ToolResultView } from './ToolResultView.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'
import { UserBodyLines } from './UserSurface.js'

/**
 * Single immutable message record projection. Shared by the stable region
 * and the active-turn region so a record renders identically before and
 * after it freezes.
 */
export function MessageColumn({
  message,
  width,
  assistantRows,
}: {
  message: Message
  width: number
  /** already projected/clipped rows owned by the active viewport. */
  assistantRows?: readonly MarkdownVisualRow[]
}) {
  const messageWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  return (
    <Box flexDirection="column" width={Math.max(1, width)} marginBottom={1}>
      <MessageHeader message={message} width={messageWidth} />
      {message.role === 'user' && <UserBodyLines content={message.content} width={messageWidth} />}
      {message.role === 'assistant' && assistantRows && <MarkdownVisualRows rows={assistantRows} width={messageWidth} />}
      {message.role === 'assistant' && !assistantRows && <MarkdownText content={message.content} width={messageWidth} streaming={message.isDraft === true} />}
      {message.role === 'tool' && message.toolPresentation && <ToolResultView presentation={message.toolPresentation} width={messageWidth} />}
      {message.role === 'tool' && !message.toolPresentation && wrapDisplayLines(sanitizeTerminalText(message.content), messageWidth, { hard: true, wordWrap: false }).map((line, index) => <Text key={index}>{line}</Text>)}
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
