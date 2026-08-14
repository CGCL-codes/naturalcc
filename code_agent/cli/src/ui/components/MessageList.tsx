import { Box, Text } from 'ink'
import type { Message } from '../types/message.js'
import { MarkdownText } from './MarkdownText.js'
import { ToolResultView } from './ToolResultView.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'

interface MessageListProps {
  messages: readonly Message[]
  width: number
  layoutGeneration?: number
}

export function MessageList({ messages, width, layoutGeneration: _layoutGeneration }: MessageListProps) {
  const messageWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {messages.map((msg) => (
        <Box key={msg.id} flexDirection="column" width={Math.max(1, width)} marginBottom={1}>
          <MessageHeader message={msg} width={messageWidth} />
          {msg.role === 'user' && <Box flexDirection="column">{wrapDisplayLines(`> ${sanitizeTerminalText(msg.content)}`, messageWidth, { hard: true, wordWrap: false }).map((line, index) => <Text color={index === 0 ? 'green' : undefined} key={index}>{line}</Text>)}</Box>}
          {msg.role === 'assistant' && <MarkdownText content={msg.content} width={messageWidth} streaming={msg.isDraft === true} />}
          {msg.role === 'tool' && msg.toolPresentation && <ToolResultView presentation={msg.toolPresentation} width={messageWidth} />}
          {msg.role === 'tool' && !msg.toolPresentation && wrapDisplayLines(sanitizeTerminalText(msg.content), messageWidth, { hard: true, wordWrap: false }).map((line, index) => <Text key={index}>{line}</Text>)}
        </Box>
      ))}
    </Box>
  )
}

function MessageHeader({ message, width }: { message: Message; width: number }) {
  const render = (value: string, color?: string) => wrapDisplayLines(sanitizeTerminalText(value), width, { hard: true, wordWrap: false }).map((line, index) => <Text color={color} key={index}>{line}</Text>)
  if (message.role === 'tool') {
    const color = message.tone === 'error' ? 'red' : 'green'
    const label = message.toolName ?? 'tool'
    return <Box flexDirection="column">{render(`● ${label}`, color)}</Box>
  }
  if (message.terminalKind === 'interrupted') {
    return <Box flexDirection="column">{render(`[Interrupted] ${message.time}`, 'red')}</Box>
  }
  if (message.tone === 'error') {
    return <Box flexDirection="column">{render(`[Error] ${message.time}`, 'red')}</Box>
  }
  return <Box flexDirection="column">{render(message.time, 'gray')}</Box>
}
