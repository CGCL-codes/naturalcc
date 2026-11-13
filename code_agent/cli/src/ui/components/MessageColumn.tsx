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
// [maint-note:f4b2aabb277b]
// 维护导航：MessageColumn；下列为本函数源码中的接口与路径索引。
// 输入 { message, width, assistantRows, }：{ message: Message width: number /** already projected/clipped rows owned by the active viewport. */ assistantRows?: readonly MarkdownVisualRow[] }。
// 返回路径：( <Box flexDirection="column" width={Math.max(1, width)} marginBottom={1}> <MessageHeader message={message} width={messageWidth} /> {message.role === 'user' && <UserBodyLines content={message.content} width={messageWidth} />} {message.role 。
// 直接调用：calculateContentWidth。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines(sanitizeTerminalText(message.content), messageWidth, { hard: true, wordWrap: false }).map。
// 直接调用：wrapDisplayLines。
// 直接调用：sanitizeTerminalText。
// [/maint-note:f4b2aabb277b]
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
// [maint-note:ad306b91048d]
// 维护导航：pausedHeading；下列为本函数源码中的接口与路径索引。
// 输入 reason：MessageTerminalReason | undefined。
// 返回契约：string。
// 分支条件：switch reason。
// 分支条件：case 'budget_exhausted'。
// 分支条件：case 'no_progress'。
// 返回路径：'Paused: model request budget exhausted'。
// 返回路径：'Paused: no progress'。
// 返回路径：'Paused'。
// [/maint-note:ad306b91048d]
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

// [maint-note:4ee2c6fe5fcc]
// 维护导航：MessageHeader；下列为本函数源码中的接口与路径索引。
// 输入 { message, width }：{ message: Message; width: number }。
// 分支条件：message.role === 'tool'。
// 分支条件：message.terminalKind === 'interrupted'。
// 分支条件：message.terminalKind === 'paused'。
// 分支条件：message.tone === 'error'。
// 返回路径：<Box flexDirection="column">{render(`● ${label}`, color)}</Box>。
// 返回路径：<Box flexDirection="column">{render(`[Interrupted] ${message.time}`, 'red')}</Box>。
// 返回路径：<Box flexDirection="column">{render(`${pausedHeading(message.terminalReason)} ${message.time}`, 'yellow')}</Box>。
// 返回路径：<Box flexDirection="column">{render(`[Error] ${message.time}`, 'red')}</Box>。
// 返回路径：<Box flexDirection="column">{render(message.time, 'gray')}</Box>。
// 直接调用：render。
// 直接调用：pausedHeading。
// [/maint-note:4ee2c6fe5fcc]
function MessageHeader({ message, width }: { message: Message; width: number }) {
  // [maint-note:bf98b37f5494]
  // 维护导航：render；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 输入 color：string（可省略）。
  // 直接调用：wrapDisplayLines(sanitizeTerminalText(value), width, { hard: true, wordWrap: false }).map。
  // 直接调用：wrapDisplayLines。
  // 直接调用：sanitizeTerminalText。
  // [/maint-note:bf98b37f5494]
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
