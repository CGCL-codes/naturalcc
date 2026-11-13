import { Box, Text } from 'ink'
import { UserBodyLines } from './UserSurface.js'
import type { Message, MessageTerminalReason } from '../types/message.js'
import { MarkdownText } from './MarkdownText.js'
import { ToolResultView } from './ToolResultView.js'
import { SessionFooter, TurnSeparator } from './TimingProjection.js'
import { calculateContentWidth, sanitizeTerminalText, wrapDisplayLines } from '../terminal/index.js'
import type { SessionTimingSnapshot } from '../state/sessionTiming.js'

// [maint-note:b1b9fc594a76]
// 数据契约导航：MessageListProps；字段含义以消费端校验为准。
// 成员 messages：readonly Message[]。
// 成员 width：number。
// 成员 layoutGeneration（可省略）：number。
// 成员 timing（可省略）：Pick<SessionTimingSnapshot, 'separators' | 'footer' | 'activeRunId'>。
// [/maint-note:b1b9fc594a76]
interface MessageListProps {
  messages: readonly Message[]
  width: number
  layoutGeneration?: number
  timing?: Pick<SessionTimingSnapshot, 'separators' | 'footer' | 'activeRunId'>
}

// [maint-note:e7d80de74f83]
// 维护导航：MessageList；下列为本函数源码中的接口与路径索引。
// 输入 { messages, width, layoutGeneration: _layoutGeneration, timing }：MessageListProps。
// 返回路径：( <Box flexDirection="column" width={Math.max(1, width)}> {messages.map((msg) => { const runId = terminalRunId(msg.correlationKey) const candidate = runId === null ? undefined : separatorsByRunId.get(runId) const boundary = candidate && !re。
// 直接调用：calculateContentWidth。
// 直接调用：separators.map。
// 直接调用：separators.at。
// 直接调用：Math.max。
// 直接调用：messages.map。
// 直接调用：separators.filter((separator) => !renderedSeparators.has(separator.id) && separator.runId !== footerRunId).map。
// 直接调用：separators.filter。
// 直接调用：separators.some。
// [/maint-note:e7d80de74f83]
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
// [maint-note:4fc20e27ea74]
// 维护导航：pausedHeading；下列为本函数源码中的接口与路径索引。
// 输入 reason：MessageTerminalReason | undefined。
// 返回契约：string。
// 分支条件：switch reason。
// 分支条件：case 'budget_exhausted'。
// 分支条件：case 'no_progress'。
// 返回路径：'Paused: model request budget exhausted'。
// 返回路径：'Paused: no progress'。
// 返回路径：'Paused'。
// [/maint-note:4fc20e27ea74]
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

// [maint-note:b67f4b57db00]
// 维护导航：terminalRunId；下列为本函数源码中的接口与路径索引。
// 输入 correlationKey：string | undefined。
// 返回契约：number | null。
// 返回路径：match ? Number(match[1]) : null。
// 直接调用：correlationKey?.match。
// 直接调用：Number。
// [/maint-note:b67f4b57db00]
function terminalRunId(correlationKey: string | undefined): number | null {
  const match = correlationKey?.match(/^run:(\d+):terminal$/u)
  return match ? Number(match[1]) : null
}

// [maint-note:df7badd33f99]
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
// [/maint-note:df7badd33f99]
function MessageHeader({ message, width }: { message: Message; width: number }) {
  // [maint-note:2a6ffb0d9ac3]
  // 维护导航：render；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 输入 color：string（可省略）。
  // 直接调用：wrapDisplayLines(sanitizeTerminalText(value), width, { hard: true, wordWrap: false }).map。
  // 直接调用：wrapDisplayLines。
  // 直接调用：sanitizeTerminalText。
  // [/maint-note:2a6ffb0d9ac3]
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
