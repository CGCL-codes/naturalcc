import { Box, Text, useStdout } from 'ink'
import type { Message } from '../types/message.js'
import { calculateContentWidth, sanitizeTerminalText, truncateDisplayText, wrapDisplayLines } from '../terminal/index.js'
import { ThinkingBar } from '../components/ThinkingBar.js'
import type { ApprovalController, ApprovalRequest } from '../../tools/approval.js'
import { SelectMenu } from '../components/SelectMenu.js'
import { useRef, useState } from 'react'
import type { TerminalSectionData } from '../components/TerminalSection.js'
import type { SelectOption } from '../state/selectMenu.js'
import type { KeyboardInputHandler } from '../state/keyboardInput.js'
import { buildMarkdownVisualRows, type MarkdownVisualRow } from '../components/MarkdownText.js'
import { MessageColumn } from '../components/MessageColumn.js'
import {
  boundApprovalPresentation,
  countActiveChromeRows,
  countMessageRows,
} from './frameLayout.js'
import type { ApprovalPresentation } from '../../tools/approval.js'
import { StreamingMarkdownRenderer, renderMarkdown } from '../markdown/renderer.js'

// [maint-note:70319c098eca]
// 数据契约导航：ActiveTurnProps；字段含义以消费端校验为准。
// 成员 messages：readonly Message[]。
// 成员 loading：boolean。
// 成员 streaming：boolean。
// 成员 streamingContent（可省略）：string。
// 成员 durationMs：number。
// 成员 approvalPaused：boolean。
// 成员 approvalPending：ApprovalRequest | null。
// 成员 width：number。
// 成员 onApprovalDecision：(requestId: string, approved: boolean) => void。
// 成员 onApprovalCancel：() => void。
// 成员 onRegisterApprovalKeyboard（可省略）：(handler: KeyboardInputHandler | null) => void。
// 成员 availableRows（可省略）：number。
// 成员 streamGeneration（可省略）：number。
// 成员 layoutGeneration（可省略）：number。
// [/maint-note:70319c098eca]
export interface ActiveTurnProps {
  messages: readonly Message[]
  loading: boolean
  streaming: boolean
  /** Legacy fixture prop; Assistant content is owned by messages. */
  streamingContent?: string
  durationMs: number
  approvalPaused: boolean
  approvalPending: ApprovalRequest | null
  width: number
  onApprovalDecision: (requestId: string, approved: boolean) => void
  onApprovalCancel: () => void
  onRegisterApprovalKeyboard?: (handler: KeyboardInputHandler | null) => void
  /** Rows available after the mutable interactive chrome is accounted for. */
  availableRows?: number
  /** Session generation invalidates a draft owner after /clear. */
  streamGeneration?: number
  /** Native width generation disambiguates same-width resize lifecycles. */
  layoutGeneration?: number
}

/**
 * dynamic region: everything that may still change in place during the
 * current turn — the bounded live streaming viewport, running/denied tool
 * activity, the approval prompt, and the current run's `Working for` timing.
 * Nothing here is frozen until the run's records finalize.
 */
/**
 * The active region receives a row budget from the top-level frame ledger.
 * Its own chrome is counted here so message records cannot push a long tool
 * result or approval prompt through the terminal height.
 */
// [maint-note:f23d5451d802]
// 维护导航：messageBodySource；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 返回契约：string。
// 分支条件：message.role === 'tool' && message.toolPresentation。
// 返回路径：message.toolPresentation.plainText || message.content。
// 返回路径：message.content。
// [/maint-note:f23d5451d802]
function messageBodySource(message: Message): string {
  if (message.role === 'tool' && message.toolPresentation) {
    return message.toolPresentation.plainText || message.content
  }
  return message.content
}

// [maint-note:c0a99761d946]
// 数据契约导航：ProjectedActiveMessage；字段含义以消费端校验为准。
// 成员 message：Message。
// 成员 assistantRows（可省略）：readonly MarkdownVisualRow[]。
// [/maint-note:c0a99761d946]
interface ProjectedActiveMessage {
  readonly message: Message
  readonly assistantRows?: readonly MarkdownVisualRow[]
}

// [maint-note:0574eaa179a1]
// 维护导航：projectMessageToRows；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 输入 width：number。
// 输入 maxRows：number。
// 输入 assistantRows：readonly MarkdownVisualRow[]（可省略）。
// 返回契约：ProjectedActiveMessage | null。
// 分支条件：message.role === 'assistant' && assistantRows。
// 分支条件：bodyBudget < 1。
// 分支条件：baseRows > maxRows。
// 分支条件：countMessageRows(projected, width) <= maxRows。
// 遍历边界：let count = Math.min(bodyBudget, wrapped.length); count >= 1; count -= 1。
// 返回路径：null。
// 返回路径：{ message, assistantRows: assistantRows.slice(-bodyBudget) }。
// 返回路径：{ message: projected }。
// 写入目标：count。
// 直接调用：countMessageRows。
// 直接调用：assistantRows.slice。
// 直接调用：Math.max。
// 直接调用：sanitizeTerminalText。
// 直接调用：messageBodySource。
// 直接调用：calculateContentWidth。
// 直接调用：wrapDisplayLines。
// 直接调用：Math.min。
// 直接调用：wrapped.slice(-count).join。
// 直接调用：wrapped.slice。
// 直接调用：candidate.startsWith。
// 直接调用：candidate.slice。
// [/maint-note:0574eaa179a1]
function projectMessageToRows(
  message: Message,
  width: number,
  maxRows: number,
  assistantRows?: readonly MarkdownVisualRow[],
): ProjectedActiveMessage | null {
  if (message.role === 'assistant' && assistantRows) {
    const fixedRows = countMessageRows({ ...message, content: '' }, width) - 1
    const bodyBudget = maxRows - fixedRows
    if (bodyBudget < 1) return null
    return { message, assistantRows: assistantRows.slice(-bodyBudget) }
  }
  const empty = { ...message, content: '', toolPresentation: undefined }
  const baseRows = countMessageRows(empty, width)
  if (baseRows > maxRows) return null
  const bodyBudget = Math.max(1, maxRows - baseRows + 1)
  const source = sanitizeTerminalText(messageBodySource(message))
  const contentWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  const prefix = message.role === 'user' ? '> ' : ''
  const wrapped = wrapDisplayLines(`${prefix}${source}`, contentWidth, { hard: true, trim: false, wordWrap: false })
  for (let count = Math.min(bodyBudget, wrapped.length); count >= 1; count -= 1) {
    const candidate = wrapped.slice(-count).join('\n')
    const content = message.role === 'user' && candidate.startsWith('> ')
      ? candidate.slice(2)
      : candidate
    const projected = { ...empty, content }
    if (countMessageRows(projected, width) <= maxRows) return { message: projected }
  }
  return null
}

/**
 * Whole-region live viewport: keeps the rendered dynamic frame below the
 * terminal height so the renderer never has to fall back to a full-screen
 * clear with static replay. Leading messages (and leading lines of the first
 * kept message) are dropped with a bounded notice; the frozen stable region
 * still receives every record in full when the run terminalizes.
 */
// [maint-note:0ed72a151c02]
// 维护导航：planActiveViewport；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly Message[]。
// 输入 width：number。
// 输入 rows：number。
// 输入 options：{ reservedRows?: number assistantRows?: ReadonlyMap<string, readonly MarkdownVisualRow[]> noticeInReservedChrome?: boolean }；缺省表达式 {}。
// 返回契约：{ kept: readonly Message[] hiddenLines: number notice: string | null assistantRows: ReadonlyMap<string, readonly MarkdownVisualRow[]> }。
// 分支条件：message.role !== 'assistant' || projectedAssistantRows.has(message.id)。
// 分支条件：counts.reduce((total, count) => total + count, 0) <= available。
// 分支条件：!message。
// 分支条件：count <= remaining。
// 分支条件：rowsForMessage。
// 分支条件：projected。
// 分支条件：projected.assistantRows。
// 分支条件：hiddenLines === 0 && kept.length === messages.length。
// 遍历边界：const message ← messages。
// 遍历边界：let index = messages.length - 1; index >= 0; index -= 1。
// 遍历边界：let older = index - 1; older >= 0; older -= 1。
// 返回路径：{ kept: messages, hiddenLines: 0, notice: null, assistantRows: projectedAssistantRows }。
// 返回路径：{ kept, hiddenLines, notice, assistantRows: keptAssistantRows }。
// 写入目标：index。
// 写入目标：remaining。
// 写入目标：hiddenLines。
// 写入目标：older。
// 直接调用：calculateContentWidth。
// 直接调用：projectedAssistantRows.has。
// 直接调用：new StreamingMarkdownRenderer({ maxWidth: contentWidth }).push。
// 直接调用：renderMarkdown。
// 直接调用：projectedAssistantRows.set。
// 直接调用：buildMarkdownVisualRows。
// 直接调用：messages.map。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：counts.reduce。
// 直接调用：countMessageRows。
// 直接调用：kept.unshift。
// 直接调用：projectedAssistantRows.get。
// 直接调用：keptAssistantRows.set。
// 直接调用：projectMessageToRows。
// [/maint-note:0ed72a151c02]
export function planActiveViewport(
  messages: readonly Message[],
  width: number,
  rows: number,
  options: {
    reservedRows?: number
    assistantRows?: ReadonlyMap<string, readonly MarkdownVisualRow[]>
    noticeInReservedChrome?: boolean
  } = {},
): {
  kept: readonly Message[]
  hiddenLines: number
  notice: string | null
  assistantRows: ReadonlyMap<string, readonly MarkdownVisualRow[]>
} {
  const projectedAssistantRows = new Map<string, readonly MarkdownVisualRow[]>(options.assistantRows)
  const contentWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  for (const message of messages) {
    if (message.role !== 'assistant' || projectedAssistantRows.has(message.id)) continue
    const projection = message.isDraft === true
      ? new StreamingMarkdownRenderer({ maxWidth: contentWidth }).push(message.content)
      : renderMarkdown(message.content, { maxWidth: contentWidth })
    projectedAssistantRows.set(message.id, buildMarkdownVisualRows(projection.blocks, contentWidth))
  }
  const counts = messages.map((message) => countMessageRows(message, width, projectedAssistantRows.get(message.id)))
  const available = Math.max(0, Math.floor(rows) - Math.max(0, options.reservedRows ?? 0))
  // The omission notice is only needed when something is actually hidden.
  // Reserving it unconditionally makes an exact-fit final batch disappear
  // one row too early, which is especially visible while the startup Header
  // reservation is being released.
  if (counts.reduce((total, count) => total + count, 0) <= available) {
    return { kept: messages, hiddenLines: 0, notice: null, assistantRows: projectedAssistantRows }
  }
  const budget = Math.max(0, available - (options.noticeInReservedChrome ? 0 : 1))
  let hiddenLines = 0
  let remaining = budget
  const kept: Message[] = []
  const keptAssistantRows = new Map<string, readonly MarkdownVisualRow[]>()
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (!message) continue
    const count = counts[index] ?? countMessageRows(message, width)
    if (count <= remaining) {
      kept.unshift(message)
      const rowsForMessage = projectedAssistantRows.get(message.id)
      if (rowsForMessage) keptAssistantRows.set(message.id, rowsForMessage)
      remaining -= count
      continue
    }
    const projected = projectMessageToRows(message, width, remaining, projectedAssistantRows.get(message.id))
    if (projected) {
      kept.unshift(projected.message)
      if (projected.assistantRows) keptAssistantRows.set(message.id, projected.assistantRows)
      hiddenLines += Math.max(0, count - countMessageRows(projected.message, width, projected.assistantRows))
      remaining = 0
    } else {
      hiddenLines += count
    }
    // Older records cannot fit once the newest record consumed the budget.
    for (let older = index - 1; older >= 0; older -= 1) hiddenLines += counts[older] ?? 0
    break
  }
  if (hiddenLines === 0 && kept.length === messages.length) {
    return { kept: messages, hiddenLines: 0, notice: null, assistantRows: projectedAssistantRows }
  }
  const notice = hiddenLines > 0
    ? '… showing latest output'
    : null
  return { kept, hiddenLines, notice, assistantRows: keptAssistantRows }
}

// [maint-note:6399dab7f781]
// 维护导航：ActiveTurn；下列为本函数源码中的接口与路径索引。
// 输入 { messages, loading, streaming, durationMs, approvalPaused, approvalPending, width, onApprovalDecision, onApprovalCancel, onRegisterApprovalKeyboard, availableRows, streamGeneration = 0, layoutGeneration = 0, }：ActiveTurnProps。
// 返回路径：( <Box flexDirection="column" width={Math.max(1, width)}> {bounded.map((message) => ( <MessageColumn key={message.id} message={message} width={width} assistantRows={plan.assistantRows.get(message.id)} /> ))} {plan.notice && !streaming ? <Te。
// 直接调用：useStdout。
// 直接调用：boundApprovalPresentation。
// 直接调用：Math.max。
// 直接调用：countActiveChromeRows。
// 直接调用：useStreamingAssistantRows。
// 直接调用：planActiveViewport。
// 直接调用：bounded.map。
// 直接调用：truncateDisplayText。
// [/maint-note:6399dab7f781]
export function ActiveTurn({
  messages, loading, streaming, durationMs, approvalPaused,
  approvalPending, width, onApprovalDecision, onApprovalCancel, onRegisterApprovalKeyboard, availableRows,
  streamGeneration = 0, layoutGeneration = 0,
}: ActiveTurnProps) {
  const { stdout } = useStdout()
  const rows = availableRows ?? (typeof stdout?.rows === 'number' && stdout.rows > 0 ? stdout.rows : 24)
  const approvalPresentation = approvalPending
    ? boundApprovalPresentation(approvalPending, width, Math.max(1, rows))
    : null
  const activeLoading = loading && !approvalPaused
  const chromeRows = countActiveChromeRows({
    loading: activeLoading,
    streaming,
    approval: approvalPresentation,
    width,
  })
  const assistantRows = useStreamingAssistantRows(messages, width, streamGeneration, layoutGeneration)
  const plan = planActiveViewport(messages, width, rows, {
    reservedRows: chromeRows,
    assistantRows,
    noticeInReservedChrome: streaming,
  })
  const bounded = plan.kept
  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {bounded.map((message) => (
        <MessageColumn key={message.id} message={message} width={width} assistantRows={plan.assistantRows.get(message.id)} />
      ))}
      {plan.notice && !streaming ? <Text color="gray">{truncateDisplayText(plan.notice, Math.max(1, width))}</Text> : null}
      <ThinkingBar
        loading={loading && !approvalPaused}
        isStreaming={streaming}
        showingLatest={streaming && plan.hiddenLines > 0}
        durationMs={durationMs}
        approvalPaused={approvalPaused}
        width={width}
      />
      {approvalPending && approvalPresentation && (
        <ApprovalPrompt
          key={approvalPending.requestId}
          request={approvalPending}
          presentation={approvalPresentation}
          width={width}
          onDecision={onApprovalDecision}
          onCancel={onApprovalCancel}
          onRegisterKeyboard={onRegisterApprovalKeyboard}
        />
      )}
    </Box>
  )
}

/** Internal read-only projection seam used by ActiveTurn and its lifecycle tests. */
// [maint-note:fb6552e918cb]
// 维护导航：useStreamingAssistantRows；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly Message[]。
// 输入 width：number。
// 输入 streamGeneration：number。
// 输入 layoutGeneration：number。
// 返回契约：ReadonlyMap<string, readonly MarkdownVisualRow[]>。
// 分支条件：!draft。
// 分支条件：!state || state.owner !== owner || state.width !== contentWidth || state.layoutGeneration !== layoutGeneration。
// 分支条件：!draft.content.startsWith(state.inputSource)。
// 分支条件：delta.length > 0。
// 分支条件：needsProjection。
// 返回路径：new Map()。
// 返回路径：state.rowMap。
// 写入目标：stateRef.current。
// 写入目标：state。
// 写入目标：needsProjection。
// 写入目标：state.inputSource。
// 写入目标：state.rows。
// 写入目标：state.rowMap。
// 直接调用：messages.findLast。
// 直接调用：useRef。
// 直接调用：calculateContentWidth。
// 直接调用：draft.content.startsWith。
// 直接调用：state.renderer.reset。
// 直接调用：draft.content.slice。
// 直接调用：state.renderer.append。
// 直接调用：state.renderer.snapshot。
// 直接调用：buildMarkdownVisualRows。
// [/maint-note:fb6552e918cb]
export function useStreamingAssistantRows(
  messages: readonly Message[],
  width: number,
  streamGeneration: number,
  layoutGeneration: number,
): ReadonlyMap<string, readonly MarkdownVisualRow[]> {
  const draft = messages.findLast(message => message.role === 'assistant' && message.isDraft === true)
  const stateRef = useRef<{
    owner: string
    width: number
    layoutGeneration: number
    inputSource: string
    renderer: StreamingMarkdownRenderer
    rows: readonly MarkdownVisualRow[]
    rowMap: ReadonlyMap<string, readonly MarkdownVisualRow[]>
  } | null>(null)
  if (!draft) {
    stateRef.current = null
    return new Map()
  }

  const contentWidth = calculateContentWidth({ terminalWidth: width, paddingRight: 4, minWidth: 1 })
  const owner = `${streamGeneration}:${draft.id}:${draft.correlationKey ?? ''}`
  let state = stateRef.current
  let needsProjection = false
  if (!state || state.owner !== owner || state.width !== contentWidth || state.layoutGeneration !== layoutGeneration) {
    state = {
      owner,
      width: contentWidth,
      layoutGeneration,
      inputSource: '',
      renderer: new StreamingMarkdownRenderer({ maxWidth: contentWidth }),
      rows: [],
      rowMap: new Map(),
    }
    needsProjection = true
  }
  if (!draft.content.startsWith(state.inputSource)) {
    state.renderer.reset()
    state.inputSource = ''
    needsProjection = true
  }
  const delta = draft.content.slice(state.inputSource.length)
  if (delta.length > 0) {
    state.renderer.append(delta)
    state.inputSource = draft.content
    needsProjection = true
  }
  if (needsProjection) {
    const projection = state.renderer.snapshot()
    state.rows = buildMarkdownVisualRows(projection.blocks, contentWidth)
    state.rowMap = new Map([[draft.id, state.rows]])
  }
  stateRef.current = state
  return state.rowMap
}

// [maint-note:81076523902b]
// 维护导航：ApprovalPrompt；下列为本函数源码中的接口与路径索引。
// 输入 { request, presentation, width, onDecision, onCancel, onRegisterKeyboard, }：{ request: ApprovalRequest presentation: ApprovalPresentation width: number onDecision: (requestId: string, approved: boolean) => void onCancel: () => void onRegisterKeyboard?: (handler: KeyboardInputHandler | null) => void }。
// 返回路径：( <SelectMenu menuId={request.requestId} title={presentation.title} sections={sections} width={width} options={options} selected={selected} active={active} onChange={setSelected} onConfirm={(choice) => { setActive(false) onDecision(request.。
// 直接调用：useState。
// [/maint-note:81076523902b]
function ApprovalPrompt({
  request,
  presentation,
  width,
  onDecision,
  onCancel,
  onRegisterKeyboard,
}: {
  request: ApprovalRequest
  presentation: ApprovalPresentation
  width: number
  onDecision: (requestId: string, approved: boolean) => void
  onCancel: () => void
  onRegisterKeyboard?: (handler: KeyboardInputHandler | null) => void
}) {
  const [selected, setSelected] = useState<'yes' | 'no'>('no')
  const [active, setActive] = useState(true)
  const options: readonly SelectOption<'yes' | 'no'>[] = [
    { id: 'yes', label: 'Yes', value: 'yes' },
    { id: 'no', label: 'No', value: 'no' },
  ]
  const sections: readonly TerminalSectionData[] = [
    { id: 'instruction', title: `${presentation.instructionLabel}:`, body: presentation.instruction },
    { id: 'reason', title: 'Reason:', body: presentation.reason },
  ]
  return (
    <SelectMenu
      menuId={request.requestId}
      title={presentation.title}
      sections={sections}
      width={width}
      options={options}
      selected={selected}
      active={active}
      onChange={setSelected}
      onConfirm={(choice) => {
        setActive(false)
        onDecision(request.requestId, choice === 'yes')
      }}
      onCancel={() => {
        setActive(false)
        onCancel()
      }}
      onRegisterKeyboard={onRegisterKeyboard}
    />
  )
}

export type { ApprovalController }
