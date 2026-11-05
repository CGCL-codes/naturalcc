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
function messageBodySource(message: Message): string {
  if (message.role === 'tool' && message.toolPresentation) {
    return message.toolPresentation.plainText || message.content
  }
  return message.content
}

interface ProjectedActiveMessage {
  readonly message: Message
  readonly assistantRows?: readonly MarkdownVisualRow[]
}

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
