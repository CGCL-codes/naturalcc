import type { AgentEvent } from '../../agent/types/event.js'
import type { ToolDisplayFacts } from '../../tools/types.js'
import { assertNeverAgentEvent } from '../../agent/terminal.js'
import type { Message, MessageTerminalKind, MessageTerminalReason, MessageType, Role, ToolMessagePhase } from '../types/message.js'
import { presentDeniedTool, presentToolResult, type ToolResultPresentation } from '../toolPresenters/index.js'
import { formatTime } from '../utils/now-time.js'
import {
  commitDetailRecord,
  createDetailStoreState,
  type DetailStoreState,
} from './detailStore.js'

export interface MessageState {
  messages: Message[]
  nextId: number
  activeRunId: number | null
  terminalRunId: number | null
  /**
   * the single bounded holder of retained detail lines. Messages keep
   * only light references; eviction here actually frees the content.
   */
  detailStore: DetailStoreState
  /**
   * content width for NEW physical projections. Committed summaries are
   * never re-presented on resize; only presentations built after the change
   * see the new width.
   */
  presentationWidth: number
  /** Native layout generation used to build the current active projection. */
  presentationGeneration: number
  /**
   * Ephemeral facts for final tool results that have not reached their first
   * Static commit. Resize may re-project these at the commit width; finish
   * releases the facts so stable history never retains a second heavy copy.
   */
  activeToolPresentationSources: ReadonlyMap<string, ActiveToolPresentationSource>
  /** Injectable wall clock; never read during component render. */
  readonly clock: () => number
}

export interface ActiveToolPresentationSource {
  readonly toolName: string
  readonly output: string
  readonly input: unknown
  readonly display?: ToolDisplayFacts
}

export function createMessageState(clock: () => number = () => Date.now()): MessageState {
  return {
    messages: [],
    nextId: 1,
    activeRunId: null,
    terminalRunId: null,
    detailStore: createDetailStoreState(),
    presentationWidth: 80,
    // -1 leaves generation 0 available for the first real terminal snapshot.
    presentationGeneration: -1,
    activeToolPresentationSources: new Map(),
    clock,
  }
}

export const initialMessageState: MessageState = createMessageState()

export type MessageAction =
  | { type: 'begin_run'; runId: number }
  | { type: 'add'; role: Role; content: string; status: MessageType }
  | { type: 'add_run_error'; runId: number; content: string }
  | { type: 'agent_event'; runId: number; event: AgentEvent }
  | { type: 'finish_run'; runId: number; presentationWidth?: number; presentationGeneration?: number }
  | { type: 'clear' }
  | { type: 'set_presentation_width'; width: number; presentationGeneration?: number }

export function messageReducer(state: MessageState, action: MessageAction): MessageState {
  switch (action.type) {
    case 'begin_run':
      if (state.activeRunId !== null) return state
      return { ...state, activeRunId: action.runId, terminalRunId: null }
    case 'add':
      return appendMessage(state, message(action.role, action.content, action.status))
    case 'add_run_error':
      if (state.activeRunId !== action.runId) return state
      return appendMessage(state, {
        ...message('assistant', action.content, 'error'),
        tone: 'error',
        terminalKind: 'error',
        workspaceMayHaveChanged: false,
        correlationKey: turnTerminalKey(action.runId),
      })
    case 'agent_event':
      if (state.activeRunId !== action.runId) return state
      return reduceAgentEvent(state, action.runId, action.event)
    case 'finish_run':
      if (state.activeRunId !== action.runId) return state
      {
        const atCommitLayout = action.presentationWidth === undefined
          ? state
          : applyPresentationLayout(state, action.presentationWidth, action.presentationGeneration)
      return {
        ...removeAssistantDraft(finalizeRunningTools(atCommitLayout, 'interrupted'), action.runId),
        activeRunId: null,
        terminalRunId: null,
        activeToolPresentationSources: new Map(),
      }
      }
    case 'clear':
      return {
        ...createMessageState(state.clock),
        presentationWidth: state.presentationWidth,
        presentationGeneration: state.presentationGeneration,
        // A new generation invalidates every retained entry; late old-run
        // events cannot resurrect detail.
        detailStore: createDetailStoreState(state.detailStore.generation + 1),
        activeToolPresentationSources: new Map(),
      }
    case 'set_presentation_width': {
      return applyPresentationLayout(state, action.width, action.presentationGeneration)
    }
  }
}

function applyPresentationLayout(
  state: MessageState,
  widthValue: number,
  generationValue?: number,
): MessageState {
  const width = Number.isFinite(widthValue) ? Math.max(1, Math.floor(widthValue)) : 1
  const generation = generationValue === undefined
    ? state.presentationGeneration + (width === state.presentationWidth ? 0 : 1)
    : Math.max(0, Math.floor(generationValue))
  if (generation < state.presentationGeneration) return state
  // One layout generation names one immutable width fact. A conflicting
  // duplicate cannot rewrite a projection that has already observed it.
  if (generation === state.presentationGeneration && width !== state.presentationWidth) return state
  if (width === state.presentationWidth && generation === state.presentationGeneration) return state
  // Stable summaries have no source entry and remain frozen. Active results
  // retain their neutral display facts until this projection is finalized.
  return reprojectActiveToolResults({
    ...state,
    presentationWidth: width,
    presentationGeneration: generation,
  })
}

function reduceAgentEvent(state: MessageState, runId: number, event: AgentEvent): MessageState {
  if (state.terminalRunId === runId) return state
  switch (event.type) {
    case 'user_message':
      return appendMessage(state, {
        ...message('user', event.content, 'normal'),
        correlationKey: turnUserKey(runId),
      })
    case 'assistant_delta':
      return appendAssistantDelta(state, runId, event.content)
    case 'assistant_segment':
      return appendAssistantSegment(state, runId, event.content)
    case 'assistant_message':
      return settleAssistantDraft(state, runId, event.content)
    case 'tool_start':
      return upsertToolStart(state, runId, event.callId, event.name, event.input)
    case 'approval_required':
      return markApprovalPending(state, runId, event.callId, event.name, event.requestId, event.input, event.reason)
    case 'approval_resolved':
      return resolveApproval(state, event.requestId, event.approved)
    case 'tool_result':
      return upsertToolResult(state, runId, event.callId, event.name, event.output, event.isError, event.code, event.display)
    case 'error':
      return {
        ...appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'failed', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'error', 'error', event.workspaceMayHaveChanged, turnTerminalKey(runId)),
        ),
        terminalRunId: runId,
      }
    case 'interrupted':
      return {
        ...appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'normal', 'interrupted', event.workspaceMayHaveChanged, turnTerminalKey(runId)),
        ),
        terminalRunId: runId,
      }
    case 'paused':
      // The paused terminal is the only carrier of the checkpoint body. A
      // streaming draft for this run is a revocable draft of the same text,
      // so it is settled in place instead of being dropped and re-appended.
      return {
        ...settlePausedCheckpoint(
          finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged),
          runId,
          event.content,
          event.reason,
          event.workspaceMayHaveChanged,
        ),
        terminalRunId: runId,
      }
    case 'done':
      return { ...state, terminalRunId: runId }
    default:
      return assertNeverAgentEvent(event)
  }
}

function appendAssistantDelta(state: MessageState, runId: number, content: string): MessageState {
  if (!content) return state
  const correlationKey = assistantDraftKey(runId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey && item.isDraft)
  if (index === -1) {
    return appendMessage(state, {
      ...message('assistant', content, 'normal'),
      correlationKey,
      isDraft: true,
    })
  }
  const existing = state.messages[index]
  if (!existing) return state
  return replaceMessage(state, index, { ...existing, content: existing.content + content })
}

function settleAssistantDraft(state: MessageState, runId: number, content: string): MessageState {
  const correlationKey = assistantDraftKey(runId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey && item.isDraft)
  if (index === -1) return appendMessage(state, {
    ...message('assistant', content, 'normal'),
    correlationKey: turnTerminalKey(runId),
  })
  const existing = state.messages[index]
  if (!existing) return state
  return replaceMessage(state, index, {
    ...existing,
    content,
    isDraft: false,
    correlationKey: turnTerminalKey(runId),
    tone: 'normal',
    terminalKind: undefined,
  })
}

function appendAssistantSegment(state: MessageState, runId: number, content: string): MessageState {
  if (!content.trim()) return state
  const draftKey = assistantDraftKey(runId)
  const draftIndex = state.messages.findIndex((item) => item.correlationKey === draftKey && item.isDraft)
  const segmentKey = assistantSegmentKey(runId, state.messages)
  if (draftIndex !== -1) {
    const draft = state.messages[draftIndex]
    if (!draft) return state
    return replaceMessage(state, draftIndex, {
      ...draft,
      content,
      isDraft: false,
      correlationKey: segmentKey,
      tone: 'normal',
      terminalKind: undefined,
    })
  }
  return appendMessage(state, {
    ...message('assistant', content, 'normal'),
    correlationKey: segmentKey,
  })
}

function settlePausedCheckpoint(
  state: MessageState,
  runId: number,
  content: string,
  reason: MessageTerminalReason,
  workspaceMayHaveChanged: boolean,
): MessageState {
  const terminal = terminalMessage('assistant', content, 'normal', 'paused', workspaceMayHaveChanged, turnTerminalKey(runId))
  const draftKey = assistantDraftKey(runId)
  const draftIndex = state.messages.findIndex((item) => item.correlationKey === draftKey && item.isDraft)
  if (draftIndex === -1) return appendMessage(state, { ...terminal, terminalReason: reason })
  const draft = state.messages[draftIndex]
  if (!draft) return appendMessage(state, { ...terminal, terminalReason: reason })
  return replaceMessage(state, draftIndex, {
    ...draft,
    ...terminal,
    terminalReason: reason,
    isDraft: false,
  })
}

function removeAssistantDraft(state: MessageState, runId: number): MessageState {
  const correlationKey = assistantDraftKey(runId)
  const messages = state.messages.filter((item) => !(item.correlationKey === correlationKey && item.isDraft))
  return messages.length === state.messages.length ? state : { ...state, messages }
}

function upsertToolStart(state: MessageState, runId: number, callId: string, toolName: string, input: unknown): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, { ...toolMessage('tool', toolName, 'running', `Running ${toolName}...`, correlationKey), toolInput: input })
  }
  const existing = state.messages[index]
  // The first approval request owns this tool call for the rest of the
  // projection. A repeated/late request must not replace its input, reason,
  // or request owner after a decision has already been made.
  if (
    !existing ||
    isFinalToolPhase(existing.toolPhase) ||
    existing.approvalRequestId !== undefined ||
    existing.approvalDecision !== undefined
  ) return state
  return replaceMessage(state, index, {
    ...existing,
    toolName,
    toolPhase: 'running',
    status: 'normal',
    tone: 'normal',
    content: `Running ${toolName}...`,
    toolInput: input,
    terminalKind: undefined,
    workspaceMayHaveChanged: undefined,
  })
}

function markApprovalPending(
  state: MessageState,
  runId: number,
  callId: string,
  toolName: string,
  requestId: string,
  input: unknown,
  reason: string,
): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, {
      ...toolMessage('tool', toolName, 'awaiting_approval', `Awaiting approval: ${toolName}...`, correlationKey, requestId),
      toolInput: input,
      approvalReason: reason,
    })
  }
  const existing = state.messages[index]
  if (
    !existing ||
    isFinalToolPhase(existing.toolPhase) ||
    existing.approvalRequestId !== undefined ||
    existing.approvalDecision !== undefined
  ) return state
  return replaceMessage(state, index, {
    ...existing,
    toolName,
    toolPhase: 'awaiting_approval',
    status: 'normal',
    tone: 'normal',
    content: `Awaiting approval: ${toolName}...`,
    approvalRequestId: requestId,
    toolInput: input,
    approvalReason: reason,
  })
}

function resolveApproval(state: MessageState, requestId: string, approved: boolean): MessageState {
  const index = state.messages.findIndex((item) => item.approvalRequestId === requestId)
  if (index === -1) return state
  const existing = state.messages[index]
  if (!existing || isFinalToolPhase(existing.toolPhase)) return state
  if (existing.approvalDecision !== undefined) return state
  if (!approved) {
    const presentation = presentDeniedTool(existing.toolName ?? 'tool', existing.toolInput, existing.approvalReason ?? 'User denied this tool call.')
    const replaced = replaceMessage(state, index, {
      ...existing,
      toolPhase: 'denied',
      status: 'normal',
      tone: 'normal',
      content: `Denied: ${existing.toolName ?? 'tool'}`,
      output: 'Denied by user.',
      approvalRequestId: undefined,
      approvalDecision: 'denied',
      toolPresentation: presentation,
    })
    return commitToolPresentation(replaced, replaced.messages[index]!, presentation, existing.toolName ?? 'tool')
  }
  return replaceMessage(state, index, {
    ...existing,
    toolPhase: 'running',
    status: 'normal',
    tone: 'normal',
    content: `Running ${existing.toolName ?? 'tool'}...`,
    approvalRequestId: undefined,
    approvalDecision: 'approved',
    toolPresentation: undefined,
  })
}

function upsertToolResult(
  state: MessageState,
  runId: number,
  callId: string,
  toolName: string,
  output: string,
  isError: boolean,
  code?: string,
  display?: import('../../tools/types.js').ToolDisplayFacts,
): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const existing = state.messages.find((item) => item.correlationKey === correlationKey)
  const presentation = presentToolResult(toolName, output, {
    input: existing?.toolInput,
    display,
    contentWidth: state.presentationWidth,
  })
  const phase: ToolMessagePhase = code === 'interrupted' ? 'interrupted' : isError ? 'failed' : 'succeeded'
  const suffix = phase === 'failed' ? ' failed' : phase === 'interrupted' ? ' interrupted' : ''
  const content = `${toolName}${suffix}: ${presentation.summary}`
  const status = phase === 'failed' ? 'error' : 'normal'
  const tone = phase === 'failed' || phase === 'interrupted' ? 'error' : 'normal'
  const label = toolName || 'tool'
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    const appended = appendMessage(state, {
      ...toolMessage('tool', toolName, phase, content, correlationKey),
      status,
      tone,
      output: formatToolOutput(output || 'completed'),
      toolPresentation: presentation,
    })
    const created = appended.messages.at(-1)!
    return rememberActiveToolPresentation(
      commitToolPresentation(appended, created, presentation, label),
      correlationKey,
      { toolName, output, input: existing?.toolInput, display },
    )
  }
  const existingAtIndex = state.messages[index]
  if (!existingAtIndex || isFinalToolPhase(existingAtIndex.toolPhase)) return state
  const {
    approvalRequestId: _approvalRequestId,
    approvalDecision: _approvalDecision,
    ...resultBase
  } = existingAtIndex
  const replaced = replaceMessage(state, index, {
    ...resultBase,
    toolName,
    toolPhase: phase,
    status,
    tone,
    content,
    output: formatToolOutput(output || 'completed'),
    toolPresentation: presentation,
  })
  return rememberActiveToolPresentation(
    commitToolPresentation(replaced, replaced.messages[index]!, presentation, label),
    correlationKey,
    { toolName, output, input: existing?.toolInput, display },
  )
}

function rememberActiveToolPresentation(
  state: MessageState,
  correlationKey: string,
  source: ActiveToolPresentationSource,
): MessageState {
  const activeToolPresentationSources = new Map(state.activeToolPresentationSources)
  activeToolPresentationSources.set(correlationKey, source)
  return { ...state, activeToolPresentationSources }
}

/**
 * End resize ownership for presentations that have atomically left the
 * active suffix. The message controller retains the same source objects
 * until Ink acknowledges their Static batch; this helper only prevents a
 * later resize from rewriting an already-frozen message projection.
 */
export function releaseActiveToolPresentationSources(
  state: MessageState,
  correlationKeys: readonly string[],
): MessageState {
  if (correlationKeys.length === 0 || state.activeToolPresentationSources.size === 0) return state
  const activeToolPresentationSources = new Map(state.activeToolPresentationSources)
  let changed = false
  for (const correlationKey of correlationKeys) {
    changed = activeToolPresentationSources.delete(correlationKey) || changed
  }
  return changed ? { ...state, activeToolPresentationSources } : state
}

function reprojectActiveToolResults(state: MessageState): MessageState {
  if (state.activeToolPresentationSources.size === 0) return state
  let changed = false
  const messages = state.messages.map((message) => {
    if (!message.correlationKey || !isFinalToolPhase(message.toolPhase)) return message
    const source = state.activeToolPresentationSources.get(message.correlationKey)
    if (!source) return message
    const projected = presentToolResult(source.toolName, source.output, {
      input: source.input,
      display: source.display,
      contentWidth: state.presentationWidth,
    })
    const { detailSource: _detailSource, ...light } = projected
    const previousSourceTruncated = message.toolPresentation?.detail.sourceTruncated === true
    const presentation: ToolResultPresentation = previousSourceTruncated
      ? {
          ...light,
          truncated: true,
          sourceTruncated: true,
          detail: { ...light.detail, sourceTruncated: true },
        }
      : light
    changed = true
    return {
      ...message,
      content: `${source.toolName}${message.toolPhase === 'failed' ? ' failed' : ''}: ${presentation.summary}`,
      toolPresentation: presentation,
      viewCollapsed: presentation.viewCollapsed,
    }
  })
  return changed ? { ...state, messages } : state
}

function finalizeRunningTools(state: MessageState, phase: 'failed' | 'interrupted', workspaceMayHaveChanged = false): MessageState {
  let changed = false
  const messages = state.messages.map((item) => {
    if (item.role !== 'tool' || !item.toolPhase || !isActiveToolPhase(item.toolPhase)) return item
    changed = true
    const toolName = item.toolName ?? 'tool'
    const terminalKind: MessageTerminalKind = phase === 'failed' ? 'error' : 'interrupted'
    return {
      ...item,
      toolPhase: phase,
      status: (phase === 'failed' ? 'error' : 'normal') as MessageType,
      tone: 'error' as const,
      terminalKind,
      workspaceMayHaveChanged,
      content: phase === 'failed' ? `${toolName} failed` : `${toolName} interrupted`,
      output: phase === 'failed' ? 'failed' : 'interrupted',
      approvalRequestId: undefined,
    }
  })
  return changed ? { ...state, messages } : state
}

/**
 * Split a freshly built presentation: the message keeps the light detail
 * reference (counters + dual truncation state); the heavy bounded lines are
 * committed to the detail store exactly once under the message's stable ID.
 */
function commitToolPresentation(
  state: MessageState,
  message: Message,
  presentation: ToolResultPresentation,
  label: string,
): MessageState {
  const source = presentation.detailSource
  if (!source || source.lines.length === 0) return state
  const { detailSource: _stripped, ...light } = presentation
  const committed = commitDetailRecord(state.detailStore, {
    recordId: message.id,
    label,
    lines: source.lines,
    sourceTruncated: source.sourceTruncated,
    viewCollapsed: source.viewCollapsed,
  })
  const merged: ToolResultPresentation = committed.clipped
    ? { ...light, detail: { ...light.detail, sourceTruncated: true } }
    : light
  const index = state.messages.indexOf(message)
  if (index === -1) return { ...state, detailStore: committed.state }
  const messages = state.messages.slice()
  messages[index] = { ...message, toolPresentation: merged, viewCollapsed: merged.viewCollapsed }
  return { ...state, messages, detailStore: committed.state }
}

function appendMessage(state: MessageState, input: Omit<Message, 'id' | 'createdAtMs' | 'time'>): MessageState {
  const createdAtMs = state.clock()
  const appended: Message = {
    ...input,
    id: `message-${state.nextId}`,
    createdAtMs,
    time: formatTime(createdAtMs),
  }
  return {
    ...state,
    messages: [...state.messages, appended],
    nextId: state.nextId + 1,
  }
}

function replaceMessage(state: MessageState, index: number, value: Message): MessageState {
  const messages = state.messages.slice()
  messages[index] = value
  return { ...state, messages }
}

function message(role: Role, content: string, status: MessageType): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return { role, status, content, tone: status === 'error' ? 'error' : 'normal' }
}

function terminalMessage(
  role: 'assistant',
  content: string,
  status: MessageType,
  terminalKind: 'error' | 'interrupted' | 'paused',
  workspaceMayHaveChanged: boolean,
  correlationKey: string,
): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return {
    ...message(role, content, status),
    tone: terminalKind === 'paused' ? 'normal' : 'error',
    terminalKind,
    workspaceMayHaveChanged,
    correlationKey,
  }
}

function toolMessage(
  role: 'tool',
  toolName: string,
  toolPhase: ToolMessagePhase,
  content: string,
  correlationKey: string,
  approvalRequestId?: string,
): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return {
    ...message(role, content, toolPhase === 'failed' ? 'error' : 'normal'),
    tone: toolPhase === 'failed' || toolPhase === 'interrupted' ? 'error' : 'normal',
    toolName,
    toolPhase,
    correlationKey,
    approvalRequestId,
  }
}

function isActiveToolPhase(phase: ToolMessagePhase): boolean {
  return phase === 'running' || phase === 'awaiting_approval'
}

function isFinalToolPhase(phase: ToolMessagePhase | undefined): boolean {
  return phase === 'succeeded' || phase === 'failed' || phase === 'denied' || phase === 'interrupted'
}

function toolCorrelationKey(runId: number, callId: string): string {
  return `run:${runId}:call:${callId}`
}

function assistantDraftKey(runId: number): string {
  return `run:${runId}:assistant-draft`
}

function assistantSegmentKey(runId: number, messages: readonly Message[]): string {
  const prefix = `run:${runId}:assistant-segment:`
  const count = messages.filter((item) => item.correlationKey?.startsWith(prefix)).length
  return `${prefix}${count + 1}`
}

export function turnTerminalKey(runId: number): string {
  return `run:${runId}:terminal`
}

function turnUserKey(runId: number): string {
  return `run:${runId}:user`
}

export function formatToolOutput(output: string): string {
  const bounded = output || 'completed'
  if (bounded.length <= MAX_TOOL_DISPLAY_CHARS) return bounded
  return `${bounded.slice(0, MAX_TOOL_DISPLAY_CHARS - 32)}… [output truncated]`
}

const MAX_TOOL_DISPLAY_CHARS = 1_200
