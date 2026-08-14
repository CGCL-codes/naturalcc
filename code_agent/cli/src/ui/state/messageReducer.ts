import type { AgentEvent } from '../../agent/types/event.js'
import type { Message, MessageTerminalKind, MessageType, Role, ToolMessagePhase } from '../types/message.js'
import { presentToolResult } from '../toolPresenters/index.js'
import { formatTime } from '../utils/now-time.js'

export interface MessageState {
  messages: Message[]
  nextId: number
  activeRunId: number | null
  /** Injectable wall clock; never read during component render. */
  readonly clock: () => number
}

export function createMessageState(clock: () => number = () => Date.now()): MessageState {
  return { messages: [], nextId: 1, activeRunId: null, clock }
}

export const initialMessageState: MessageState = createMessageState()

export type MessageAction =
  | { type: 'begin_run'; runId: number }
  | { type: 'add'; role: Role; content: string; status: MessageType }
  | { type: 'agent_event'; runId: number; event: AgentEvent }
  | { type: 'finish_run'; runId: number }
  | { type: 'clear' }

export function messageReducer(state: MessageState, action: MessageAction): MessageState {
  switch (action.type) {
    case 'begin_run':
      if (state.activeRunId !== null) return state
      return { ...state, activeRunId: action.runId }
    case 'add':
      return appendMessage(state, message(action.role, action.content, action.status))
    case 'agent_event':
      if (state.activeRunId !== action.runId) return state
      return reduceAgentEvent(state, action.runId, action.event)
    case 'finish_run':
      if (state.activeRunId !== action.runId) return state
      return {
        ...removeAssistantDraft(finalizeRunningTools(state, 'interrupted'), action.runId),
        activeRunId: null,
      }
    case 'clear':
      return createMessageState(state.clock)
  }
}

function reduceAgentEvent(state: MessageState, runId: number, event: AgentEvent): MessageState {
  switch (event.type) {
    case 'user_message':
      return appendMessage(state, message('user', event.content, 'normal'))
    case 'assistant_delta':
      return appendAssistantDelta(state, runId, event.content)
    case 'assistant_message':
      return settleAssistantDraft(state, runId, event.content)
    case 'tool_start':
      return upsertToolStart(removeAssistantDraft(state, runId), runId, event.callId, event.name, event.input)
    case 'approval_required':
      return markApprovalPending(state, runId, event.callId, event.name, event.requestId)
    case 'approval_resolved':
      return resolveApproval(state, event.requestId, event.approved)
    case 'tool_result':
      return upsertToolResult(state, runId, event.callId, event.name, event.output, event.isError)
    case 'error':
      return appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'failed', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'error', 'error', event.workspaceMayHaveChanged),
      )
    case 'interrupted':
      return appendMessage(
        removeAssistantDraft(finalizeRunningTools(state, 'interrupted', event.workspaceMayHaveChanged), runId),
        terminalMessage('assistant', event.message, 'normal', 'interrupted', event.workspaceMayHaveChanged),
      )
    case 'done':
      return state
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
  if (index === -1) return appendMessage(state, message('assistant', content, 'normal'))
  const existing = state.messages[index]
  if (!existing) return state
  return replaceMessage(state, index, {
    ...existing,
    content,
    isDraft: false,
    correlationKey: undefined,
    tone: 'normal',
    terminalKind: undefined,
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
  if (!existing || isFinalToolPhase(existing.toolPhase)) return state
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

function markApprovalPending(state: MessageState, runId: number, callId: string, toolName: string, requestId: string): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, toolMessage('tool', toolName, 'awaiting_approval', `Awaiting approval: ${toolName}...`, correlationKey, requestId))
  }
  const existing = state.messages[index]
  if (!existing || isFinalToolPhase(existing.toolPhase)) return state
  return replaceMessage(state, index, {
    ...existing,
    toolName,
    toolPhase: 'awaiting_approval',
    status: 'normal',
    tone: 'normal',
    content: `Awaiting approval: ${toolName}...`,
    approvalRequestId: requestId,
  })
}

function resolveApproval(state: MessageState, requestId: string, approved: boolean): MessageState {
  const index = state.messages.findIndex((item) => item.approvalRequestId === requestId)
  if (index === -1) return state
  const existing = state.messages[index]
  if (!existing || isFinalToolPhase(existing.toolPhase) || !approved) return state
  return replaceMessage(state, index, {
    ...existing,
    toolPhase: 'running',
    status: 'normal',
    tone: 'normal',
    content: `Running ${existing.toolName ?? 'tool'}...`,
  })
}

function upsertToolResult(
  state: MessageState,
  runId: number,
  callId: string,
  toolName: string,
  output: string,
  isError: boolean,
): MessageState {
  const correlationKey = toolCorrelationKey(runId, callId)
  const existing = state.messages.find((item) => item.correlationKey === correlationKey)
  const presentation = presentToolResult(toolName, output, { input: existing?.toolInput })
  const phase: ToolMessagePhase = isError ? 'failed' : 'succeeded'
  const content = `${toolName}${isError ? ' failed' : ''}: ${presentation.summary}`
  const index = state.messages.findIndex((item) => item.correlationKey === correlationKey)
  if (index === -1) {
    return appendMessage(state, {
      ...toolMessage('tool', toolName, phase, content, correlationKey),
      status: isError ? 'error' : 'normal',
      tone: isError ? 'error' : 'normal',
      output: formatToolOutput(output || 'completed'),
      toolPresentation: presentation,
    })
  }
  const existingAtIndex = state.messages[index]
  if (!existingAtIndex || isFinalToolPhase(existingAtIndex.toolPhase)) return state
  return replaceMessage(state, index, {
    ...existingAtIndex,
    toolName,
    toolPhase: phase,
    status: isError ? 'error' : 'normal',
    tone: isError ? 'error' : 'normal',
    content,
    output: formatToolOutput(output || 'completed'),
    approvalRequestId: undefined,
    toolPresentation: presentation,
  })
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

function appendMessage(state: MessageState, input: Omit<Message, 'id' | 'createdAtMs' | 'time'>): MessageState {
  const createdAtMs = state.clock()
  return {
    ...state,
    messages: [...state.messages, {
      ...input,
      id: `message-${state.nextId}`,
      createdAtMs,
      time: formatTime(createdAtMs),
    }],
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
  terminalKind: 'error' | 'interrupted',
  workspaceMayHaveChanged: boolean,
): Omit<Message, 'id' | 'createdAtMs' | 'time'> {
  return {
    ...message(role, content, status),
    tone: 'error',
    terminalKind,
    workspaceMayHaveChanged,
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
  return phase === 'succeeded' || phase === 'failed' || phase === 'interrupted'
}

function toolCorrelationKey(runId: number, callId: string): string {
  return `run:${runId}:call:${callId}`
}

function assistantDraftKey(runId: number): string {
  return `run:${runId}:assistant-draft`
}

export function formatToolOutput(output: string): string {
  const bounded = output || 'completed'
  if (bounded.length <= MAX_TOOL_DISPLAY_CHARS) return bounded
  return `${bounded.slice(0, MAX_TOOL_DISPLAY_CHARS - 32)}… [output truncated]`
}

const MAX_TOOL_DISPLAY_CHARS = 1_200
