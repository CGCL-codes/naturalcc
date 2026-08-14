import type { AgentEvent } from '../../agent/types/event.js'
import type { ApprovalRequest } from '../../tools/approval.js'

export interface ExecutorState {
  status: 'idle' | 'running'
  phase: 'thinking' | 'streaming' | 'tool' | 'approval'
  streamingContent: string
  approvalPending: ApprovalRequest | null
  startedAt: number
  thinkTime: number | null
  activeRunId: number | null
}

export const initialExecutorState: ExecutorState = {
  status: 'idle',
  phase: 'thinking',
  streamingContent: '',
  approvalPending: null,
  startedAt: 0,
  thinkTime: null,
  activeRunId: null,
}

export type ExecutorAction =
  | { type: 'start'; runId: number; startedAt: number }
  | { type: 'event'; runId: number; event: AgentEvent }
  | { type: 'finish'; runId: number }

export function executorReducer(
  state: ExecutorState,
  action: ExecutorAction,
): ExecutorState {
  switch (action.type) {
    case 'start':
      if (state.activeRunId !== null) return state
      return {
        ...initialExecutorState,
        status: 'running',
        startedAt: action.startedAt,
        activeRunId: action.runId,
      }
    case 'event':
      if (state.activeRunId !== action.runId) return state
      return reduceAgentEvent(state, action.event)
    case 'finish':
      if (state.activeRunId !== action.runId) return state
      return {
        ...state,
        status: 'idle',
        phase: 'thinking',
        streamingContent: '',
        approvalPending: null,
        activeRunId: null,
      }
  }
}

function reduceAgentEvent(state: ExecutorState, event: AgentEvent): ExecutorState {
  switch (event.type) {
    case 'assistant_delta':
      return {
        ...state,
        phase: 'streaming',
        streamingContent: state.streamingContent + event.content,
      }
    case 'tool_start':
      return { ...state, phase: 'tool', streamingContent: '' }
    case 'tool_result':
      return { ...state, phase: 'tool', streamingContent: '' }
    case 'approval_required':
      return {
        ...state,
        phase: 'approval',
        streamingContent: '',
        approvalPending: {
          requestId: event.requestId,
          callId: event.callId,
          toolName: event.name,
          input: isRecord(event.input) ? event.input : {},
          reason: event.reason,
        },
      }
    case 'approval_resolved':
      if (state.approvalPending?.requestId !== event.requestId) return state
      return { ...state, phase: 'tool', approvalPending: null }
    case 'done':
      return {
        ...state,
        thinkTime: elapsedSeconds(state.startedAt),
      }
    case 'user_message':
    case 'assistant_message':
      return state
    case 'error':
    case 'interrupted':
      return { ...state, approvalPending: null }
  }
}

function elapsedSeconds(startedAt: number): number | null {
  return startedAt > 0 ? (Date.now() - startedAt) / 1000 : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
