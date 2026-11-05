import type { AgentEvent } from '../../agent/types/event.js'
import { assertNeverAgentEvent } from '../../agent/terminal.js'
import type { ApprovalRequest } from '../../tools/approval.js'

export interface ExecutorState {
  status: 'idle' | 'running'
  phase: 'thinking' | 'streaming' | 'tool' | 'approval'
  approvalPending: ApprovalRequest | null
  startedAt: number
  thinkTime: number | null
  activeRunId: number | null
  terminalRunId: number | null
  /** Run-local approval ledger: the first request owns a call until it resolves. */
  approvalLedger: Readonly<Record<string, { requestId: string; settled: boolean }>>
}

export const initialExecutorState: ExecutorState = {
  status: 'idle',
  phase: 'thinking',
  approvalPending: null,
  startedAt: 0,
  thinkTime: null,
  activeRunId: null,
  terminalRunId: null,
  approvalLedger: {},
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
        terminalRunId: null,
        approvalLedger: {},
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
        approvalPending: null,
        activeRunId: null,
        terminalRunId: null,
        approvalLedger: {},
      }
  }
}

function reduceAgentEvent(state: ExecutorState, event: AgentEvent): ExecutorState {
  if (state.terminalRunId !== null) return state
  switch (event.type) {
    case 'assistant_delta':
      // The message-controller refresh owner batches the visible draft.
      // Once this phase is established, returning the same executor object
      // prevents every canonical delta from independently invalidating the
      // whole Repl tree while preserving immediate AgentEvent consumption.
      return state.phase === 'streaming'
        ? state
        : { ...state, phase: 'streaming' }
    case 'assistant_segment':
      // A segment is a committed UI boundary before tool execution, not a
      // terminal event and not a second streaming buffer.
      return state
    case 'tool_start':
      // A result settles the canonical call even when the start event was
      // missing or arrived after it. Late replay must not move the executor
      // back into a tool phase or reopen an approval projection.
      if (state.approvalLedger[event.callId]?.settled) return state
      return { ...state, phase: 'tool' }
    case 'tool_result': {
      const pending = state.approvalPending
      const settlesPending = pending?.callId === event.callId
      const owner = state.approvalLedger[event.callId]
      return {
        ...state,
        phase: pending && !settlesPending ? 'approval' : 'tool',
        approvalPending: settlesPending ? null : pending,
        approvalLedger: {
          ...state.approvalLedger,
          [event.callId]: {
            requestId: owner?.requestId ?? '',
            settled: true,
          },
        },
      }
    }
    case 'approval_required': {
      if (state.approvalLedger[event.callId]) return state
      return {
        ...state,
        phase: 'approval',
        approvalPending: {
          requestId: event.requestId,
          callId: event.callId,
          toolName: event.name,
          input: isRecord(event.input) ? event.input : {},
          reason: event.reason,
        },
        approvalLedger: {
          ...state.approvalLedger,
          [event.callId]: { requestId: event.requestId, settled: false },
        },
      }
    }
    case 'approval_resolved': {
      const pending = state.approvalPending
      const owner = pending ? state.approvalLedger[pending.callId] : undefined
      if (!pending || !owner || owner.requestId !== event.requestId || owner.settled) return state
      return {
        ...state,
        phase: 'tool',
        approvalPending: null,
        approvalLedger: {
          ...state.approvalLedger,
          [pending.callId]: { ...owner, settled: true },
        },
      }
    }
    case 'done':
      return { ...state, terminalRunId: state.activeRunId }
    case 'user_message':
    case 'assistant_message':
      return state
    case 'error':
    case 'interrupted':
    case 'paused':
      return { ...state, approvalPending: null, terminalRunId: state.activeRunId }
    default:
      return assertNeverAgentEvent(event)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
