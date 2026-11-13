import type { AgentEvent } from '../../agent/types/event.js'
import { assertNeverAgentEvent } from '../../agent/terminal.js'
import type { ApprovalRequest } from '../../tools/approval.js'

// [maint-note:ae7a0e664f48]
// 数据契约导航：ExecutorState；字段含义以消费端校验为准。
// 成员 status：'idle' | 'running'。
// 成员 phase：'thinking' | 'streaming' | 'tool' | 'approval'。
// 成员 approvalPending：ApprovalRequest | null。
// 成员 startedAt：number。
// 成员 thinkTime：number | null。
// 成员 activeRunId：number | null。
// 成员 terminalRunId：number | null。
// 成员 approvalLedger：Readonly<Record<string, { requestId: string; settled: boolean }>>。
// [/maint-note:ae7a0e664f48]
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

// [maint-note:11a10dbb487c]
// 数据契约导航：ExecutorAction；字段含义以消费端校验为准。
// 类型组合：| { type: 'start'; runId: number; startedAt: number } | { type: 'event'; runId: number; event: AgentEvent } | { type: 'finish'; runId: number }。
// [/maint-note:11a10dbb487c]
export type ExecutorAction =
  | { type: 'start'; runId: number; startedAt: number }
  | { type: 'event'; runId: number; event: AgentEvent }
  | { type: 'finish'; runId: number }

// [maint-note:0b5579b77d41]
// 维护导航：executorReducer；下列为本函数源码中的接口与路径索引。
// 输入 state：ExecutorState。
// 输入 action：ExecutorAction。
// 返回契约：ExecutorState。
// 分支条件：switch action.type。
// 分支条件：case 'start'。
// 分支条件：state.activeRunId !== null。
// 分支条件：case 'event'。
// 分支条件：state.activeRunId !== action.runId。
// 分支条件：case 'finish'。
// 返回路径：state。
// 返回路径：{ ...initialExecutorState, status: 'running', startedAt: action.startedAt, activeRunId: action.runId, terminalRunId: null, approvalLedger: {}, }。
// 返回路径：reduceAgentEvent(state, action.event)。
// 返回路径：{ ...state, status: 'idle', phase: 'thinking', approvalPending: null, activeRunId: null, terminalRunId: null, approvalLedger: {}, }。
// 直接调用：reduceAgentEvent。
// [/maint-note:0b5579b77d41]
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

// [maint-note:29600a877c08]
// 维护导航：reduceAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 state：ExecutorState。
// 输入 event：AgentEvent。
// 返回契约：ExecutorState。
// 分支条件：state.terminalRunId !== null。
// 分支条件：switch event.type。
// 分支条件：case 'assistant_delta'。
// 分支条件：case 'assistant_segment'。
// 分支条件：case 'tool_start'。
// 分支条件：state.approvalLedger[event.callId]?.settled。
// 分支条件：case 'tool_result'。
// 分支条件：case 'approval_required'。
// 分支条件：state.approvalLedger[event.callId]。
// 分支条件：case 'approval_resolved'。
// 分支条件：!pending || !owner || owner.requestId !== event.requestId || owner.settled。
// 分支条件：case 'done'。
// 分支条件：case 'user_message'。
// 分支条件：case 'assistant_message'。
// 分支条件：case 'error'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'paused'。
// 返回路径：state。
// 返回路径：state.phase === 'streaming' ? state : { ...state, phase: 'streaming' }。
// 返回路径：{ ...state, phase: 'tool' }。
// 返回路径：{ ...state, phase: pending && !settlesPending ? 'approval' : 'tool', approvalPending: settlesPending ? null : pending, approvalLedger: { ...state.approvalLedger, [event.callId]: { requestId: owner?.requestId ?? '', settled: true, }, }, }。
// 返回路径：{ ...state, phase: 'approval', approvalPending: { requestId: event.requestId, callId: event.callId, toolName: event.name, input: isRecord(event.input) ? event.input : {}, reason: event.reason, }, approvalLedger: { ...state.approvalLedger, [。
// 返回路径：{ ...state, phase: 'tool', approvalPending: null, approvalLedger: { ...state.approvalLedger, [pending.callId]: { ...owner, settled: true }, }, }。
// 返回路径：{ ...state, terminalRunId: state.activeRunId }。
// 返回路径：{ ...state, approvalPending: null, terminalRunId: state.activeRunId }。
// 返回路径：assertNeverAgentEvent(event)。
// 直接调用：isRecord。
// 直接调用：assertNeverAgentEvent。
// [/maint-note:29600a877c08]
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

// [maint-note:e49b38a17f13]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:e49b38a17f13]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
