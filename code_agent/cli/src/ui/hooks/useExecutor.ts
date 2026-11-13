import { useCallback, useReducer, useRef } from 'react'
import type { Ctx } from '../types/ctx.js'
import type { AgentEvent } from '../../agent/types/event.js'
import { isTerminalEvent } from '../../agent/terminal.js'
import type { SessionManager } from '../../app/session/SessionManager.js'
import type { ApprovalController } from '../../tools/approval.js'
import {
  executorReducer,
  initialExecutorState,
} from '../state/executorReducer.js'
import { ExecutionOwner } from '../state/executionOwner.js'
import { DeliveryDispatcher, DeliveryEnvelopeOwner, type DeliveryGate } from '../state/deliveryGate.js'

// [maint-note:5d7961f328c8]
// 数据契约导航：ExecutorDeps；字段含义以消费端校验为准。
// 成员 sessionManager：SessionManager。
// 成员 addMsg：Ctx['addMsg']。
// 成员 addRunError：(runId: number, content: string) => void。
// 成员 beginRun：(runId: number) => void。
// 成员 handleAgentEvent：(runId: number, event: AgentEvent) => void。
// 成员 finishRun：(runId: number) => void。
// 成员 onRunStarted（可省略）：(runId: number) => void。
// 成员 onRunEvent（可省略）：(runId: number, event: AgentEvent) => void。
// 成员 onRunTerminal（可省略）：(runId: number, event: AgentEvent) => void。
// 成员 onRunFinished（可省略）：(runId: number) => void。
// 成员 approvalProvider（可省略）：ApprovalController。
// 成员 deliveryGate（可省略）：DeliveryGate。
// 成员 deliveryOwner（可省略）：DeliveryEnvelopeOwner。
// [/maint-note:5d7961f328c8]
interface ExecutorDeps {
  sessionManager: SessionManager
  addMsg: Ctx['addMsg']
  addRunError: (runId: number, content: string) => void
  beginRun: (runId: number) => void
  handleAgentEvent: (runId: number, event: AgentEvent) => void
  finishRun: (runId: number) => void
  onRunStarted?: (runId: number) => void
  onRunEvent?: (runId: number, event: AgentEvent) => void
  onRunTerminal?: (runId: number, event: AgentEvent) => void
  onRunFinished?: (runId: number) => void
  approvalProvider?: ApprovalController
  /**
   * delivery gate: every AgentEvent is stamped with the session
   * generation and a per-run strictly increasing sequence before it may
   * touch the executor, message, timing, or approval state. Exact replays
   * are ignored; stale generations/runs and sequence regressions are
   * dropped without updating any owner.
   */
  deliveryGate?: DeliveryGate
  /** Producer-side owner; one immutable envelope is created for each event. */
  deliveryOwner?: DeliveryEnvelopeOwner
}

// [maint-note:ba5417807977]
// 维护导航：useExecutor；下列为本函数源码中的接口与路径索引。
// 输入 deps：ExecutorDeps。
// 返回路径：{ loading: state.status === 'running', thinkTime: state.thinkTime, isStreaming: state.phase === 'streaming', approvalPending: state.approvalPending, execute, interrupt, resolveApproval, }。
// 直接调用：useReducer。
// 直接调用：useRef。
// 直接调用：useCallback。
// [/maint-note:ba5417807977]
export function useExecutor(deps: ExecutorDeps) {
  const [state, dispatch] = useReducer(executorReducer, initialExecutorState)
  const ownerRef = useRef(new ExecutionOwner())
  const deliveryOwnerRef = useRef(deps.deliveryOwner ?? new DeliveryEnvelopeOwner(deps.deliveryGate?.sessionGeneration ?? 1))
  const deliveryDispatcherRef = useRef(deps.deliveryGate === undefined ? null : new DeliveryDispatcher(deps.deliveryGate, deliveryOwnerRef.current))

  const deliver = useCallback((runId: number, deliveredEvent: AgentEvent): void => {
    deps.onRunEvent?.(runId, deliveredEvent)
    dispatch({ type: 'event', runId, event: deliveredEvent })
    deps.handleAgentEvent(runId, deliveredEvent)
  }, [deps.handleAgentEvent, deps.onRunEvent])

  const handleEvent = useCallback((runId: number, event: AgentEvent): boolean => {
    if (deliveryDispatcherRef.current !== null) {
      return deliveryDispatcherRef.current.dispatch(runId, event, deliveredEvent => deliver(runId, deliveredEvent)) !== null
    }
    deliver(runId, event)
    return true
  }, [deps.deliveryGate, deliver])

  const executeAgent = useCallback(async (
    value: string,
    runId: number,
    controller: AbortController,
  ) => {
    let terminalTimingClosed = false
    try {
      const session = deps.sessionManager.current()
      if (!session) throw new Error('No active coding agent session')
      for await (const event of session.runTurn({
        prompt: value,
        signal: controller.signal,
      })) {
        const delivered = handleEvent(runId, event)
        if (delivered && !terminalTimingClosed && isTerminalEvent(event)) {
          terminalTimingClosed = true
          // The terminal event has been handed to the message projection
          // first. Timing closes at this boundary, rather than waiting for a
          // potentially slow iterator/finally path.
          deps.onRunTerminal?.(runId, event)
        }
      }
    }
    catch(error){
      if (ownerRef.current.current()?.runId === runId) {
        deps.addRunError(runId, formatError(error))
      }
    }
    finally{
      deps.finishRun(runId)
      deliveryDispatcherRef.current?.endRun(runId)
      if (!terminalTimingClosed) deps.onRunFinished?.(runId)
      if (ownerRef.current.release(runId)) {
        dispatch({ type: 'finish', runId })
      }
    }
  }, [deps.addRunError, deps.finishRun, deps.onRunFinished, deps.onRunTerminal, deps.sessionManager, handleEvent])

  const execute = useCallback((value: string): boolean => {
    const execution = ownerRef.current.acquire()
    if (!execution) return false
    const { runId, controller } = execution
    deliveryDispatcherRef.current?.beginRun(runId)
    deps.beginRun(runId)
    deps.onRunStarted?.(runId)
    // SessionTimingController is the sole UI timing source. The reducer field
    // remains only for compatibility with existing reducer consumers.
    dispatch({ type: 'start', runId, startedAt: 0 })
    void executeAgent(value, runId, controller)
    return true
  }, [deps.beginRun, deps.onRunStarted, executeAgent])

  const interrupt = useCallback(() => {
    ownerRef.current.interrupt()
  }, [])

  const resolveApproval = useCallback((requestId: string, approved: boolean): void => {
    if (state.approvalPending?.requestId !== requestId) return
    deps.approvalProvider?.decide(requestId, approved)
  }, [deps.approvalProvider, state.approvalPending?.requestId])

  return {
    loading: state.status === 'running',
    thinkTime: state.thinkTime,
    isStreaming: state.phase === 'streaming',
    approvalPending: state.approvalPending,
    execute,
    interrupt,
    resolveApproval,
  }
}

// [maint-note:602afca55fa5]
// 维护导航：formatError；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：string。
// 返回路径：error instanceof Error ? error.message : String(error)。
// 直接调用：String。
// [/maint-note:602afca55fa5]
function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
