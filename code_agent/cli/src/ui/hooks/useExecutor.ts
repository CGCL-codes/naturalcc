import { useCallback, useReducer, useRef } from 'react'
import type { Ctx } from '../types/ctx.js'
import type { AgentEvent } from '../../agent/types/event.js'
import type { SessionManager } from '../../app/session/SessionManager.js'
import type { ApprovalController } from '../../tools/approval.js'
import {
  executorReducer,
  initialExecutorState,
} from '../state/executorReducer.js'
import { ExecutionOwner } from '../state/executionOwner.js'

interface ExecutorDeps {
  sessionManager: SessionManager
  addMsg: Ctx['addMsg']
  beginRun: (runId: number) => void
  handleAgentEvent: (runId: number, event: AgentEvent) => void
  finishRun: (runId: number) => void
  approvalProvider?: ApprovalController
}

export function useExecutor(deps: ExecutorDeps) {
  const [state, dispatch] = useReducer(executorReducer, initialExecutorState)
  const ownerRef = useRef(new ExecutionOwner())

  const handleEvent = useCallback((runId: number, event: AgentEvent) => {
    if (ownerRef.current.current()?.runId !== runId) return
    dispatch({ type: 'event', runId, event })
    deps.handleAgentEvent(runId, event)
  }, [deps.handleAgentEvent])

  const executeAgent = useCallback(async (
    value: string,
    runId: number,
    controller: AbortController,
  ) => {
    try {
      const session = deps.sessionManager.current()
      if (!session) throw new Error('No active coding agent session')
      for await (const event of session.runTurn({
        prompt: value,
        signal: controller.signal,
      })) {
        handleEvent(runId, event)
      }
    }
    catch(error){
      if (ownerRef.current.current()?.runId === runId) {
        deps.addMsg('assistant', formatError(error), 'error')
      }
    }
    finally{
      deps.finishRun(runId)
      if (ownerRef.current.release(runId)) {
        dispatch({ type: 'finish', runId })
      }
    }
  }, [deps.addMsg, deps.sessionManager, handleEvent])

  const execute = useCallback((value: string): boolean => {
    const execution = ownerRef.current.acquire()
    if (!execution) return false
    const { runId, controller } = execution
    deps.beginRun(runId)
    dispatch({ type: 'start', runId, startedAt: Date.now() })
    void executeAgent(value, runId, controller)
    return true
  }, [deps.beginRun, executeAgent])

  const interrupt = useCallback(() => {
    ownerRef.current.interrupt()
  }, [])

  const resolveApproval = useCallback((requestId: string, approved: boolean): void => {
    if (state.approvalPending?.requestId !== requestId) return
    deps.approvalProvider?.decide(requestId, approved)
  }, [deps.approvalProvider, state.approvalPending?.requestId])

  return {
    loading: state.status === 'running',
    streamingContent: state.streamingContent,
    thinkTime: state.thinkTime,
    isStreaming: state.phase === 'streaming' || state.phase === 'tool' || state.phase === 'approval',
    approvalPending: state.approvalPending,
    execute,
    interrupt,
    resolveApproval,
  }
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
