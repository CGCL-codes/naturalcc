import { useCallback, useReducer } from 'react'
import type { AgentEvent } from '../../agent/types/event.js'
import type { MessageType, Role } from '../types/message.js'
import {
  initialMessageState,
  messageReducer,
} from '../state/messageReducer.js'

// [maint-note:c35efec353e2]
// 维护导航：useMessages；下列为本函数源码中的接口与路径索引。
// 返回路径：{ messages: state.messages, addMsg, addRunError, beginRun, handleAgentEvent, finishRun, clearMessages, }。
// 直接调用：useReducer。
// 直接调用：useCallback。
// [/maint-note:c35efec353e2]
export function useMessages() {
    const [state, dispatch] = useReducer(messageReducer, initialMessageState)

    const addMsg = useCallback((role: Role, content: string, status: MessageType = 'normal'): void => {
        dispatch({ type: 'add', role, content, status })
    }, [])

    const addRunError = useCallback((runId: number, content: string): void => {
        dispatch({ type: 'add_run_error', runId, content })
    }, [])

    const beginRun = useCallback((runId: number): void => {
        dispatch({ type: 'begin_run', runId })
    }, [])

    const handleAgentEvent = useCallback((runId: number, event: AgentEvent): void => {
        dispatch({ type: 'agent_event', runId, event })
    }, [])

    const finishRun = useCallback((runId: number): void => {
        dispatch({ type: 'finish_run', runId })
    }, [])

    const clearMessages = useCallback((): void => {
        dispatch({ type: 'clear' })
    }, [])

    return{
        messages: state.messages,
        addMsg,
        addRunError,
        beginRun,
        handleAgentEvent,
        finishRun,
        clearMessages,
    }
}
