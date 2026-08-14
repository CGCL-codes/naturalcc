import { useCallback, useReducer } from 'react'
import type { AgentEvent } from '../../agent/types/event.js'
import type { MessageType, Role } from '../types/message.js'
import {
  initialMessageState,
  messageReducer,
} from '../state/messageReducer.js'

export function useMessages() {
    const [state, dispatch] = useReducer(messageReducer, initialMessageState)

    const addMsg = useCallback((role: Role, content: string, status: MessageType = 'normal'): void => {
        dispatch({ type: 'add', role, content, status })
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
        beginRun,
        handleAgentEvent,
        finishRun,
        clearMessages,
    }
}
