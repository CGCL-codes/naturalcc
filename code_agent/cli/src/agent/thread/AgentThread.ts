import { randomUUID } from 'node:crypto'

import { AgentError } from '../errors.js'
import type { ModelMessage } from '../../model/types/types.js'
import type { ConversationState } from './conversationState.js'

export interface AgentThread {
  readonly id: string
  readonly status: ConversationState['status']
  snapshot(): readonly ModelMessage[]
  /** Build a bounded request view without changing the complete in-memory history. */
  requestView(
    maxHistoryChars: number,
    pendingMessages?: readonly ModelMessage[],
  ): readonly ModelMessage[]
}

interface ThreadController {
  begin(): void
  commit(messages: readonly ModelMessage[]): void
  rollback(): void
}

const states = new WeakMap<AgentThread, ConversationState>()

export function createAgentThread(
  initialMessages: readonly ModelMessage[] = [],
  id: string = randomUUID(),
): AgentThread {
  const state: ConversationState = {
    id,
    messages: cloneMessages(initialMessages),
    status: 'idle',
  }

  const thread: AgentThread = {
    get id() {
      return state.id
    },
    get status() {
      return state.status
    },
    snapshot() {
      return freezeMessages(state.messages)
    },
    requestView(maxHistoryChars, pendingMessages = []) {
      const systemMessages = state.messages.filter((message) => message.role === 'system')
      const groups = groupHistory(state.messages)
      const systemChars = messageChars(systemMessages)
      let remaining = Math.max(0, Math.floor(maxHistoryChars) - systemChars)
      const selectedGroups: ModelMessage[][] = []

      for (let index = groups.length - 1; index >= 0; index -= 1) {
        const group = groups[index]!
        const chars = messageChars(group)
        // Do not skip a recent oversized exchange to resurrect an older one.
        // Complete tool exchanges are kept or dropped as a unit.
        if (chars > remaining) break
        selectedGroups.unshift(group)
        remaining -= chars
      }

      return freezeMessages([
        ...systemMessages,
        ...selectedGroups.flat(),
        ...pendingMessages,
      ])
    },
  }
  states.set(thread, state)
  return Object.freeze(thread)
}

/** @internal Used by AgentRunner to keep thread mutation out of UI/tool code. */
export function getThreadController(thread: AgentThread): ThreadController {
  const state = states.get(thread)
  if (!state) {
    throw new AgentError('config_error', 'The supplied agent thread is invalid')
  }

  return {
    begin() {
      if (state.status === 'running') {
        throw new AgentError(
          'session_busy',
          'This agent thread is already running a turn',
          true,
        )
      }
      state.status = 'running'
    },
    commit(messages) {
      if (state.status !== 'running') {
        throw new AgentError('config_error', 'The agent thread is not running')
      }
      state.messages.push(...cloneMessages(messages))
      state.status = 'idle'
    },
    rollback() {
      state.status = 'idle'
    },
  }
}

function cloneMessages(messages: readonly ModelMessage[]): ModelMessage[] {
  return structuredClone([...messages])
}

function groupHistory(messages: readonly ModelMessage[]): ModelMessage[][] {
  const groups: ModelMessage[][] = []
  let current: ModelMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    if (message.role === 'user' && current.length > 0) {
      groups.push(current)
      current = []
    }
    current.push(message)
  }
  if (current.length > 0) groups.push(current)
  return groups
}

function messageChars(messages: readonly ModelMessage[]): number {
  return JSON.stringify(messages).length
}

function freezeMessages(messages: readonly ModelMessage[]): readonly ModelMessage[] {
  const cloned = cloneMessages(messages)
  for (const message of cloned) {
    Object.freeze(message.content)
    for (const block of message.content) Object.freeze(block)
    Object.freeze(message)
  }
  return Object.freeze(cloned)
}
