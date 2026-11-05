import type { AgentErrorCode, AgentEvent } from './types/event.js'

export type TerminalAgentEvent = Extract<
  AgentEvent,
  { type: 'done' | 'error' | 'interrupted' | 'paused' }
>

export type TerminalOutcome =
  | { type: 'done'; content: string }
  | { type: 'error'; code: AgentErrorCode; message: string; workspaceMayHaveChanged: boolean }
  | { type: 'interrupted'; message: string; workspaceMayHaveChanged: boolean }
  | {
      type: 'paused'
      reason: 'budget_exhausted' | 'no_progress'
      content: string
      workspaceMayHaveChanged: boolean
    }

/** The single terminal classifier shared by session and UI execution code. */
export function isTerminalEvent(event: AgentEvent): event is TerminalAgentEvent {
  return terminalOutcome(event) !== undefined
}

export function terminalType(event: AgentEvent): TerminalAgentEvent['type'] | undefined {
  return isTerminalEvent(event) ? event.type : undefined
}

/**
 * Maps a terminal event to the stable outcome shape consumed by adapters.
 * The switch is intentionally exhaustive for the current AgentEvent union;
 * Terminal event additions must update this mapper and its consumers together.
 */
export function terminalOutcome(event: AgentEvent): TerminalOutcome | undefined {
  switch (event.type) {
    case 'done':
      return { type: 'done', content: event.content }
    case 'error':
      return {
        type: 'error',
        code: event.code,
        message: event.message,
        workspaceMayHaveChanged: event.workspaceMayHaveChanged,
      }
    case 'interrupted':
      return {
        type: 'interrupted',
        message: event.message,
        workspaceMayHaveChanged: event.workspaceMayHaveChanged,
      }
    case 'paused':
      return {
        type: 'paused',
        reason: event.reason,
        content: event.content,
        workspaceMayHaveChanged: event.workspaceMayHaveChanged,
      }
    case 'user_message':
    case 'assistant_delta':
    case 'assistant_segment':
    case 'assistant_message':
    case 'tool_start':
    case 'tool_result':
    case 'approval_required':
    case 'approval_resolved':
      return undefined
    default:
      return futureNonTerminal(event)
  }
}

/** Compile-time exhaustive guard for event consumers. */
export function assertNeverAgentEvent(value: never): never {
  throw new Error(`Unhandled agent event: ${String(value)}`)
}

function futureNonTerminal(value: never): undefined {
  // Runtime adapters may still observe an event from a newer producer. It is
  // non-terminal until the shared union is deliberately extended; the never
  // parameter keeps that extension visible to TypeScript callers.
  void value
  return undefined
}
