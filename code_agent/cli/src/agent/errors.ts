import type { AgentErrorCode } from './types/event.js'

export class AgentError extends Error {
  readonly code: AgentErrorCode
  readonly recoverable: boolean

  constructor(
    code: AgentErrorCode,
    message: string,
    recoverable = false,
  ) {
    super(message)
    this.name = 'AgentError'
    this.code = code
    this.recoverable = recoverable
  }
}

export function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true
  return (
    error instanceof Error &&
    (error.name === 'AbortError' || error.name === 'CanceledError')
  )
}

export function createAbortError(message = 'The operation was aborted'): Error {
  const error = new Error(message)
  error.name = 'AbortError'
  return error
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
