import type { TaskPlanOperation } from '../../task/contracts.js'
import type { ToolExecutionContext, ToolExecutionResult } from '../types.js'
import { formatToolErrorJson } from '../diagnostics.js'

export async function executeUpdateTaskPlan(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
): Promise<ToolExecutionResult> {
  const operation = input as unknown as TaskPlanOperation
  const port = context.taskPlan
  const callId = context.canonicalToolCallId
  if (!port || !callId) {
    return boundedTaskError('tool_execution_error', 'No active run task owner is available', context.maxOutputChars)
  }
  if (context.signal?.aborted) {
    const error = new Error('Task plan operation interrupted')
    error.name = 'AbortError'
    throw error
  }
  const result = port.executeTaskPlanOperation(callId, operation, context.signal)
  if (!result.ok && result.value.error && typeof result.value.error === 'object') {
    return { content: formatToolErrorJson(result.value.error as Record<string, unknown>, context.maxOutputChars),
      isError: true, errorCode: result.errorCode }
  }
  const content = JSON.stringify(result.value)
  if (context.maxOutputChars !== undefined && content.length > context.maxOutputChars) {
    // the budget-rejection result is itself bounded by the SAME
    // configured budget — a parseable failure with the correct code that a
    // minimal supported configuration (96 chars) can still carry.
    return boundedTaskError(
      'tool_validation_error',
      'Task plan result exceeds the configured tool output budget',
      context.maxOutputChars,
    )
  }
  return {
    content,
    isError: !result.ok,
    errorCode: result.errorCode,
    display: result.ok && result.display
      ? { kind: result.display.kind, facts: result.display.facts }
      : undefined,
  }
}

/**
 * Build a legal JSON error whose serialized length fits the configured
 * output budget. The message shrinks code-point-safely; the shell and code
 * are never damaged, and the empty-message shell (66 chars) fits every
 * supported configuration (floor 96).
 */
function boundedTaskError(
  code: 'tool_validation_error' | 'tool_execution_error',
  message: string,
  maxOutputChars: number | undefined,
): ToolExecutionResult {
  if (maxOutputChars === undefined) return taskError(code, message)
  const shell = (messageText: string): string =>
    JSON.stringify({ ok: false, error: { code, message: messageText } })
  if (shell(message).length <= maxOutputChars) return taskError(code, message)
  let characters = [...message]
  while (characters.length > 0 && shell(characters.join('')).length > maxOutputChars) {
    characters = characters.slice(0, Math.floor(characters.length / 2))
  }
  let text = characters.join('')
  while (characters.length > 0 && shell(text).length > maxOutputChars) {
    characters = characters.slice(0, -1)
    text = characters.join('')
  }
  return { ...taskError(code, text) }
}

function taskError(
  code: 'tool_validation_error' | 'tool_execution_error',
  message: string,
): ToolExecutionResult {
  return {
    content: JSON.stringify({ ok: false, error: { code, message } }),
    isError: true,
    errorCode: code,
  }
}
