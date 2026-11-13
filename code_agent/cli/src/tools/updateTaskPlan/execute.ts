import type { TaskPlanOperation } from '../../task/contracts.js'
import type { ToolExecutionContext, ToolExecutionResult } from '../types.js'
import { formatToolErrorJson } from '../diagnostics.js'

// [maint-note:e38fd509478d]
// 维护导航：executeUpdateTaskPlan；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 返回契约：Promise<ToolExecutionResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!port || !callId。
// 分支条件：context.signal?.aborted。
// 分支条件：!result.ok && result.value.error && typeof result.value.error === 'object'。
// 分支条件：context.maxOutputChars !== undefined && content.length > context.maxOutputChars。
// 返回路径：boundedTaskError('tool_execution_error', 'No active run task owner is available', context.maxOutputChars)。
// 返回路径：{ content: formatToolErrorJson(result.value.error as Record<string, unknown>, context.maxOutputChars), isError: true, errorCode: result.errorCode }。
// 返回路径：boundedTaskError( 'tool_validation_error', 'Task plan result exceeds the configured tool output budget', context.maxOutputChars, )。
// 返回路径：{ content, isError: !result.ok, errorCode: result.errorCode, display: result.ok && result.display ? { kind: result.display.kind, facts: result.display.facts } : undefined, }。
// 显式抛出：error。
// 写入目标：error.name。
// 直接调用：boundedTaskError。
// 直接调用：port.executeTaskPlanOperation。
// 直接调用：formatToolErrorJson。
// 直接调用：JSON.stringify。
// [/maint-note:e38fd509478d]
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
// [maint-note:17e2167cbb4a]
// 维护导航：boundedTaskError；下列为本函数源码中的接口与路径索引。
// 输入 code：'tool_validation_error' | 'tool_execution_error'。
// 输入 message：string。
// 输入 maxOutputChars：number | undefined。
// 返回契约：ToolExecutionResult。
// 分支条件：maxOutputChars === undefined。
// 分支条件：shell(message).length <= maxOutputChars。
// 遍历边界：characters.length > 0 && shell(characters.join('')).length > maxOutputChars。
// 遍历边界：characters.length > 0 && shell(text).length > maxOutputChars。
// 返回路径：taskError(code, message)。
// 返回路径：{ ...taskError(code, text) }。
// 写入目标：characters。
// 写入目标：text。
// 直接调用：taskError。
// 直接调用：shell。
// 直接调用：characters.join。
// 直接调用：characters.slice。
// 直接调用：Math.floor。
// [/maint-note:17e2167cbb4a]
function boundedTaskError(
  code: 'tool_validation_error' | 'tool_execution_error',
  message: string,
  maxOutputChars: number | undefined,
): ToolExecutionResult {
  if (maxOutputChars === undefined) return taskError(code, message)
  // [maint-note:e85689db8826]
  // 维护导航：shell；下列为本函数源码中的接口与路径索引。
  // 输入 messageText：string。
  // 返回契约：string。
  // 直接调用：JSON.stringify。
  // [/maint-note:e85689db8826]
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

// [maint-note:2786f11a1672]
// 维护导航：taskError；下列为本函数源码中的接口与路径索引。
// 输入 code：'tool_validation_error' | 'tool_execution_error'。
// 输入 message：string。
// 返回契约：ToolExecutionResult。
// 返回路径：{ content: JSON.stringify({ ok: false, error: { code, message } }), isError: true, errorCode: code, }。
// 直接调用：JSON.stringify。
// [/maint-note:2786f11a1672]
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
