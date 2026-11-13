import type { AgentErrorCode } from './types/event.js'
import type { ToolDiagnostic } from '../tools/types.js'

export class AgentError extends Error {
  readonly code: AgentErrorCode
  readonly recoverable: boolean

  // [maint-note:4f817c7010c5]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 code：AgentErrorCode。
  // 输入 message：string。
  // 输入 recoverable：类型由调用方与函数体约束；缺省表达式 false。
  // 输入 diagnostic：ToolDiagnostic（可省略）。
  // 写入目标：this.name。
  // 写入目标：this.code。
  // 写入目标：this.recoverable。
  // 直接调用：super。
  // [/maint-note:4f817c7010c5]
  constructor(
    code: AgentErrorCode,
    message: string,
    recoverable = false,
    readonly diagnostic?: ToolDiagnostic,
  ) {
    super(message)
    this.name = 'AgentError'
    this.code = code
    this.recoverable = recoverable
  }
}

// [maint-note:3c9791196304]
// 维护导航：isAbortError；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：boolean。
// 分支条件：signal?.aborted。
// 返回路径：true。
// 返回路径：( error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError') )。
// [/maint-note:3c9791196304]
export function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true
  return (
    error instanceof Error &&
    (error.name === 'AbortError' || error.name === 'CanceledError')
  )
}

// [maint-note:92859ceb1e3f]
// 维护导航：createAbortError；下列为本函数源码中的接口与路径索引。
// 输入 message：类型由调用方与函数体约束；缺省表达式 'The operation was aborted'。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:92859ceb1e3f]
export function createAbortError(message = 'The operation was aborted'): Error {
  const error = new Error(message)
  error.name = 'AbortError'
  return error
}

// [maint-note:fe2ce6d8e07c]
// 维护导航：errorMessage；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：string。
// 返回路径：error instanceof Error ? error.message : String(error)。
// 直接调用：String。
// [/maint-note:fe2ce6d8e07c]
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
