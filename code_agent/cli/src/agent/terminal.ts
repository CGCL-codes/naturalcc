import type { AgentErrorCode, AgentEvent } from './types/event.js'

// [maint-note:c49136c95129]
// 数据契约导航：TerminalAgentEvent；字段含义以消费端校验为准。
// 类型组合：Extract< AgentEvent, { type: 'done' | 'error' | 'interrupted' | 'paused' } >。
// [/maint-note:c49136c95129]
export type TerminalAgentEvent = Extract<
  AgentEvent,
  { type: 'done' | 'error' | 'interrupted' | 'paused' }
>

// [maint-note:0cd0b82887a4]
// 数据契约导航：TerminalOutcome；字段含义以消费端校验为准。
// 类型组合：| { type: 'done'; content: string } | { type: 'error'; code: AgentErrorCode; message: string; workspaceMayHaveChanged: boolean } | { type: 'interrupted'; message: string; workspaceMayHaveChanged: boolean } | { type: 'paused' reason: 'budget。
// [/maint-note:0cd0b82887a4]
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
// [maint-note:1d196fe2ca32]
// 维护导航：isTerminalEvent；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：event is TerminalAgentEvent。
// 返回路径：terminalOutcome(event) !== undefined。
// 直接调用：terminalOutcome。
// [/maint-note:1d196fe2ca32]
export function isTerminalEvent(event: AgentEvent): event is TerminalAgentEvent {
  return terminalOutcome(event) !== undefined
}

// [maint-note:94caed7ebed9]
// 维护导航：terminalType；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：TerminalAgentEvent['type'] | undefined。
// 返回路径：isTerminalEvent(event) ? event.type : undefined。
// 直接调用：isTerminalEvent。
// [/maint-note:94caed7ebed9]
export function terminalType(event: AgentEvent): TerminalAgentEvent['type'] | undefined {
  return isTerminalEvent(event) ? event.type : undefined
}

/**
 * Maps a terminal event to the stable outcome shape consumed by adapters.
 * The switch is intentionally exhaustive for the current AgentEvent union;
 * Terminal event additions must update this mapper and its consumers together.
 */
// [maint-note:1d7a87822aef]
// 维护导航：terminalOutcome；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：TerminalOutcome | undefined。
// 分支条件：switch event.type。
// 分支条件：case 'done'。
// 分支条件：case 'error'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'paused'。
// 分支条件：case 'user_message'。
// 分支条件：case 'assistant_delta'。
// 分支条件：case 'assistant_segment'。
// 分支条件：case 'assistant_message'。
// 分支条件：case 'tool_start'。
// 分支条件：case 'tool_result'。
// 分支条件：case 'approval_required'。
// 分支条件：case 'approval_resolved'。
// 返回路径：{ type: 'done', content: event.content }。
// 返回路径：{ type: 'error', code: event.code, message: event.message, workspaceMayHaveChanged: event.workspaceMayHaveChanged, }。
// 返回路径：{ type: 'interrupted', message: event.message, workspaceMayHaveChanged: event.workspaceMayHaveChanged, }。
// 返回路径：{ type: 'paused', reason: event.reason, content: event.content, workspaceMayHaveChanged: event.workspaceMayHaveChanged, }。
// 返回路径：undefined。
// 返回路径：futureNonTerminal(event)。
// 直接调用：futureNonTerminal。
// [/maint-note:1d7a87822aef]
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
// [maint-note:ec5190f57fba]
// 维护导航：assertNeverAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 value：never。
// 返回契约：never。
// 显式抛出：new Error(`Unhandled agent event: ${String(value)}`)。
// 直接调用：String。
// [/maint-note:ec5190f57fba]
export function assertNeverAgentEvent(value: never): never {
  throw new Error(`Unhandled agent event: ${String(value)}`)
}

// [maint-note:0fc99d8681ce]
// 维护导航：futureNonTerminal；下列为本函数源码中的接口与路径索引。
// 输入 value：never。
// 返回契约：undefined。
// 返回路径：undefined。
// [/maint-note:0fc99d8681ce]
function futureNonTerminal(value: never): undefined {
  // Runtime adapters may still observe an event from a newer producer. It is
  // non-terminal until the shared union is deliberately extended; the never
  // parameter keeps that extension visible to TypeScript callers.
  void value
  return undefined
}
