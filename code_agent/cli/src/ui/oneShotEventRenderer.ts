import type { AgentEvent } from '../agent/types/event.js'
import { assertNeverAgentEvent, isTerminalEvent } from '../agent/terminal.js'
import { formatToolOutput } from './state/messageReducer.js'
import { buildApprovalPresentation } from '../tools/approval.js'

// [maint-note:683292c669e9]
// 数据契约导航：EventWriter；字段含义以消费端校验为准。
// 成员 write：void。
// [/maint-note:683292c669e9]
export interface EventWriter {
  write(chunk: string): void
}

// [maint-note:3524ac95c3be]
// 数据契约导航：OneShotEventRendererOptions；字段含义以消费端校验为准。
// 成员 approvalPresentationOwner（可省略）：'provider' | 'renderer' | 'none'。
// [/maint-note:3524ac95c3be]
export interface OneShotEventRendererOptions {
  approvalPresentationOwner?: 'provider' | 'renderer' | 'none'
}

export class OneShotEventRenderer {
  private code = 0
  private closed = false
  private pendingAssistantMessage: string | undefined
  private readonly toolPhases = new Map<string, 'pending' | 'rendered' | 'denied'>()
  private readonly approvalNotices = new Set<string>()
  private readonly approvalRequests = new Map<string, {
    callId: string
    name: string
    input: Record<string, unknown>
    reason: string
  }>()
  private readonly approvalDecisions = new Map<string, 'approved' | 'denied'>()

  // [maint-note:912d502c5691]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 stdout：EventWriter。
  // 输入 stderr：EventWriter。
  // 输入 options：OneShotEventRendererOptions；缺省表达式 {}。
  // [/maint-note:912d502c5691]
  constructor(
    private readonly stdout: EventWriter,
    private readonly stderr: EventWriter,
    private readonly options: OneShotEventRendererOptions = {},
  ) {}

  // [maint-note:0e83cab88f4d]
  // 维护导航：exitCode；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.code。
  // [/maint-note:0e83cab88f4d]
  get exitCode(): number {
    return this.code
  }

  // [maint-note:8883a10bd5bc]
  // 维护导航：handle；下列为本函数源码中的接口与路径索引。
  // 输入 event：AgentEvent。
  // 返回契约：void。
  // 分支条件：this.closed。
  // 分支条件：isTerminalEvent(event)。
  // 分支条件：switch event.type。
  // 分支条件：case 'assistant_delta'。
  // 分支条件：case 'assistant_segment'。
  // 分支条件：case 'assistant_message'。
  // 分支条件：case 'tool_start'。
  // 分支条件：!this.toolPhases.has(event.callId)。
  // 分支条件：case 'tool_result'。
  // 分支条件：this.toolPhases.get(event.callId) === 'rendered' || this.toolPhases.get(event.callId) === 'denied'。
  // 分支条件：case 'approval_required'。
  // 分支条件：this.toolPhases.get(event.callId) === 'rendered' || this.toolPhases.get(event.callId) === 'denied' || this.approvalDecisions.has(event.callId) || [...this.approvalRequests.values()].some((request) => request.callId === event.callId)。
  // 分支条件：this.options.approvalPresentationOwner === 'provider' || this.options.approvalPresentationOwner === 'none'。
  // 分支条件：!this.approvalNotices.has(event.callId)。
  // 分支条件：case 'approval_resolved'。
  // 分支条件：!request || this.approvalDecisions.has(request.callId)。
  // 分支条件：!event.approved && this.toolPhases.get(callId) !== 'rendered' && this.toolPhases.get(callId) !== 'denied'。
  // 分支条件：case 'error'。
  // 分支条件：case 'interrupted'。
  // 分支条件：case 'paused'。
  // 分支条件：case 'done'。
  // 分支条件：case 'user_message'。
  // 返回路径：无返回值。
  // 写入目标：this.closed。
  // 写入目标：this.pendingAssistantMessage。
  // 写入目标：this.code。
  // 直接调用：isTerminalEvent。
  // 直接调用：this.toolPhases.has。
  // 直接调用：this.toolPhases.set。
  // 直接调用：this.toolPhases.get。
  // 直接调用：this.stderr.write。
  // 直接调用：formatToolOutput。
  // 直接调用：this.approvalDecisions.has。
  // 直接调用：[...this.approvalRequests.values()].some。
  // 直接调用：this.approvalRequests.values。
  // 直接调用：this.approvalRequests.set。
  // 直接调用：isRecord。
  // 直接调用：this.approvalNotices.has。
  // 直接调用：this.approvalNotices.add。
  // 直接调用：buildApprovalPresentation。
  // 直接调用：this.approvalRequests.get。
  // 直接调用：this.approvalDecisions.set。
  // 直接调用：deniedInstructionLabel。
  // 直接调用：Math.max。
  // 直接调用：this.stdout.write。
  // 直接调用：assertNeverAgentEvent。
  // [/maint-note:8883a10bd5bc]
  handle(event: AgentEvent): void {
    if (this.closed) return
    if (isTerminalEvent(event)) this.closed = true

    switch (event.type) {
      case 'assistant_delta':
        // stdout is a scripting interface: only a committed final assistant
        // message may be written there. REPL owns live delta rendering.
        break
      case 'assistant_segment':
        // Intermediate text before a tool-call batch is REPL-only. One-shot
        // stdout remains final-answer-only and must not expose a half turn.
        break
      case 'assistant_message':
        // A pause checkpoint is delivered only by the paused terminal, so this
        // event now precedes done alone. The candidate is still held until the
        // terminal confirms it, and stdout stays final-answer-only.
        this.pendingAssistantMessage = event.content
        break
      case 'tool_start':
        // one-shot output is not an in-place UI: render only the final result.
        if (!this.toolPhases.has(event.callId)) this.toolPhases.set(event.callId, 'pending')
        break
      case 'tool_result':
        if (this.toolPhases.get(event.callId) === 'rendered' || this.toolPhases.get(event.callId) === 'denied') break
        this.toolPhases.set(event.callId, 'rendered')
        this.stderr.write(
          `[tool${event.isError ? ' error' : ''}] ${event.name}: ${formatToolOutput(event.output)}\n`,
        )
        break
      case 'approval_required':
        if (
          this.toolPhases.get(event.callId) === 'rendered' ||
          this.toolPhases.get(event.callId) === 'denied' ||
          this.approvalDecisions.has(event.callId) ||
          [...this.approvalRequests.values()].some((request) => request.callId === event.callId)
        ) return
        if (!this.toolPhases.has(event.callId)) this.toolPhases.set(event.callId, 'pending')
        this.approvalRequests.set(event.requestId, {
          callId: event.callId,
          name: event.name,
          input: isRecord(event.input) ? event.input : {},
          reason: event.reason,
        })
        if (this.options.approvalPresentationOwner === 'provider' || this.options.approvalPresentationOwner === 'none') return
        if (!this.approvalNotices.has(event.callId)) {
          this.approvalNotices.add(event.callId)
          const presentation = buildApprovalPresentation({
            requestId: event.requestId,
            callId: event.callId,
            toolName: event.name,
            input: isRecord(event.input) ? event.input : {},
            reason: event.reason,
          })
          this.stderr.write(
            `[approval] ${presentation.title}\n${presentation.instructionLabel}:\n${presentation.instruction}\nReason:\n${presentation.reason}\n`,
          )
        }
        return
      case 'approval_resolved':
        {
          const request = this.approvalRequests.get(event.requestId)
          if (!request || this.approvalDecisions.has(request.callId)) break
          const callId = request.callId
          this.approvalDecisions.set(callId, event.approved ? 'approved' : 'denied')
          if (!event.approved && this.toolPhases.get(callId) !== 'rendered' && this.toolPhases.get(callId) !== 'denied') {
            const presentation = buildApprovalPresentation({
              requestId: '',
              callId: '',
              toolName: request.name,
              input: request.input,
              reason: request.reason,
            })
            this.toolPhases.set(callId, 'denied')
            this.stderr.write(
              `[tool denied] ${request.name}\n${deniedInstructionLabel(request.name, presentation.instructionLabel)}:\n${presentation.instruction}\nReason:\n${presentation.reason}\n`,
            )
          }
        }
        break
      case 'error':
        this.code = Math.max(this.code, event.code === 'config_error' ? 2 : 1)
        this.pendingAssistantMessage = undefined
        this.stderr.write(`[error:${event.code}] ${event.message}\n`)
        this.closed = true
        break
      case 'interrupted':
        this.code = 130
        this.pendingAssistantMessage = undefined
        this.stderr.write(`${event.message}\n`)
        this.closed = true
        break
      case 'paused':
        this.code = Math.max(this.code, 1)
        this.pendingAssistantMessage = undefined
        this.stderr.write(`[paused:${event.reason}] ${event.content}\n`)
        this.closed = true
        break
      case 'done':
        this.stdout.write(`${event.content}\n`)
        this.pendingAssistantMessage = undefined
        this.closed = true
        break
      case 'user_message':
        break
      default:
        assertNeverAgentEvent(event)
    }
  }
}

// [maint-note:fa27639b957e]
// 维护导航：deniedInstructionLabel；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 label：string。
// 返回契约：string。
// 返回路径：toolName === 'apply_patch' && label === 'Patch' ? 'Patch (not executed)' : label。
// [/maint-note:fa27639b957e]
function deniedInstructionLabel(toolName: string, label: string): string {
  return toolName === 'apply_patch' && label === 'Patch' ? 'Patch (not executed)' : label
}

// [maint-note:66cf0961af41]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:66cf0961af41]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
