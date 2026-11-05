import type { AgentEvent } from '../agent/types/event.js'
import { assertNeverAgentEvent, isTerminalEvent } from '../agent/terminal.js'
import { formatToolOutput } from './state/messageReducer.js'
import { buildApprovalPresentation } from '../tools/approval.js'

export interface EventWriter {
  write(chunk: string): void
}

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

  constructor(
    private readonly stdout: EventWriter,
    private readonly stderr: EventWriter,
    private readonly options: OneShotEventRendererOptions = {},
  ) {}

  get exitCode(): number {
    return this.code
  }

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

function deniedInstructionLabel(toolName: string, label: string): string {
  return toolName === 'apply_patch' && label === 'Patch' ? 'Patch (not executed)' : label
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
