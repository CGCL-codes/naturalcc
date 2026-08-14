import type { AgentEvent } from '../agent/types/event.js'
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
  private readonly toolPhases = new Map<string, 'pending' | 'rendered'>()
  private readonly approvalNotices = new Set<string>()

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

    switch (event.type) {
      case 'assistant_delta':
        // stdout is a scripting interface: only a committed final assistant
        // message may be written there. REPL owns live delta rendering.
        break
      case 'assistant_message':
        this.stdout.write(`${event.content}\n`)
        break
      case 'tool_start':
        // one-shot output is not an in-place UI: render only the final result.
        if (!this.toolPhases.has(event.callId)) this.toolPhases.set(event.callId, 'pending')
        break
      case 'tool_result':
        if (this.toolPhases.get(event.callId) === 'rendered') break
        this.toolPhases.set(event.callId, 'rendered')
        this.stderr.write(
          `[tool${event.isError ? ' error' : ''}] ${event.name}: ${formatToolOutput(event.output)}\n`,
        )
        break
      case 'approval_required':
        if (this.toolPhases.get(event.callId) === 'rendered') return
        if (!this.toolPhases.has(event.callId)) this.toolPhases.set(event.callId, 'pending')
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
        break
      case 'error':
        this.code = Math.max(this.code, event.code === 'config_error' ? 2 : 1)
        this.stderr.write(`[error:${event.code}] ${event.message}\n`)
        this.closed = true
        break
      case 'interrupted':
        this.code = 130
        this.stderr.write(`${event.message}\n`)
        this.closed = true
        break
      case 'done':
        this.closed = true
        break
      case 'user_message':
        break
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
