import type { AgentEvent } from './types/event.js'
import { AgentError, isAbortError } from './errors.js'
import { formatToolError, toolDiagnostic } from '../tools/diagnostics.js'
import type { ToolDiagnostic } from '../tools/types.js'
import {
  type ToolAttemptLedger,
  type ToolAttemptTransitionError,
  type ToolResultOrigin,
  type ToolResultReceipt,
} from './toolAttemptLedger.js'
import type { ToolCall, ToolDisplayFacts, ToolSchema } from '../tools/types.js'
import { normalizeApprovalResponse, type NormalizedApproval, type ToolPolicy } from '../tools/approval.js'
import type { ToolRepeatPolicy } from './runLoopPolicy.js'

export interface BatchToolResult {
  content: string
  isError: boolean
  /** Stable canonical outcome code; presentation must not parse it from text. */
  code?: string
  /** UI-only structured display facts; never used to reconstruct history. */
  display?: ToolDisplayFacts
}

export type ToolBatchUpdate =
  | { type: 'event'; event: Extract<AgentEvent, { type: 'tool_start' | 'approval_required' | 'approval_resolved' }> }
  | { type: 'execution_started'; callId: string; mutating: boolean }
  | { type: 'result'; call: ToolCall; result: BatchToolResult; receipt: ToolResultReceipt }

export interface ToolBatchExecutorOptions {
  calls: readonly ToolCall[]
  schemas: readonly ToolSchema[]
  policy: ToolPolicy
  ledger: ToolAttemptLedger
  signal?: AbortSignal
  maxOutputChars?: number
  batchValidationError?: string
  /** Whole-batch admission failure; no validation policy, approval or tool runs. */
  batchBudgetError?: { readonly code: string; readonly message: string }
  createRequestId: (callId: string, ordinal: number) => string
  validate: (call: ToolCall) => AgentError | undefined
  execute: (call: ToolCall) => Promise<BatchToolResult>
  repeatPolicy?: ToolRepeatPolicy
}

/** Executes one canonical model tool batch while owning attempt transitions. */
export class ToolBatchExecutor {
  private registered = false
  private readonly repeatObserved = new Set<number>()

  constructor(private readonly options: ToolBatchExecutorOptions) {}

  async *run(): AsyncGenerator<ToolBatchUpdate> {
    this.ensureRegistered()

    for (const [ordinal, call] of this.options.calls.entries()) {
      yield {
        type: 'event',
        event: {
          type: 'tool_start',
          callId: call.id,
          name: call.name,
          input: call.arguments,
        },
      }
      throwIfAborted(this.options.signal)

      const validationError = this.options.batchValidationError
        ? new AgentError('tool_validation_error', this.options.batchValidationError, true)
        : this.options.validate(call)
      const schema = this.schemaFor(call)
      const validated = this.options.ledger.validate(call.id)
      if (!isAttemptRecord(validated)) {
        yield* this.internalFailure(call, validated, ordinal)
        continue
      }
      if (this.options.batchBudgetError) {
        yield* this.result(call, {
          content: this.formatError(
            this.options.batchBudgetError.code,
            this.options.batchBudgetError.message,
          ),
          isError: true,
          code: this.options.batchBudgetError.code,
        }, 'validation', this.options.batchBudgetError.code, schema)
        continue
      }
      const repeated = schema?.taskManagement === true ? undefined : this.options.repeatPolicy?.beforeAttempt({
        call,
        schema,
        validationCode: validationError?.code,
      })
      if (repeated) {
        yield* this.result(call, {
          content: this.formatError(repeated.code, repeated.message),
          isError: true,
        }, 'validation', repeated.code, schema)
        continue
      }
      if (validationError) {
        yield* this.result(call, {
          content: this.formatError(validationError.code, validationError.message, validationError.diagnostic),
          isError: true,
        }, 'validation', validationError.code, schema)
        continue
      }

      if (schema && this.options.policy.requiresApproval(schema)) {
        const reason = typeof call.arguments.reason === 'string' ? call.arguments.reason : ''
        let requestId: string
        try {
          requestId = this.options.createRequestId(call.id, ordinal)
        } catch {
          yield* this.internalFailure(call, 'invalid_request_id', ordinal)
          continue
        }
        const pending = this.options.ledger.requestApproval(call.id, requestId)
        if (!isAttemptRecord(pending)) {
          yield* this.internalFailure(call, pending, ordinal)
          continue
        }

        yield {
          type: 'event',
          event: {
            type: 'approval_required',
            requestId,
            callId: call.id,
            name: call.name,
            input: call.arguments,
            reason,
          },
        }

        let decision: NormalizedApproval
        try {
          throwIfAborted(this.options.signal)
          const response = await this.options.policy.approvalProvider.request({
            requestId,
            callId: call.id,
            toolName: call.name,
            input: call.arguments,
            reason,
          }, this.options.signal)
          throwIfAborted(this.options.signal)
          decision = normalizeApprovalResponse(response)
        } catch (error) {
          if (isAbortError(error, this.options.signal)) throw error
          this.options.ledger.failApproval(call.id, requestId)
          yield* this.result(call, {
            content: this.formatError('approval_provider_error', '', toolDiagnostic('approval_provider_failed', {
              message: error instanceof Error ? error.message : undefined,
            })),
            isError: true,
          }, 'approval', 'approval_provider_error', schema)
          continue
        }

        throwIfAborted(this.options.signal)
        const approved = decision.approved
        const resolved = this.options.ledger.resolveApproval(call.id, requestId, approved)
        if (!isAttemptRecord(resolved)) {
          yield* this.internalFailure(call, resolved, ordinal)
          continue
        }
        // Confirm a denial before the approval-resolved yield. A consumer may
        // stop at that boundary; closure must still recover the original fact.
        const deniedReceipt = approved
          ? undefined
          : this.confirmResult(call, {
              content: this.formatError('tool_denied', decision.denial.message),
              isError: true,
              code: 'tool_denied',
            }, 'approval', schema)
        yield {
          type: 'event',
          event: { type: 'approval_resolved', requestId, approved },
        }
        throwIfAborted(this.options.signal)

        if (!approved) {
          yield* this.deliver(call, deniedReceipt!, schema)
          continue
        }
      }

      // This is the last cancellation check before invocation. The execution-start fact
      // and the call expression are adjacent in one synchronous stack: no
      // observer, yield, await, or second abort check can split them.
      throwIfAborted(this.options.signal)
      const started = this.options.ledger.beginExecution(call.id)
      if (!isAttemptRecord(started)) {
        yield* this.internalFailure(call, started, ordinal)
        continue
      }
      let execution: Promise<ExecutionOutcome>
      try {
        const promise = this.options.execute(call)
        // Attach both handlers before any yield so an abandoned consumer can
        // never turn an already-started rejection into an unhandled promise.
        execution = Promise.resolve(promise).then(
          (result) => ({ ok: true as const, result }),
          (error: unknown) => ({ ok: false as const, error }),
        )
      } catch (error) {
        execution = Promise.resolve({ ok: false as const, error })
      }
      yield {
        type: 'execution_started',
        callId: call.id,
        mutating: schema?.readOnly !== true,
      }

      // Do not depend on a tool implementation honoring AbortSignal. The
      // invocation promise already has fulfillment/rejection handlers, so
      // cancellation can close the protocol immediately while a late tool
      // outcome remains inert and cannot become an unhandled rejection.
      const outcome = await awaitExecutionOutcome(execution, this.options.signal)
      if (!outcome.ok) {
        if (!isAbortError(outcome.error, this.options.signal)) throw outcome.error
        const receipt = this.closeCall(call)
        yield* this.deliver(call, receipt, schema)
        throw outcome.error
      }
      if (this.options.signal?.aborted) {
        const receipt = this.closeCall(call)
        yield* this.deliver(call, receipt, schema)
        throw createAbortError()
      }
      yield* this.result(call, outcome.result, 'executor', undefined, schema)
    }
  }

  /**
   * Synchronously closes every unconsumed call in canonical order. It never
   * requests approval, invokes a tool, or observes repeat progress.
   */
  closeInterrupted(): readonly Extract<ToolBatchUpdate, { type: 'result' }>[] {
    this.ensureRegistered()
    const updates: Array<Extract<ToolBatchUpdate, { type: 'result' }>> = []
    for (const call of this.options.calls) {
      if (this.options.ledger.resultWasAcknowledged(call.id)) continue
      const existing = this.options.ledger.resultReceipt(call.id)
      const receipt = existing ?? this.closeCall(call)
      if (!receipt.delivered) {
        updates.push({ type: 'result', call, result: receipt.result, receipt })
      }
    }
    return updates
  }

  acknowledge(receipt: ToolResultReceipt): void {
    const acknowledged = this.options.ledger.acknowledgeResult(receipt.callId, receipt.receiptId)
    if (!isAttemptRecord(acknowledged)) {
      throw new AgentError('model_protocol_error', 'Tool result receipt acknowledgement was rejected')
    }
  }

  private ensureRegistered(): void {
    if (this.registered) return
    if (!this.options.ledger.canRegister(this.options.calls.length)) {
      throw new AgentError('model_protocol_error', 'Tool batch exceeds the run attempt capacity')
    }
    for (const [ordinal, call] of this.options.calls.entries()) {
      const registered = this.options.ledger.register(call)
      if (!isAttemptRecord(registered)) {
        throw new AgentError(
          'model_protocol_error',
          `Tool protocol transition rejected at call ordinal ${ordinal + 1}`,
        )
      }
    }
    this.registered = true
  }

  private closeCall(call: ToolCall): ToolResultReceipt {
    const record = this.options.ledger.get(call.id)
    if (!record) {
      throw new AgentError('model_protocol_error', 'Tool closure could not find its canonical attempt')
    }
    const mutating = this.schemaFor(call)?.readOnly !== true
    if (record.state !== 'aborted') {
      const aborted = this.options.ledger.abort(call.id)
      if (!isAttemptRecord(aborted)) {
        const existing = this.options.ledger.resultReceipt(call.id)
        if (existing) return existing
        throw new AgentError('model_protocol_error', 'Tool cancellation lost its first-wins result')
      }
    }
    const content = record.invoked
      ? mutating
        ? 'Tool execution interrupted after invocation; the final effect is unknown.'
        : 'Tool execution interrupted after invocation; no usable result was observed.'
      : 'Tool execution interrupted before invocation; the tool was not executed.'
    const receipt = this.options.ledger.confirmResult(call.id, {
      content,
      isError: true,
      code: 'interrupted',
    }, {
      origin: 'framework_close',
      mutating,
    })
    if (!isResultReceipt(receipt)) {
      throw new AgentError('model_protocol_error', 'Tool cancellation result could not be confirmed')
    }
    return receipt
  }

  private *result(
    call: ToolCall,
    result: BatchToolResult,
    origin: ToolResultOrigin,
    resultCode?: string,
    schema?: ToolSchema,
  ): Generator<ToolBatchUpdate> {
    const normalized = resultCode && result.code === undefined
      ? { ...result, code: resultCode }
      : result
    const receipt = this.confirmResult(call, normalized, origin, schema)
    yield* this.deliver(call, receipt, schema)
  }

  private confirmResult(
    call: ToolCall,
    result: BatchToolResult,
    origin: ToolResultOrigin,
    schema?: ToolSchema,
  ): ToolResultReceipt {
    const receipt = this.options.ledger.confirmResult(call.id, result, {
      origin,
      mutating: schema?.readOnly !== true,
    })
    if (!isResultReceipt(receipt)) {
      const existing = this.options.ledger.resultReceipt(call.id)
      if (existing) return existing
      throw new AgentError('model_protocol_error', 'Tool protocol ledger rejected its canonical result')
    }
    return receipt
  }

  private *deliver(
    call: ToolCall,
    receipt: ToolResultReceipt,
    schema?: ToolSchema,
  ): Generator<ToolBatchUpdate> {
    if (
      receipt.origin !== 'framework_close' &&
      schema?.taskManagement !== true &&
      !this.repeatObserved.has(receipt.receiptId)
    ) {
      // Mark before the internal observer: if an invariant violation throws,
      // cleanup must not apply a partially observed result a second time.
      this.repeatObserved.add(receipt.receiptId)
      this.options.repeatPolicy?.observe({
        call,
        schema,
        result: receipt.result,
        resultCode: receipt.result.code,
      })
    }
    yield { type: 'result', call, result: receipt.result, receipt }
  }

  private *internalFailure(
    call: ToolCall,
    transition: ToolAttemptTransitionError,
    ordinal?: number,
  ): Generator<ToolBatchUpdate> {
    if (!this.options.ledger.get(call.id)) {
      throw new AgentError('model_protocol_error', 'Tool protocol ledger could not register the requested attempt')
    }
    const receipt = this.confirmResult(call, {
      content: this.formatError(
        'model_protocol_error',
        `Tool protocol transition rejected at call ordinal ${(ordinal ?? 0) + 1}: ${transition}`,
      ),
      isError: true,
      code: 'model_protocol_error',
    }, 'validation', this.schemaFor(call))
    yield* this.deliver(call, receipt, this.schemaFor(call))
  }

  private formatError(code: string, message: string, diagnostic?: ToolDiagnostic): string {
    return formatToolError(code, message, this.options.maxOutputChars, diagnostic)
  }

  private schemaFor(call: ToolCall): ToolSchema | undefined {
    return this.options.schemas.find((candidate) => candidate.name === call.name)
  }
}

type ExecutionOutcome =
  | { readonly ok: true; readonly result: BatchToolResult }
  | { readonly ok: false; readonly error: unknown }

function isAttemptRecord(value: unknown): value is { state: string } {
  return value !== null && typeof value === 'object' && 'state' in value
}

function isResultReceipt(value: unknown): value is ToolResultReceipt {
  return value !== null && typeof value === 'object' && 'receiptId' in value
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

function createAbortError(): Error {
  const error = new Error('The agent turn was aborted')
  error.name = 'AbortError'
  return error
}

function awaitExecutionOutcome(
  execution: Promise<ExecutionOutcome>,
  signal?: AbortSignal,
): Promise<ExecutionOutcome> {
  if (!signal) return execution
  if (signal.aborted) return Promise.resolve({ ok: false, error: createAbortError() })
  return new Promise((resolve) => {
    let decided = false
    const finish = (outcome: ExecutionOutcome) => {
      if (decided) return
      decided = true
      signal.removeEventListener('abort', onAbort)
      resolve(outcome)
    }
    const onAbort = () => finish({ ok: false, error: createAbortError() })
    signal.addEventListener('abort', onAbort, { once: true })
    void execution.then(finish)
  })
}
