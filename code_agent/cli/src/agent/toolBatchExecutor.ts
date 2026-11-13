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

// [maint-note:e55c3ee1c043]
// 数据契约导航：BatchToolResult；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 isError：boolean。
// 成员 code（可省略）：string。
// 成员 display（可省略）：ToolDisplayFacts。
// [/maint-note:e55c3ee1c043]
export interface BatchToolResult {
  content: string
  isError: boolean
  /** Stable canonical outcome code; presentation must not parse it from text. */
  code?: string
  /** UI-only structured display facts; never used to reconstruct history. */
  display?: ToolDisplayFacts
}

// [maint-note:bbce50da40f3]
// 数据契约导航：ToolBatchUpdate；字段含义以消费端校验为准。
// 类型组合：| { type: 'event'; event: Extract<AgentEvent, { type: 'tool_start' | 'approval_required' | 'approval_resolved' }> } | { type: 'execution_started'; callId: string; mutating: boolean } | { type: 'result'; call: ToolCall; result: BatchToolResu。
// [/maint-note:bbce50da40f3]
export type ToolBatchUpdate =
  | { type: 'event'; event: Extract<AgentEvent, { type: 'tool_start' | 'approval_required' | 'approval_resolved' }> }
  | { type: 'execution_started'; callId: string; mutating: boolean }
  | { type: 'result'; call: ToolCall; result: BatchToolResult; receipt: ToolResultReceipt }

// [maint-note:75a34456eaea]
// 数据契约导航：ToolBatchExecutorOptions；字段含义以消费端校验为准。
// 成员 calls：readonly ToolCall[]。
// 成员 schemas：readonly ToolSchema[]。
// 成员 policy：ToolPolicy。
// 成员 ledger：ToolAttemptLedger。
// 成员 signal（可省略）：AbortSignal。
// 成员 maxOutputChars（可省略）：number。
// 成员 batchValidationError（可省略）：string。
// 成员 batchBudgetError（可省略）：{ readonly code: string; readonly message: string }。
// 成员 createRequestId：(callId: string, ordinal: number) => string。
// 成员 validate：(call: ToolCall) => AgentError | undefined。
// 成员 execute：(call: ToolCall) => Promise<BatchToolResult>。
// 成员 repeatPolicy（可省略）：ToolRepeatPolicy。
// [/maint-note:75a34456eaea]
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

  // [maint-note:1d85e4d1cb82]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：ToolBatchExecutorOptions。
  // [/maint-note:1d85e4d1cb82]
  constructor(private readonly options: ToolBatchExecutorOptions) {}

  // [maint-note:216083be1403]
  // 维护导航：run；下列为本函数源码中的接口与路径索引。
  // 返回契约：AsyncGenerator<ToolBatchUpdate>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!isAttemptRecord(validated)。
  // 分支条件：this.options.batchBudgetError。
  // 分支条件：repeated。
  // 分支条件：validationError。
  // 分支条件：schema && this.options.policy.requiresApproval(schema)。
  // 分支条件：!isAttemptRecord(pending)。
  // 分支条件：isAbortError(error, this.options.signal)。
  // 分支条件：!isAttemptRecord(resolved)。
  // 分支条件：!approved。
  // 分支条件：!isAttemptRecord(started)。
  // 分支条件：!outcome.ok。
  // 分支条件：!isAbortError(outcome.error, this.options.signal)。
  // 分支条件：this.options.signal?.aborted。
  // 遍历边界：const [ordinal, call] ← this.options.calls.entries()。
  // 显式抛出：error。
  // 显式抛出：outcome.error。
  // 显式抛出：createAbortError()。
  // 写入目标：requestId。
  // 写入目标：decision。
  // 写入目标：execution。
  // 直接调用：this.ensureRegistered。
  // 直接调用：this.options.calls.entries。
  // 直接调用：throwIfAborted。
  // 直接调用：this.options.validate。
  // 直接调用：this.schemaFor。
  // 直接调用：this.options.ledger.validate。
  // 直接调用：isAttemptRecord。
  // 直接调用：this.internalFailure。
  // 直接调用：this.result。
  // 直接调用：this.formatError。
  // 直接调用：this.options.repeatPolicy?.beforeAttempt。
  // 直接调用：this.options.policy.requiresApproval。
  // 直接调用：this.options.createRequestId。
  // 直接调用：this.options.ledger.requestApproval。
  // 直接调用：this.options.policy.approvalProvider.request。
  // 直接调用：normalizeApprovalResponse。
  // 直接调用：isAbortError。
  // 直接调用：this.options.ledger.failApproval。
  // 直接调用：toolDiagnostic。
  // 直接调用：this.options.ledger.resolveApproval。
  // 直接调用：this.confirmResult。
  // 直接调用：this.deliver。
  // 直接调用：this.options.ledger.beginExecution。
  // 直接调用：this.options.execute。
  // 直接调用：Promise.resolve(promise).then。
  // 直接调用：Promise.resolve。
  // 直接调用：awaitExecutionOutcome。
  // 直接调用：this.closeCall。
  // 直接调用：createAbortError。
  // [/maint-note:216083be1403]
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
  // [maint-note:6b7d3dded2cd]
  // 维护导航：closeInterrupted；下列为本函数源码中的接口与路径索引。
  // 返回契约：readonly Extract<ToolBatchUpdate, { type: 'result' }>[]。
  // 分支条件：this.options.ledger.resultWasAcknowledged(call.id)。
  // 分支条件：!receipt.delivered。
  // 遍历边界：const call ← this.options.calls。
  // 返回路径：updates。
  // 直接调用：this.ensureRegistered。
  // 直接调用：this.options.ledger.resultWasAcknowledged。
  // 直接调用：this.options.ledger.resultReceipt。
  // 直接调用：this.closeCall。
  // 直接调用：updates.push。
  // [/maint-note:6b7d3dded2cd]
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

  // [maint-note:e81daf5300fc]
  // 维护导航：acknowledge；下列为本函数源码中的接口与路径索引。
  // 输入 receipt：ToolResultReceipt。
  // 返回契约：void。
  // 分支条件：!isAttemptRecord(acknowledged)。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool result receipt acknowledgement was rejected')。
  // 直接调用：this.options.ledger.acknowledgeResult。
  // 直接调用：isAttemptRecord。
  // [/maint-note:e81daf5300fc]
  acknowledge(receipt: ToolResultReceipt): void {
    const acknowledged = this.options.ledger.acknowledgeResult(receipt.callId, receipt.receiptId)
    if (!isAttemptRecord(acknowledged)) {
      throw new AgentError('model_protocol_error', 'Tool result receipt acknowledgement was rejected')
    }
  }

  // [maint-note:069a3bf61e40]
  // 维护导航：ensureRegistered；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：this.registered。
  // 分支条件：!this.options.ledger.canRegister(this.options.calls.length)。
  // 分支条件：!isAttemptRecord(registered)。
  // 遍历边界：const [ordinal, call] ← this.options.calls.entries()。
  // 返回路径：无返回值。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool batch exceeds the run attempt capacity')。
  // 显式抛出：new AgentError( 'model_protocol_error', `Tool protocol transition rejected at call ordinal ${ordinal + 1}`, )。
  // 写入目标：this.registered。
  // 直接调用：this.options.ledger.canRegister。
  // 直接调用：this.options.calls.entries。
  // 直接调用：this.options.ledger.register。
  // 直接调用：isAttemptRecord。
  // [/maint-note:069a3bf61e40]
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

  // [maint-note:f722120085ac]
  // 维护导航：closeCall；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 返回契约：ToolResultReceipt。
  // 分支条件：!record。
  // 分支条件：record.state !== 'aborted'。
  // 分支条件：!isAttemptRecord(aborted)。
  // 分支条件：existing。
  // 分支条件：!isResultReceipt(receipt)。
  // 返回路径：existing。
  // 返回路径：receipt。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool closure could not find its canonical attempt')。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool cancellation lost its first-wins result')。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool cancellation result could not be confirmed')。
  // 直接调用：this.options.ledger.get。
  // 直接调用：this.schemaFor。
  // 直接调用：this.options.ledger.abort。
  // 直接调用：isAttemptRecord。
  // 直接调用：this.options.ledger.resultReceipt。
  // 直接调用：this.options.ledger.confirmResult。
  // 直接调用：isResultReceipt。
  // [/maint-note:f722120085ac]
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

  // [maint-note:9f0e5fabf622]
  // 维护导航：result；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 result：BatchToolResult。
  // 输入 origin：ToolResultOrigin。
  // 输入 resultCode：string（可省略）。
  // 输入 schema：ToolSchema（可省略）。
  // 返回契约：Generator<ToolBatchUpdate>。
  // 直接调用：this.confirmResult。
  // 直接调用：this.deliver。
  // [/maint-note:9f0e5fabf622]
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

  // [maint-note:7fbc0d4b6683]
  // 维护导航：confirmResult；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 result：BatchToolResult。
  // 输入 origin：ToolResultOrigin。
  // 输入 schema：ToolSchema（可省略）。
  // 返回契约：ToolResultReceipt。
  // 分支条件：!isResultReceipt(receipt)。
  // 分支条件：existing。
  // 返回路径：existing。
  // 返回路径：receipt。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool protocol ledger rejected its canonical result')。
  // 直接调用：this.options.ledger.confirmResult。
  // 直接调用：isResultReceipt。
  // 直接调用：this.options.ledger.resultReceipt。
  // [/maint-note:7fbc0d4b6683]
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

  // [maint-note:a9857ec75937]
  // 维护导航：deliver；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 receipt：ToolResultReceipt。
  // 输入 schema：ToolSchema（可省略）。
  // 返回契约：Generator<ToolBatchUpdate>。
  // 分支条件：receipt.origin !== 'framework_close' && schema?.taskManagement !== true && !this.repeatObserved.has(receipt.receiptId)。
  // 直接调用：this.repeatObserved.has。
  // 直接调用：this.repeatObserved.add。
  // 直接调用：this.options.repeatPolicy?.observe。
  // [/maint-note:a9857ec75937]
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

  // [maint-note:3eed6fc0c0fd]
  // 维护导航：internalFailure；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 transition：ToolAttemptTransitionError。
  // 输入 ordinal：number（可省略）。
  // 返回契约：Generator<ToolBatchUpdate>。
  // 分支条件：!this.options.ledger.get(call.id)。
  // 显式抛出：new AgentError('model_protocol_error', 'Tool protocol ledger could not register the requested attempt')。
  // 直接调用：this.options.ledger.get。
  // 直接调用：this.confirmResult。
  // 直接调用：this.formatError。
  // 直接调用：this.schemaFor。
  // 直接调用：this.deliver。
  // [/maint-note:3eed6fc0c0fd]
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

  // [maint-note:14609f57450b]
  // 维护导航：formatError；下列为本函数源码中的接口与路径索引。
  // 输入 code：string。
  // 输入 message：string。
  // 输入 diagnostic：ToolDiagnostic（可省略）。
  // 返回契约：string。
  // 返回路径：formatToolError(code, message, this.options.maxOutputChars, diagnostic)。
  // 直接调用：formatToolError。
  // [/maint-note:14609f57450b]
  private formatError(code: string, message: string, diagnostic?: ToolDiagnostic): string {
    return formatToolError(code, message, this.options.maxOutputChars, diagnostic)
  }

  // [maint-note:72d221506071]
  // 维护导航：schemaFor；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 返回契约：ToolSchema | undefined。
  // 返回路径：this.options.schemas.find((candidate) => candidate.name === call.name)。
  // 直接调用：this.options.schemas.find。
  // [/maint-note:72d221506071]
  private schemaFor(call: ToolCall): ToolSchema | undefined {
    return this.options.schemas.find((candidate) => candidate.name === call.name)
  }
}

// [maint-note:982a97cc11dd]
// 数据契约导航：ExecutionOutcome；字段含义以消费端校验为准。
// 类型组合：| { readonly ok: true; readonly result: BatchToolResult } | { readonly ok: false; readonly error: unknown }。
// [/maint-note:982a97cc11dd]
type ExecutionOutcome =
  | { readonly ok: true; readonly result: BatchToolResult }
  | { readonly ok: false; readonly error: unknown }

// [maint-note:2a8dfdbfae5c]
// 维护导航：isAttemptRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is { state: string }。
// 返回路径：value !== null && typeof value === 'object' && 'state' in value。
// [/maint-note:2a8dfdbfae5c]
function isAttemptRecord(value: unknown): value is { state: string } {
  return value !== null && typeof value === 'object' && 'state' in value
}

// [maint-note:e5371d2833c0]
// 维护导航：isResultReceipt；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is ToolResultReceipt。
// 返回路径：value !== null && typeof value === 'object' && 'receiptId' in value。
// [/maint-note:e5371d2833c0]
function isResultReceipt(value: unknown): value is ToolResultReceipt {
  return value !== null && typeof value === 'object' && 'receiptId' in value
}

// [maint-note:3fa4e5591cb8]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：createAbortError()。
// 直接调用：createAbortError。
// [/maint-note:3fa4e5591cb8]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

// [maint-note:d1fbb3145815]
// 维护导航：createAbortError；下列为本函数源码中的接口与路径索引。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:d1fbb3145815]
function createAbortError(): Error {
  const error = new Error('The agent turn was aborted')
  error.name = 'AbortError'
  return error
}

// [maint-note:576c5c4ea05b]
// 维护导航：awaitExecutionOutcome；下列为本函数源码中的接口与路径索引。
// 输入 execution：Promise<ExecutionOutcome>。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：Promise<ExecutionOutcome>。
// 分支条件：!signal。
// 分支条件：signal.aborted。
// 返回路径：execution。
// 返回路径：Promise.resolve({ ok: false, error: createAbortError() })。
// 返回路径：new Promise((resolve) => { let decided = false const finish = (outcome: ExecutionOutcome) => { if (decided) return decided = true signal.removeEventListener('abort', onAbort) resolve(outcome) } const onAbort = () => finish({ ok: false, erro。
// 直接调用：Promise.resolve。
// 直接调用：createAbortError。
// [/maint-note:576c5c4ea05b]
function awaitExecutionOutcome(
  execution: Promise<ExecutionOutcome>,
  signal?: AbortSignal,
): Promise<ExecutionOutcome> {
  if (!signal) return execution
  if (signal.aborted) return Promise.resolve({ ok: false, error: createAbortError() })
  return new Promise((resolve) => {
    let decided = false
    // [maint-note:9cd25e76c8b4]
    // 维护导航：finish；下列为本函数源码中的接口与路径索引。
    // 输入 outcome：ExecutionOutcome。
    // 分支条件：decided。
    // 返回路径：无返回值。
    // 写入目标：decided。
    // 直接调用：signal.removeEventListener。
    // 直接调用：resolve。
    // [/maint-note:9cd25e76c8b4]
    const finish = (outcome: ExecutionOutcome) => {
      if (decided) return
      decided = true
      signal.removeEventListener('abort', onAbort)
      resolve(outcome)
    }
    // [maint-note:be4b105a2b4f]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 直接调用：finish。
    // 直接调用：createAbortError。
    // [/maint-note:be4b105a2b4f]
    const onAbort = () => finish({ ok: false, error: createAbortError() })
    signal.addEventListener('abort', onAbort, { once: true })
    void execution.then(finish)
  })
}
