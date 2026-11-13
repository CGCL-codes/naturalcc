import type { ToolCall, ToolDisplayFacts } from '../tools/types.js'

// [maint-note:1b7acb625995]
// 数据契约导航：ToolAttemptState；字段含义以消费端校验为准。
// 类型组合：| 'registered' | 'validated' | 'approval_pending' | 'approved' | 'denied' | 'approval_failed' | 'execution_started' | 'result_emitted' | 'aborted' | 'settled'。
// [/maint-note:1b7acb625995]
export type ToolAttemptState =
  | 'registered'
  | 'validated'
  | 'approval_pending'
  | 'approved'
  | 'denied'
  | 'approval_failed'
  | 'execution_started'
  | 'result_emitted'
  | 'aborted'
  | 'settled'

// [maint-note:cdc1eb4ee944]
// 数据契约导航：ToolAttemptRecord；字段含义以消费端校验为准。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 state：ToolAttemptState。
// 成员 requestId（可省略）：string。
// 成员 approvalDecision（可省略）：'approved' | 'denied'。
// 成员 resultCode（可省略）：string。
// 成员 invoked：boolean。
// [/maint-note:cdc1eb4ee944]
export interface ToolAttemptRecord {
  readonly callId: string
  readonly toolName: string
  readonly state: ToolAttemptState
  readonly requestId?: string
  readonly approvalDecision?: 'approved' | 'denied'
  readonly resultCode?: string
  /** True only after the executor function itself was synchronously entered. */
  readonly invoked: boolean
}

// [maint-note:3edba00597a8]
// 数据契约导航：CanonicalToolResultPayload；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 isError：boolean。
// 成员 code（可省略）：string。
// 成员 display（可省略）：ToolDisplayFacts。
// [/maint-note:3edba00597a8]
export interface CanonicalToolResultPayload {
  readonly content: string
  readonly isError: boolean
  readonly code?: string
  readonly display?: ToolDisplayFacts
}

// [maint-note:803bebafbbb9]
// 数据契约导航：ToolResultOrigin；字段含义以消费端校验为准。
// 类型组合：'executor' | 'validation' | 'approval' | 'framework_close'。
// [/maint-note:803bebafbbb9]
export type ToolResultOrigin = 'executor' | 'validation' | 'approval' | 'framework_close'

/** Run-local recovery receipt; canonical payload is released after append. */
// [maint-note:d4ee6a979fde]
// 数据契约导航：ToolResultReceipt；字段含义以消费端校验为准。
// 成员 receiptId：number。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 result：CanonicalToolResultPayload。
// 成员 origin：ToolResultOrigin。
// 成员 invoked：boolean。
// 成员 mutating：boolean。
// 成员 delivered：boolean。
// [/maint-note:d4ee6a979fde]
export interface ToolResultReceipt {
  readonly receiptId: number
  readonly callId: string
  readonly toolName: string
  readonly result: CanonicalToolResultPayload
  readonly origin: ToolResultOrigin
  readonly invoked: boolean
  readonly mutating: boolean
  readonly delivered: boolean
}

// [maint-note:a3b645cdb21c]
// 数据契约导航：ToolAttemptTransitionError；字段含义以消费端校验为准。
// 类型组合：| 'duplicate_call' | 'ledger_capacity' | 'invalid_transition' | 'invalid_request_id' | 'duplicate_request_id' | 'duplicate_approval' | 'duplicate_execution' | 'duplicate_result'。
// [/maint-note:a3b645cdb21c]
export type ToolAttemptTransitionError =
  | 'duplicate_call'
  | 'ledger_capacity'
  | 'invalid_transition'
  | 'invalid_request_id'
  | 'duplicate_request_id'
  | 'duplicate_approval'
  | 'duplicate_execution'
  | 'duplicate_result'

/**
 * Run-local ownership ledger for one canonical tool-call identity.
 *
 * This is deliberately a small protocol state machine, not a general
 * workflow engine. It owns attempt/approval/result cardinality while the
 * AgentRunner continues to own model rounds and the thread transaction.
 */
export class ToolAttemptLedger {
  private readonly attempts = new Map<string, MutableToolAttemptRecord>()
  private readonly requestOwners = new Map<string, string>()
  private readonly maxAttempts: number
  private nextReceiptId = 1

  // [maint-note:39f592303fde]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 maxAttempts：类型由调用方与函数体约束；缺省表达式 1_024。
  // 写入目标：this.maxAttempts。
  // 直接调用：Number.isSafeInteger。
  // [/maint-note:39f592303fde]
  constructor(maxAttempts = 1_024) {
    this.maxAttempts = Number.isSafeInteger(maxAttempts) && maxAttempts > 0
      ? maxAttempts
      : 1_024
  }

  // [maint-note:44f02a6c4b88]
  // 维护导航：register；下列为本函数源码中的接口与路径索引。
  // 输入 call：Pick<ToolCall, 'id' | 'name'>。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：this.attempts.has(call.id)。
  // 分支条件：this.attempts.size >= this.maxAttempts。
  // 返回路径：'duplicate_call'。
  // 返回路径：'ledger_capacity'。
  // 返回路径：snapshot(record)。
  // 直接调用：this.attempts.has。
  // 直接调用：this.attempts.set。
  // 直接调用：snapshot。
  // [/maint-note:44f02a6c4b88]
  register(call: Pick<ToolCall, 'id' | 'name'>): ToolAttemptRecord | ToolAttemptTransitionError {
    if (this.attempts.has(call.id)) return 'duplicate_call'
    if (this.attempts.size >= this.maxAttempts) return 'ledger_capacity'
    const record: MutableToolAttemptRecord = {
      callId: call.id,
      toolName: call.name,
      state: 'registered',
      invoked: false,
    }
    this.attempts.set(call.id, record)
    return snapshot(record)
  }

  // [maint-note:a541e80d9170]
  // 维护导航：get；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolAttemptRecord | undefined。
  // 返回路径：record ? snapshot(record) : undefined。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:a541e80d9170]
  get(callId: string): ToolAttemptRecord | undefined {
    const record = this.attempts.get(callId)
    return record ? snapshot(record) : undefined
  }

  // [maint-note:4f85bd2db55c]
  // 维护导航：size；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.attempts.size。
  // [/maint-note:4f85bd2db55c]
  size(): number {
    return this.attempts.size
  }

  // [maint-note:d13e2cbfcee0]
  // 维护导航：remainingCapacity；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：Math.max(0, this.maxAttempts - this.attempts.size)。
  // 直接调用：Math.max。
  // [/maint-note:d13e2cbfcee0]
  remainingCapacity(): number {
    return Math.max(0, this.maxAttempts - this.attempts.size)
  }

  // [maint-note:4c37815136bb]
  // 维护导航：canRegister；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 返回契约：boolean。
  // 返回路径：Number.isSafeInteger(count) && count >= 0 && count <= this.remainingCapacity()。
  // 直接调用：Number.isSafeInteger。
  // 直接调用：this.remainingCapacity。
  // [/maint-note:4c37815136bb]
  canRegister(count: number): boolean {
    return Number.isSafeInteger(count) && count >= 0 && count <= this.remainingCapacity()
  }

  // [maint-note:9549a06538ed]
  // 维护导航：validate；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 返回路径：this.transition(callId, 'validated')。
  // 直接调用：this.transition。
  // [/maint-note:9549a06538ed]
  validate(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    return this.transition(callId, 'validated')
  }

  // [maint-note:fd94b254bceb]
  // 维护导航：requestApproval；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 requestId：string。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record。
  // 分支条件：typeof requestId !== 'string' || !requestId.trim()。
  // 分支条件：record.requestId !== undefined。
  // 分支条件：record.state !== 'validated'。
  // 分支条件：owner !== undefined && owner !== callId。
  // 返回路径：'invalid_transition'。
  // 返回路径：'invalid_request_id'。
  // 返回路径：'duplicate_approval'。
  // 返回路径：'duplicate_request_id'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.requestId。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：requestId.trim。
  // 直接调用：this.requestOwners.get。
  // 直接调用：this.requestOwners.set。
  // 直接调用：snapshot。
  // [/maint-note:fd94b254bceb]
  requestApproval(
    callId: string,
    requestId: string,
  ): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (typeof requestId !== 'string' || !requestId.trim()) return 'invalid_request_id'
    if (record.requestId !== undefined) return 'duplicate_approval'
    if (record.state !== 'validated') return 'invalid_transition'
    const owner = this.requestOwners.get(requestId)
    if (owner !== undefined && owner !== callId) return 'duplicate_request_id'
    record.requestId = requestId
    this.requestOwners.set(requestId, callId)
    record.state = 'approval_pending'
    return snapshot(record)
  }

  // [maint-note:c73d49e5cb6b]
  // 维护导航：resolveApproval；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 requestId：string。
  // 输入 approved：boolean。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record || record.requestId !== requestId。
  // 分支条件：record.state !== 'approval_pending'。
  // 返回路径：'invalid_transition'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.approvalDecision。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:c73d49e5cb6b]
  resolveApproval(
    callId: string,
    requestId: string,
    approved: boolean,
  ): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record || record.requestId !== requestId) return 'invalid_transition'
    if (record.state !== 'approval_pending') return 'invalid_transition'
    record.approvalDecision = approved ? 'approved' : 'denied'
    record.state = approved ? 'approved' : 'denied'
    return snapshot(record)
  }

  // [maint-note:e4592c81288b]
  // 维护导航：failApproval；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 requestId：string。
  // 输入 resultCode：类型由调用方与函数体约束；缺省表达式 'approval_provider_error'。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record || record.requestId !== requestId。
  // 分支条件：record.state !== 'approval_pending'。
  // 返回路径：'invalid_transition'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.resultCode。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:e4592c81288b]
  failApproval(
    callId: string,
    requestId: string,
    resultCode = 'approval_provider_error',
  ): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record || record.requestId !== requestId) return 'invalid_transition'
    if (record.state !== 'approval_pending') return 'invalid_transition'
    record.resultCode = resultCode
    record.state = 'approval_failed'
    return snapshot(record)
  }

  // [maint-note:07c6a6d1473f]
  // 维护导航：beginExecution；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record。
  // 分支条件：record.state === 'execution_started'。
  // 分支条件：record.state !== 'validated' && record.state !== 'approved'。
  // 返回路径：'invalid_transition'。
  // 返回路径：'duplicate_execution'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.invoked。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:07c6a6d1473f]
  beginExecution(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state === 'execution_started') return 'duplicate_execution'
    if (record.state !== 'validated' && record.state !== 'approved') {
      return 'invalid_transition'
    }
    record.invoked = true
    record.state = 'execution_started'
    return snapshot(record)
  }

  // [maint-note:9be6f8f295c1]
  // 维护导航：confirmResult；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 result：CanonicalToolResultPayload。
  // 输入 input：{ readonly origin: ToolResultOrigin; readonly mutating: boolean }。
  // 返回契约：ToolResultReceipt | ToolAttemptTransitionError。
  // 分支条件：typeof emitted === 'string'。
  // 分支条件：typeof settled === 'string'。
  // 返回路径：emitted。
  // 返回路径：settled。
  // 返回路径：receiptSnapshot(record.receipt)。
  // 写入目标：record.receipt。
  // 直接调用：deepFreeze。
  // 直接调用：structuredClone。
  // 直接调用：this.emitResult。
  // 直接调用：this.settle。
  // 直接调用：this.attempts.get。
  // 直接调用：receiptSnapshot。
  // [/maint-note:9be6f8f295c1]
  confirmResult(
    callId: string,
    result: CanonicalToolResultPayload,
    input: { readonly origin: ToolResultOrigin; readonly mutating: boolean },
  ): ToolResultReceipt | ToolAttemptTransitionError {
    const canonical = deepFreeze(structuredClone(result))
    const emitted = this.emitResult(callId, canonical.code)
    if (typeof emitted === 'string') return emitted
    const settled = this.settle(callId)
    if (typeof settled === 'string') return settled
    const record = this.attempts.get(callId)!
    record.receipt = {
      receiptId: this.nextReceiptId++,
      callId: record.callId,
      toolName: record.toolName,
      result: canonical,
      origin: input.origin,
      invoked: record.invoked,
      mutating: input.mutating,
      delivered: false,
    }
    return receiptSnapshot(record.receipt)
  }

  // [maint-note:be2ca68b36aa]
  // 维护导航：resultReceipt；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolResultReceipt | undefined。
  // 返回路径：receipt?.result ? receiptSnapshot(receipt) : undefined。
  // 直接调用：this.attempts.get。
  // 直接调用：receiptSnapshot。
  // [/maint-note:be2ca68b36aa]
  resultReceipt(callId: string): ToolResultReceipt | undefined {
    const receipt = this.attempts.get(callId)?.receipt
    return receipt?.result ? receiptSnapshot(receipt) : undefined
  }

  // [maint-note:8783b66a532b]
  // 维护导航：resultWasAcknowledged；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：boolean。
  // 返回路径：this.attempts.get(callId)?.receipt?.delivered === true。
  // 直接调用：this.attempts.get。
  // [/maint-note:8783b66a532b]
  resultWasAcknowledged(callId: string): boolean {
    return this.attempts.get(callId)?.receipt?.delivered === true
  }

  // [maint-note:2e21f7d2e2e1]
  // 维护导航：pendingResultReceipts；下列为本函数源码中的接口与路径索引。
  // 输入 callIds：readonly string[]（可省略）。
  // 返回契约：readonly ToolResultReceipt[]。
  // 返回路径：selected .map((record) => record!.receipt) .filter((receipt): receipt is MutableToolResultReceipt => receipt !== undefined && !receipt.delivered) .map(receiptSnapshot)。
  // 直接调用：callIds.map((callId) => this.attempts.get(callId)).filter。
  // 直接调用：callIds.map。
  // 直接调用：this.attempts.values。
  // 直接调用：selected .map((record) => record!.receipt) .filter((receipt): receipt is MutableToolResultReceipt => receipt !== undefined && !receipt.delivered) .map。
  // 直接调用：selected .map((record) => record!.receipt) .filter。
  // 直接调用：selected .map。
  // [/maint-note:2e21f7d2e2e1]
  pendingResultReceipts(callIds?: readonly string[]): readonly ToolResultReceipt[] {
    const selected = callIds
      ? callIds.map((callId) => this.attempts.get(callId)).filter((record) => record !== undefined)
      : [...this.attempts.values()]
    return selected
      .map((record) => record!.receipt)
      .filter((receipt): receipt is MutableToolResultReceipt => receipt !== undefined && !receipt.delivered)
      .map(receiptSnapshot)
  }

  // [maint-note:d7a7dabfc07a]
  // 维护导航：acknowledgeResult；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 receiptId：number。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record?.receipt || record.receipt.receiptId !== receiptId。
  // 分支条件：record.receipt.delivered。
  // 返回路径：'invalid_transition'。
  // 返回路径：'duplicate_result'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.receipt.delivered。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:d7a7dabfc07a]
  acknowledgeResult(callId: string, receiptId: number): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record?.receipt || record.receipt.receiptId !== receiptId) return 'invalid_transition'
    if (record.receipt.delivered) return 'duplicate_result'
    record.receipt.delivered = true
    // The canonical message is now owned by the Runner/Thread batch buffer;
    // do not retain a second long-lived payload in the attempt ledger.
    delete record.receipt.result
    return snapshot(record)
  }

  // [maint-note:b6191156e3f3]
  // 维护导航：emitResult；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 resultCode：string（可省略）。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record。
  // 分支条件：record.state === 'result_emitted' || record.state === 'settled'。
  // 分支条件：record.state !== 'registered' && record.state !== 'validated' && record.state !== 'denied' && record.state !== 'approval_failed' && record.state !== 'execution_started' && record.state !== 'aborted'。
  // 返回路径：'invalid_transition'。
  // 返回路径：'duplicate_result'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.resultCode。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:b6191156e3f3]
  emitResult(
    callId: string,
    resultCode?: string,
  ): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state === 'result_emitted' || record.state === 'settled') {
      return 'duplicate_result'
    }
    if (
      record.state !== 'registered' &&
      record.state !== 'validated' &&
      record.state !== 'denied' &&
      record.state !== 'approval_failed' &&
      record.state !== 'execution_started' &&
      record.state !== 'aborted'
    ) {
      return 'invalid_transition'
    }
    record.resultCode = resultCode ?? record.resultCode
    record.state = 'result_emitted'
    return snapshot(record)
  }

  // [maint-note:97e03b05ace7]
  // 维护导航：abort；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record。
  // 分支条件：record.state === 'settled' || record.state === 'result_emitted'。
  // 返回路径：'invalid_transition'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:97e03b05ace7]
  abort(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state === 'settled' || record.state === 'result_emitted') {
      return 'invalid_transition'
    }
    record.state = 'aborted'
    return snapshot(record)
  }

  // [maint-note:c4b7404d9039]
  // 维护导航：settle；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record。
  // 分支条件：record.state !== 'result_emitted' && record.state !== 'aborted'。
  // 返回路径：'invalid_transition'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:c4b7404d9039]
  settle(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state !== 'result_emitted' && record.state !== 'aborted') {
      return 'invalid_transition'
    }
    record.state = 'settled'
    return snapshot(record)
  }

  // [maint-note:4e8620fe160d]
  // 维护导航：records；下列为本函数源码中的接口与路径索引。
  // 返回契约：readonly ToolAttemptRecord[]。
  // 返回路径：[...this.attempts.values()].map(snapshot)。
  // 直接调用：[...this.attempts.values()].map。
  // 直接调用：this.attempts.values。
  // [/maint-note:4e8620fe160d]
  records(): readonly ToolAttemptRecord[] {
    return [...this.attempts.values()].map(snapshot)
  }

  // [maint-note:f0b8aa528ef3]
  // 维护导航：transition；下列为本函数源码中的接口与路径索引。
  // 输入 callId：string。
  // 输入 state：'validated'。
  // 返回契约：ToolAttemptRecord | ToolAttemptTransitionError。
  // 分支条件：!record || record.state !== 'registered'。
  // 返回路径：'invalid_transition'。
  // 返回路径：snapshot(record)。
  // 写入目标：record.state。
  // 直接调用：this.attempts.get。
  // 直接调用：snapshot。
  // [/maint-note:f0b8aa528ef3]
  private transition(
    callId: string,
    state: 'validated',
  ): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record || record.state !== 'registered') return 'invalid_transition'
    record.state = state
    return snapshot(record)
  }
}

// [maint-note:94cc3e745801]
// 数据契约导航：MutableToolAttemptRecord；字段含义以消费端校验为准。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 state：ToolAttemptState。
// 成员 requestId（可省略）：string。
// 成员 approvalDecision（可省略）：'approved' | 'denied'。
// 成员 resultCode（可省略）：string。
// 成员 invoked：boolean。
// 成员 receipt（可省略）：MutableToolResultReceipt。
// [/maint-note:94cc3e745801]
interface MutableToolAttemptRecord {
  callId: string
  toolName: string
  state: ToolAttemptState
  requestId?: string
  approvalDecision?: 'approved' | 'denied'
  resultCode?: string
  invoked: boolean
  receipt?: MutableToolResultReceipt
}

// [maint-note:01d95cebc0f7]
// 数据契约导航：MutableToolResultReceipt；字段含义以消费端校验为准。
// 成员 receiptId：number。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 result（可省略）：CanonicalToolResultPayload。
// 成员 origin：ToolResultOrigin。
// 成员 invoked：boolean。
// 成员 mutating：boolean。
// 成员 delivered：boolean。
// [/maint-note:01d95cebc0f7]
interface MutableToolResultReceipt {
  receiptId: number
  callId: string
  toolName: string
  result?: CanonicalToolResultPayload
  origin: ToolResultOrigin
  invoked: boolean
  mutating: boolean
  delivered: boolean
}

// [maint-note:69bc4dc78188]
// 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
// 输入 record：MutableToolAttemptRecord。
// 返回契约：ToolAttemptRecord。
// 返回路径：{ ...value }。
// [/maint-note:69bc4dc78188]
function snapshot(record: MutableToolAttemptRecord): ToolAttemptRecord {
  const { receipt: _receipt, ...value } = record
  return { ...value }
}

// [maint-note:aed0953f57ae]
// 维护导航：receiptSnapshot；下列为本函数源码中的接口与路径索引。
// 输入 receipt：MutableToolResultReceipt。
// 返回契约：ToolResultReceipt。
// 分支条件：!receipt.result。
// 返回路径：Object.freeze({ receiptId: receipt.receiptId, callId: receipt.callId, toolName: receipt.toolName, result: receipt.result, origin: receipt.origin, invoked: receipt.invoked, mutating: receipt.mutating, delivered: receipt.delivered, })。
// 显式抛出：new Error('The canonical tool result payload has already been released')。
// 直接调用：Object.freeze。
// [/maint-note:aed0953f57ae]
function receiptSnapshot(receipt: MutableToolResultReceipt): ToolResultReceipt {
  if (!receipt.result) throw new Error('The canonical tool result payload has already been released')
  return Object.freeze({
    receiptId: receipt.receiptId,
    callId: receipt.callId,
    toolName: receipt.toolName,
    result: receipt.result,
    origin: receipt.origin,
    invoked: receipt.invoked,
    mutating: receipt.mutating,
    delivered: receipt.delivered,
  })
}

// [maint-note:dbf6ff7f30d8]
// 维护导航：deepFreeze；下列为本函数源码中的接口与路径索引。
// 输入 value：T。
// 返回契约：T。
// 分支条件：!value || typeof value !== 'object' || Object.isFrozen(value)。
// 遍历边界：const child ← Object.values(value as Record<string, unknown>)。
// 返回路径：value。
// 返回路径：Object.freeze(value)。
// 直接调用：Object.isFrozen。
// 直接调用：Object.values。
// 直接调用：deepFreeze。
// 直接调用：Object.freeze。
// [/maint-note:dbf6ff7f30d8]
function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
  return Object.freeze(value)
}
