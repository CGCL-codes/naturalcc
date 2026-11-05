import type { ToolCall, ToolDisplayFacts } from '../tools/types.js'

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

export interface CanonicalToolResultPayload {
  readonly content: string
  readonly isError: boolean
  readonly code?: string
  readonly display?: ToolDisplayFacts
}

export type ToolResultOrigin = 'executor' | 'validation' | 'approval' | 'framework_close'

/** Run-local recovery receipt; canonical payload is released after append. */
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

  constructor(maxAttempts = 1_024) {
    this.maxAttempts = Number.isSafeInteger(maxAttempts) && maxAttempts > 0
      ? maxAttempts
      : 1_024
  }

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

  get(callId: string): ToolAttemptRecord | undefined {
    const record = this.attempts.get(callId)
    return record ? snapshot(record) : undefined
  }

  size(): number {
    return this.attempts.size
  }

  remainingCapacity(): number {
    return Math.max(0, this.maxAttempts - this.attempts.size)
  }

  canRegister(count: number): boolean {
    return Number.isSafeInteger(count) && count >= 0 && count <= this.remainingCapacity()
  }

  validate(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    return this.transition(callId, 'validated')
  }

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

  resultReceipt(callId: string): ToolResultReceipt | undefined {
    const receipt = this.attempts.get(callId)?.receipt
    return receipt?.result ? receiptSnapshot(receipt) : undefined
  }

  resultWasAcknowledged(callId: string): boolean {
    return this.attempts.get(callId)?.receipt?.delivered === true
  }

  pendingResultReceipts(callIds?: readonly string[]): readonly ToolResultReceipt[] {
    const selected = callIds
      ? callIds.map((callId) => this.attempts.get(callId)).filter((record) => record !== undefined)
      : [...this.attempts.values()]
    return selected
      .map((record) => record!.receipt)
      .filter((receipt): receipt is MutableToolResultReceipt => receipt !== undefined && !receipt.delivered)
      .map(receiptSnapshot)
  }

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

  abort(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state === 'settled' || record.state === 'result_emitted') {
      return 'invalid_transition'
    }
    record.state = 'aborted'
    return snapshot(record)
  }

  settle(callId: string): ToolAttemptRecord | ToolAttemptTransitionError {
    const record = this.attempts.get(callId)
    if (!record) return 'invalid_transition'
    if (record.state !== 'result_emitted' && record.state !== 'aborted') {
      return 'invalid_transition'
    }
    record.state = 'settled'
    return snapshot(record)
  }

  records(): readonly ToolAttemptRecord[] {
    return [...this.attempts.values()].map(snapshot)
  }

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

function snapshot(record: MutableToolAttemptRecord): ToolAttemptRecord {
  const { receipt: _receipt, ...value } = record
  return { ...value }
}

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

function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
  return Object.freeze(value)
}
