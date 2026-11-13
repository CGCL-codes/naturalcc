import { createHash } from 'node:crypto'
import type { AgentEvent } from '../../agent/types/event.js'

/**
 * delivery envelope: the application/UI boundary wraps every AgentEvent
 * with its session generation, owning run, and a per-run strictly increasing
 * sequence. The envelope belongs to application/UI delivery only — it never
 * modifies the provider-neutral AgentEvent, AgentThread, or model history.
 */
// [maint-note:fe55f3850d6d]
// 数据契约导航：DeliveryEnvelope；字段含义以消费端校验为准。
// 成员 sessionGeneration：number。
// 成员 runId：number。
// 成员 sequence：number。
// 成员 event：AgentEvent。
// [/maint-note:fe55f3850d6d]
export interface DeliveryEnvelope {
  sessionGeneration: number
  runId: number
  sequence: number
  event: AgentEvent
}

// [maint-note:5f04234ea1ff]
// 数据契约导航：DeliveryDropReason；字段含义以消费端校验为准。
// 类型组合：| 'stale-generation' | 'stale-run' | 'duplicate-sequence' | 'sequence-regression' | 'sequence-gap' | 'conflicting-payload'。
// [/maint-note:5f04234ea1ff]
export type DeliveryDropReason =
  | 'stale-generation'
  | 'stale-run'
  | 'duplicate-sequence'
  | 'sequence-regression'
  | 'sequence-gap'
  | 'conflicting-payload'

// [maint-note:4397c7086e98]
// 数据契约导航：DeliveryDiagnostic；字段含义以消费端校验为准。
// 成员 bounded：true。
// 成员 reason：DeliveryDropReason。
// 成员 sessionGeneration：number。
// 成员 runId：number。
// 成员 sequence：number。
// [/maint-note:4397c7086e98]
export interface DeliveryDiagnostic {
  bounded: true
  reason: DeliveryDropReason
  sessionGeneration: number
  runId: number
  sequence: number
}

// [maint-note:509d0ea7a7a9]
// 数据契约导航：DeliveryGateState；字段含义以消费端校验为准。
// 成员 sessionGeneration：number。
// 成员 activeRunId：number | null。
// 成员 closedRuns：Set<number>。
// 成员 lastByRun：Map<number, { sequence: number; digest: string }>。
// 成员 diagnostics：DeliveryDiagnostic[]。
// [/maint-note:509d0ea7a7a9]
export interface DeliveryGateState {
  sessionGeneration: number
  /** The only run currently allowed to reach UI owners. */
  activeRunId: number | null
  /** Run IDs that have closed; late events cannot reopen their owners. */
  closedRuns: Set<number>
  /** Last accepted sequence per run, with the digest of its payload. */
  lastByRun: Map<number, { sequence: number; digest: string }>
  /** Bounded ring of diagnostics: only the most recent are retained. */
  diagnostics: DeliveryDiagnostic[]
}

export const MAX_DELIVERY_DIAGNOSTICS = 16

// [maint-note:7b011da59508]
// 维护导航：createDeliveryGateState；下列为本函数源码中的接口与路径索引。
// 输入 sessionGeneration：类型由调用方与函数体约束；缺省表达式 1。
// 返回契约：DeliveryGateState。
// 返回路径：{ sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: [] }。
// [/maint-note:7b011da59508]
export function createDeliveryGateState(sessionGeneration = 1): DeliveryGateState {
  return { sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: [] }
}

/**
 * Establish the owner before feeding a stamped envelope to the pure gate.
 * An idle gate is intentionally fail-closed: sequence one is not proof that
 * the producer belongs to this session or is the current run.
 */
// [maint-note:7147fce7412f]
// 维护导航：beginDeliveryGateRun；下列为本函数源码中的接口与路径索引。
// 输入 state：DeliveryGateState。
// 输入 runId：number。
// 返回契约：DeliveryGateState。
// 返回路径：{ ...state, activeRunId: runId, closedRuns: new Set([...state.closedRuns].filter((closedRunId) => closedRunId !== runId)), lastByRun: new Map(state.lastByRun).set(runId, { sequence: 0, digest: '' }), }。
// 直接调用：[...state.closedRuns].filter。
// 直接调用：new Map(state.lastByRun).set。
// [/maint-note:7147fce7412f]
export function beginDeliveryGateRun(state: DeliveryGateState, runId: number): DeliveryGateState {
  return {
    ...state,
    activeRunId: runId,
    closedRuns: new Set([...state.closedRuns].filter((closedRunId) => closedRunId !== runId)),
    lastByRun: new Map(state.lastByRun).set(runId, { sequence: 0, digest: '' }),
  }
}

// [maint-note:a0170afdb0fc]
// 维护导航：payloadDigest；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：string。
// 返回路径：createHash('sha256').update(JSON.stringify(event)).digest('hex').slice(0, 16)。
// 直接调用：createHash('sha256').update(JSON.stringify(event)).digest('hex').slice。
// 直接调用：createHash('sha256').update(JSON.stringify(event)).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:a0170afdb0fc]
function payloadDigest(event: AgentEvent): string {
  return createHash('sha256').update(JSON.stringify(event)).digest('hex').slice(0, 16)
}

// [maint-note:8e57f2bcee43]
// 维护导航：recordDiagnostic；下列为本函数源码中的接口与路径索引。
// 输入 state：DeliveryGateState。
// 输入 reason：DeliveryDropReason。
// 输入 envelope：DeliveryEnvelope。
// 返回契约：DeliveryGateState。
// 返回路径：{ ...state, diagnostics: diagnostics.slice(-MAX_DELIVERY_DIAGNOSTICS) }。
// 直接调用：diagnostics.slice。
// [/maint-note:8e57f2bcee43]
function recordDiagnostic(state: DeliveryGateState, reason: DeliveryDropReason, envelope: DeliveryEnvelope): DeliveryGateState {
  const diagnostics = [
    ...state.diagnostics,
    { bounded: true as const, reason, sessionGeneration: envelope.sessionGeneration, runId: envelope.runId, sequence: envelope.sequence },
  ]
  return { ...state, diagnostics: diagnostics.slice(-MAX_DELIVERY_DIAGNOSTICS) }
}

/**
 * Pure gate application. Accepted envelopes update the per-run sequence
 * watermark; rejected ones are classified. A repeated sequence with a
 * DIFFERENT payload is recorded as a bounded internal diagnostic rather than
 * silently taking either payload.
 */
// [maint-note:288d4f7597b3]
// 维护导航：applyDeliveryEnvelope；下列为本函数源码中的接口与路径索引。
// 输入 state：DeliveryGateState。
// 输入 envelope：DeliveryEnvelope。
// 返回契约：{ state: DeliveryGateState; accepted: boolean; reason?: DeliveryDropReason }。
// 分支条件：envelope.sessionGeneration !== state.sessionGeneration。
// 分支条件：state.activeRunId === null || state.closedRuns.has(envelope.runId) || envelope.runId !== state.activeRunId。
// 分支条件：last !== undefined。
// 分支条件：envelope.sequence === last.sequence。
// 分支条件：payloadDigest(envelope.event) === last.digest。
// 分支条件：envelope.sequence < last.sequence。
// 分支条件：envelope.sequence > last.sequence + 1。
// 分支条件：envelope.sequence !== 1。
// 返回路径：{ state: recordDiagnostic(state, 'stale-generation', envelope), accepted: false, reason: 'stale-generation' }。
// 返回路径：{ state: recordDiagnostic(state, 'stale-run', envelope), accepted: false, reason: 'stale-run' }。
// 返回路径：{ state, accepted: false, reason: 'duplicate-sequence' }。
// 返回路径：{ state: recordDiagnostic(state, 'conflicting-payload', envelope), accepted: false, reason: 'conflicting-payload' }。
// 返回路径：{ state: recordDiagnostic(state, 'sequence-regression', envelope), accepted: false, reason: 'sequence-regression' }。
// 返回路径：{ state: recordDiagnostic(state, 'sequence-gap', envelope), accepted: false, reason: 'sequence-gap' }。
// 返回路径：{ state: next, accepted: true }。
// 直接调用：recordDiagnostic。
// 直接调用：state.closedRuns.has。
// 直接调用：state.lastByRun.get。
// 直接调用：payloadDigest。
// 直接调用：new Map(state.lastByRun).set。
// [/maint-note:288d4f7597b3]
export function applyDeliveryEnvelope(state: DeliveryGateState, envelope: DeliveryEnvelope): { state: DeliveryGateState; accepted: boolean; reason?: DeliveryDropReason } {
  if (envelope.sessionGeneration !== state.sessionGeneration) {
    return { state: recordDiagnostic(state, 'stale-generation', envelope), accepted: false, reason: 'stale-generation' }
  }
  if (state.activeRunId === null || state.closedRuns.has(envelope.runId) || envelope.runId !== state.activeRunId) {
    return { state: recordDiagnostic(state, 'stale-run', envelope), accepted: false, reason: 'stale-run' }
  }
  const last = state.lastByRun.get(envelope.runId)
  if (last !== undefined) {
    if (envelope.sequence === last.sequence) {
      if (payloadDigest(envelope.event) === last.digest) {
        // Exact replay of an already-applied envelope: idempotent ignore.
        return { state, accepted: false, reason: 'duplicate-sequence' }
      }
      return { state: recordDiagnostic(state, 'conflicting-payload', envelope), accepted: false, reason: 'conflicting-payload' }
    }
    if (envelope.sequence < last.sequence) {
      return { state: recordDiagnostic(state, 'sequence-regression', envelope), accepted: false, reason: 'sequence-regression' }
    }
    if (envelope.sequence > last.sequence + 1) {
      return { state: recordDiagnostic(state, 'sequence-gap', envelope), accepted: false, reason: 'sequence-gap' }
    }
  } else if (envelope.sequence !== 1) {
    return { state: recordDiagnostic(state, 'sequence-gap', envelope), accepted: false, reason: 'sequence-gap' }
  }
  const next: DeliveryGateState = {
    ...state,
    lastByRun: new Map(state.lastByRun).set(envelope.runId, { sequence: envelope.sequence, digest: payloadDigest(envelope.event) }),
  }
  return { state: next, accepted: true }
}

/** A new session generation (e.g. /clear) invalidates every prior envelope. */
// [maint-note:f755967e36ac]
// 维护导航：resetDeliveryGate；下列为本函数源码中的接口与路径索引。
// 输入 state：DeliveryGateState。
// 输入 sessionGeneration：number。
// 返回契约：DeliveryGateState。
// 返回路径：{ sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: state.diagnostics }。
// [/maint-note:f755967e36ac]
export function resetDeliveryGate(state: DeliveryGateState, sessionGeneration: number): DeliveryGateState {
  return { sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: state.diagnostics }
}

/**
 * Imperative wrapper used by the executor boundary: one gate instance per
 * REPL, reset on /clear, applied once per delivered event.
 */
export class DeliveryGate {
  private state: DeliveryGateState

  // [maint-note:512daa819401]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 sessionGeneration：类型由调用方与函数体约束；缺省表达式 1。
  // 写入目标：this.state。
  // 直接调用：createDeliveryGateState。
  // [/maint-note:512daa819401]
  constructor(sessionGeneration = 1) {
    this.state = createDeliveryGateState(sessionGeneration)
  }

  // [maint-note:97f1bc44c107]
  // 维护导航：sessionGeneration；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.state.sessionGeneration。
  // [/maint-note:97f1bc44c107]
  get sessionGeneration(): number {
    return this.state.sessionGeneration
  }

  // [maint-note:fd98a712dd48]
  // 维护导航：activeRunId；下列为本函数源码中的接口与路径索引。
  // 返回契约：number | null。
  // 返回路径：this.state.activeRunId。
  // [/maint-note:fd98a712dd48]
  get activeRunId(): number | null {
    return this.state.activeRunId
  }

  // [maint-note:d943151d4226]
  // 维护导航：diagnostics；下列为本函数源码中的接口与路径索引。
  // 返回契约：readonly DeliveryDiagnostic[]。
  // 返回路径：this.state.diagnostics。
  // [/maint-note:d943151d4226]
  get diagnostics(): readonly DeliveryDiagnostic[] {
    return this.state.diagnostics
  }

  /** The sequence to stamp on the next envelope of the given run. */
  // [maint-note:1294a8048d6c]
  // 维护导航：nextSequence；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：number。
  // 返回路径：last === undefined ? 1 : last.sequence + 1。
  // 直接调用：this.state.lastByRun.get。
  // [/maint-note:1294a8048d6c]
  nextSequence(runId: number): number {
    const last = this.state.lastByRun.get(runId)
    return last === undefined ? 1 : last.sequence + 1
  }

  /** Starts the single delivery owner for a run before its first event. */
  // [maint-note:1382930fe456]
  // 维护导航：beginRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 写入目标：this.state。
  // 直接调用：beginDeliveryGateRun。
  // [/maint-note:1382930fe456]
  beginRun(runId: number): void {
    this.state = beginDeliveryGateRun(this.state, runId)
  }

  /** Closes the owner so late events are classified as stale-run. */
  // [maint-note:c11e96bd3b53]
  // 维护导航：endRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 分支条件：this.state.activeRunId !== runId。
  // 返回路径：无返回值。
  // 写入目标：this.state。
  // 直接调用：new Set(this.state.closedRuns).add。
  // [/maint-note:c11e96bd3b53]
  endRun(runId: number): void {
    if (this.state.activeRunId !== runId) return
    this.state = { ...this.state, activeRunId: null, closedRuns: new Set(this.state.closedRuns).add(runId) }
  }

  // [maint-note:d573c850ce38]
  // 维护导航：apply；下列为本函数源码中的接口与路径索引。
  // 输入 envelope：Omit<DeliveryEnvelope, 'sessionGeneration'> & { sessionGeneration?: number }。
  // 返回契约：boolean。
  // 返回路径：result.accepted。
  // 写入目标：this.state。
  // 直接调用：applyDeliveryEnvelope。
  // [/maint-note:d573c850ce38]
  apply(envelope: Omit<DeliveryEnvelope, 'sessionGeneration'> & { sessionGeneration?: number }): boolean {
    const complete: DeliveryEnvelope = {
      ...envelope,
      sessionGeneration: envelope.sessionGeneration ?? this.state.sessionGeneration,
    }
    const result = applyDeliveryEnvelope(this.state, complete)
    this.state = result.state
    return result.accepted
  }

  // [maint-note:ea7f535e01ca]
  // 维护导航：reset；下列为本函数源码中的接口与路径索引。
  // 输入 sessionGeneration：number。
  // 返回契约：void。
  // 写入目标：this.state。
  // 直接调用：resetDeliveryGate。
  // [/maint-note:ea7f535e01ca]
  reset(sessionGeneration: number): void {
    this.state = resetDeliveryGate(this.state, sessionGeneration)
  }
}

/**
 * The producer side of the boundary. It is deliberately separate from
 * DeliveryGate: the producer creates one immutable envelope per event and
 * the gate only validates/replays it. This prevents a consumer from
 * manufacturing a fresh "next" sequence while handling a duplicate event.
 */
export class DeliveryEnvelopeOwner {
  private sessionGeneration: number
  private activeRunId: number | null = null
  private readonly nextByRun = new Map<number, number>()

  // [maint-note:2ef7176bd904]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 sessionGeneration：类型由调用方与函数体约束；缺省表达式 1。
  // 写入目标：this.sessionGeneration。
  // [/maint-note:2ef7176bd904]
  constructor(sessionGeneration = 1) {
    this.sessionGeneration = sessionGeneration
  }

  // [maint-note:ba933e0caec3]
  // 维护导航：beginRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 写入目标：this.activeRunId。
  // 直接调用：this.nextByRun.set。
  // [/maint-note:ba933e0caec3]
  beginRun(runId: number): void {
    this.activeRunId = runId
    this.nextByRun.set(runId, 1)
  }

  // [maint-note:abb678f1b193]
  // 维护导航：envelope；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 event：AgentEvent。
  // 返回契约：DeliveryEnvelope。
  // 返回路径：Object.freeze({ sessionGeneration: this.sessionGeneration, runId, sequence, event })。
  // 直接调用：this.nextByRun.get。
  // 直接调用：this.nextByRun.set。
  // 直接调用：Object.freeze。
  // [/maint-note:abb678f1b193]
  envelope(runId: number, event: AgentEvent): DeliveryEnvelope {
    const sequence = this.nextByRun.get(runId) ?? 1
    this.nextByRun.set(runId, sequence + 1)
    return Object.freeze({ sessionGeneration: this.sessionGeneration, runId, sequence, event })
  }

  // [maint-note:2126e623714b]
  // 维护导航：endRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 分支条件：this.activeRunId === runId。
  // 写入目标：this.activeRunId。
  // [/maint-note:2126e623714b]
  endRun(runId: number): void {
    if (this.activeRunId === runId) this.activeRunId = null
  }

  // [maint-note:09d4bf051258]
  // 维护导航：reset；下列为本函数源码中的接口与路径索引。
  // 输入 sessionGeneration：number。
  // 返回契约：void。
  // 写入目标：this.sessionGeneration。
  // 写入目标：this.activeRunId。
  // 直接调用：this.nextByRun.clear。
  // [/maint-note:09d4bf051258]
  reset(sessionGeneration: number): void {
    this.sessionGeneration = sessionGeneration
    this.activeRunId = null
    this.nextByRun.clear()
  }
}

/**
 * The single application delivery boundary used by the executor. Production
 * events use dispatch(), while replay tooling can feed an already stamped
 * envelope through accept() without manufacturing a new sequence.
 */
export class DeliveryDispatcher {
  // [maint-note:01fd5230c585]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 gate：DeliveryGate。
  // 输入 owner：DeliveryEnvelopeOwner。
  // [/maint-note:01fd5230c585]
  constructor(
    private readonly gate: DeliveryGate,
    private readonly owner: DeliveryEnvelopeOwner,
  ) {}

  // [maint-note:efccdc913626]
  // 维护导航：beginRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 直接调用：this.owner.beginRun。
  // 直接调用：this.gate.beginRun。
  // [/maint-note:efccdc913626]
  beginRun(runId: number): void {
    this.owner.beginRun(runId)
    this.gate.beginRun(runId)
  }

  // [maint-note:1da6a46c4589]
  // 维护导航：endRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 返回契约：void。
  // 直接调用：this.owner.endRun。
  // 直接调用：this.gate.endRun。
  // [/maint-note:1da6a46c4589]
  endRun(runId: number): void {
    this.owner.endRun(runId)
    this.gate.endRun(runId)
  }

  // [maint-note:bb0ee9584edb]
  // 维护导航：dispatch；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 event：AgentEvent。
  // 输入 consumer：(event: AgentEvent, envelope: DeliveryEnvelope) => void。
  // 返回契约：DeliveryEnvelope | null。
  // 返回路径：this.accept(envelope, consumer) ? envelope : null。
  // 直接调用：this.owner.envelope。
  // 直接调用：this.accept。
  // [/maint-note:bb0ee9584edb]
  dispatch(runId: number, event: AgentEvent, consumer: (event: AgentEvent, envelope: DeliveryEnvelope) => void): DeliveryEnvelope | null {
    const envelope = this.owner.envelope(runId, event)
    return this.accept(envelope, consumer) ? envelope : null
  }

  // [maint-note:e80b1f8d77bb]
  // 维护导航：accept；下列为本函数源码中的接口与路径索引。
  // 输入 envelope：DeliveryEnvelope。
  // 输入 consumer：(event: AgentEvent, envelope: DeliveryEnvelope) => void。
  // 返回契约：boolean。
  // 分支条件：!this.gate.apply(envelope)。
  // 返回路径：false。
  // 返回路径：true。
  // 直接调用：this.gate.apply。
  // 直接调用：consumer。
  // [/maint-note:e80b1f8d77bb]
  accept(envelope: DeliveryEnvelope, consumer: (event: AgentEvent, envelope: DeliveryEnvelope) => void): boolean {
    if (!this.gate.apply(envelope)) return false
    consumer(envelope.event, envelope)
    return true
  }
}
