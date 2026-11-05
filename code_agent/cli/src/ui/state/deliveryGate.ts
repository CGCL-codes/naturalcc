import { createHash } from 'node:crypto'
import type { AgentEvent } from '../../agent/types/event.js'

/**
 * delivery envelope: the application/UI boundary wraps every AgentEvent
 * with its session generation, owning run, and a per-run strictly increasing
 * sequence. The envelope belongs to application/UI delivery only — it never
 * modifies the provider-neutral AgentEvent, AgentThread, or model history.
 */
export interface DeliveryEnvelope {
  sessionGeneration: number
  runId: number
  sequence: number
  event: AgentEvent
}

export type DeliveryDropReason =
  | 'stale-generation'
  | 'stale-run'
  | 'duplicate-sequence'
  | 'sequence-regression'
  | 'sequence-gap'
  | 'conflicting-payload'

export interface DeliveryDiagnostic {
  bounded: true
  reason: DeliveryDropReason
  sessionGeneration: number
  runId: number
  sequence: number
}

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

export function createDeliveryGateState(sessionGeneration = 1): DeliveryGateState {
  return { sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: [] }
}

/**
 * Establish the owner before feeding a stamped envelope to the pure gate.
 * An idle gate is intentionally fail-closed: sequence one is not proof that
 * the producer belongs to this session or is the current run.
 */
export function beginDeliveryGateRun(state: DeliveryGateState, runId: number): DeliveryGateState {
  return {
    ...state,
    activeRunId: runId,
    closedRuns: new Set([...state.closedRuns].filter((closedRunId) => closedRunId !== runId)),
    lastByRun: new Map(state.lastByRun).set(runId, { sequence: 0, digest: '' }),
  }
}

function payloadDigest(event: AgentEvent): string {
  return createHash('sha256').update(JSON.stringify(event)).digest('hex').slice(0, 16)
}

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
export function resetDeliveryGate(state: DeliveryGateState, sessionGeneration: number): DeliveryGateState {
  return { sessionGeneration, activeRunId: null, closedRuns: new Set(), lastByRun: new Map(), diagnostics: state.diagnostics }
}

/**
 * Imperative wrapper used by the executor boundary: one gate instance per
 * REPL, reset on /clear, applied once per delivered event.
 */
export class DeliveryGate {
  private state: DeliveryGateState

  constructor(sessionGeneration = 1) {
    this.state = createDeliveryGateState(sessionGeneration)
  }

  get sessionGeneration(): number {
    return this.state.sessionGeneration
  }

  get activeRunId(): number | null {
    return this.state.activeRunId
  }

  get diagnostics(): readonly DeliveryDiagnostic[] {
    return this.state.diagnostics
  }

  /** The sequence to stamp on the next envelope of the given run. */
  nextSequence(runId: number): number {
    const last = this.state.lastByRun.get(runId)
    return last === undefined ? 1 : last.sequence + 1
  }

  /** Starts the single delivery owner for a run before its first event. */
  beginRun(runId: number): void {
    this.state = beginDeliveryGateRun(this.state, runId)
  }

  /** Closes the owner so late events are classified as stale-run. */
  endRun(runId: number): void {
    if (this.state.activeRunId !== runId) return
    this.state = { ...this.state, activeRunId: null, closedRuns: new Set(this.state.closedRuns).add(runId) }
  }

  apply(envelope: Omit<DeliveryEnvelope, 'sessionGeneration'> & { sessionGeneration?: number }): boolean {
    const complete: DeliveryEnvelope = {
      ...envelope,
      sessionGeneration: envelope.sessionGeneration ?? this.state.sessionGeneration,
    }
    const result = applyDeliveryEnvelope(this.state, complete)
    this.state = result.state
    return result.accepted
  }

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

  constructor(sessionGeneration = 1) {
    this.sessionGeneration = sessionGeneration
  }

  beginRun(runId: number): void {
    this.activeRunId = runId
    this.nextByRun.set(runId, 1)
  }

  envelope(runId: number, event: AgentEvent): DeliveryEnvelope {
    const sequence = this.nextByRun.get(runId) ?? 1
    this.nextByRun.set(runId, sequence + 1)
    return Object.freeze({ sessionGeneration: this.sessionGeneration, runId, sequence, event })
  }

  endRun(runId: number): void {
    if (this.activeRunId === runId) this.activeRunId = null
  }

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
  constructor(
    private readonly gate: DeliveryGate,
    private readonly owner: DeliveryEnvelopeOwner,
  ) {}

  beginRun(runId: number): void {
    this.owner.beginRun(runId)
    this.gate.beginRun(runId)
  }

  endRun(runId: number): void {
    this.owner.endRun(runId)
    this.gate.endRun(runId)
  }

  dispatch(runId: number, event: AgentEvent, consumer: (event: AgentEvent, envelope: DeliveryEnvelope) => void): DeliveryEnvelope | null {
    const envelope = this.owner.envelope(runId, event)
    return this.accept(envelope, consumer) ? envelope : null
  }

  accept(envelope: DeliveryEnvelope, consumer: (event: AgentEvent, envelope: DeliveryEnvelope) => void): boolean {
    if (!this.gate.apply(envelope)) return false
    consumer(envelope.event, envelope)
    return true
  }
}
