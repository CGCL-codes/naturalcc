import type { AgentEvent } from '../../agent/types/event.js'

export type SessionGeneration = string | number
export type MonotonicClock = () => number

export interface TurnSeparatorDescriptor {
  readonly id: string
  readonly type: 'turn_separator'
  readonly runId: number
}

export interface SessionFooterDescriptor {
  readonly id: string
  readonly type: 'session_footer'
  readonly sessionGeneration: SessionGeneration
  readonly durationMs: number
  readonly completedTurnCount: number
  readonly finalized: boolean
}

export interface SessionTimingSnapshot {
  readonly sessionGeneration: SessionGeneration
  readonly accumulatedActiveMs: number
  readonly completedTurnCount: number
  readonly footerId: string
  readonly footerFinalized: boolean
  readonly activeRunId: number | null
  readonly currentTurnMs: number
  readonly approvalPaused: boolean
  readonly live: boolean
  readonly separators: readonly TurnSeparatorDescriptor[]
  readonly footer: SessionFooterDescriptor | null
}

export interface FinishedTurnTiming {
  readonly runId: number
  readonly turnDurationMs: number
  readonly sessionDurationMs: number
  readonly completedTurnCount: number
  readonly separator: TurnSeparatorDescriptor
  readonly footer: SessionFooterDescriptor
}

interface ActiveRun {
  runId: number
  accumulatedMs: number
  segmentStartedAt: number | null
  pausedRequestId: string | null
  finalized: boolean
}

const MAX_SAFE_DURATION_MS = Number.MAX_SAFE_INTEGER
const monotonicOrigin = typeof process !== 'undefined' && typeof process.hrtime?.bigint === 'function'
  ? process.hrtime.bigint()
  : 0n

/**
 * UI-only timing state. It deliberately stores no rendered strings and never
 * crosses the AgentThread/ModelClient boundary.
 */
export class SessionTimingController {
  private generation: SessionGeneration
  private accumulatedMs = 0
  private completedTurns = 0
  private footerFinalized = false
  private footerSerial = 0
  private footerId: string
  private activeRun: ActiveRun | null = null
  private readonly separatorList: TurnSeparatorDescriptor[] = []

  constructor(private readonly clock: MonotonicClock = defaultMonotonicClock, generation: SessionGeneration = 'initial') {
    this.generation = generation
    this.footerId = this.makeFooterId()
  }

  beginSession(generation: SessionGeneration = this.generation): void {
    this.generation = generation
    this.accumulatedMs = 0
    this.completedTurns = 0
    this.footerFinalized = false
    this.activeRun = null
    this.separatorList.length = 0
    this.footerId = this.makeFooterId()
  }

  clearSession(generation: SessionGeneration = this.generation): void {
    this.beginSession(generation)
  }

  beginRun(runId: number, at = this.clock()): boolean {
    if (this.activeRun || this.footerFinalized) return false
    this.activeRun = {
      runId,
      accumulatedMs: 0,
      segmentStartedAt: normalizeTimestamp(at),
      pausedRequestId: null,
      finalized: false,
    }
    return true
  }

  pauseApproval(runId: number, requestId: string, at = this.clock()): boolean {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized || run.pausedRequestId !== null) return false
    run.accumulatedMs = addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))
    run.segmentStartedAt = null
    run.pausedRequestId = requestId
    return true
  }

  resumeApproval(runId: number, requestId: string, at = this.clock()): boolean {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized || run.pausedRequestId !== requestId) return false
    run.pausedRequestId = null
    run.segmentStartedAt = normalizeTimestamp(at)
    return true
  }

  handleEvent(runId: number, event: AgentEvent, at = this.clock()): boolean {
    if (event.type === 'approval_required') return this.pauseApproval(runId, event.requestId, at)
    if (event.type === 'approval_resolved') return this.resumeApproval(runId, event.requestId, at)
    return false
  }

  currentTurnDuration(runId: number, at = this.clock()): number {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized) return 0
    return addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))
  }

  finishRun(runId: number, _outcome?: string, at = this.clock()): FinishedTurnTiming | null {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized) return null

    run.accumulatedMs = addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))
    run.segmentStartedAt = null
    run.pausedRequestId = null
    run.finalized = true

    const turnDurationMs = normalizeDuration(run.accumulatedMs)
    this.accumulatedMs = addDuration(this.accumulatedMs, turnDurationMs)
    this.completedTurns += 1
    this.activeRun = null

    const separator: TurnSeparatorDescriptor = Object.freeze({
      id: `turn-separator-${runId}`,
      type: 'turn_separator',
      runId,
    })
    this.separatorList.push(separator)

    return {
      runId,
      turnDurationMs,
      sessionDurationMs: this.accumulatedMs,
      completedTurnCount: this.completedTurns,
      separator,
      footer: this.makeFooter(),
    }
  }

  finalizeSession(at = this.clock()): SessionFooterDescriptor | null {
    // A session close is not a terminal event. Do not turn an active run into
    // a completed turn merely because the UI is being finalized.
    if (this.activeRun) return null
    if (this.completedTurns === 0) return null
    this.footerFinalized = true
    return this.makeFooter()
  }

  snapshot(at = this.clock()): SessionTimingSnapshot {
    const activeRunId = this.activeRun?.runId ?? null
    const currentTurnMs = activeRunId === null ? 0 : this.currentTurnDuration(activeRunId, at)
    const approvalPaused = this.activeRun?.pausedRequestId !== null && this.activeRun?.pausedRequestId !== undefined
    const footer = activeRunId === null && this.completedTurns > 0 ? this.makeFooter() : null
    return {
      sessionGeneration: this.generation,
      accumulatedActiveMs: this.accumulatedMs,
      completedTurnCount: this.completedTurns,
      footerId: this.footerId,
      footerFinalized: this.footerFinalized,
      activeRunId,
      currentTurnMs,
      approvalPaused,
      live: activeRunId !== null && !approvalPaused,
      separators: this.separatorList.slice(),
      footer,
    }
  }

  private makeFooter(): SessionFooterDescriptor {
    return Object.freeze({
      id: this.footerId,
      type: 'session_footer',
      sessionGeneration: this.generation,
      durationMs: this.accumulatedMs,
      completedTurnCount: this.completedTurns,
      finalized: this.footerFinalized,
    })
  }

  private makeFooterId(): string {
    this.footerSerial += 1
    return `session-footer-${this.footerSerial}`
  }
}

export function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.floor(normalizeDuration(milliseconds) / 1_000)
  if (totalSeconds < 60) return `${totalSeconds} s`

  const seconds = totalSeconds % 60
  const totalMinutes = Math.floor(totalSeconds / 60)
  if (totalMinutes < 60) return `${totalMinutes} m ${seconds} s`

  const minutes = totalMinutes % 60
  const hours = Math.floor(totalMinutes / 60)
  return `${hours} h ${minutes} m ${seconds} s`
}

export function normalizeDuration(milliseconds: number): number {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return 0
  return Math.min(Math.floor(milliseconds), MAX_SAFE_DURATION_MS)
}

function activeDelta(startedAt: number | null, now: number): number {
  if (startedAt === null) return 0
  return normalizeDuration(normalizeTimestamp(now) - startedAt)
}

function addDuration(left: number, right: number): number {
  return Math.min(MAX_SAFE_DURATION_MS, normalizeDuration(left) + normalizeDuration(right))
}

function normalizeTimestamp(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0
  return Math.min(value, MAX_SAFE_DURATION_MS)
}

function defaultMonotonicClock(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') return performance.now()
  if (monotonicOrigin !== 0n && typeof process !== 'undefined' && typeof process.hrtime?.bigint === 'function') {
    return Number(process.hrtime.bigint() - monotonicOrigin) / 1_000_000
  }
  return 0
}
