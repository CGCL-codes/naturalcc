import type { AgentEvent } from '../../agent/types/event.js'

// [maint-note:c66204179de2]
// 数据契约导航：SessionGeneration；字段含义以消费端校验为准。
// 类型组合：string | number。
// [/maint-note:c66204179de2]
export type SessionGeneration = string | number
// [maint-note:675ee1d4f53f]
// 数据契约导航：MonotonicClock；字段含义以消费端校验为准。
// 类型组合：() => number。
// [/maint-note:675ee1d4f53f]
export type MonotonicClock = () => number

// [maint-note:83c8428f497b]
// 数据契约导航：TurnSeparatorDescriptor；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 type：'turn_separator'。
// 成员 runId：number。
// [/maint-note:83c8428f497b]
export interface TurnSeparatorDescriptor {
  readonly id: string
  readonly type: 'turn_separator'
  readonly runId: number
}

// [maint-note:39fa0820e2b0]
// 数据契约导航：SessionFooterDescriptor；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 type：'session_footer'。
// 成员 sessionGeneration：SessionGeneration。
// 成员 durationMs：number。
// 成员 completedTurnCount：number。
// 成员 finalized：boolean。
// [/maint-note:39fa0820e2b0]
export interface SessionFooterDescriptor {
  readonly id: string
  readonly type: 'session_footer'
  readonly sessionGeneration: SessionGeneration
  readonly durationMs: number
  readonly completedTurnCount: number
  readonly finalized: boolean
}

// [maint-note:f1a542a261a2]
// 数据契约导航：SessionTimingSnapshot；字段含义以消费端校验为准。
// 成员 sessionGeneration：SessionGeneration。
// 成员 accumulatedActiveMs：number。
// 成员 completedTurnCount：number。
// 成员 footerId：string。
// 成员 footerFinalized：boolean。
// 成员 activeRunId：number | null。
// 成员 currentTurnMs：number。
// 成员 approvalPaused：boolean。
// 成员 live：boolean。
// 成员 separators：readonly TurnSeparatorDescriptor[]。
// 成员 footer：SessionFooterDescriptor | null。
// [/maint-note:f1a542a261a2]
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

// [maint-note:6a5e2e1be462]
// 数据契约导航：FinishedTurnTiming；字段含义以消费端校验为准。
// 成员 runId：number。
// 成员 turnDurationMs：number。
// 成员 sessionDurationMs：number。
// 成员 completedTurnCount：number。
// 成员 separator：TurnSeparatorDescriptor。
// 成员 footer：SessionFooterDescriptor。
// [/maint-note:6a5e2e1be462]
export interface FinishedTurnTiming {
  readonly runId: number
  readonly turnDurationMs: number
  readonly sessionDurationMs: number
  readonly completedTurnCount: number
  readonly separator: TurnSeparatorDescriptor
  readonly footer: SessionFooterDescriptor
}

// [maint-note:a7efa3efa1b3]
// 数据契约导航：ActiveRun；字段含义以消费端校验为准。
// 成员 runId：number。
// 成员 accumulatedMs：number。
// 成员 segmentStartedAt：number | null。
// 成员 pausedRequestId：string | null。
// 成员 finalized：boolean。
// [/maint-note:a7efa3efa1b3]
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

  // [maint-note:eeadb239e7a3]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 clock：MonotonicClock；缺省表达式 defaultMonotonicClock。
  // 输入 generation：SessionGeneration；缺省表达式 'initial'。
  // 写入目标：this.generation。
  // 写入目标：this.footerId。
  // 直接调用：this.makeFooterId。
  // [/maint-note:eeadb239e7a3]
  constructor(private readonly clock: MonotonicClock = defaultMonotonicClock, generation: SessionGeneration = 'initial') {
    this.generation = generation
    this.footerId = this.makeFooterId()
  }

  // [maint-note:195c0a3d3438]
  // 维护导航：beginSession；下列为本函数源码中的接口与路径索引。
  // 输入 generation：SessionGeneration；缺省表达式 this.generation。
  // 返回契约：void。
  // 写入目标：this.generation。
  // 写入目标：this.accumulatedMs。
  // 写入目标：this.completedTurns。
  // 写入目标：this.footerFinalized。
  // 写入目标：this.activeRun。
  // 写入目标：this.separatorList.length。
  // 写入目标：this.footerId。
  // 直接调用：this.makeFooterId。
  // [/maint-note:195c0a3d3438]
  beginSession(generation: SessionGeneration = this.generation): void {
    this.generation = generation
    this.accumulatedMs = 0
    this.completedTurns = 0
    this.footerFinalized = false
    this.activeRun = null
    this.separatorList.length = 0
    this.footerId = this.makeFooterId()
  }

  // [maint-note:53520e96e0ab]
  // 维护导航：clearSession；下列为本函数源码中的接口与路径索引。
  // 输入 generation：SessionGeneration；缺省表达式 this.generation。
  // 返回契约：void。
  // 直接调用：this.beginSession。
  // [/maint-note:53520e96e0ab]
  clearSession(generation: SessionGeneration = this.generation): void {
    this.beginSession(generation)
  }

  // [maint-note:6349577fb7a5]
  // 维护导航：beginRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：boolean。
  // 分支条件：this.activeRun || this.footerFinalized。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.activeRun。
  // 直接调用：normalizeTimestamp。
  // [/maint-note:6349577fb7a5]
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

  // [maint-note:9b7a84f8e116]
  // 维护导航：pauseApproval；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 requestId：string。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：boolean。
  // 分支条件：!run || run.runId !== runId || run.finalized || run.pausedRequestId !== null。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：run.accumulatedMs。
  // 写入目标：run.segmentStartedAt。
  // 写入目标：run.pausedRequestId。
  // 直接调用：addDuration。
  // 直接调用：activeDelta。
  // [/maint-note:9b7a84f8e116]
  pauseApproval(runId: number, requestId: string, at = this.clock()): boolean {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized || run.pausedRequestId !== null) return false
    run.accumulatedMs = addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))
    run.segmentStartedAt = null
    run.pausedRequestId = requestId
    return true
  }

  // [maint-note:52b63c260f5d]
  // 维护导航：resumeApproval；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 requestId：string。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：boolean。
  // 分支条件：!run || run.runId !== runId || run.finalized || run.pausedRequestId !== requestId。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：run.pausedRequestId。
  // 写入目标：run.segmentStartedAt。
  // 直接调用：normalizeTimestamp。
  // [/maint-note:52b63c260f5d]
  resumeApproval(runId: number, requestId: string, at = this.clock()): boolean {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized || run.pausedRequestId !== requestId) return false
    run.pausedRequestId = null
    run.segmentStartedAt = normalizeTimestamp(at)
    return true
  }

  // [maint-note:6e9acc251450]
  // 维护导航：handleEvent；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 event：AgentEvent。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：boolean。
  // 分支条件：event.type === 'approval_required'。
  // 分支条件：event.type === 'approval_resolved'。
  // 返回路径：this.pauseApproval(runId, event.requestId, at)。
  // 返回路径：this.resumeApproval(runId, event.requestId, at)。
  // 返回路径：false。
  // 直接调用：this.pauseApproval。
  // 直接调用：this.resumeApproval。
  // [/maint-note:6e9acc251450]
  handleEvent(runId: number, event: AgentEvent, at = this.clock()): boolean {
    if (event.type === 'approval_required') return this.pauseApproval(runId, event.requestId, at)
    if (event.type === 'approval_resolved') return this.resumeApproval(runId, event.requestId, at)
    return false
  }

  // [maint-note:4e202550cedb]
  // 维护导航：currentTurnDuration；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：number。
  // 分支条件：!run || run.runId !== runId || run.finalized。
  // 返回路径：0。
  // 返回路径：addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))。
  // 直接调用：addDuration。
  // 直接调用：activeDelta。
  // [/maint-note:4e202550cedb]
  currentTurnDuration(runId: number, at = this.clock()): number {
    const run = this.activeRun
    if (!run || run.runId !== runId || run.finalized) return 0
    return addDuration(run.accumulatedMs, activeDelta(run.segmentStartedAt, at))
  }

  // [maint-note:41e7cbfdea49]
  // 维护导航：finishRun；下列为本函数源码中的接口与路径索引。
  // 输入 runId：number。
  // 输入 _outcome：string（可省略）。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：FinishedTurnTiming | null。
  // 分支条件：!run || run.runId !== runId || run.finalized。
  // 返回路径：null。
  // 返回路径：{ runId, turnDurationMs, sessionDurationMs: this.accumulatedMs, completedTurnCount: this.completedTurns, separator, footer: this.makeFooter(), }。
  // 写入目标：run.accumulatedMs。
  // 写入目标：run.segmentStartedAt。
  // 写入目标：run.pausedRequestId。
  // 写入目标：run.finalized。
  // 写入目标：this.accumulatedMs。
  // 写入目标：this.completedTurns。
  // 写入目标：this.activeRun。
  // 直接调用：addDuration。
  // 直接调用：activeDelta。
  // 直接调用：normalizeDuration。
  // 直接调用：Object.freeze。
  // 直接调用：this.separatorList.push。
  // 直接调用：this.makeFooter。
  // [/maint-note:41e7cbfdea49]
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

  // [maint-note:7866bf3d0cd2]
  // 维护导航：finalizeSession；下列为本函数源码中的接口与路径索引。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：SessionFooterDescriptor | null。
  // 分支条件：this.activeRun。
  // 分支条件：this.completedTurns === 0。
  // 返回路径：null。
  // 返回路径：this.makeFooter()。
  // 写入目标：this.footerFinalized。
  // 直接调用：this.makeFooter。
  // [/maint-note:7866bf3d0cd2]
  finalizeSession(at = this.clock()): SessionFooterDescriptor | null {
    // A session close is not a terminal event. Do not turn an active run into
    // a completed turn merely because the UI is being finalized.
    if (this.activeRun) return null
    if (this.completedTurns === 0) return null
    this.footerFinalized = true
    return this.makeFooter()
  }

  // [maint-note:a25ce9ad6706]
  // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
  // 输入 at：类型由调用方与函数体约束；缺省表达式 this.clock()。
  // 返回契约：SessionTimingSnapshot。
  // 返回路径：{ sessionGeneration: this.generation, accumulatedActiveMs: this.accumulatedMs, completedTurnCount: this.completedTurns, footerId: this.footerId, footerFinalized: this.footerFinalized, activeRunId, currentTurnMs, approvalPaused, live: active。
  // 直接调用：this.currentTurnDuration。
  // 直接调用：this.makeFooter。
  // 直接调用：this.separatorList.slice。
  // [/maint-note:a25ce9ad6706]
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

  // [maint-note:c3e02b96804e]
  // 维护导航：makeFooter；下列为本函数源码中的接口与路径索引。
  // 返回契约：SessionFooterDescriptor。
  // 返回路径：Object.freeze({ id: this.footerId, type: 'session_footer', sessionGeneration: this.generation, durationMs: this.accumulatedMs, completedTurnCount: this.completedTurns, finalized: this.footerFinalized, })。
  // 直接调用：Object.freeze。
  // [/maint-note:c3e02b96804e]
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

  // [maint-note:69068364c0ab]
  // 维护导航：makeFooterId；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 返回路径：`session-footer-${this.footerSerial}`。
  // 写入目标：this.footerSerial。
  // [/maint-note:69068364c0ab]
  private makeFooterId(): string {
    this.footerSerial += 1
    return `session-footer-${this.footerSerial}`
  }
}

// [maint-note:544a4979a1cd]
// 维护导航：formatDuration；下列为本函数源码中的接口与路径索引。
// 输入 milliseconds：number。
// 返回契约：string。
// 分支条件：totalSeconds < 60。
// 分支条件：totalMinutes < 60。
// 返回路径：`${totalSeconds} s`。
// 返回路径：`${totalMinutes} m ${seconds} s`。
// 返回路径：`${hours} h ${minutes} m ${seconds} s`。
// 直接调用：Math.floor。
// 直接调用：normalizeDuration。
// [/maint-note:544a4979a1cd]
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

// [maint-note:96816b05e4c6]
// 维护导航：normalizeDuration；下列为本函数源码中的接口与路径索引。
// 输入 milliseconds：number。
// 返回契约：number。
// 分支条件：!Number.isFinite(milliseconds) || milliseconds <= 0。
// 返回路径：0。
// 返回路径：Math.min(Math.floor(milliseconds), MAX_SAFE_DURATION_MS)。
// 直接调用：Number.isFinite。
// 直接调用：Math.min。
// 直接调用：Math.floor。
// [/maint-note:96816b05e4c6]
export function normalizeDuration(milliseconds: number): number {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return 0
  return Math.min(Math.floor(milliseconds), MAX_SAFE_DURATION_MS)
}

// [maint-note:7396c77a16ae]
// 维护导航：activeDelta；下列为本函数源码中的接口与路径索引。
// 输入 startedAt：number | null。
// 输入 now：number。
// 返回契约：number。
// 分支条件：startedAt === null。
// 返回路径：0。
// 返回路径：normalizeDuration(normalizeTimestamp(now) - startedAt)。
// 直接调用：normalizeDuration。
// 直接调用：normalizeTimestamp。
// [/maint-note:7396c77a16ae]
function activeDelta(startedAt: number | null, now: number): number {
  if (startedAt === null) return 0
  return normalizeDuration(normalizeTimestamp(now) - startedAt)
}

// [maint-note:7d8fa8dce607]
// 维护导航：addDuration；下列为本函数源码中的接口与路径索引。
// 输入 left：number。
// 输入 right：number。
// 返回契约：number。
// 返回路径：Math.min(MAX_SAFE_DURATION_MS, normalizeDuration(left) + normalizeDuration(right))。
// 直接调用：Math.min。
// 直接调用：normalizeDuration。
// [/maint-note:7d8fa8dce607]
function addDuration(left: number, right: number): number {
  return Math.min(MAX_SAFE_DURATION_MS, normalizeDuration(left) + normalizeDuration(right))
}

// [maint-note:3ffb86785e32]
// 维护导航：normalizeTimestamp；下列为本函数源码中的接口与路径索引。
// 输入 value：number。
// 返回契约：number。
// 分支条件：!Number.isFinite(value) || value <= 0。
// 返回路径：0。
// 返回路径：Math.min(value, MAX_SAFE_DURATION_MS)。
// 直接调用：Number.isFinite。
// 直接调用：Math.min。
// [/maint-note:3ffb86785e32]
function normalizeTimestamp(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0
  return Math.min(value, MAX_SAFE_DURATION_MS)
}

// [maint-note:38baa86edfc1]
// 维护导航：defaultMonotonicClock；下列为本函数源码中的接口与路径索引。
// 返回契约：number。
// 分支条件：typeof performance !== 'undefined' && typeof performance.now === 'function'。
// 分支条件：monotonicOrigin !== 0n && typeof process !== 'undefined' && typeof process.hrtime?.bigint === 'function'。
// 返回路径：performance.now()。
// 返回路径：Number(process.hrtime.bigint() - monotonicOrigin) / 1_000_000。
// 返回路径：0。
// 直接调用：performance.now。
// 直接调用：Number。
// 直接调用：process.hrtime.bigint。
// [/maint-note:38baa86edfc1]
function defaultMonotonicClock(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') return performance.now()
  if (monotonicOrigin !== 0n && typeof process !== 'undefined' && typeof process.hrtime?.bigint === 'function') {
    return Number(process.hrtime.bigint() - monotonicOrigin) / 1_000_000
  }
  return 0
}
