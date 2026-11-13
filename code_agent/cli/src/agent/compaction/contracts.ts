import type {
  ModelBudgetCapabilities,
  ModelMessage,
  ModelRequestMeasurement,
} from '../../model/types/types.js'

/**
 * process-internal, sourced context compaction.
 *
 * Mechanism (plan 13.1.2): freeze compressible evidence → bounded summary →
 * schema/source validation → safe-point install. Raw messages stay with their
 * original owner (AgentThread); a summary only changes the model REQUEST
 * VIEW, never the raw revision, saved cursor, close receipt or running lock.
 */

/** First-version engineering caps. */
export const COMPACTION_LIMITS = Object.freeze({
  /** Max raw source characters fed to one summarization request. */
  maxInputChars: 400_000,
  /** Max compressible units (complete exchanges) fed to one request. */
  maxInputUnits: 128,
  /** Max raw response characters accepted from the provider. */
  maxOutputChars: 16_384,
  /** Max serialized UTF-8 bytes of the full CompactionSummary envelope. */
  maxSummaryBytes: 16_384,
  /** Max compaction cycles per run (a cycle = freeze + at most one repair). */
  maxCyclesPerRun: 2,
  /** Max total maintenance model requests per run (summary + repair). */
  maxMaintenanceCallsPerRun: 3,
  /** Max schema/source repair attempts per frozen source (at most one). */
  maxRepairsPerSource: 1,
  /** Wall-clock deadline for one maintenance request. */
  requestTimeoutMs: 30_000,
  /** Cumulative wall-clock budget for maintenance in one run. */
  cumulativeTimeoutMs: 60_000,
  /** Bounded wait for a non-cooperative maintenance promise/iterator. */
  cleanupGraceMs: 1_000,
})

// [maint-note:9730bb44ffda]
// 数据契约导航：CompactionSourceKind；字段含义以消费端校验为准。
// 类型组合：| 'historical_user' | 'historical_exchange' | 'historical_terminal' | 'current_saved_exchange' /** A complete-but-not-yet-appended exchange in the pending buffer. */ | 'current_closed_pending_exchange'。
// [/maint-note:9730bb44ffda]
export type CompactionSourceKind =
  | 'historical_user'
  | 'historical_exchange'
  | 'historical_terminal'
  | 'current_saved_exchange'
  /** A complete-but-not-yet-appended exchange in the pending buffer. */
  | 'current_closed_pending_exchange'

/** Stable identity of one complete raw source range (never content hashes alone). */
// [maint-note:3a387552a564]
// 数据契约导航：CompactionSourceUnit；字段含义以消费端校验为准。
// 成员 kind：CompactionSourceKind。
// 成员 exchangeId：string。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 start：number。
// 成员 end：number。
// 成员 revision：number。
// 成员 contentHash：string。
// [/maint-note:3a387552a564]
export interface CompactionSourceUnit {
  readonly kind: CompactionSourceKind
  /** Thread-generated request, exchange, terminal, or pending-range identity. */
  readonly exchangeId: string
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  /** Half-open raw message range in the frozen source snapshot (-1 = pending). */
  readonly start: number
  readonly end: number
  /** Thread revision at which the range was appended (-1 = pending). */
  readonly revision: number
  /** sha256 of the exact serialized raw messages — content check only. */
  readonly contentHash: string
}

/** The frozen source handed to the summarizer; immutable evidence. */
// [maint-note:5dbebf37871c]
// 数据契约导航：CompactionSource；字段含义以消费端校验为准。
// 成员 schemaVersion：1。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 frozenAt：number。
// 成员 units：readonly CompactionSourceUnit[]。
// 成员 messages：readonly ModelMessage[]。
// [/maint-note:5dbebf37871c]
export interface CompactionSource {
  readonly schemaVersion: 1
  readonly threadId: string
  readonly generation: string
  readonly frozenAt: number
  readonly units: readonly CompactionSourceUnit[]
  readonly messages: readonly ModelMessage[]
}

// [maint-note:d883fe0ad75e]
// 数据契约导航：CompactionUsage；字段含义以消费端校验为准。
// 成员 inputTokens（可省略）：number。
// 成员 outputTokens（可省略）：number。
// 成员 totalTokens（可省略）：number。
// 成员 usageComplete：boolean。
// [/maint-note:d883fe0ad75e]
export interface CompactionUsage {
  readonly inputTokens?: number
  readonly outputTokens?: number
  /** Provider-reported total when present; never input+output inference. */
  readonly totalTokens?: number
  readonly usageComplete: boolean
}

// [maint-note:0bf8e32572c6]
// 数据契约导航：CompactionFailureKind；字段含义以消费端校验为准。
// 类型组合：| 'aborted' | 'timeout' | 'output_overflow' | 'transport_error' | 'invalid_output' | 'no_growth' | 'budget_exhausted'。
// [/maint-note:0bf8e32572c6]
export type CompactionFailureKind =
  | 'aborted'
  | 'timeout'
  | 'output_overflow'
  | 'transport_error'
  | 'invalid_output'
  | 'no_growth'
  | 'budget_exhausted'

// [maint-note:6fe618d1054b]
// 数据契约导航：CompactionFailure；字段含义以消费端校验为准。
// 成员 kind：CompactionFailureKind。
// 成员 message：string。
// [/maint-note:6fe618d1054b]
export interface CompactionFailure {
  readonly kind: CompactionFailureKind
  readonly message: string
}

/** The installed, derived summary — the only thing the request view consumes. */
// [maint-note:554d2cbd2cd6]
// 数据契约导航：CompactionSummary；字段含义以消费端校验为准。
// 成员 schemaVersion：1。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 createdAt：number。
// 成员 verification：'schema_and_source_valid'。
// 成员 coveredUnits：readonly CompactionSourceUnit[]。
// 成员 text：string。
// 成员 usage（可省略）：CompactionUsage。
// [/maint-note:554d2cbd2cd6]
export interface CompactionSummary {
  readonly schemaVersion: 1
  readonly threadId: string
  readonly generation: string
  readonly createdAt: number
  readonly verification: 'schema_and_source_valid'
  /** Units this summary replaces, in raw order; strict superset of the previous. */
  readonly coveredUnits: readonly CompactionSourceUnit[]
  /** Rendered summary body (structured sections, inference marked). */
  readonly text: string
  readonly usage?: CompactionUsage
}

// [maint-note:8ed970e4cfff]
// 数据契约导航：CompactionCandidate；字段含义以消费端校验为准。
// 成员 summary：CompactionSummary。
// 成员 rawChars：number。
// 成员 summaryChars：number。
// [/maint-note:8ed970e4cfff]
export interface CompactionCandidate {
  readonly summary: CompactionSummary
  /** Raw→summary net saving measured on the same covered source. */
  readonly rawChars: number
  readonly summaryChars: number
}

/** Result of one maintenance attempt. */
// [maint-note:fdd72be998a4]
// 数据契约导航：CompactionOutcome；字段含义以消费端校验为准。
// 类型组合：| { readonly ok: true; readonly candidate: CompactionCandidate } | { readonly ok: false; readonly failure: CompactionFailure }。
// [/maint-note:fdd72be998a4]
export type CompactionOutcome =
  | { readonly ok: true; readonly candidate: CompactionCandidate }
  | { readonly ok: false; readonly failure: CompactionFailure }

/** One structured entry inside the summary body; sources are cited, not trusted. */
// [maint-note:6f8d75fbf730]
// 数据契约导航：CompactionBodyEntry；字段含义以消费端校验为准。
// 成员 text：string。
// 成员 sourceIds：readonly string[]。
// [/maint-note:6f8d75fbf730]
export interface CompactionBodyEntry {
  readonly text: string
  readonly sourceIds: readonly string[]
}

/** The single-stage structured summary body. */
// [maint-note:fc2c18583877]
// 数据契约导航：CompactionBody；字段含义以消费端校验为准。
// 成员 facts：readonly CompactionBodyEntry[]。
// 成员 failures：readonly CompactionBodyEntry[]。
// 成员 unknowns：readonly CompactionBodyEntry[]。
// 成员 unfinished：readonly CompactionBodyEntry[]。
// 成员 suggestions：readonly CompactionBodyEntry[]。
// [/maint-note:fc2c18583877]
export interface CompactionBody {
  readonly facts: readonly CompactionBodyEntry[]
  readonly failures: readonly CompactionBodyEntry[]
  readonly unknowns: readonly CompactionBodyEntry[]
  readonly unfinished: readonly CompactionBodyEntry[]
  readonly suggestions: readonly CompactionBodyEntry[]
}

/** Body caps: entries per section, chars per entry/text, total JSON bytes. */
export const COMPACTION_BODY_LIMITS = Object.freeze({
  maxEntriesPerSection: 64,
  maxEntryChars: 2_000,
  maxBodyBytes: 16_384,
})

/**
 * The UNTRUSTED provider result: a raw JSON body string plus any known usage.
 * Identity, coverage, verification and the envelope are rebuilt by the Thread
 * core — the provider can never grant itself verification or alter coverage.
 */
// [maint-note:86f0974c263b]
// 数据契约导航：ProviderBodyResult；字段含义以消费端校验为准。
// 类型组合：| { readonly ok: true; readonly body: string; readonly usage?: CompactionUsage } | { readonly ok: false; readonly kind: 'aborted' | 'timeout' | 'output_overflow' | 'transport_error' | 'invalid_output'; readonly message: string; readonly usa。
// [/maint-note:86f0974c263b]
export type ProviderBodyResult =
  | { readonly ok: true; readonly body: string; readonly usage?: CompactionUsage }
  | { readonly ok: false; readonly kind: 'aborted' | 'timeout' | 'output_overflow' | 'transport_error' | 'invalid_output'; readonly message: string; readonly usage?: CompactionUsage; readonly body?: string }

/** Narrow port the Runner consumes; app injects the model-backed default. */
// [maint-note:069b2a2d0ea0]
// 数据契约导航：CompactionProvider；字段含义以消费端校验为准。
// 成员 budgetCapabilities（可省略）：ModelBudgetCapabilities。
// 成员 prepare（可省略）：{ readonly measurement: ModelRequestMeasurement dispatch(): Promise<ProviderBodyResult> }。
// 成员 compact：Promise<ProviderBodyResult>。
// [/maint-note:069b2a2d0ea0]
export interface CompactionProvider {
  /** Capability of this maintenance provider; it may differ from the Agent model. */
  readonly budgetCapabilities?: ModelBudgetCapabilities
  /** Optional provider-aware preparation used by the unified run budget. */
  prepare?(request: {
    readonly source: CompactionSource
    readonly requestMessages: readonly ModelMessage[]
    readonly signal?: AbortSignal
    readonly deadlineMs: number
  }): {
    readonly measurement: ModelRequestMeasurement
    dispatch(): Promise<ProviderBodyResult>
  }
  compact(request: {
    readonly source: CompactionSource
    /** Pre-built outbound messages (measured against the input cap by the engine). */
    readonly requestMessages: readonly ModelMessage[]
    readonly signal?: AbortSignal
    readonly deadlineMs: number
  }): Promise<ProviderBodyResult>
}

/**
 * Policy derived soft target: a summary is attempted when the
 * compressible history exceeds this fraction/absolute floor of the history
 * allowance. It is a SOFT trigger — never a new rejection of legal requests.
 */
// [maint-note:fad490cc1d16]
// 维护导航：compactionSoftTarget；下列为本函数源码中的接口与路径索引。
// 输入 maxHistoryChars：number。
// 返回契约：number。
// 返回路径：Math.max(2_048, Math.floor(maxHistoryChars / 2))。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// [/maint-note:fad490cc1d16]
export function compactionSoftTarget(maxHistoryChars: number): number {
  return Math.max(2_048, Math.floor(maxHistoryChars / 2))
}
