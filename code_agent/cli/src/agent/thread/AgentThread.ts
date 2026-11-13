import { createHash, randomUUID } from 'node:crypto'

import { AgentError } from '../errors.js'
import type {
  ModelAssistantMessage,
  ModelMessage,
  ModelUserMessage,
} from '../../model/types/types.js'
import type { ConversationState } from './conversationState.js'
import { canonicalToolCallRef } from '../toolCallIdentity.js'
import type { CompactionSummary } from '../compaction/contracts.js'
import {
  compactionCandidateSummary,
  finalizeCompactionReceiptTrim,
  freezeCompactionSource,
  installCompactionSummary,
  prepareCompactionCandidate,
  summaryStillValid,
  summaryViewMessage,
  type FreezeReceipt,
  type ThreadCompactionMutableState,
} from './threadCompaction.js'
import {
  serializedFinalRequestChars,
  serializedRequestChars,
} from '../compaction/viewSerializer.js'
import type { FinalRequestProjection } from '../types/frameworkContext.js'

// [maint-note:3ec4b901961b]
// 数据契约导航：ThreadRunIdentity；字段含义以消费端校验为准。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 start：number。
// [/maint-note:3ec4b901961b]
export interface ThreadRunIdentity {
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly start: number
}

// [maint-note:8cf0694bbfc2]
// 数据契约导航：ThreadRunSourceRef；字段含义以消费端校验为准。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 sourceId：string。
// 成员 start：number。
// 成员 end：number。
// 成员 revision：number。
// 成员 sha256：string。
// [/maint-note:8cf0694bbfc2]
export interface ThreadRunSourceRef {
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly sourceId: string
  /** Message-index range in the canonical Thread history. */
  readonly start: number
  readonly end: number
  readonly revision: number
  readonly sha256: string
}

// [maint-note:6babc1ede87b]
// 数据契约导航：ThreadExchangeRange；字段含义以消费端校验为准。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 exchangeId：string。
// 成员 start：number。
// 成员 end：number。
// 成员 revision：number。
// 成员 toolCallRefs：readonly string[]。
// [/maint-note:6babc1ede87b]
export interface ThreadExchangeRange {
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly exchangeId: string
  /** Message-index range containing Assistant tool calls and every result. */
  readonly start: number
  readonly end: number
  readonly revision: number
  /** Bounded identities for every call in this indivisible exchange. */
  readonly toolCallRefs: readonly string[]
}

// [maint-note:749ee2a019a0]
// 数据契约导航：ThreadTerminalFact；字段含义以消费端校验为准。
// 类型组合：| { readonly type: 'done'; readonly workspaceMayHaveChanged: boolean } | { readonly type: 'paused' readonly reason: 'budget_exhausted' | 'no_progress' readonly workspaceMayHaveChanged: boolean } | { readonly type: 'interrupted'; readonly wo。
// [/maint-note:749ee2a019a0]
export type ThreadTerminalFact =
  | { readonly type: 'done'; readonly workspaceMayHaveChanged: boolean }
  | {
      readonly type: 'paused'
      readonly reason: 'budget_exhausted' | 'no_progress'
      readonly workspaceMayHaveChanged: boolean
    }
  | { readonly type: 'interrupted'; readonly workspaceMayHaveChanged: boolean }
  | { readonly type: 'error'; readonly code: string; readonly workspaceMayHaveChanged: boolean }

// [maint-note:2923a4c97e88]
// 数据契约导航：ThreadRunHistoryReceipt；字段含义以消费端校验为准。
// 成员 requestRetained：boolean。
// 成员 closedExchangeCount：number。
// 成员 terminalAssistantRetained：boolean。
// 成员 source：ThreadRunSourceRef | null。
// [/maint-note:2923a4c97e88]
export interface ThreadRunHistoryReceipt {
  readonly requestRetained: boolean
  readonly closedExchangeCount: number
  readonly terminalAssistantRetained: boolean
  readonly source: ThreadRunSourceRef | null
}

// [maint-note:155e640e0420]
// 数据契约导航：ThreadCloseReceipt；字段含义以消费端校验为准。
// 成员 threadId：string。
// 成员 generation：string。
// 成员 runKey：string。
// 成员 runStart：number。
// 成员 runEnd：number。
// 成员 revision：number。
// 成员 terminal：ThreadTerminalFact。
// 成员 history：ThreadRunHistoryReceipt。
// 成员 exchangeRanges：readonly ThreadExchangeRange[]。
// [/maint-note:155e640e0420]
export interface ThreadCloseReceipt {
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly runStart: number
  readonly runEnd: number
  readonly revision: number
  readonly terminal: ThreadTerminalFact
  readonly history: ThreadRunHistoryReceipt
  readonly exchangeRanges: readonly ThreadExchangeRange[]
}

// [maint-note:5259a7eb5ad3]
// 数据契约导航：ThreadSourceReader；字段含义以消费端校验为准。
// 成员 totalChars：number。
// 成员 read：{ readonly text: string; readonly offset: number; readonly nextOffset: number | null }。
// [/maint-note:5259a7eb5ad3]
export interface ThreadSourceReader {
  readonly totalChars: number
  read(input: {
    readonly offset: number
    readonly limit: number
  }): { readonly text: string; readonly offset: number; readonly nextOffset: number | null }
}

// [maint-note:458ab43185a2]
// 数据契约导航：ThreadSourceResolution；字段含义以消费端校验为准。
// 类型组合：| { readonly status: 'readable'; readonly reader: ThreadSourceReader } | { readonly status: 'unavailable'; readonly reason: string }。
// [/maint-note:458ab43185a2]
export type ThreadSourceResolution =
  | { readonly status: 'readable'; readonly reader: ThreadSourceReader }
  | { readonly status: 'unavailable'; readonly reason: string }

// [maint-note:dc921e01dc8a]
// 数据契约导航：AgentThread；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 status：ConversationState['status']。
// 成员 snapshot：readonly ModelMessage[]。
// 成员 requestView：readonly ModelMessage[]。
// 成员 requestViewWithMetadata：AgentRequestView。
// 成员 resolveSource：ThreadSourceResolution。
// 成员 activeCompactionSummary：CompactionSummary | undefined。
// 成员 freezeCompactionSource：FreezeReceipt。
// 成员 installCompactionSummary：boolean。
// 成员 invalidateCompactionSummaryIfStale：void。
// 成员 rememberFailedCompactionFrontier：void。
// 成员 hasFailedCompactionFrontier：boolean。
// 成员 compactionLeaseValid：boolean。
// 成员 prepareCompactionCandidate：{ ok: true; candidate: import('./threadCompaction.js').SealedCompactionCandidate; measurement: import('./threadCompaction.js').CandidateMeasurement } | { ok: false; reason: string }。
// 成员 finalizeReceiptTrim：FreezeReceipt | undefined。
// [/maint-note:dc921e01dc8a]
export interface AgentThread {
  readonly id: string
  readonly status: ConversationState['status']
  snapshot(): readonly ModelMessage[]
  /** Build a bounded request view without changing the complete in-memory history. */
  requestView(
    maxHistoryChars: number,
    pendingMessages?: readonly ModelMessage[],
  ): readonly ModelMessage[]
  requestViewWithMetadata(
    maxHistoryChars: number,
    pendingMessages?: readonly ModelMessage[],
  ): AgentRequestView
  /** Resolve only an exact Thread-generated source identity and range. */
  resolveSource(ref: ThreadRunSourceRef): ThreadSourceResolution
  /** the currently installed derived-view summary, if any. */
  activeCompactionSummary(): CompactionSummary | undefined
  /** Freeze the current compressible evidence (raw stays untouched). */
  freezeCompactionSource(options?: { readonly pendingClosed?: readonly ModelMessage[] }): FreezeReceipt
  /** Install a CORE-SEALED candidate; callers cannot forge one. */
  installCompactionSummary(
    candidate: import('./threadCompaction.js').SealedCompactionCandidate,
    _unused?: undefined,
    options?: { readonly pendingClosed?: readonly ModelMessage[] },
  ): boolean
  /** Drop the derived summary when its source no longer validates. */
  invalidateCompactionSummaryIfStale(options?: { readonly pendingClosed?: readonly ModelMessage[] }): void
  /** bounded cross-run latch of deterministically failed frontiers. */
  rememberFailedCompactionFrontier(key: string): void
  hasFailedCompactionFrontier(key: string): boolean
  /** verify the one core-owned lease (never registerable publicly). */
  compactionLeaseValid(leaseId: string): boolean
  /**
   * core-in-closure candidate preparation. The live state
   * never leaves the controller — callers pass only the frozen receipt, the
   * untrusted body and optional usage, and get an opaque registry handle.
   * The optional request-only sections are the boundary's READ-ONLY peek:
   * they only parameterize the common final serializer measurement and are
   * never consumed here (the real dispatch takes its own sections).
   */
  prepareCompactionCandidate(
    receipt: FreezeReceipt,
    rawBody: string,
    usage?: import('../compaction/contracts.js').CompactionUsage,
    requestOnlySections?: readonly string[],
    view?: {
      readonly maxHistoryChars: number
      readonly pendingMessages?: readonly ModelMessage[]
      readonly requestOnlySectionsFor?: (view: AgentRequestView) => readonly string[]
      readonly requestProjection?: FinalRequestProjection
      readonly requestProjectionFor?: (view: AgentRequestView) => FinalRequestProjection
    },
  ): { ok: true; candidate: import('./threadCompaction.js').SealedCompactionCandidate; measurement: import('./threadCompaction.js').CandidateMeasurement } | { ok: false; reason: string }
  /**
   * core finalization of an input-shell trim. The engine submits only
   * the SELECTED unit IDs of a receipt the core issued; the core rebuilds
   * messages/counts/order from its own frozen record, re-issues the receipt
   * under the SAME lease, and returns the FINAL version that dispatch,
   * prepare and install must all share. Undefined = the selection was refused.
   */
  finalizeReceiptTrim(
    receipt: FreezeReceipt,
    selectedUnitIds: readonly string[],
  ): FreezeReceipt | undefined
}

// [maint-note:51b14c14519f]
// 数据契约导航：AgentRequestView；字段含义以消费端校验为准。
// 成员 messages：readonly ModelMessage[]。
// 成员 truncated：boolean。
// 成员 omittedGroupCount：number。
// 成员 omittedCharsApprox：number。
// 成员 frameworkContext（可省略）：string。
// [/maint-note:51b14c14519f]
export interface AgentRequestView {
  readonly messages: readonly ModelMessage[]
  readonly truncated: boolean
  readonly omittedGroupCount: number
  readonly omittedCharsApprox: number
  /** Trusted, request-only framework facts; never persisted as model-authored text. */
  readonly frameworkContext?: string
}

// [maint-note:e10e7c1cb17e]
// 数据契约导航：ThreadState；字段含义以消费端校验为准。
// 成员 generation：string。
// 成员 revision：number。
// 成员 nextRunOrdinal：number。
// 成员 activeRun（可省略）：MutableThreadRun。
// 成员 completedRuns：CompletedThreadRun[]。
// 成员 activeSummary（可省略）：CompactionSummary。
// 成员 failedFrontiers（可省略）：readonly string[]。
// 成员 compactionLease（可省略）：string。
// [/maint-note:e10e7c1cb17e]
interface ThreadState extends ConversationState {
  readonly generation: string
  revision: number
  nextRunOrdinal: number
  activeRun?: MutableThreadRun
  readonly completedRuns: CompletedThreadRun[]
  /** derived view summary; raw history is never rewritten. */
  activeSummary?: CompactionSummary
  /** Bounded cross-run latch of deterministically failed source frontiers. */
  failedFrontiers?: readonly string[]
  /** the one live install lease; closed candidates lose install rights. */
  compactionLease?: string
}

// [maint-note:7c2dab8c6afb]
// 数据契约导航：MutableThreadRun；字段含义以消费端校验为准。
// 成员 token：symbol。
// 成员 runKey：string。
// 成员 prompt：string。
// 成员 start：number。
// 成员 savedCursor：number。
// 成员 nextExchangeOrdinal：number。
// 成员 source：ThreadRunSourceRef | null。
// 成员 exchangeRanges：ThreadExchangeRange[]。
// 成员 appendReceipts：Map<number, { readonly digest: string; readonly receipt: ThreadExchangeRange }>。
// 成员 published（可省略）：ThreadCloseCandidate。
// [/maint-note:7c2dab8c6afb]
interface MutableThreadRun {
  readonly token: symbol
  readonly runKey: string
  readonly prompt: string
  readonly start: number
  savedCursor: number
  nextExchangeOrdinal: number
  source: ThreadRunSourceRef | null
  readonly exchangeRanges: ThreadExchangeRange[]
  readonly appendReceipts: Map<number, { readonly digest: string; readonly receipt: ThreadExchangeRange }>
  published?: ThreadCloseCandidate
}

// [maint-note:d31b89631ea0]
// 数据契约导航：CompletedThreadRun；字段含义以消费端校验为准。
// 成员 runKey：string。
// 成员 start：number。
// 成员 end：number。
// 成员 revision：number。
// 成员 source：ThreadRunSourceRef。
// 成员 exchangeRanges：readonly ThreadExchangeRange[]。
// 成员 terminal：ThreadTerminalFact。
// [/maint-note:d31b89631ea0]
interface CompletedThreadRun {
  readonly runKey: string
  readonly start: number
  readonly end: number
  readonly revision: number
  readonly source: ThreadRunSourceRef
  readonly exchangeRanges: readonly ThreadExchangeRange[]
  readonly terminal: ThreadTerminalFact
}

// [maint-note:3e4fdda843dd]
// 数据契约导航：ThreadCloseCandidate；字段含义以消费端校验为准。
// 成员 expectedRevision：number。
// 成员 expectedCursor：number。
// 成员 remainingMessages：readonly ModelMessage[]。
// 成员 runStart：number。
// 成员 runEnd：number。
// [/maint-note:3e4fdda843dd]
export interface ThreadCloseCandidate extends ThreadCloseReceipt {
  readonly expectedRevision: number
  readonly expectedCursor: number
  readonly remainingMessages: readonly ModelMessage[]
  readonly runStart: number
  readonly runEnd: number
}

// [maint-note:aec586cfd67d]
// 数据契约导航：ThreadRunController；字段含义以消费端校验为准。
// 成员 identity：ThreadRunIdentity。
// 成员 runKey：string。
// 成员 savedCursor：number。
// 成员 requestViewWithMetadata：AgentRequestView。
// 成员 appendClosedExchange：ThreadExchangeRange。
// 成员 prepareClose：ThreadCloseCandidate。
// 成员 publishClose：ThreadCloseReceipt。
// 成员 release：void。
// [/maint-note:aec586cfd67d]
export interface ThreadRunController {
  readonly identity: ThreadRunIdentity
  readonly runKey: string
  readonly savedCursor: number
  requestViewWithMetadata(
    maxHistoryChars: number,
    pendingMessages: readonly ModelMessage[],
  ): AgentRequestView
  appendClosedExchange(input: {
    readonly expectedCursor: number
    readonly messages: readonly ModelMessage[]
  }): ThreadExchangeRange
  prepareClose(input: {
    readonly terminal: ThreadTerminalFact
    readonly expectedCursor: number
    readonly remainingCanonicalMessages?: readonly ModelMessage[]
  }): ThreadCloseCandidate
  publishClose(candidate: ThreadCloseCandidate): ThreadCloseReceipt
  release(receipt: ThreadCloseReceipt): void
}

// [maint-note:82c9048a7f66]
// 数据契约导航：ThreadController；字段含义以消费端校验为准。
// 成员 begin：ThreadRunController。
// [/maint-note:82c9048a7f66]
interface ThreadController {
  begin(prompt: string): ThreadRunController
}

const states = new WeakMap<AgentThread, ThreadState>()

// [maint-note:4b175a25bec0]
// 维护导航：createAgentThread；下列为本函数源码中的接口与路径索引。
// 输入 initialMessages：readonly ModelMessage[]；缺省表达式 []。
// 输入 id：string；缺省表达式 randomUUID()。
// 返回契约：AgentThread。
// 返回路径：Object.freeze(thread)。
// 直接调用：cloneMessages。
// 直接调用：randomUUID。
// 直接调用：states.set。
// 直接调用：Object.freeze。
// [/maint-note:4b175a25bec0]
export function createAgentThread(
  initialMessages: readonly ModelMessage[] = [],
  id: string = randomUUID(),
): AgentThread {
  const state: ThreadState = {
    id,
    messages: cloneMessages(initialMessages),
    status: 'idle',
    generation: randomUUID(),
    revision: 0,
    nextRunOrdinal: 1,
    completedRuns: [],
  }

  const thread: AgentThread = {
    // [maint-note:2f5ab80b48be]
    // 维护导航：id；下列为本函数源码中的接口与路径索引。
    // 返回路径：state.id。
    // [/maint-note:2f5ab80b48be]
    get id() {
      return state.id
    },
    // [maint-note:2b685561db8e]
    // 维护导航：status；下列为本函数源码中的接口与路径索引。
    // 返回路径：state.status。
    // [/maint-note:2b685561db8e]
    get status() {
      return state.status
    },
    // [maint-note:a88b47f4c1fd]
    // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
    // 返回路径：freezeMessages(state.messages)。
    // 直接调用：freezeMessages。
    // [/maint-note:a88b47f4c1fd]
    snapshot() {
      return freezeMessages(state.messages)
    },
    // [maint-note:446464d21197]
    // 维护导航：requestView；下列为本函数源码中的接口与路径索引。
    // 输入 maxHistoryChars：类型由调用方与函数体约束。
    // 输入 pendingMessages：类型由调用方与函数体约束；缺省表达式 []。
    // 返回路径：this.requestViewWithMetadata(maxHistoryChars, pendingMessages).messages。
    // 直接调用：this.requestViewWithMetadata。
    // [/maint-note:446464d21197]
    requestView(maxHistoryChars, pendingMessages = []) {
      return this.requestViewWithMetadata(maxHistoryChars, pendingMessages).messages
    },
    // [maint-note:d9df151d35db]
    // 维护导航：requestViewWithMetadata；下列为本函数源码中的接口与路径索引。
    // 输入 maxHistoryChars：类型由调用方与函数体约束。
    // 输入 pendingMessages：类型由调用方与函数体约束；缺省表达式 []。
    // 返回路径：buildRequestView(state, maxHistoryChars, pendingMessages)。
    // 直接调用：buildRequestView。
    // [/maint-note:d9df151d35db]
    requestViewWithMetadata(maxHistoryChars, pendingMessages = []) {
      return buildRequestView(state, maxHistoryChars, pendingMessages)
    },
    // [maint-note:ba00c2ec9f10]
    // 维护导航：resolveSource；下列为本函数源码中的接口与路径索引。
    // 输入 ref：类型由调用方与函数体约束。
    // 返回路径：resolveThreadSource(state, ref)。
    // 直接调用：resolveThreadSource。
    // [/maint-note:ba00c2ec9f10]
    resolveSource(ref) {
      return resolveThreadSource(state, ref)
    },
    // ------------------------------------------------------------------
    // the Thread owns the DERIVED request-view summary. Freezing
    // reads raw evidence; installing validates and swaps only the derived
    // view. Raw history, revision, cursor and receipts never change here.
    // ------------------------------------------------------------------
    // [maint-note:3938ca486bed]
    // 维护导航：activeCompactionSummary；下列为本函数源码中的接口与路径索引。
    // 返回路径：state.activeSummary ? structuredClone(state.activeSummary) : undefined。
    // 直接调用：structuredClone。
    // [/maint-note:3938ca486bed]
    activeCompactionSummary() {
      // Read-only projection: callers can never mutate the installed state.
      return state.activeSummary ? structuredClone(state.activeSummary) : undefined
    },
    // [maint-note:205f82a07f9c]
    // 维护导航：freezeCompactionSource；下列为本函数源码中的接口与路径索引。
    // 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
    // 返回路径：receipt。
    // 写入目标：state.compactionLease。
    // 直接调用：freezeCompactionSource。
    // 直接调用：compactionView。
    // [/maint-note:205f82a07f9c]
    freezeCompactionSource(options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      const receipt = freezeCompactionSource(compactionView(state), options)
      // Registering the new lease revokes any earlier candidate: only the
      // newest freeze on the live owner can publish.
      state.compactionLease = receipt.leaseId
      return receipt
    },
    // [maint-note:2376d44105e7]
    // 维护导航：installCompactionSummary；下列为本函数源码中的接口与路径索引。
    // 输入 candidate：import('./threadCompaction.js').SealedCompactionCandidate。
    // 输入 _unused：undefined（可省略）。
    // 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
    // 返回路径：installCompactionSummary(compactionView(state), candidate, undefined, options)。
    // 直接调用：installCompactionSummary。
    // 直接调用：compactionView。
    // [/maint-note:2376d44105e7]
    installCompactionSummary(candidate: import('./threadCompaction.js').SealedCompactionCandidate, _unused?: undefined, options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      return installCompactionSummary(compactionView(state), candidate, undefined, options)
    },
    // [maint-note:3a5b2d94477d]
    // 维护导航：invalidateCompactionSummaryIfStale；下列为本函数源码中的接口与路径索引。
    // 输入 options：{ readonly pendingClosed?: readonly ModelMessage[] }；缺省表达式 {}。
    // 分支条件：!summaryStillValid(compactionView(state), options)。
    // 写入目标：state.activeSummary。
    // 直接调用：summaryStillValid。
    // 直接调用：compactionView。
    // [/maint-note:3a5b2d94477d]
    invalidateCompactionSummaryIfStale(options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      if (!summaryStillValid(compactionView(state), options)) state.activeSummary = undefined
    },
    // [maint-note:2aebe9566f97]
    // 维护导航：rememberFailedCompactionFrontier；下列为本函数源码中的接口与路径索引。
    // 输入 key：string。
    // 分支条件：existing.includes(key)。
    // 返回路径：无返回值。
    // 写入目标：state.failedFrontiers。
    // 直接调用：existing.includes。
    // 直接调用：existing.slice。
    // [/maint-note:2aebe9566f97]
    rememberFailedCompactionFrontier(key: string) {
      const existing = state.failedFrontiers ?? []
      if (existing.includes(key)) return
      state.failedFrontiers = [...existing.slice(-7), key]
    },
    // [maint-note:3380f143310c]
    // 维护导航：hasFailedCompactionFrontier；下列为本函数源码中的接口与路径索引。
    // 输入 key：string。
    // 返回路径：(state.failedFrontiers ?? []).includes(key)。
    // 直接调用：(state.failedFrontiers ?? []).includes。
    // [/maint-note:3380f143310c]
    hasFailedCompactionFrontier(key: string) {
      return (state.failedFrontiers ?? []).includes(key)
    },
    // [maint-note:ef8f3b3bf056]
    // 维护导航：compactionLeaseValid；下列为本函数源码中的接口与路径索引。
    // 输入 leaseId：string。
    // 返回路径：state.compactionLease === leaseId。
    // [/maint-note:ef8f3b3bf056]
    compactionLeaseValid(leaseId: string) {
      return state.compactionLease === leaseId
    },
    // [maint-note:0ae1ab1d2357]
    // 维护导航：prepareCompactionCandidate；下列为本函数源码中的接口与路径索引。
    // 输入 receipt：类型由调用方与函数体约束。
    // 输入 rawBody：类型由调用方与函数体约束。
    // 输入 usage：类型由调用方与函数体约束。
    // 输入 requestOnlySections：类型由调用方与函数体约束。
    // 输入 view：类型由调用方与函数体约束。
    // 分支条件：!prepared.ok || !view。
    // 分支条件：!summary。
    // 返回路径：prepared。
    // 返回路径：{ ok: false, reason: 'The prepared candidate is unavailable for request measurement' }。
    // 返回路径：{ ...prepared, measurement: { ...prepared.measurement, currentRequestChars: serializedFinalRequestChars(currentRequest.messages, currentProjection), candidateRequestChars: serializedFinalRequestChars(candidateRequest.messages, candidateProj。
    // 直接调用：prepareCompactionCandidate。
    // 直接调用：compactionView。
    // 直接调用：compactionCandidateSummary。
    // 直接调用：buildRequestView。
    // 直接调用：projectionFor。
    // 直接调用：serializedFinalRequestChars。
    // [/maint-note:0ae1ab1d2357]
    prepareCompactionCandidate(receipt, rawBody, usage, requestOnlySections, view) {
      // prepare runs against the LIVE state inside the core closure.
      const effectiveRequestOnlySections = requestOnlySections ?? view?.requestProjection?.requestOnlySections
      const prepared = prepareCompactionCandidate(
        compactionView(state),
        receipt,
        rawBody,
        usage,
        effectiveRequestOnlySections,
      )
      if (!prepared.ok || !view) return prepared
      const summary = compactionCandidateSummary(prepared.candidate)
      if (!summary) return { ok: false, reason: 'The prepared candidate is unavailable for request measurement' }
      const pending = view.pendingMessages ?? []
      const currentRequest = buildRequestView(state, view.maxHistoryChars, pending, state.activeRun)
      const candidateRequest = buildRequestView(
        state,
        view.maxHistoryChars,
        pending,
        state.activeRun,
        summary,
      )
      const fallbackProjection: FinalRequestProjection = view.requestProjection ?? {
        requestOnlySections: effectiveRequestOnlySections ?? [],
        trailingFrameworkMessages: [],
      }
      // [maint-note:4b6f5f6e1dc3]
      // 维护导航：projectionFor；下列为本函数源码中的接口与路径索引。
      // 输入 request：AgentRequestView。
      // 返回契约：FinalRequestProjection。
      // 分支条件：view.requestProjectionFor。
      // 返回路径：view.requestProjectionFor(request)。
      // 返回路径：sections ? { requestOnlySections: sections, trailingFrameworkMessages: fallbackProjection.trailingFrameworkMessages, } : fallbackProjection。
      // 直接调用：view.requestProjectionFor。
      // 直接调用：view.requestOnlySectionsFor。
      // [/maint-note:4b6f5f6e1dc3]
      const projectionFor = (request: AgentRequestView): FinalRequestProjection => {
        if (view.requestProjectionFor) return view.requestProjectionFor(request)
        const sections = view.requestOnlySectionsFor?.(request)
        return sections
          ? {
              requestOnlySections: sections,
              trailingFrameworkMessages: fallbackProjection.trailingFrameworkMessages,
            }
          : fallbackProjection
      }
      const currentProjection = projectionFor(currentRequest)
      const candidateProjection = projectionFor(candidateRequest)
      return {
        ...prepared,
        measurement: {
          ...prepared.measurement,
          currentRequestChars: serializedFinalRequestChars(currentRequest.messages, currentProjection),
          candidateRequestChars: serializedFinalRequestChars(candidateRequest.messages, candidateProjection),
        },
      }
    },
    // [maint-note:b45cd92ed7c6]
    // 维护导航：finalizeReceiptTrim；下列为本函数源码中的接口与路径索引。
    // 输入 receipt：类型由调用方与函数体约束。
    // 输入 selectedUnitIds：类型由调用方与函数体约束。
    // 返回路径：finalizeCompactionReceiptTrim(compactionView(state), receipt, selectedUnitIds)。
    // 直接调用：finalizeCompactionReceiptTrim。
    // 直接调用：compactionView。
    // [/maint-note:b45cd92ed7c6]
    finalizeReceiptTrim(receipt, selectedUnitIds) {
      // only the owner's registry can finalize — the engine never
      // writes sources, messages or counts itself.
      return finalizeCompactionReceiptTrim(compactionView(state), receipt, selectedUnitIds)
    },
  }
  states.set(thread, state)
  return Object.freeze(thread)
}

// [maint-note:40402fe167ce]
// 维护导航：compactionView；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 返回契约：ThreadCompactionMutableState。
// 返回路径：state as unknown as ThreadCompactionMutableState。
// [/maint-note:40402fe167ce]
function compactionView(state: ThreadState): ThreadCompactionMutableState {
  return state as unknown as ThreadCompactionMutableState
}

/** @internal Used by AgentRunner to keep thread mutation out of UI/tool code. */
// [maint-note:cadb7d50cf78]
// 维护导航：getThreadController；下列为本函数源码中的接口与路径索引。
// 输入 thread：AgentThread。
// 返回契约：ThreadController。
// 分支条件：!state。
// 返回路径：{ begin(prompt) { if (state.status === 'running') { throw new AgentError( 'session_busy', 'This agent thread is already running a turn', true, ) } const run: MutableThreadRun = { token: Symbol(`thread-run-${state.nextRunOrdinal}`), runKey: 。
// 显式抛出：new AgentError('config_error', 'The supplied agent thread is invalid')。
// 直接调用：states.get。
// [/maint-note:cadb7d50cf78]
export function getThreadController(thread: AgentThread): ThreadController {
  const state = states.get(thread)
  if (!state) {
    throw new AgentError('config_error', 'The supplied agent thread is invalid')
  }

  return {
    // [maint-note:6a1d760ec4cd]
    // 维护导航：begin；下列为本函数源码中的接口与路径索引。
    // 输入 prompt：类型由调用方与函数体约束。
    // 分支条件：state.status === 'running'。
    // 返回路径：createRunController(state, run)。
    // 显式抛出：new AgentError( 'session_busy', 'This agent thread is already running a turn', true, )。
    // 写入目标：state.status。
    // 写入目标：state.activeRun。
    // 直接调用：Symbol。
    // 直接调用：createRunController。
    // [/maint-note:6a1d760ec4cd]
    begin(prompt) {
      if (state.status === 'running') {
        throw new AgentError(
          'session_busy',
          'This agent thread is already running a turn',
          true,
        )
      }
      const run: MutableThreadRun = {
        token: Symbol(`thread-run-${state.nextRunOrdinal}`),
        runKey: `thread-run-${state.nextRunOrdinal++}`,
        prompt,
        start: state.messages.length,
        savedCursor: 0,
        nextExchangeOrdinal: 1,
        source: null,
        exchangeRanges: [],
        appendReceipts: new Map(),
      }
      state.status = 'running'
      state.activeRun = run
      return createRunController(state, run)
    },
  }
}

// [maint-note:410a98387640]
// 维护导航：createRunController；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 run：MutableThreadRun。
// 返回契约：ThreadRunController。
// 返回路径：{ get identity() { return Object.freeze({ threadId: state.id, generation: state.generation, runKey: run.runKey, start: run.start, }) }, get runKey() { return run.runKey }, get savedCursor() { return run.savedCursor }, requestViewWithMetadat。
// [/maint-note:410a98387640]
function createRunController(state: ThreadState, run: MutableThreadRun): ThreadRunController {
  return {
    // [maint-note:59725c13a029]
    // 维护导航：identity；下列为本函数源码中的接口与路径索引。
    // 返回路径：Object.freeze({ threadId: state.id, generation: state.generation, runKey: run.runKey, start: run.start, })。
    // 直接调用：Object.freeze。
    // [/maint-note:59725c13a029]
    get identity() {
      return Object.freeze({
        threadId: state.id,
        generation: state.generation,
        runKey: run.runKey,
        start: run.start,
      })
    },
    // [maint-note:3713531149f2]
    // 维护导航：runKey；下列为本函数源码中的接口与路径索引。
    // 返回路径：run.runKey。
    // [/maint-note:3713531149f2]
    get runKey() {
      return run.runKey
    },
    // [maint-note:8836efbc7234]
    // 维护导航：savedCursor；下列为本函数源码中的接口与路径索引。
    // 返回路径：run.savedCursor。
    // [/maint-note:8836efbc7234]
    get savedCursor() {
      return run.savedCursor
    },
    // [maint-note:44afa98c65eb]
    // 维护导航：requestViewWithMetadata；下列为本函数源码中的接口与路径索引。
    // 输入 maxHistoryChars：类型由调用方与函数体约束。
    // 输入 pendingMessages：类型由调用方与函数体约束。
    // 返回路径：buildRequestView(state, maxHistoryChars, pendingMessages, run)。
    // 直接调用：assertRunOwner。
    // 直接调用：buildRequestView。
    // [/maint-note:44afa98c65eb]
    requestViewWithMetadata(maxHistoryChars, pendingMessages) {
      assertRunOwner(state, run)
      return buildRequestView(state, maxHistoryChars, pendingMessages, run)
    },
    // [maint-note:f3a30056e8c2]
    // 维护导航：appendClosedExchange；下列为本函数源码中的接口与路径索引。
    // 输入 input：类型由调用方与函数体约束。
    // 分支条件：replay。
    // 分支条件：replay.digest !== digest。
    // 分支条件：run.published || input.expectedCursor !== run.savedCursor。
    // 分支条件：!assistant || assistant.role !== 'assistant'。
    // 分支条件：!run.source。
    // 返回路径：replay.receipt。
    // 返回路径：receipt。
    // 显式抛出：new AgentError('model_protocol_error', 'Conflicting closed-exchange replay')。
    // 显式抛出：new AgentError('model_protocol_error', 'Closed-exchange cursor is stale or out of order')。
    // 显式抛出：new AgentError('model_protocol_error', 'A closed exchange has no canonical Assistant message')。
    // 写入目标：state.revision。
    // 写入目标：run.source。
    // 写入目标：run.savedCursor。
    // 直接调用：assertRunOwner。
    // 直接调用：cloneMessages。
    // 直接调用：messageDigest。
    // 直接调用：run.appendReceipts.get。
    // 直接调用：validateClosedExchange。
    // 直接调用：cloned.slice。
    // 直接调用：state.messages.push。
    // 直接调用：sourceRefForUser。
    // 直接调用：Object.freeze。
    // 直接调用：toolCallsOf(assistant).map。
    // 直接调用：toolCallsOf。
    // 直接调用：run.exchangeRanges.push。
    // 直接调用：run.appendReceipts.set。
    // [/maint-note:f3a30056e8c2]
    appendClosedExchange(input) {
      assertRunOwner(state, run)
      const cloned = cloneMessages(input.messages)
      const digest = messageDigest(cloned)
      const replay = run.appendReceipts.get(input.expectedCursor)
      if (replay) {
        if (replay.digest !== digest) {
          throw new AgentError('model_protocol_error', 'Conflicting closed-exchange replay')
        }
        return replay.receipt
      }
      if (run.published || input.expectedCursor !== run.savedCursor) {
        throw new AgentError('model_protocol_error', 'Closed-exchange cursor is stale or out of order')
      }
      validateClosedExchange(run, cloned)
      const exchange = run.savedCursor === 0 ? cloned.slice(1) : cloned
      const assistant = exchange[0]
      if (!assistant || assistant.role !== 'assistant') {
        throw new AgentError('model_protocol_error', 'A closed exchange has no canonical Assistant message')
      }

      const appendStart = state.messages.length
      state.messages.push(...cloned)
      state.revision += 1
      if (!run.source) {
        run.source = sourceRefForUser(state, run, appendStart, state.revision, cloned[0])
      }
      const exchangeStart = appendStart + (cloned[0]?.role === 'user' ? 1 : 0)
      const receipt = Object.freeze({
        threadId: state.id,
        generation: state.generation,
        runKey: run.runKey,
        exchangeId: `${run.runKey}:exchange-${run.nextExchangeOrdinal++}`,
        start: exchangeStart,
        end: appendStart + cloned.length,
        revision: state.revision,
        toolCallRefs: Object.freeze(toolCallsOf(assistant).map((call) => canonicalToolCallRef(call.id))),
      })
      run.exchangeRanges.push(receipt)
      run.appendReceipts.set(input.expectedCursor, { digest, receipt })
      run.savedCursor += cloned.length
      return receipt
    },
    // [maint-note:756083caf25d]
    // 维护导航：prepareClose；下列为本函数源码中的接口与路径索引。
    // 输入 input：类型由调用方与函数体约束。
    // 分支条件：run.published || input.expectedCursor !== run.savedCursor。
    // 返回路径：deepFreeze({ threadId: state.id, generation: state.generation, runKey: run.runKey, runStart: run.start, runEnd: state.messages.length + remainingMessages.length, revision: nextRevision, terminal: structuredClone(input.terminal), history: { 。
    // 显式抛出：new AgentError('model_protocol_error', 'Thread close cursor is stale or already published')。
    // 直接调用：assertRunOwner。
    // 直接调用：freezeMessages。
    // 直接调用：validateCloseTail。
    // 直接调用：sourceRefForUser。
    // 直接调用：deepFreeze。
    // 直接调用：structuredClone。
    // 直接调用：remainingMessages.some。
    // [/maint-note:756083caf25d]
    prepareClose(input) {
      assertRunOwner(state, run)
      if (run.published || input.expectedCursor !== run.savedCursor) {
        throw new AgentError('model_protocol_error', 'Thread close cursor is stale or already published')
      }
      const remainingMessages = freezeMessages(input.remainingCanonicalMessages ?? [])
      validateCloseTail(run, remainingMessages, input.terminal)
      const nextRevision = state.revision + 1
      const source = run.source ?? sourceRefForUser(
        state,
        run,
        state.messages.length,
        nextRevision,
        remainingMessages[0],
      )
      return deepFreeze({
        threadId: state.id,
        generation: state.generation,
        runKey: run.runKey,
        runStart: run.start,
        runEnd: state.messages.length + remainingMessages.length,
        revision: nextRevision,
        terminal: structuredClone(input.terminal),
        history: {
          requestRetained: true,
          closedExchangeCount: run.exchangeRanges.length,
          terminalAssistantRetained: remainingMessages.some((message) => message.role === 'assistant'),
          source,
        },
        exchangeRanges: structuredClone(run.exchangeRanges),
        expectedRevision: state.revision,
        expectedCursor: run.savedCursor,
        remainingMessages,
      })
    },
    // [maint-note:a70169565f31]
    // 维护导航：publishClose；下列为本函数源码中的接口与路径索引。
    // 输入 candidate：类型由调用方与函数体约束。
    // 分支条件：run.published || candidate.threadId !== state.id || candidate.generation !== state.generation || candidate.runKey !== run.runKey || candidate.revision !== state.revision + 1 || candidate.expectedRevision !== state.revision || candidate.expe。
    // 返回路径：candidate。
    // 显式抛出：new AgentError('model_protocol_error', 'Thread close candidate is stale or invalid')。
    // 写入目标：state.revision。
    // 写入目标：run.source。
    // 写入目标：run.published。
    // 直接调用：assertRunOwner。
    // 直接调用：state.messages.push。
    // 直接调用：state.completedRuns.push。
    // 直接调用：Object.freeze。
    // [/maint-note:a70169565f31]
    publishClose(candidate) {
      assertRunOwner(state, run)
      if (
        run.published ||
        candidate.threadId !== state.id ||
        candidate.generation !== state.generation ||
        candidate.runKey !== run.runKey ||
        candidate.revision !== state.revision + 1 ||
        candidate.expectedRevision !== state.revision ||
        candidate.expectedCursor !== run.savedCursor ||
        candidate.runStart !== run.start ||
        candidate.runEnd !== state.messages.length + candidate.remainingMessages.length
      ) {
        throw new AgentError('model_protocol_error', 'Thread close candidate is stale or invalid')
      }

      // All cloning, validation, range creation and hashing happened during
      // prepareClose. This publish segment only installs prebuilt state.
      state.messages.push(...candidate.remainingMessages)
      state.revision += 1
      run.source = candidate.history.source
      run.published = candidate
      state.completedRuns.push(Object.freeze({
        runKey: run.runKey,
        start: run.start,
        end: candidate.runEnd,
        revision: candidate.revision,
        source: candidate.history.source!,
        exchangeRanges: candidate.exchangeRanges,
        terminal: candidate.terminal,
      }))
      return candidate
    },
    // [maint-note:551255bf95de]
    // 维护导航：release；下列为本函数源码中的接口与路径索引。
    // 输入 receipt：类型由调用方与函数体约束。
    // 分支条件：run.published !== receipt。
    // 显式抛出：new AgentError('model_protocol_error', 'Thread run cannot release before its close is published')。
    // 写入目标：state.compactionLease。
    // 写入目标：state.activeRun。
    // 写入目标：state.status。
    // 直接调用：assertRunOwner。
    // [/maint-note:551255bf95de]
    release(receipt) {
      assertRunOwner(state, run)
      if (run.published !== receipt) {
        throw new AgentError('model_protocol_error', 'Thread run cannot release before its close is published')
      }
      // closing the run revokes every install lease it issued —
      // late candidates from this run can never publish. Already PUBLISHED
      // saved-only summaries stay readable (that is a view read, not an
      // install) and remain valid across runs.
      state.compactionLease = undefined
      state.activeRun = undefined
      state.status = 'idle'
    },
  }
}

// [maint-note:bc49be5e914e]
// 维护导航：buildRequestView；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 maxHistoryChars：number。
// 输入 pendingMessages：readonly ModelMessage[]。
// 输入 activeRun：MutableThreadRun（可省略）。
// 输入 summaryOverride：CompactionSummary（可省略）。
// 返回契约：AgentRequestView。
// 返回路径：{ messages: freezeMessages([ ...systemMessages, ...finalMessages, ...pendingMessages, ]), truncated: effectiveOmitted > 0, omittedGroupCount: effectiveOmitted, omittedCharsApprox: effectiveChars, ...(frameworkContext ? { frameworkContext } 。
// 直接调用：state.messages.filter。
// 直接调用：state.messages.slice。
// 直接调用：state.messages .slice(0, historyEnd) .filter。
// 直接调用：state.messages .slice。
// 直接调用：buildHistoricalRuns。
// 直接调用：serializedRequestChars。
// 直接调用：summaryViewMessage。
// 直接调用：selectHistoricalUnits。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：renderFrameworkContext。
// 直接调用：substituteSummaryIn。
// 直接调用：recoveredDroppedCoveredRuns。
// 直接调用：freezeMessages。
// [/maint-note:bc49be5e914e]
function buildRequestView(
  state: ThreadState,
  maxHistoryChars: number,
  pendingMessages: readonly ModelMessage[],
  activeRun?: MutableThreadRun,
  summaryOverride?: CompactionSummary,
): AgentRequestView {
  const activeSummary = summaryOverride ?? state.activeSummary
  const systemMessages = state.messages.filter((message) => message.role === 'system')
  const activeSaved = activeRun ? state.messages.slice(activeRun.start) : []
  const historyEnd = activeRun?.start ?? state.messages.length
  const historicalMessages = state.messages
    .slice(0, historyEnd)
    .filter((message) => message.role !== 'system')
  const historicalRuns = buildHistoricalRuns(state, historyEnd, historicalMessages)
  // 77.4 summary SOFT SLOT: the installed summary's real request-view
  // projection (framework note + body, serialized as it travels) is part of
  // the historical budget. Selection plans around the reservation, but the
  // summary itself is never evicted (its reserved slot keeps it visible) and this adds no
  // whole-request hard cap — the protected current run may still exceed the
  // allowance exactly as before.
  const summarySlotChars = activeSummary
    ? serializedRequestChars([summaryViewMessage(activeSummary)], [])
    : 0
  const selected = selectHistoricalUnits(
    systemMessages,
    historicalRuns,
    Math.max(0, Math.floor(maxHistoryChars) - summarySlotChars),
  )

  const frameworkContext = renderFrameworkContext(state.completedRuns)
  const assembled = substituteSummaryIn(
    state,
    activeSummary,
    [...selected.messages, ...activeSaved],
  )
  // : the derived summary must be visible even when every covered
  // raw group was dropped by the history budget — the small summary replaces
  // what was omitted, it is not lost with it. If substitution found no
  // surviving covered message, inject the summary right after the system
  // messages (the position of the oldest covered content).
  const summaryInjected = activeSummary !== undefined && assembled.summaryEmitted
  const finalMessages = summaryInjected || !activeSummary
    ? assembled.messages
    : [summaryViewMessage(activeSummary), ...assembled.messages]
  // omission reports UNRECOVERED loss by SOURCE RANGE — a visible run
  // covered by the summary never offsets a DIFFERENT older run's omission.
  // Count the omitted units the old way, then subtract only those whose
  // covered ranges belong to runs that are still EXPRESSED (i.e. their
  // anchor survived selection); fully-covered-but-dropped runs ARE recovered
  // by the injected summary.
  const recovered = recoveredDroppedCoveredRuns(state, selected.messages, activeSummary)
  const effectiveOmitted = Math.max(0, selected.omittedGroupCount - recovered.count)
  const effectiveChars = Math.max(0, selected.omittedCharsApprox - recovered.chars)
  return {
    messages: freezeMessages([
      ...systemMessages,
      ...finalMessages,
      ...pendingMessages,
    ]),
    truncated: effectiveOmitted > 0,
    omittedGroupCount: effectiveOmitted,
    omittedCharsApprox: effectiveChars,
    ...(frameworkContext ? { frameworkContext } : {}),
  }
}

/**
 * of the fully-covered runs, count only those whose CONTENT would have
 * been dropped by selection (their anchor user message is absent from the
 * selected view) — the summary genuinely recovered them. A covered run that
 * is still visible never offsets an uncovered older run's omission.
 */
// [maint-note:09090101c71a]
// 维护导航：recoveredDroppedCoveredRuns；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 selectedHistorical：readonly ModelMessage[]。
// 输入 summary：CompactionSummary | undefined。
// 返回契约：{ count: number; chars: number }。
// 分支条件：!summary。
// 分支条件：covered.has(run.source.sourceId) && anchor && !selectedObjects.has(anchor)。
// 分支条件：run.exchangeRanges.length === 0 && !hasTerminalUnit。
// 分支条件：!covered.has(range.exchangeId)。
// 分支条件：!rawPresent。
// 分支条件：occupied.has(index)。
// 分支条件：!terminal || terminal.role !== 'assistant' || toolCallsOf(terminal).length > 0。
// 分支条件：covered.has(`${run.runKey}:terminal`) && !selectedObjects.has(terminal)。
// 遍历边界：const run ← state.completedRuns。
// 遍历边界：const range ← run.exchangeRanges。
// 遍历边界：let index = range.start; index < range.end; index += 1。
// 遍历边界：let index = run.end - 1; index >= run.start; index -= 1。
// 返回路径：{ count: 0, chars: 0 }。
// 返回路径：{ count, chars }。
// 写入目标：count。
// 写入目标：chars。
// 写入目标：index。
// 直接调用：summary.coveredUnits.map。
// 直接调用：state.messages .slice(run.start, run.end) .some。
// 直接调用：state.messages .slice。
// 直接调用：covered.has。
// 直接调用：selectedObjects.has。
// 直接调用：messageChars。
// 直接调用：state.messages.slice。
// 直接调用：raw.some。
// 直接调用：JSON.stringify。
// 直接调用：occupied.add。
// 直接调用：occupied.has。
// 直接调用：toolCallsOf。
// [/maint-note:09090101c71a]
function recoveredDroppedCoveredRuns(
  state: ThreadState,
  selectedHistorical: readonly ModelMessage[],
  summary: CompactionSummary | undefined,
): { count: number; chars: number } {
  if (!summary) return { count: 0, chars: 0 }
  const covered = new Set(summary.coveredUnits.map((unit) => unit.exchangeId))
  // count (and size) per EXCHANGE RANGE from the same final
  // representation set. A covered exchange whose raw messages are absent
  // from the selected view is represented by the summary — its group AND its
  // characters are recovered; a visible anchor never implies visibility.
  const selectedObjects = new Set(selectedHistorical)
  let count = 0
  let chars = 0
  for (const run of state.completedRuns) {
    const anchor = state.messages[run.source.start]
    const hasTerminalUnit = state.messages
      .slice(run.start, run.end)
      .some((message) => message.role === 'assistant' && toolCallsOf(message).length === 0)
    if (
      covered.has(run.source.sourceId) &&
      anchor &&
      !selectedObjects.has(anchor)
    ) {
      // The historical selector accounts the run anchor as character cost,
      // but not as a separate omitted group when the run has an exchange or
      // terminal unit. An error/interrupted run can contain only the anchor;
      // in that shape the selector counted the anchor as its one omitted
      // group, so the summary must recover that count as well as its chars.
      if (run.exchangeRanges.length === 0 && !hasTerminalUnit) count += 1
      chars += messageChars([anchor])
    }
    for (const range of run.exchangeRanges) {
      if (!covered.has(range.exchangeId)) continue
      const raw = state.messages.slice(range.start, range.end)
      const rawPresent = raw.some((message) => selectedObjects.has(message))
      if (!rawPresent) {
        count += 1
        chars += JSON.stringify(raw).length
      }
    }
    const occupied = new Set<number>([run.source.start])
    for (const range of run.exchangeRanges) {
      for (let index = range.start; index < range.end; index += 1) occupied.add(index)
    }
    for (let index = run.end - 1; index >= run.start; index -= 1) {
      if (occupied.has(index)) continue
      const terminal = state.messages[index]
      if (!terminal || terminal.role !== 'assistant' || toolCallsOf(terminal).length > 0) continue
      if (covered.has(`${run.runKey}:terminal`) && !selectedObjects.has(terminal)) {
        count += 1
        chars += messageChars([terminal])
      }
      break
    }
  }
  return { count, chars }
}

/**
 * derived-view substitution: every message covered by the installed
 * summary is replaced by ONE summary message at the position of the first
 * covered message; protected holes (user prompts, the newest exchange) keep
 * their raw form. The raw array itself is never modified.
 */
// [maint-note:c7b4a3fd8e4a]
// 维护导航：substituteSummaryIn；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 summary：CompactionSummary | undefined。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：{ messages: ModelMessage[]; summaryEmitted: boolean }。
// 分支条件：!summary || summary.coveredUnits.length === 0。
// 分支条件：unit.kind === 'current_closed_pending_exchange'。
// 分支条件：!covered.has(message)。
// 分支条件：!emitted。
// 遍历边界：const unit ← summary.coveredUnits。
// 遍历边界：let index = unit.start; index < unit.end && index < state.messages.length; index += 1。
// 遍历边界：const message ← messages。
// 返回路径：{ messages: [...messages], summaryEmitted: false }。
// 返回路径：{ messages: output, summaryEmitted: emitted }。
// 写入目标：index。
// 写入目标：emitted。
// 直接调用：covered.add。
// 直接调用：covered.has。
// 直接调用：output.push。
// 直接调用：summaryViewMessage。
// [/maint-note:c7b4a3fd8e4a]
function substituteSummaryIn(
  state: ThreadState,
  summary: CompactionSummary | undefined,
  messages: readonly ModelMessage[],
): { messages: ModelMessage[]; summaryEmitted: boolean } {
  if (!summary || summary.coveredUnits.length === 0) return { messages: [...messages], summaryEmitted: false }
  const covered = new Set<ModelMessage>()
  for (const unit of summary.coveredUnits) {
    if (unit.kind === 'current_closed_pending_exchange') continue
    for (let index = unit.start; index < unit.end && index < state.messages.length; index += 1) {
      covered.add(state.messages[index]!)
    }
  }
  const output: ModelMessage[] = []
  let emitted = false
  for (const message of messages) {
    if (!covered.has(message)) {
      output.push(message)
      continue
    }
    if (!emitted) {
      output.push(summaryViewMessage(summary))
      emitted = true
    }
  }
  return { messages: output, summaryEmitted: emitted }
}

// [maint-note:0a8ef964c8a7]
// 维护导航：validateClosedExchange；下列为本函数源码中的接口与路径索引。
// 输入 run：MutableThreadRun。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：void。
// 分支条件：run.savedCursor === 0。
// 分支条件：!user || user.role !== 'user' || userText(user) !== run.prompt。
// 分支条件：exchange.length < 2 || exchange[0]?.role !== 'assistant'。
// 分支条件：calls.length === 0 || results.length !== calls.length || results.some((message, index) => ( message.role !== 'toolResult' || message.toolCallId !== calls[index]?.id || message.toolName !== calls[index]?.name ))。
// 显式抛出：new AgentError('model_protocol_error', 'The first closed exchange must retain the accepted user request')。
// 显式抛出：new AgentError('model_protocol_error', 'A closed exchange must begin with an Assistant tool-call message')。
// 显式抛出：new AgentError('model_protocol_error', 'A closed exchange must contain one ordered result for every tool call')。
// 直接调用：messages.slice。
// 直接调用：userText。
// 直接调用：toolCallsOf。
// 直接调用：exchange.slice。
// 直接调用：results.some。
// [/maint-note:0a8ef964c8a7]
function validateClosedExchange(run: MutableThreadRun, messages: readonly ModelMessage[]): void {
  const exchange = run.savedCursor === 0 ? messages.slice(1) : messages
  if (run.savedCursor === 0) {
    const user = messages[0]
    if (!user || user.role !== 'user' || userText(user) !== run.prompt) {
      throw new AgentError('model_protocol_error', 'The first closed exchange must retain the accepted user request')
    }
  }
  if (exchange.length < 2 || exchange[0]?.role !== 'assistant') {
    throw new AgentError('model_protocol_error', 'A closed exchange must begin with an Assistant tool-call message')
  }
  const calls = toolCallsOf(exchange[0])
  const results = exchange.slice(1)
  if (
    calls.length === 0 ||
    results.length !== calls.length ||
    results.some((message, index) => (
      message.role !== 'toolResult' ||
      message.toolCallId !== calls[index]?.id ||
      message.toolName !== calls[index]?.name
    ))
  ) {
    throw new AgentError('model_protocol_error', 'A closed exchange must contain one ordered result for every tool call')
  }
}

// [maint-note:0bf3e3b114d9]
// 维护导航：validateCloseTail；下列为本函数源码中的接口与路径索引。
// 输入 run：MutableThreadRun。
// 输入 messages：readonly ModelMessage[]。
// 输入 terminal：ThreadTerminalFact。
// 返回契约：void。
// 分支条件：run.savedCursor === 0。
// 分支条件：!user || user.role !== 'user' || userText(user) !== run.prompt。
// 分支条件：tail.some((message) => message.role === 'user' || message.role === 'system' || message.role === 'toolResult')。
// 分支条件：tail.length > 1。
// 分支条件：assistant?.role === 'assistant' && toolCallsOf(assistant).length > 0。
// 分支条件：(terminal.type === 'done' || terminal.type === 'paused') && assistant?.role !== 'assistant'。
// 分支条件：(terminal.type === 'error' || terminal.type === 'interrupted') && assistant。
// 显式抛出：new AgentError('model_protocol_error', 'Thread close must retain the accepted user request')。
// 显式抛出：new AgentError('model_protocol_error', 'Thread close contains an incomplete or misplaced protocol message')。
// 显式抛出：new AgentError('model_protocol_error', 'Thread close contains more than one terminal Assistant message')。
// 显式抛出：new AgentError('model_protocol_error', 'Thread close cannot retain an unpaired tool call')。
// 显式抛出：new AgentError('model_protocol_error', 'A normal or paused close requires one complete Assistant message')。
// 显式抛出：new AgentError('model_protocol_error', 'An error close cannot retain an unaccepted Assistant draft')。
// 写入目标：tail。
// 直接调用：userText。
// 直接调用：messages.slice。
// 直接调用：tail.some。
// 直接调用：toolCallsOf。
// [/maint-note:0bf3e3b114d9]
function validateCloseTail(
  run: MutableThreadRun,
  messages: readonly ModelMessage[],
  terminal: ThreadTerminalFact,
): void {
  let tail = messages
  if (run.savedCursor === 0) {
    const user = messages[0]
    if (!user || user.role !== 'user' || userText(user) !== run.prompt) {
      throw new AgentError('model_protocol_error', 'Thread close must retain the accepted user request')
    }
    tail = messages.slice(1)
  }
  if (tail.some((message) => message.role === 'user' || message.role === 'system' || message.role === 'toolResult')) {
    throw new AgentError('model_protocol_error', 'Thread close contains an incomplete or misplaced protocol message')
  }
  if (tail.length > 1) {
    throw new AgentError('model_protocol_error', 'Thread close contains more than one terminal Assistant message')
  }
  const assistant = tail[0]
  if (assistant?.role === 'assistant' && toolCallsOf(assistant).length > 0) {
    throw new AgentError('model_protocol_error', 'Thread close cannot retain an unpaired tool call')
  }
  if ((terminal.type === 'done' || terminal.type === 'paused') && assistant?.role !== 'assistant') {
    throw new AgentError('model_protocol_error', 'A normal or paused close requires one complete Assistant message')
  }
  if ((terminal.type === 'error' || terminal.type === 'interrupted') && assistant) {
    throw new AgentError('model_protocol_error', 'An error close cannot retain an unaccepted Assistant draft')
  }
}

// [maint-note:ac740a854c0c]
// 维护导航：sourceRefForUser；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 run：MutableThreadRun。
// 输入 start：number。
// 输入 revision：number。
// 输入 message：ModelMessage | undefined。
// 返回契约：ThreadRunSourceRef。
// 分支条件：!message || message.role !== 'user' || userText(message) !== run.prompt。
// 返回路径：Object.freeze({ threadId: state.id, generation: state.generation, runKey: run.runKey, sourceId: `${run.runKey}:request`, start, end: start + 1, revision, sha256: createHash('sha256').update(run.prompt).digest('hex'), })。
// 显式抛出：new AgentError('model_protocol_error', 'The accepted run source is missing from its canonical range')。
// 直接调用：userText。
// 直接调用：Object.freeze。
// 直接调用：createHash('sha256').update(run.prompt).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// [/maint-note:ac740a854c0c]
function sourceRefForUser(
  state: ThreadState,
  run: MutableThreadRun,
  start: number,
  revision: number,
  message: ModelMessage | undefined,
): ThreadRunSourceRef {
  if (!message || message.role !== 'user' || userText(message) !== run.prompt) {
    throw new AgentError('model_protocol_error', 'The accepted run source is missing from its canonical range')
  }
  return Object.freeze({
    threadId: state.id,
    generation: state.generation,
    runKey: run.runKey,
    sourceId: `${run.runKey}:request`,
    start,
    end: start + 1,
    revision,
    sha256: createHash('sha256').update(run.prompt).digest('hex'),
  })
}

// [maint-note:479172a6cdc4]
// 维护导航：resolveThreadSource；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 ref：ThreadRunSourceRef。
// 返回契约：ThreadSourceResolution。
// 分支条件：ref.threadId !== state.id || ref.generation !== state.generation || ref.start < 0 || ref.end !== ref.start + 1 || ref.end > state.messages.length。
// 分支条件：!run || !message || message.role !== 'user'。
// 分支条件：createHash('sha256').update(text).digest('hex') !== ref.sha256。
// 返回路径：{ status: 'unavailable', reason: 'the source identity or range does not belong to this thread' }。
// 返回路径：{ status: 'unavailable', reason: 'the source run or user message is unavailable' }。
// 返回路径：{ status: 'unavailable', reason: 'the source content no longer matches its integrity digest' }。
// 返回路径：{ status: 'readable', reader: { totalChars: characters.length, read(input) { const offset = Math.max(0, Math.min(input.offset, characters.length)) const end = Math.min(characters.length, offset + Math.max(0, input.limit)) return { text: cha。
// 直接调用：state.completedRuns.find。
// 直接调用：userText。
// 直接调用：createHash('sha256').update(text).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// [/maint-note:479172a6cdc4]
function resolveThreadSource(state: ThreadState, ref: ThreadRunSourceRef): ThreadSourceResolution {
  if (
    ref.threadId !== state.id ||
    ref.generation !== state.generation ||
    ref.start < 0 ||
    ref.end !== ref.start + 1 ||
    ref.end > state.messages.length
  ) {
    return { status: 'unavailable', reason: 'the source identity or range does not belong to this thread' }
  }
  const run = state.completedRuns.find((candidate) => (
    candidate.runKey === ref.runKey && sameSourceRef(candidate.source, ref)
  ))
  const message = state.messages[ref.start]
  if (!run || !message || message.role !== 'user') {
    return { status: 'unavailable', reason: 'the source run or user message is unavailable' }
  }
  const text = userText(message)
  if (createHash('sha256').update(text).digest('hex') !== ref.sha256) {
    return { status: 'unavailable', reason: 'the source content no longer matches its integrity digest' }
  }
  const characters = [...text]
  return {
    status: 'readable',
    reader: {
      totalChars: characters.length,
      // [maint-note:2b9f9ec847f3]
      // 维护导航：read；下列为本函数源码中的接口与路径索引。
      // 输入 input：类型由调用方与函数体约束。
      // 返回路径：{ text: characters.slice(offset, end).join(''), offset, nextOffset: end < characters.length ? end : null, }。
      // 直接调用：Math.max。
      // 直接调用：Math.min。
      // 直接调用：characters.slice(offset, end).join。
      // 直接调用：characters.slice。
      // [/maint-note:2b9f9ec847f3]
      read(input) {
        const offset = Math.max(0, Math.min(input.offset, characters.length))
        const end = Math.min(characters.length, offset + Math.max(0, input.limit))
        return {
          text: characters.slice(offset, end).join(''),
          offset,
          nextOffset: end < characters.length ? end : null,
        }
      },
    },
  }
}

// [maint-note:4a7a377ba6ad]
// 维护导航：renderFrameworkContext；下列为本函数源码中的接口与路径索引。
// 输入 runs：readonly CompletedThreadRun[]。
// 返回契约：string | undefined。
// 分支条件：relevant.length === 0。
// 返回路径：undefined。
// 返回路径：`Trusted prior run state (framework-generated, not model-authored):\n${facts.join('\n')}`。
// 直接调用：runs.filter((run) => run.terminal.type !== 'done').slice。
// 直接调用：runs.filter。
// 直接调用：relevant.map。
// 直接调用：facts.join。
// [/maint-note:4a7a377ba6ad]
function renderFrameworkContext(runs: readonly CompletedThreadRun[]): string | undefined {
  const relevant = runs.filter((run) => run.terminal.type !== 'done').slice(-4)
  if (relevant.length === 0) return undefined
  const facts = relevant.map((run) => {
    const terminal = run.terminal.type === 'error'
      ? `error(${run.terminal.code})`
      : run.terminal.type === 'paused'
        ? `paused(${run.terminal.reason})`
        : run.terminal.type
    return `${run.runKey} ended ${terminal}; ${run.exchangeRanges.length} complete tool exchange(s) were retained${run.terminal.workspaceMayHaveChanged ? '; the workspace may have changed and must be rechecked' : ''}.`
  })
  return `Trusted prior run state (framework-generated, not model-authored):\n${facts.join('\n')}`
}

// [maint-note:b1c1252a14ed]
// 维护导航：assertRunOwner；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 run：MutableThreadRun。
// 返回契约：void。
// 分支条件：state.status !== 'running' || state.activeRun?.token !== run.token。
// 显式抛出：new AgentError('config_error', 'The thread run owner is unavailable or stale')。
// [/maint-note:b1c1252a14ed]
function assertRunOwner(state: ThreadState, run: MutableThreadRun): void {
  if (state.status !== 'running' || state.activeRun?.token !== run.token) {
    throw new AgentError('config_error', 'The thread run owner is unavailable or stale')
  }
}

// [maint-note:aecbd7c45d69]
// 维护导航：sameSourceRef；下列为本函数源码中的接口与路径索引。
// 输入 left：ThreadRunSourceRef。
// 输入 right：ThreadRunSourceRef。
// 返回契约：boolean。
// 返回路径：left.threadId === right.threadId && left.generation === right.generation && left.runKey === right.runKey && left.sourceId === right.sourceId && left.start === right.start && left.end === right.end && left.revision === right.revision && left。
// [/maint-note:aecbd7c45d69]
function sameSourceRef(left: ThreadRunSourceRef, right: ThreadRunSourceRef): boolean {
  return left.threadId === right.threadId &&
    left.generation === right.generation &&
    left.runKey === right.runKey &&
    left.sourceId === right.sourceId &&
    left.start === right.start &&
    left.end === right.end &&
    left.revision === right.revision &&
    left.sha256 === right.sha256
}

// [maint-note:ebcbf7d1c2d3]
// 维护导航：cloneMessages；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：ModelMessage[]。
// 返回路径：structuredClone([...messages])。
// 直接调用：structuredClone。
// [/maint-note:ebcbf7d1c2d3]
function cloneMessages(messages: readonly ModelMessage[]): ModelMessage[] {
  return structuredClone([...messages])
}

// [maint-note:821ff07de9ee]
// 维护导航：groupHistory；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：ModelMessage[][]。
// 分支条件：message.role === 'system'。
// 分支条件：message.role === 'user' && current.length > 0。
// 分支条件：current.length > 0。
// 遍历边界：const message ← messages。
// 返回路径：groups。
// 写入目标：current。
// 直接调用：groups.push。
// 直接调用：current.push。
// [/maint-note:821ff07de9ee]
function groupHistory(messages: readonly ModelMessage[]): ModelMessage[][] {
  const groups: ModelMessage[][] = []
  let current: ModelMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    if (message.role === 'user' && current.length > 0) {
      groups.push(current)
      current = []
    }
    current.push(message)
  }
  if (current.length > 0) groups.push(current)
  return groups
}

// [maint-note:59fd5741f2ad]
// 数据契约导航：HistoricalRunUnits；字段含义以消费端校验为准。
// 成员 anchor（可省略）：ModelMessage。
// 成员 units：readonly (readonly ModelMessage[])[]。
// [/maint-note:59fd5741f2ad]
interface HistoricalRunUnits {
  readonly anchor?: ModelMessage
  readonly units: readonly (readonly ModelMessage[])[]
}

// [maint-note:07c755346da2]
// 维护导航：buildHistoricalRuns；下列为本函数源码中的接口与路径索引。
// 输入 state：ThreadState。
// 输入 historyEnd：number。
// 输入 historicalMessages：readonly ModelMessage[]。
// 返回契约：HistoricalRunUnits[]。
// 分支条件：completed.length === 0。
// 分支条件：cursor < run.start。
// 分支条件：!anchor || anchor.role !== 'user'。
// 分支条件：tail.length > 0。
// 分支条件：cursor < historyEnd。
// 遍历边界：const run ← completed。
// 遍历边界：const range ← run.exchangeRanges。
// 遍历边界：let index = range.start; index < range.end; index += 1。
// 返回路径：legacyHistoricalRuns(historicalMessages)。
// 返回路径：runs。
// 写入目标：cursor。
// 写入目标：index。
// 直接调用：state.completedRuns.filter。
// 直接调用：legacyHistoricalRuns。
// 直接调用：runs.push。
// 直接调用：state.messages.slice。
// 直接调用：run.exchangeRanges.map。
// 直接调用：covered.add。
// 直接调用：state.messages.slice(run.start, run.end).filter。
// 直接调用：units.push。
// [/maint-note:07c755346da2]
function buildHistoricalRuns(
  state: ThreadState,
  historyEnd: number,
  historicalMessages: readonly ModelMessage[],
): HistoricalRunUnits[] {
  const completed = state.completedRuns.filter((run) => run.end <= historyEnd)
  if (completed.length === 0) return legacyHistoricalRuns(historicalMessages)

  const runs: HistoricalRunUnits[] = []
  let cursor = 0
  for (const run of completed) {
    if (cursor < run.start) {
      runs.push(...legacyHistoricalRuns(state.messages.slice(cursor, run.start)))
    }
    const anchor = state.messages[run.source.start]
    if (!anchor || anchor.role !== 'user') {
      runs.push(...legacyHistoricalRuns(state.messages.slice(run.start, run.end)))
      cursor = run.end
      continue
    }
    const units = run.exchangeRanges.map((range) => state.messages.slice(range.start, range.end))
    const covered = new Set<number>([run.source.start])
    for (const range of run.exchangeRanges) {
      for (let index = range.start; index < range.end; index += 1) covered.add(index)
    }
    const tail = state.messages.slice(run.start, run.end).filter((message, offset) => (
      message.role !== 'system' && !covered.has(run.start + offset)
    ))
    if (tail.length > 0) units.push(tail)
    runs.push({ anchor, units })
    cursor = run.end
  }
  if (cursor < historyEnd) {
    runs.push(...legacyHistoricalRuns(state.messages.slice(cursor, historyEnd)))
  }
  return runs
}

// [maint-note:6429936da6e6]
// 维护导航：legacyHistoricalRuns；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：HistoricalRunUnits[]。
// 返回路径：groupHistory(messages.filter((message) => message.role !== 'system')).map((group) => { const first = group[0] if (!first || first.role !== 'user') return { units: [group] } const units: ModelMessage[][] = [] let cursor = 1 while (cursor < g。
// 直接调用：groupHistory(messages.filter((message) => message.role !== 'system')).map。
// 直接调用：groupHistory。
// 直接调用：messages.filter。
// [/maint-note:6429936da6e6]
function legacyHistoricalRuns(messages: readonly ModelMessage[]): HistoricalRunUnits[] {
  return groupHistory(messages.filter((message) => message.role !== 'system')).map((group) => {
    const first = group[0]
    if (!first || first.role !== 'user') return { units: [group] }
    const units: ModelMessage[][] = []
    let cursor = 1
    while (cursor < group.length) {
      const assistant = group[cursor]
      if (!assistant || assistant.role !== 'assistant') {
        return { units: [group] }
      }
      const calls = toolCallsOf(assistant)
      if (calls.length === 0) {
        units.push([assistant])
        cursor += 1
        continue
      }
      const results = group.slice(cursor + 1, cursor + 1 + calls.length)
      if (
        results.length !== calls.length ||
        results.some((message, index) => (
          message.role !== 'toolResult' ||
          message.toolCallId !== calls[index]?.id ||
          message.toolName !== calls[index]?.name
        ))
      ) {
        // Unknown legacy shapes remain all-or-nothing; never manufacture an
        // orphan tool call/result while trying to compact them.
        return { units: [group] }
      }
      units.push([assistant, ...results])
      cursor += 1 + results.length
    }
    return { anchor: first, units }
  })
}

// [maint-note:d230e5c42901]
// 维护导航：selectHistoricalUnits；下列为本函数源码中的接口与路径索引。
// 输入 systemMessages：readonly ModelMessage[]。
// 输入 runs：readonly HistoricalRunUnits[]。
// 输入 maxHistoryChars：number。
// 返回契约：{ readonly messages: readonly ModelMessage[] readonly omittedGroupCount: number readonly omittedCharsApprox: number }。
// 分支条件：addedAnchor。
// 分支条件：fits()。
// 分支条件：indexes.size === 0。
// 分支条件：run.anchor && !selectedAnchors.has(runIndex)。
// 分支条件：run.units.length === 0。
// 分支条件：!selectedFromRun。
// 遍历边界：let runIndex = runs.length - 1; runIndex >= 0; runIndex -= 1。
// 遍历边界：let unitIndex = run.units.length - 1; unitIndex >= 0; unitIndex -= 1。
// 遍历边界：let older = 0; older < runIndex; older += 1。
// 返回路径：{ messages: selectedMessages(), omittedGroupCount, omittedCharsApprox, }。
// 写入目标：runIndex。
// 写入目标：unitIndex。
// 写入目标：selectedFromRun。
// 写入目标：omittedGroupCount。
// 写入目标：omittedCharsApprox。
// 写入目标：older。
// 直接调用：selectedUnits.get。
// 直接调用：selectedUnits.set。
// 直接调用：indexes.add。
// 直接调用：Boolean。
// 直接调用：selectedAnchors.has。
// 直接调用：selectedAnchors.add。
// 直接调用：fits。
// 直接调用：indexes.delete。
// 直接调用：selectedUnits.delete。
// 直接调用：selectedAnchors.delete。
// 直接调用：messageChars。
// 直接调用：Math.max。
// 直接调用：omitted.units.reduce。
// 直接调用：selectedMessages。
// [/maint-note:d230e5c42901]
function selectHistoricalUnits(
  systemMessages: readonly ModelMessage[],
  runs: readonly HistoricalRunUnits[],
  maxHistoryChars: number,
): {
  readonly messages: readonly ModelMessage[]
  readonly omittedGroupCount: number
  readonly omittedCharsApprox: number
} {
  const selectedAnchors = new Set<number>()
  const selectedUnits = new Map<number, Set<number>>()
  let omittedGroupCount = 0
  let omittedCharsApprox = 0

  // [maint-note:f01b00d3a13d]
  // 维护导航：selectedMessages；下列为本函数源码中的接口与路径索引。
  // 返回契约：ModelMessage[]。
  // 直接调用：runs.flatMap。
  // [/maint-note:f01b00d3a13d]
  const selectedMessages = (): ModelMessage[] => runs.flatMap((run, runIndex) => {
    const indexes = selectedUnits.get(runIndex)
    const messages: ModelMessage[] = []
    if (selectedAnchors.has(runIndex) && run.anchor) messages.push(run.anchor)
    if (indexes) {
      for (const [unitIndex, unit] of run.units.entries()) {
        if (indexes.has(unitIndex)) messages.push(...unit)
      }
    }
    return messages
  })
  // [maint-note:b4abf7203d1d]
  // 维护导航：fits；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 直接调用：messageChars。
  // 直接调用：selectedMessages。
  // [/maint-note:b4abf7203d1d]
  const fits = (): boolean => messageChars([...systemMessages, ...selectedMessages()]) <= maxHistoryChars

  for (let runIndex = runs.length - 1; runIndex >= 0; runIndex -= 1) {
    const run = runs[runIndex]!
    let selectedFromRun = false
    for (let unitIndex = run.units.length - 1; unitIndex >= 0; unitIndex -= 1) {
      const indexes = selectedUnits.get(runIndex) ?? new Set<number>()
      selectedUnits.set(runIndex, indexes)
      indexes.add(unitIndex)
      const addedAnchor = Boolean(run.anchor && !selectedAnchors.has(runIndex))
      if (addedAnchor) selectedAnchors.add(runIndex)
      if (fits()) {
        selectedFromRun = true
        continue
      }
      indexes.delete(unitIndex)
      if (indexes.size === 0) selectedUnits.delete(runIndex)
      if (addedAnchor) selectedAnchors.delete(runIndex)
      omittedGroupCount += 1
      omittedCharsApprox += messageChars(run.units[unitIndex]!)
    }

    if (run.anchor && !selectedAnchors.has(runIndex)) {
      selectedAnchors.add(runIndex)
      if (fits()) {
        selectedFromRun = true
      } else {
        selectedAnchors.delete(runIndex)
        if (run.units.length === 0) omittedGroupCount += 1
        omittedCharsApprox += messageChars([run.anchor])
      }
    }

    // Preserve the established no-resurrection rule: if the newest
    // remaining run contributes nothing, older runs cannot jump over it.
    if (!selectedFromRun) {
      for (let older = 0; older < runIndex; older += 1) {
        const omitted = runs[older]!
        omittedGroupCount += Math.max(1, omitted.units.length)
        omittedCharsApprox += omitted.units.reduce(
          (total, unit) => total + messageChars(unit),
          omitted.anchor ? messageChars([omitted.anchor]) : 0,
        )
      }
      break
    }
  }

  return {
    messages: selectedMessages(),
    omittedGroupCount,
    omittedCharsApprox,
  }
}

// [maint-note:f23f3c70536a]
// 维护导航：messageChars；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：number。
// 返回路径：JSON.stringify(messages).length。
// 直接调用：JSON.stringify。
// [/maint-note:f23f3c70536a]
function messageChars(messages: readonly ModelMessage[]): number {
  return JSON.stringify(messages).length
}

// [maint-note:ed0f8d177dd0]
// 维护导航：messageDigest；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：string。
// 返回路径：createHash('sha256').update(JSON.stringify(messages)).digest('hex')。
// 直接调用：createHash('sha256').update(JSON.stringify(messages)).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:ed0f8d177dd0]
function messageDigest(messages: readonly ModelMessage[]): string {
  return createHash('sha256').update(JSON.stringify(messages)).digest('hex')
}

// [maint-note:9a9a26708ef1]
// 维护导航：toolCallsOf；下列为本函数源码中的接口与路径索引。
// 输入 message：ModelAssistantMessage。
// 返回契约：Array<{ id: string; name: string }>。
// 返回路径：message.content .filter((block) => block.type === 'toolCall') .map((block) => ({ id: block.id, name: block.name }))。
// 直接调用：message.content .filter((block) => block.type === 'toolCall') .map。
// 直接调用：message.content .filter。
// [/maint-note:9a9a26708ef1]
export function toolCallsOf(message: ModelAssistantMessage): Array<{ id: string; name: string }> {
  return message.content
    .filter((block) => block.type === 'toolCall')
    .map((block) => ({ id: block.id, name: block.name }))
}

// [maint-note:52c6a0e3f618]
// 维护导航：userText；下列为本函数源码中的接口与路径索引。
// 输入 message：ModelUserMessage。
// 返回契约：string。
// 返回路径：message.content .filter((block) => block.type === 'text') .map((block) => block.text) .join('')。
// 直接调用：message.content .filter((block) => block.type === 'text') .map((block) => block.text) .join。
// 直接调用：message.content .filter((block) => block.type === 'text') .map。
// 直接调用：message.content .filter。
// [/maint-note:52c6a0e3f618]
function userText(message: ModelUserMessage): string {
  return message.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('')
}

// [maint-note:c80283a37f44]
// 维护导航：freezeMessages；下列为本函数源码中的接口与路径索引。
// 输入 messages：readonly ModelMessage[]。
// 返回契约：readonly ModelMessage[]。
// 返回路径：deepFreeze(cloneMessages(messages))。
// 直接调用：deepFreeze。
// 直接调用：cloneMessages。
// [/maint-note:c80283a37f44]
function freezeMessages(messages: readonly ModelMessage[]): readonly ModelMessage[] {
  return deepFreeze(cloneMessages(messages))
}

// [maint-note:05f39773b277]
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
// [/maint-note:05f39773b277]
function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
  return Object.freeze(value)
}
