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

export interface ThreadRunIdentity {
  readonly threadId: string
  readonly generation: string
  readonly runKey: string
  readonly start: number
}

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

export type ThreadTerminalFact =
  | { readonly type: 'done'; readonly workspaceMayHaveChanged: boolean }
  | {
      readonly type: 'paused'
      readonly reason: 'budget_exhausted' | 'no_progress'
      readonly workspaceMayHaveChanged: boolean
    }
  | { readonly type: 'interrupted'; readonly workspaceMayHaveChanged: boolean }
  | { readonly type: 'error'; readonly code: string; readonly workspaceMayHaveChanged: boolean }

export interface ThreadRunHistoryReceipt {
  readonly requestRetained: boolean
  readonly closedExchangeCount: number
  readonly terminalAssistantRetained: boolean
  readonly source: ThreadRunSourceRef | null
}

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

export interface ThreadSourceReader {
  readonly totalChars: number
  read(input: {
    readonly offset: number
    readonly limit: number
  }): { readonly text: string; readonly offset: number; readonly nextOffset: number | null }
}

export type ThreadSourceResolution =
  | { readonly status: 'readable'; readonly reader: ThreadSourceReader }
  | { readonly status: 'unavailable'; readonly reason: string }

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

export interface AgentRequestView {
  readonly messages: readonly ModelMessage[]
  readonly truncated: boolean
  readonly omittedGroupCount: number
  readonly omittedCharsApprox: number
  /** Trusted, request-only framework facts; never persisted as model-authored text. */
  readonly frameworkContext?: string
}

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

interface CompletedThreadRun {
  readonly runKey: string
  readonly start: number
  readonly end: number
  readonly revision: number
  readonly source: ThreadRunSourceRef
  readonly exchangeRanges: readonly ThreadExchangeRange[]
  readonly terminal: ThreadTerminalFact
}

export interface ThreadCloseCandidate extends ThreadCloseReceipt {
  readonly expectedRevision: number
  readonly expectedCursor: number
  readonly remainingMessages: readonly ModelMessage[]
  readonly runStart: number
  readonly runEnd: number
}

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

interface ThreadController {
  begin(prompt: string): ThreadRunController
}

const states = new WeakMap<AgentThread, ThreadState>()

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
    get id() {
      return state.id
    },
    get status() {
      return state.status
    },
    snapshot() {
      return freezeMessages(state.messages)
    },
    requestView(maxHistoryChars, pendingMessages = []) {
      return this.requestViewWithMetadata(maxHistoryChars, pendingMessages).messages
    },
    requestViewWithMetadata(maxHistoryChars, pendingMessages = []) {
      return buildRequestView(state, maxHistoryChars, pendingMessages)
    },
    resolveSource(ref) {
      return resolveThreadSource(state, ref)
    },
    // ------------------------------------------------------------------
    // the Thread owns the DERIVED request-view summary. Freezing
    // reads raw evidence; installing validates and swaps only the derived
    // view. Raw history, revision, cursor and receipts never change here.
    // ------------------------------------------------------------------
    activeCompactionSummary() {
      // Read-only projection: callers can never mutate the installed state.
      return state.activeSummary ? structuredClone(state.activeSummary) : undefined
    },
    freezeCompactionSource(options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      const receipt = freezeCompactionSource(compactionView(state), options)
      // Registering the new lease revokes any earlier candidate: only the
      // newest freeze on the live owner can publish.
      state.compactionLease = receipt.leaseId
      return receipt
    },
    installCompactionSummary(candidate: import('./threadCompaction.js').SealedCompactionCandidate, _unused?: undefined, options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      return installCompactionSummary(compactionView(state), candidate, undefined, options)
    },
    invalidateCompactionSummaryIfStale(options: { readonly pendingClosed?: readonly ModelMessage[] } = {}) {
      if (!summaryStillValid(compactionView(state), options)) state.activeSummary = undefined
    },
    rememberFailedCompactionFrontier(key: string) {
      const existing = state.failedFrontiers ?? []
      if (existing.includes(key)) return
      state.failedFrontiers = [...existing.slice(-7), key]
    },
    hasFailedCompactionFrontier(key: string) {
      return (state.failedFrontiers ?? []).includes(key)
    },
    compactionLeaseValid(leaseId: string) {
      return state.compactionLease === leaseId
    },
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
    finalizeReceiptTrim(receipt, selectedUnitIds) {
      // only the owner's registry can finalize — the engine never
      // writes sources, messages or counts itself.
      return finalizeCompactionReceiptTrim(compactionView(state), receipt, selectedUnitIds)
    },
  }
  states.set(thread, state)
  return Object.freeze(thread)
}

function compactionView(state: ThreadState): ThreadCompactionMutableState {
  return state as unknown as ThreadCompactionMutableState
}

/** @internal Used by AgentRunner to keep thread mutation out of UI/tool code. */
export function getThreadController(thread: AgentThread): ThreadController {
  const state = states.get(thread)
  if (!state) {
    throw new AgentError('config_error', 'The supplied agent thread is invalid')
  }

  return {
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

function createRunController(state: ThreadState, run: MutableThreadRun): ThreadRunController {
  return {
    get identity() {
      return Object.freeze({
        threadId: state.id,
        generation: state.generation,
        runKey: run.runKey,
        start: run.start,
      })
    },
    get runKey() {
      return run.runKey
    },
    get savedCursor() {
      return run.savedCursor
    },
    requestViewWithMetadata(maxHistoryChars, pendingMessages) {
      assertRunOwner(state, run)
      return buildRequestView(state, maxHistoryChars, pendingMessages, run)
    },
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

function assertRunOwner(state: ThreadState, run: MutableThreadRun): void {
  if (state.status !== 'running' || state.activeRun?.token !== run.token) {
    throw new AgentError('config_error', 'The thread run owner is unavailable or stale')
  }
}

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

function cloneMessages(messages: readonly ModelMessage[]): ModelMessage[] {
  return structuredClone([...messages])
}

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

interface HistoricalRunUnits {
  readonly anchor?: ModelMessage
  readonly units: readonly (readonly ModelMessage[])[]
}

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

function messageChars(messages: readonly ModelMessage[]): number {
  return JSON.stringify(messages).length
}

function messageDigest(messages: readonly ModelMessage[]): string {
  return createHash('sha256').update(JSON.stringify(messages)).digest('hex')
}

export function toolCallsOf(message: ModelAssistantMessage): Array<{ id: string; name: string }> {
  return message.content
    .filter((block) => block.type === 'toolCall')
    .map((block) => ({ id: block.id, name: block.name }))
}

function userText(message: ModelUserMessage): string {
  return message.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('')
}

function freezeMessages(messages: readonly ModelMessage[]): readonly ModelMessage[] {
  return deepFreeze(cloneMessages(messages))
}

function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
  return Object.freeze(value)
}
