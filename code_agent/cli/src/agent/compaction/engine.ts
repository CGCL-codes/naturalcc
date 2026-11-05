import type {
  ModelBudgetCapabilities,
  ModelMessage,
  ModelRequestMeasurement,
} from '../../model/types/types.js'
import type { AgentRequestView } from '../thread/AgentThread.js'
import type { FinalRequestProjection } from '../types/frameworkContext.js'
import {
  COMPACTION_LIMITS,
  compactionSoftTarget,
  type CompactionSource,
  type CompactionSourceUnit,
  type CompactionUsage,
  type ProviderBodyResult,
} from './contracts.js'
import { maintenanceRequestMessages, repairRequestMessages } from './provider.js'
import {
  frontierKey,
  parseCompactionBody,
  type FreezeReceipt,
} from '../thread/threadCompaction.js'

/**
 * per-run maintenance coordinator.
 *
 * ONE total ledger: maintenance requests are counted at LOGICAL DISPATCH
 * time (before the await), consume ordinary slots, and are tracked separately
 * for observability. Before every compact/repair attempt the coordinator
 * re-checks: closing state, at least THREE remaining total slots (the normal
 * Agent request plus a potential tool-free closing), cycle/call/repair and
 * cumulative-time budgets. Maintenance never calls guard.beginRound/
 * finishRound, never consumes task reminders, and never fabricates a
 * terminal — failure degrades to the previous legal request view; an
 * external abort installs nothing, repairs nothing and stops the run.
 */

export interface MaintenanceLedger {
  readonly cycles: number
  readonly calls: number
  readonly repairs: number
  readonly maintenanceMs: number
  readonly installedSummaries: number
  readonly lastFailure?: { readonly kind: string; readonly message: string }
  readonly maintenanceFirstCause?: string
  /** Maintenance usage kept independently of install success. */
  readonly usage?: CompactionUsage
}

export interface CompactionPort {
  /** Freeze + maybe run maintenance; returns whether a summary was installed. */
  maintainAtBoundary(input: {
    readonly pendingTail: readonly ModelMessage[]
    readonly signal?: AbortSignal
    readonly remainingRequests: number
    readonly maxHistoryChars: number
    /** The exact next final provider body is over its trusted context window. */
    readonly contextPressure?: boolean
    /** True when the run is already closing / high-priority correction pending. */
    readonly closing: boolean
    /**
     * 77.4 request-only PEEK: a read-only snapshot of the sections the next
     * normal request would carry (framework context, task context, omission
     * notice). It parameterizes the common final serializer measurement —
     * nothing is consumed; the real dispatch assembles and consumes its own.
     */
    readonly requestOnlyPeek?: readonly string[]
    /** complete frozen projection used by current/candidate measurement. */
    readonly requestProjectionPeek?: FinalRequestProjection
    /**
     * Build a complete read-only section set for a concrete current/candidate
     * request view after the coordinator's projected maintenance dispatches.
     */
    readonly prepareRequestOnly?: (
      view: AgentRequestView,
      maintenanceCalls: number,
    ) => readonly string[]
    readonly prepareRequestProjection?: (
      view: AgentRequestView,
      maintenanceCalls: number,
    ) => FinalRequestProjection
    /** unified run admission for each actual maintenance dispatch. */
    readonly prepareDispatch?: (
      kind: 'compaction' | 'repair',
      measurement?: ModelRequestMeasurement,
      capabilities?: ModelBudgetCapabilities,
    ) => {
      validate(): boolean
      publish(): boolean
      settle(usage?: CompactionUsage): void
    }
  }): Promise<boolean>
  readonly ledger: MaintenanceLedger
}

export interface CompactionEngineThread {
  freezeCompactionSource(options?: { readonly pendingClosed?: readonly ModelMessage[] }): FreezeReceipt
  installCompactionSummary(
    candidate: import('../thread/threadCompaction.js').SealedCompactionCandidate,
    _unused?: undefined,
    options?: { readonly pendingClosed?: readonly ModelMessage[] },
  ): boolean
  invalidateCompactionSummaryIfStale(options?: { readonly pendingClosed?: readonly ModelMessage[] }): void
  activeCompactionSummary(): import('./contracts.js').CompactionSummary | undefined
  rememberFailedCompactionFrontier(key: string): void
  hasFailedCompactionFrontier(key: string): boolean
  prepareCompactionCandidate(
    receipt: FreezeReceipt,
    rawBody: string,
    usage?: CompactionUsage,
    requestOnlySections?: readonly string[],
    view?: {
      readonly maxHistoryChars: number
      readonly pendingMessages?: readonly ModelMessage[]
      readonly requestOnlySectionsFor?: (view: AgentRequestView) => readonly string[]
      readonly requestProjection?: FinalRequestProjection
      readonly requestProjectionFor?: (view: AgentRequestView) => FinalRequestProjection
    },
  ): { ok: true; candidate: import('../thread/threadCompaction.js').SealedCompactionCandidate; measurement: import('../thread/threadCompaction.js').CandidateMeasurement } | { ok: false; reason: string }
  /**
   * core finalization of a shell trim. The engine submits only the
   * selected unit IDs of a receipt the core issued; the core rebuilds
   * messages/counts/order from its own record and re-issues the FINAL receipt
   * under the same lease. Undefined = the selection was refused.
   */
  finalizeReceiptTrim(receipt: FreezeReceipt, selectedUnitIds: readonly string[]): FreezeReceipt | undefined
}

type BoundedProvider = { compact(request: {
  readonly source: CompactionSource
  readonly requestMessages: readonly ModelMessage[]
  readonly signal?: AbortSignal
  readonly deadlineMs: number
}): Promise<ProviderBodyResult>;
  prepare?: import('./contracts.js').CompactionProvider['prepare']
  readonly budgetCapabilities?: ModelBudgetCapabilities
}

export function createRunCompactionCoordinator(options: {
  readonly provider: BoundedProvider
  readonly thread: CompactionEngineThread
  readonly now?: () => number
}): CompactionPort {
  const now = options.now ?? Date.now
  let cycles = 0
  let calls = 0
  let repairs = 0
  let maintenanceMs = 0
  let installedSummaries = 0
  let lastFailure: { kind: string; message: string } | undefined
  let maintenanceFirstCause: string | undefined
  let usage: CompactionUsage | undefined

  // every DISPATCH settles its usage fact — including the
  // absence of one. A dispatch without usage marks the ledger unknown and
  // that sticks for the whole run (later known values accumulate but never
  // restore completeness); an empty boundary (no new dispatch) changes
  // nothing.
  const mergeUsage = (incoming: CompactionUsage | undefined): void => {
    if (incoming === undefined) {
      if (usage === undefined) usage = { usageComplete: false }
      else usage = { ...usage, usageComplete: false }
      return
    }
    if (usage === undefined) usage = { ...incoming }
    else {
      usage = {
        inputTokens: (usage.inputTokens ?? 0) + (incoming.inputTokens ?? 0),
        outputTokens: (usage.outputTokens ?? 0) + (incoming.outputTokens ?? 0),
        // the provider-REPORTED total accumulates; it is never inferred
        // from input+output. A dispatch that did not report it adds nothing.
        ...(usage.totalTokens !== undefined || incoming.totalTokens !== undefined
          ? { totalTokens: (usage.totalTokens ?? 0) + (incoming.totalTokens ?? 0) }
          : {}),
        // partial-missing also sticks incomplete.
        usageComplete: usage.usageComplete && incoming.usageComplete,
      }
    }
  }

  const ledger = (): MaintenanceLedger => ({
    cycles,
    calls,
    repairs,
    maintenanceMs,
    installedSummaries,
    ...(usage ? { usage } : {}),
    ...(lastFailure ? { lastFailure } : {}),
    ...(maintenanceFirstCause ? { maintenanceFirstCause } : {}),
  })

  return {
    get ledger() {
      return ledger()
    },
    async maintainAtBoundary(input) {
      // Safety point: the pending tail must be empty, exactly one complete
      // closed exchange, or the CURRENT USER message awaiting its first model
      // request (a legal preparation state — ). The current user is
      // protected and never a compressible unit; its presence must not block
      // maintenance of older closed history.
      const pendingClosed = closedTailOrNull(input.pendingTail)
      const userOnlyPending = pendingClosed === null && input.pendingTail.length === 1 && input.pendingTail[0]?.role === 'user'
      if (pendingClosed === null && !userOnlyPending && input.pendingTail.length > 0) return false
      // No maintenance once the run is closing or in a high-priority
      // correction window; those own the remaining slots.
      if (input.closing) return false

      options.thread.invalidateCompactionSummaryIfStale(pendingClosed ? { pendingClosed } : undefined)

      const freezeInput = pendingClosed ? { pendingClosed } : undefined
      const receipt = options.thread.freezeCompactionSource(freezeInput)
      if (receipt.source.units.length === 0) return false

      // the trigger set is the NEWLY compressible units only —
      // re-read old coverage never re-triggers by itself.
      const newUnits = receipt.source.units.filter((unit) => receipt.newUnitIds.includes(unit.exchangeId))
      if (newUnits.length === 0) return false
      // the soft-target gate measures the FROZEN HISTORY (pre-trim).
      // The shell trim below is a transport constraint on the outbound
      // request; history that already crossed the target must still be
      // maintainable even when only a small prefix survives the trim.
      const rawChars = JSON.stringify(receipt.source.messages).length
      if (!input.contextPressure && rawChars <= compactionSoftTarget(input.maxHistoryChars)) return false

      // real input measurement — the outbound maintenance shell
      // (JSON escaping included) must fit maxInputChars. The engine keeps ONLY
      // the bounded estimation and selection strategy: it picks the fitting
      // unit IDs and asks the CORE to finalize the trim. The core re-issues
      // the receipt from its OWN frozen record (messages, counts, order) under
      // the same lease — that FINAL core-issued receipt is what dispatch,
      // parse, prepare and install all bind to; the engine never writes a
      // source, so the pre-trim receipt is simply never used again.
      const finalReceipt = finalizeTrimAtCore(options.thread, receipt)
      if (finalReceipt === undefined) return false
      const source = finalReceipt.source
      if (source.units.length === 0) return false
      // If the previous summary exists and the trim dropped any of its old
      // coverage, a strict-superset replacement is impossible: keep the old
      // summary (bounded retreat) instead of installing partial coverage.
      const previousCoverage = options.thread.activeCompactionSummary()?.coveredUnits ?? []
      if (previousCoverage.length > 0) {
        const finalIds = new Set(source.units.map((unit) => unit.exchangeId))
        const lostOld = previousCoverage.some((unit) => !finalIds.has(unit.exchangeId))
        if (lostOld) {
          latchFailure('budget_exhausted', 'The input cap would drop previously covered units; keeping the installed summary')
          return false
        }
      }
      // The DISPATCHED selection must still carry new evidence: a trim that
      // drops every newly compressible unit would only re-summarize old
      // coverage — decline and keep the previous legal view.
      if (source.units.every((unit) => !receipt.newUnitIds.includes(unit.exchangeId))) return false

      const frontier = frontierKey(source.units)
      if (options.thread.hasFailedCompactionFrontier(frontier)) return false
      if (cycles >= COMPACTION_LIMITS.maxCyclesPerRun) return false
      if (calls >= COMPACTION_LIMITS.maxMaintenanceCallsPerRun) return false
      // reserve the normal Agent request AND a potential tool-free
      // closing — maintenance may only consume the surplus beyond two slots.
      if (input.remainingRequests < 3) return false
      const remainingMs = COMPACTION_LIMITS.cumulativeTimeoutMs - maintenanceMs
      if (remainingMs <= 0) {
        latchFailure('budget_exhausted', 'The cumulative maintenance time budget is exhausted')
        return false
      }
      const deadlineMs = Math.min(COMPACTION_LIMITS.requestTimeoutMs, remainingMs)
      const requestMessages = maintenanceRequestMessages(source)

      const providerRequest = { source, requestMessages, signal: input.signal, deadlineMs }
      const providerPrepared = options.provider.prepare?.(providerRequest)
      const preparedDispatch = input.prepareDispatch?.(
        'compaction',
        providerPrepared?.measurement,
        options.provider.budgetCapabilities,
      )
      if (preparedDispatch && !preparedDispatch.validate()) return false

      if (preparedDispatch && !preparedDispatch.publish()) return false
      cycles += 1
      // Dispatch-time accounting: the slot is consumed when the request is
      // logically dispatched, not when the await returns.
      calls += 1
      const startedAt = now()
      let outcome = await boundedCompact(
        providerPrepared ? { compact: () => providerPrepared.dispatch() } : options.provider,
        providerRequest,
      )
      maintenanceMs += now() - startedAt
      mergeUsage(outcome.usage)
      preparedDispatch?.settle(outcome.usage)

      // an external abort wins over any late success — nothing is
      // installed, nothing is repaired, and no further work is attempted.
      if (input.signal?.aborted) {
        latchFailure('aborted', 'The run was aborted during maintenance')
        return false
      }

      // a provider ok:true means the model protocol succeeded —
      // the BODY may still be bad JSON. Parse it here; a parse failure is
      // the one repairable class (same frozen source, at most one retry).
      let outcomeNext: ProviderBodyResult = outcome
      if (outcomeNext.ok) {
        const pre = parseCompactionBody(outcomeNext.body, source)
        if (!pre.ok) {
          const canRepair = repairs < COMPACTION_LIMITS.maxRepairsPerSource &&
            calls < COMPACTION_LIMITS.maxMaintenanceCallsPerRun &&
            input.remainingRequests - calls >= 3 &&
            COMPACTION_LIMITS.cumulativeTimeoutMs - maintenanceMs > 0
          // 77.4: only DETERMINISTIC schema failures latch the frontier
          // across runs. A transient repair dispatch (timeout/transport)
          // degrades once and leaves the source retryable later.
          let deterministicSchemaFailure = true
          if (canRepair) {
            const repairMessages = repairRequestMessages(source, outcomeNext.body, pre.reason)
            if (repairMessages === undefined) {
              latchFailure('budget_exhausted', 'The maintenance repair request does not fit the input budget')
              return false
            }
            const repairRequest = {
              source,
              requestMessages: repairMessages,
              signal: input.signal,
              deadlineMs: Math.min(
                COMPACTION_LIMITS.requestTimeoutMs,
                COMPACTION_LIMITS.cumulativeTimeoutMs - maintenanceMs,
              ),
            }
            const providerPreparedRepair = options.provider.prepare?.(repairRequest)
            const preparedRepair = input.prepareDispatch?.(
              'repair',
              providerPreparedRepair?.measurement,
              options.provider.budgetCapabilities,
            )
            if (preparedRepair && !preparedRepair.validate()) return false
            if (preparedRepair && !preparedRepair.publish()) return false
            repairs += 1
            calls += 1
            const retryStarted = now()
            const retry = await boundedCompact(
              providerPreparedRepair
                ? { compact: () => providerPreparedRepair.dispatch() }
                : options.provider,
              repairRequest,
            )
            maintenanceMs += now() - retryStarted
            mergeUsage(retry.usage)
            preparedRepair?.settle(retry.usage)
            if (input.signal?.aborted) {
              latchFailure('aborted', 'The run was aborted during the maintenance repair')
              return false
            }
            if (retry.ok) {
              const reparsed = parseCompactionBody(retry.body, source)
              if (reparsed.ok) {
                return installIfWorthwhile(
                  finalReceipt,
                  retry,
                  source,
                  pendingClosed,
                  input.pendingTail,
                  input.requestOnlyPeek,
                  input.requestProjectionPeek,
                  input.maxHistoryChars,
                  input.prepareRequestOnly,
                  input.prepareRequestProjection,
                )
              }
              latchFailure('invalid_output', reparsed.reason)
            } else {
              latchFailure(retry.kind, retry.message)
              deterministicSchemaFailure = retry.kind === 'invalid_output'
            }
          } else {
            latchFailure('invalid_output', pre.reason)
          }
          if (deterministicSchemaFailure) {
            options.thread.rememberFailedCompactionFrontier(frontier)
          }
          return false
        }
      }

      if (!outcomeNext.ok) {
        // protocol failures (length/tool/envelope/overflow) carry no
        // body to repair — degrade once. Only deterministic structural
        // failures latch the frontier across runs (transport is transient).
        latchFailure(outcomeNext.kind, outcomeNext.message)
        if (outcomeNext.kind === 'invalid_output') {
          options.thread.rememberFailedCompactionFrontier(frontier)
        }
        return false
      }
      outcome = outcomeNext
      if (outcome.ok) return installIfWorthwhile(
        finalReceipt,
        outcome,
        source,
        pendingClosed,
        input.pendingTail,
        input.requestOnlyPeek,
        input.requestProjectionPeek,
        input.maxHistoryChars,
        input.prepareRequestOnly,
        input.prepareRequestProjection,
      )
      return false
    },
  }

  /** same-source net gain — no install when the view would not shrink. */
  function installIfWorthwhile(
    finalReceipt: FreezeReceipt,
    outcome: ProviderBodyResult & { ok: true },
    source: CompactionSource,
    pendingClosed: readonly ModelMessage[] | null,
    requestPendingMessages: readonly ModelMessage[],
    requestOnlyPeek: readonly string[] | undefined,
    requestProjectionPeek: FinalRequestProjection | undefined,
    maxHistoryChars: number,
    prepareRequestOnly: ((view: AgentRequestView, maintenanceCalls: number) => readonly string[]) | undefined,
    prepareRequestProjection: ((view: AgentRequestView, maintenanceCalls: number) => FinalRequestProjection) | undefined,
  ): boolean {
    // the core controller prepares INSIDE its own closure — the
    // engine passes only the frozen receipt and the untrusted body.
    // the engine passes the receipt the CORE issued and still
    // owns; any caller-side rewriting of that object is rejected by the
    // registry before a candidate can exist. 77.4: the boundary's
    // read-only peek parameterizes the shared final serializer — nothing
    // here consumes a reminder or checks in a work batch.
    const prepared = options.thread.prepareCompactionCandidate(
      finalReceipt,
      outcome.body,
      outcome.usage,
      requestOnlyPeek ?? requestProjectionPeek?.requestOnlySections,
      {
        maxHistoryChars,
        // source validation and complete request measurement have
        // different pending domains. Only a closed exchange may participate
        // in the compaction source, while the next ordinary request also
        // includes a not-yet-saved current user message.
        pendingMessages: requestPendingMessages,
        ...(requestProjectionPeek ? { requestProjection: requestProjectionPeek } : {}),
        ...(prepareRequestProjection
          ? { requestProjectionFor: (view: AgentRequestView) => prepareRequestProjection(view, calls) }
          : {}),
        ...(prepareRequestOnly
          ? { requestOnlySectionsFor: (view: AgentRequestView) => prepareRequestOnly(view, calls) }
          : {}),
      },
    )
    if (!prepared.ok) {
      // expected envelope/budget failures are classified by content —
      // an overflow is output_overflow (bounded maintenance retreat), a body
      // problem is invalid_output; neither escapes into a run error.
      const overflow = /envelope|byte cap/i.test(prepared.reason)
      latchFailure(overflow ? 'output_overflow' : 'invalid_output', prepared.reason)
      if (!overflow) options.thread.rememberFailedCompactionFrontier(frontierKey(source.units))
      return false
    }
    // net gain uses the CORE-COMPUTED measurement bound to the sealed
    // candidate — the real serialized summary envelope, never a constant.
    if (prepared.measurement.summaryEnvelopeChars >= prepared.measurement.rawChars) {
      latchFailure('no_growth', 'The summary would not shrink the covered source')
      options.thread.rememberFailedCompactionFrontier(frontierKey(source.units))
      return false
    }
    const installed = options.thread.installCompactionSummary(
      prepared.candidate,
      undefined,
      pendingClosed ? { pendingClosed } : undefined,
    )
    if (installed) installedSummaries += 1
    return installed
  }

  function latchFailure(kind: string, message: string): void {
    lastFailure = { kind, message }
    if (maintenanceFirstCause === undefined) maintenanceFirstCause = kind
  }
}

/**
 * the coordinator is the safety boundary for EVERY
 * CompactionProvider — including app-injected ones. The compact() promise is
 * raced against the external signal and the attempt deadline:
 * - an external abort ends the wait as soon as the signal fires (never
 *   waiting for the provider to cooperate);
 * - a deadline overrun returns a timeout outcome;
 * - a provider that throws synchronously or rejects is DEGRADED to a
 *   transport_error outcome — it never bubbles into a run error;
 * - late resolutions/rejections after the race are absorbed by detached
 *   handlers so nothing leaks and the first cause stays intact.
 */
async function boundedCompact(
  provider: BoundedProvider,
  request: {
    readonly source: CompactionSource
    readonly requestMessages: readonly ModelMessage[]
    readonly signal?: AbortSignal
    readonly deadlineMs: number
  },
): Promise<ProviderBodyResult> {
  const deadline = Math.max(1, request.deadlineMs)
  // every attempt gets its OWN child cancellation domain. The user's
  // run signal is only FORWARDED; a deadline overrun aborts the child so the
  // provider actually observes cancellation, while the parent run stays
  // untouched (a maintenance timeout degrades, never interrupts the run).
  const child = new AbortController()
  const forwardAbort = () => child.abort()
  request.signal?.addEventListener('abort', forwardAbort, { once: true })
  return new Promise<ProviderBodyResult>((resolve) => {
    let settled = false
    const finish = (value: ProviderBodyResult): void => {
      if (settled) return
      settled = true
      if (timer !== undefined) clearTimeout(timer)
      request.signal?.removeEventListener('abort', onAbort)
      // Detached late handlers: the child is aborted either way; a late
      // resolve/reject is absorbed and never changes the settled outcome.
      if (!child.signal.aborted) child.abort()
      resolve(value)
    }
    const timer = setTimeout(() => {
      child.abort()
      finish({ ok: false, kind: 'timeout', message: 'The maintenance request exceeded its deadline' })
    }, deadline)
    const onAbort = () => {
      finish({ ok: false, kind: 'aborted', message: 'The maintenance request was aborted' })
    }
    if (request.signal?.aborted) {
      child.abort()
      finish({ ok: false, kind: 'aborted', message: 'The maintenance request was aborted' })
      return
    }
    request.signal?.addEventListener('abort', onAbort, { once: true })
    try {
      Promise.resolve(provider.compact({ ...request, signal: child.signal })).then(
        (outcome) => finish(outcome),
        (error) => finish({
          ok: false,
          kind: 'transport_error',
          message: error instanceof Error ? error.message : String(error),
        }),
      )
    } catch (error) {
      finish({
        ok: false,
        kind: 'transport_error',
        message: error instanceof Error ? error.message : String(error),
      })
    }
  }).finally(() => {
    request.signal?.removeEventListener('abort', forwardAbort)
  })
}

/**
 * the engine keeps ONLY the bounded shell estimation and selection
 * strategy. It measures the real outbound shell of a candidate prefix by
 * reading the frozen receipt (measurement never mutates anything), picks the
 * fitting contiguous unit IDs, and asks the CORE to finalize the trim. A
 * no-op selection returns the original receipt unchanged; undefined means the
 * core refused the selection (not issued / not a closed span) and maintenance
 * degrades without locking the frontier.
 */
function finalizeTrimAtCore(
  thread: CompactionEngineThread,
  receipt: FreezeReceipt,
): FreezeReceipt | undefined {
  const selectedIds = selectUnitsForInputShell(receipt)
  if (selectedIds.length === receipt.source.units.length) return receipt
  return thread.finalizeReceiptTrim(receipt, selectedIds)
}

/**
 * Bounded selection: the unit IDs of a contiguous PREFIX of the issued
 * canonical order whose real outbound shell fits maxInputChars. The newest
 * compressible units are dropped from the tail first — old (re-read)
 * coverage sits at the front, so a strict-superset replacement stays possible.
 */
function selectUnitsForInputShell(receipt: FreezeReceipt): readonly string[] {
  if (shellChars(receipt, receipt.source.units.length) <= COMPACTION_LIMITS.maxInputChars) {
    return receipt.source.units.map((unit) => unit.exchangeId)
  }
  let keep = receipt.source.units.length - 1
  while (keep > 0 && shellChars(receipt, keep) > COMPACTION_LIMITS.maxInputChars) keep -= 1
  return receipt.source.units.slice(0, keep).map((unit) => unit.exchangeId)
}

/** Real shell measurement of the first `keep` units of the issued receipt. */
function shellChars(receipt: FreezeReceipt, keep: number): number {
  const units = receipt.source.units.slice(0, keep)
  const messages: ModelMessage[] = []
  let cursor = 0
  for (let index = 0; index < receipt.source.units.length; index += 1) {
    const count = receipt.unitMessageCounts[index] ?? 0
    if (index < keep) messages.push(...receipt.source.messages.slice(cursor, cursor + count))
    cursor += count
  }
  return JSON.stringify(maintenanceRequestMessages({ ...receipt.source, units, messages })).length
}

function closedTailOrNull(tail: readonly ModelMessage[]): readonly ModelMessage[] | null {
  if (tail.length === 0) return []
  const first = tail[0]
  if (!first) return null
  // Either the user-anchored batch (first exchange of a run) or the
  // assistant-tool-call head of a later already-saved batch.
  if (first.role === 'user') {
    const rest = tail.slice(1)
    if (rest.length === 0) return null
    return rest[0]?.role === 'assistant' ? tail : null
  }
  if (first.role === 'assistant') return tail
  return null
}
