import type { AgentRuntime } from './types/types.js'
import type { AgentEvent, AgentErrorCode } from './types/event.js'
import type {
  ModelAssistantMessage,
  ModelClient,
  ModelContentBlock,
  ModelResponse,
  ModelRequest,
  PreparedModelDispatch,
  ModelToolCall,
  ModelToolSchema,
  ModelMessage,
  ModelToolResultMessage,
} from '../model/types/types.js'
import {
  ToolInputError,
  type ToolCall,
  type ToolDisplayFacts,
  type ToolExecutionResult,
  type ToolSchema,
} from '../tools/types.js'
import type {
  AgentThread,
  ThreadCloseCandidate,
  ThreadRunController,
  ThreadTerminalFact,
} from './thread/AgentThread.js'
import { errorMessage, isAbortError, AgentError } from './errors.js'
import { boundedErrorContent, errorDiagnostic, formatToolError, toolDiagnostic } from '../tools/diagnostics.js'
import { getThreadController } from './thread/AgentThread.js'
import { createRunCompactionCoordinator, type CompactionPort } from './compaction/engine.js'
import { buildFinalRequestMessages, omissionRequestNotice } from './compaction/viewSerializer.js'
import { ModelProtocolError, ModelResponseLimitError } from '../model/errors.js'
import {
  detectToolControlEnvelope,
  StreamingControlEnvelopeGate,
} from '../model/controlEnvelope.js'
import {
  createToolPolicy,
  validateApprovalInput,
  validateApprovalSchema,
} from '../tools/approval.js'
import {
  DEFAULT_MODEL_RESPONSE_LIMITS,
  ModelStreamAccumulator,
  type ModelResponseLimits,
  validateModelResponse,
} from '../model/stream.js'
import { randomUUID } from 'node:crypto'
import { ToolAttemptLedger } from './toolAttemptLedger.js'
import {
  ToolBatchExecutor,
  type ToolBatchUpdate,
} from './toolBatchExecutor.js'
import {
  RunLoopGuard,
  createRunLoopPolicy,
  type RunLoopPolicy,
} from './runLoopPolicy.js'
import type {
  PreparedRunTaskSettlement,
  RunTaskPort,
  RunTaskTerminal,
} from '../task/contracts.js'
import type {
  FinalRequestProjection,
  FrameworkContextPort,
  PreparedFrameworkContext,
} from './types/frameworkContext.js'
import {
  RunBudgetLedger,
  type BudgetDispatchReceipt,
  type ModelDispatchKind,
  type RunBudgetRejection,
} from './runBudget.js'

/** Maximum raw value accepted from a custom approval request-id factory. */
export const MAX_APPROVAL_FACTORY_VALUE_CHARS = 256

export interface RunTurnInput {
  thread: AgentThread
  prompt: string
  runtime: AgentRuntime
  signal?: AbortSignal
  /** Session-owned, run-scoped task state. Direct Runner callers may omit it. */
  taskRun?: RunTaskPort
  /** Session-owned frozen framework facts; direct Runner callers may omit it. */
  frameworkContext?: FrameworkContextPort
}

export interface AgentRunnerOptions {
  modelResponseLimits?: ModelResponseLimits
  runLoopPolicy?: Partial<RunLoopPolicy>
  onRunMetrics?: (metrics: RunMetrics) => void
  /** @internal Repeat-failure guard test seam; production runs keep this enabled. */
  r6a?: { enabled?: boolean }
  /** Deterministic elapsed-admission clock for tests. */
  now?: () => number
}

export interface RunMetrics {
  modelRequestCount: number
  toolCallCount: number
  approvalCount: number
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  usageComplete: boolean
  /** run-local budget projection. */
  segmentModelRequestCount?: number
  segmentRenewalCount?: number
  observedToolCallCount?: number
  admittedToolAttemptCount?: number
  reservedTokens?: number
  knownActualTokens?: number
  unknownTokenDebt?: number
  budgetOverspent?: boolean
  /** maintenance breakdown; agent + maintenance = modelRequestCount. */
  maintenanceRequestCount?: number
  compactionInstalledSummaries?: number
  compactionLastFailure?: { readonly kind: string; readonly message: string }
}

export class AgentRunner {
  private readonly modelResponseLimits: ModelResponseLimits
  private readonly runLoopPolicy?: Partial<RunLoopPolicy>
  private readonly onRunMetrics?: (metrics: RunMetrics) => void
  private readonly r6aEnabled: boolean
  private readonly now: () => number

  constructor(options: AgentRunnerOptions = {}) {
    this.modelResponseLimits = options.modelResponseLimits ?? DEFAULT_MODEL_RESPONSE_LIMITS
    this.runLoopPolicy = options.runLoopPolicy
    this.onRunMetrics = options.onRunMetrics
    this.r6aEnabled = options.r6a?.enabled ?? true
    this.now = options.now ?? Date.now
  }

  async *runTurn(input: RunTurnInput): AsyncGenerator<AgentEvent> {
    let started = false
    let finalized = false
    let sideEffectToolExecuted = false
    const pendingMessages: ModelMessage[] = [
      { role: 'user', content: textContent(input.prompt) },
    ]
    const metrics: RunMetrics = {
      modelRequestCount: 0,
      toolCallCount: 0,
      approvalCount: 0,
      usageComplete: true,
    }
    const budget = new RunBudgetLedger(
      input.runtime.loop.maxTurns,
      input.runtime.loop.runBudget,
      this.now,
    )
    // Provider call IDs are only meaningful within one turn. Keep a turn-wide
    // set so a later model response cannot accidentally reuse a terminal UI
    // correlation key for a new tool batch.
    const usedCallIds = new Set<string>()
    const attemptLedger = new ToolAttemptLedger(
      Math.max(1, Math.min(4_096, input.runtime.loop.maxTurns * this.modelResponseLimits.maxToolCalls)),
    )
    let threadRun: ThreadRunController | undefined
    let activeBatch: ActiveToolBatch | undefined
    let guard: RunLoopGuard | undefined
    let terminalIntent: AgentEvent | undefined

    try {
      if (!input.prompt.trim()) {
        yield errorEvent(
          'config_error',
          'A non-empty prompt is required',
          false,
        )
        return
      }
      validateRuntimeBudget(input.runtime)

      threadRun = getThreadController(input.thread).begin(input.prompt)
      started = true
      const turnPolicy = input.runtime.toolPolicy ?? createToolPolicy('deny')
      turnPolicy.beginTurn?.()
      const fallbackTurnId = randomUUID()
      const factoryRequestIds = new Map<string, string>()

      input.taskRun?.begin({ threadRun: threadRun.identity })
      yield { type: 'user_message', content: input.prompt }

      const toolSchemas = validateToolSchemas(input.runtime.tools.schemas())
      const policy = createRunLoopPolicy(input.runtime.loop.maxTurns, this.runLoopPolicy)
      guard = this.r6aEnabled ? new RunLoopGuard(policy) : undefined
      let recoveryPending = false
      let softReminderUsed = false
      let pauseReason: 'budget_exhausted' | 'no_progress' | undefined
      let closingRetryUsed = false
      let closingRetryPending = false
      let completionCorrectionUsed = false
      let completionCorrectionPending = false
      let eligibleProgressBaseline = 0
      // one maintenance coordinator per run; it consumes ordinary
      // request slots from the SAME total ledger and never touches the loop
      // guard, task reminders, or the run terminal.
      let compaction: CompactionPort | undefined = input.runtime.compactionProvider
        ? createRunCompactionCoordinator({
          provider: input.runtime.compactionProvider,
          thread: input.thread,
        })
        : undefined
      /**
       * Read-only preparation for every request-only section. The returned
       * acknowledgement is the sole mutation seam and is invoked exactly at
       * a normal logical dispatch; maintenance previews are simply discarded.
       */
      const prepareRequestOnly = (
        requestView: ReturnType<ThreadRunController['requestViewWithMetadata']>,
        projectedRequestCount: number,
      ): {
        readonly sections: readonly string[]
        readonly closing: boolean
        validate(): boolean
        publish(): void
      } => {
        const sections: string[] = []
        const validations: Array<() => boolean> = []
        const publications: Array<() => void> = []
        if (requestView.frameworkContext) sections.push(requestView.frameworkContext)
        if (input.taskRun?.started) sections.push(input.taskRun.requestContext())
        if (requestView.truncated) {
          sections.push(omissionRequestNotice(requestView.omittedGroupCount, requestView.omittedCharsApprox))
        }

        let higherPriority = false
        if (pauseReason) {
          higherPriority = true
          sections.push(pauseCheckpointInstruction(pauseReason))
          if (closingRetryPending) {
            sections.push(plainTextClosingInstruction())
            validations.push(() => closingRetryPending)
            publications.push(() => { closingRetryPending = false })
          }
        } else if (completionCorrectionPending) {
          higherPriority = true
          sections.push(missingCompletionClaimInstruction())
          validations.push(() => completionCorrectionPending)
          publications.push(() => { completionCorrectionPending = false })
        } else if (recoveryPending && guard) {
          const recoveryGuard = guard
          higherPriority = true
          sections.push(runtimeRecoveryInstruction(recoveryGuard.recoverySummary()))
          validations.push(() => recoveryPending && recoveryGuard.shouldRequestRecovery())
          publications.push(() => {
            recoveryGuard.consumeRecoveryPrompt()
            recoveryPending = false
          })
        } else if (
          this.r6aEnabled &&
          !softReminderUsed &&
          policy.softTurnThreshold > 0 &&
          projectedRequestCount >= policy.softTurnThreshold
        ) {
          higherPriority = true
          sections.push(softProgressInstruction())
          validations.push(() => !softReminderUsed)
          publications.push(() => { softReminderUsed = true })
        }

        if (!pauseReason && !higherPriority) {
          const reminder = input.taskRun?.preparePlanReminder()
          if (reminder) {
            sections.push(reminder.text)
            validations.push(() => input.taskRun?.validatePlanReminder(reminder) ?? false)
            publications.push(() => {
              input.taskRun?.publishPlanReminder(reminder)
            })
          }
        }

        let published = false
        return {
          sections: Object.freeze(sections),
          closing: pauseReason !== undefined,
          validate: () => !published && validations.every((validate) => validate()),
          publish() {
            if (published) return
            published = true
            for (const publish of publications) publish()
          },
        }
      }

      while (metrics.modelRequestCount < input.runtime.loop.maxTurns) {
        throwIfAborted(input.signal)
        // One clock observation for this preparation boundary. Every
        // current/candidate measurement below reuses this frozen value.
        let preparedFramework = freezeFrameworkContext(input.frameworkContext)
        // safe point: the previous complete tool batch is backfilled
        // (the pending tail is empty or one complete closed exchange) before
        // the next model request is prepared. Maintenance failure degrades
        // to the previous legal view — it never blocks the request.
        if (compaction) {
          const maintenanceClosing = pauseReason !== undefined || completionCorrectionPending || closingRetryPending
          // 77.4 request-only PEEK: a read-only snapshot of the sections the
          // NEXT normal request would carry (framework context, task context,
          // omission notice). It only parameterizes the common final
          // serializer's net-gain measurement; nothing is consumed here —
          // Plan reminders and one-shot guard sections are acknowledged only
          // at the real dispatch (the "dispatch ack").
          const peekView = threadRun.requestViewWithMetadata(
            input.runtime.loop.maxHistoryChars,
            pendingMessages.slice(threadRun.savedCursor),
          )
          const requestProjectionPeek = finalProjection(
            prepareRequestOnly(peekView, metrics.modelRequestCount).sections,
            preparedFramework,
          )
          // The history soft target is not the only useful trigger.
          // Probe the exact next final provider body without publishing any
          // request-only/framework receipt; trusted context pressure may ask
          // the same A coordinator to compact legal closed source once.
          const contextPressure = maintenanceClosing
            ? false
            : budget.exceedsContextWindow(
                prepareModelDispatch(
                  input.runtime.modelClient,
                  createModelRequest(
                    input,
                    buildFinalRequestMessages(peekView.messages, requestProjectionPeek),
                    toModelToolSchemas(toolSchemas),
                  ),
                ).measurement,
                input.runtime.modelClient.budgetCapabilities,
              )
          const callsBeforeMaintenance = compaction.ledger.calls
          await compaction.maintainAtBoundary({
            pendingTail: pendingMessages.slice(threadRun.savedCursor),
            signal: input.signal,
            remainingRequests: budget.remainingAbsoluteRequests,
            maxHistoryChars: input.runtime.loop.maxHistoryChars,
            contextPressure,
            closing: maintenanceClosing,
            requestProjectionPeek,
            prepareRequestProjection: (view, _projectedMaintenanceCalls) => finalProjection(
              prepareRequestOnly(
                view,
                metrics.modelRequestCount,
              ).sections,
              preparedFramework,
            ),
            prepareDispatch: (kind, measurement, capabilities) => {
              const candidate = budget.prepareDispatch({
                kind,
                closing: false,
                measurement,
                capabilities,
              })
              let receipt: BudgetDispatchReceipt | undefined
              return {
                validate: () => candidate.validate() === undefined,
                publish: () => {
                  receipt = candidate.publish()
                  if (!receipt) return false
                  syncBudgetMetrics(metrics, budget)
                  return true
                },
                settle: (usage) => {
                  if (!receipt) return
                  budget.settleDispatch(receipt, usage && {
                    inputTokens: usage.inputTokens,
                    outputTokens: usage.outputTokens,
                    totalTokens: usage.totalTokens,
                  }, { usageComplete: usage?.usageComplete })
                  syncBudgetMetrics(metrics, budget)
                },
              }
            },
          })
          syncCompactionMetrics(metrics, compaction)
          syncBudgetMetrics(metrics, budget)
          // an abort during maintenance ends the run here — no normal
          // request is dispatched from this iteration.
          throwIfAborted(input.signal)
          if (metrics.modelRequestCount >= input.runtime.loop.maxTurns) break
          // Maintenance dispatches consumed time and a request slot. Re-read
          // the clock and rebuild the final ordinary projection even when no
          // summary was installed.
          if (compaction.ledger.calls > callsBeforeMaintenance) {
            preparedFramework = freezeFrameworkContext(input.frameworkContext)
          }
        }
        // A maintenance request is not an Agent work round. Start the guard
        // only after maintenance has yielded to an actual normal request.
        guard?.beginRound()
        const requestView = threadRun.requestViewWithMetadata(
          input.runtime.loop.maxHistoryChars,
          pendingMessages.slice(threadRun.savedCursor),
        )
        const preparedRequest = prepareRequestOnly(requestView, metrics.modelRequestCount)
        const requestOnlySections = preparedRequest.sections
        const projection = finalProjection(requestOnlySections, preparedFramework)
        const closing = preparedRequest.closing
        let response: ModelResponse | undefined
        const requestMessages = buildFinalRequestMessages(requestView.messages, projection)
        const request = createModelRequest(
          input,
          requestMessages,
          closing ? [] : toModelToolSchemas(toolSchemas),
          closing ? 'none' : undefined,
        )
        const preparedModel = prepareModelDispatch(input.runtime.modelClient, request)
        const budgetInput = {
          kind: closing ? 'closing' : 'agent',
          closing,
          measurement: preparedModel.measurement,
          capabilities: input.runtime.modelClient.budgetCapabilities,
          firstAgentCompatibility: !closing && (
            !this.r6aEnabled ||
            (metrics.modelRequestCount === 0 && input.runtime.loop.maxTurns === 1)
          ),
        } as const
        let budgetCandidate = budget.prepareDispatch(budgetInput)
        throwIfAborted(input.signal)
        if (!preparedRequest.validate()) {
          throw new AgentError('model_protocol_error', 'The prepared request-only context became stale before dispatch')
        }
        if (!validateFramework(preparedFramework)) {
          throw new AgentError(
            'model_protocol_error',
            'The prepared framework context became stale before dispatch',
          )
        }
        let budgetError = budgetCandidate.validate()
        if (
          budgetError?.reason === 'segment_requests' &&
          !closing &&
          this.r6aEnabled
        ) {
          const eligible = guard?.eligibleProgressReceipt().serial ?? 0
          const capability = input.runtime.modelClient.budgetCapabilities
          const trustedContext = Boolean(
            input.runtime.modelClient.prepare &&
            capability?.contextWindowTokens &&
            (capability.inputTokenEstimation === 'exact_tokenizer' || capability.inputTokenEstimation === 'provider_estimate') &&
            capability.outputLimit?.covers === 'completion_and_reasoning' &&
            input.runtime.loop.runBudget?.maxOutputTokensPerRequest,
          )
          const renewalError = eligible > eligibleProgressBaseline && trustedContext
            ? budget.previewAfterSegmentRenewal({
                kind: budgetInput.kind,
                closing: false,
                measurement: preparedModel.measurement,
                capabilities: input.runtime.modelClient.budgetCapabilities,
              })
            : budgetError
          if (!renewalError && budget.renewSegment()) {
            eligibleProgressBaseline = eligible
            syncBudgetMetrics(metrics, budget)
            budgetCandidate = budget.prepareDispatch(budgetInput)
            budgetError = budgetCandidate.validate()
          } else {
            pauseReason = eligible > eligibleProgressBaseline ? 'budget_exhausted' : 'no_progress'
            continue
          }
        }
        if (
          budgetError &&
          !closing &&
          metrics.modelRequestCount > 0 &&
          canAttemptBudgetClosing(budgetError, budget)
        ) {
          // The denied ordinary candidate was never published. Rebuild the
          // next iteration as a real tools-disabled closing request and put
          // that final body through the same prepare/validate/commit path.
          pauseReason = 'budget_exhausted'
          continue
        }
        if (budgetError) throw new AgentError(budgetError.code, budgetError.message)
        throwIfAborted(input.signal)
        // No await is permitted in this publish segment. Every candidate has
        // already validated and publish cannot reject.
        preparedRequest.publish()
        publishFramework(preparedFramework)
        const dispatchReceipt = budgetCandidate.publish()
        if (!dispatchReceipt) throw new AgentError('model_protocol_error', 'The prepared budget reservation became stale before dispatch')
        syncBudgetMetrics(metrics, budget)
        try {
          for await (const update of consumePreparedModel(
            preparedModel,
            input.signal,
            this.modelResponseLimits,
            !closing && toolSchemas.length > 0,
            !closing && input.taskRun?.completionGate() !== 'missing_claim',
          )) {
            if (update.type === 'assistant_delta') {
              yield update
            } else {
              response = update.response
            }
          }
        } catch (error) {
          // Preserve any validated usage that arrived before a later stream
          // failure. Only the still-unobserved axes remain unknown.
          const observedUsage = modelUsageFromError(error)
          if (observedUsage) budget.settleDispatch(dispatchReceipt, observedUsage)
          else budget.settleUnknown(dispatchReceipt)
          syncBudgetMetrics(metrics, budget)
          throw error
        }
        if (!response) {
          metrics.usageComplete = false
          throw new AgentError(
            'model_protocol_error',
            'Model stream ended without a complete response',
          )
        }

        budget.settleDispatch(dispatchReceipt, response.usage)
        syncBudgetMetrics(metrics, budget)

        if (closing) {
          if (response.toolCalls.length > 0 || response.stopReason !== 'stop') {
            throw new AgentError(
              response.toolCalls.length > 0 ? 'model_protocol_error' : 'model_incomplete',
              response.toolCalls.length > 0
                ? 'The pause checkpoint request must not call tools'
                : 'The pause checkpoint response did not finish normally',
            )
          }
          if (!response.content.trim()) {
            throw new AgentError('model_protocol_error', 'Model returned an empty pause checkpoint')
          }
          // toolChoice/empty tools/request-only wording are provider
          // constraints, not a guarantee. A tool-free closing response whose
          // whole shape is a provider tool-control envelope is a protocol
          // failure, never a checkpoint: it is not parsed, canonicalized, or
          // executed here or anywhere downstream.
          if (detectToolControlEnvelope(response.content)) {
            // One plain-text retry is allowed only while the existing hard
            // budget still has a request left. Without budget the pause fails
            // closed instead of fabricating a checkpoint.
            if (!closingRetryUsed && metrics.modelRequestCount < input.runtime.loop.maxTurns) {
              closingRetryUsed = true
              closingRetryPending = true
              continue
            }
            throw new AgentError(
              'model_protocol_error',
              'The pause checkpoint response was a provider tool-control envelope, not a summary',
            )
          }
          // The Runner still owns the canonical Assistant checkpoint in the
          // retained thread, but the event stream has exactly one carrier of
          // that text: the paused terminal.
          pendingMessages.push(normalizeAssistantMessage(response, []))
          const terminal = finalizeRun({
            type: 'paused',
            reason: pauseReason!,
            content: response.content,
            workspaceMayHaveChanged: sideEffectToolExecuted,
          }, threadRun, pendingMessages, input.taskRun, sideEffectToolExecuted)
          finalized = true
          started = false
          yield terminal
          return
        }

        if (response.toolCalls.length > 0) {
          if (response.stopReason !== 'toolUse') {
            if (
              response.stopReason === 'length' ||
              response.stopReason === 'contentFilter'
            ) {
              throw new AgentError(
                'model_incomplete',
                response.stopReason === 'length'
                  ? 'Model response was truncated by the length limit'
                  : 'Model response was stopped by the content filter',
              )
            }
            throw new AgentError(
              'model_protocol_error',
              `Model returned tool calls with stop reason: ${response.stopReason}`,
            )
          }

          const toolBatch = normalizeToolCalls(response.toolCalls, usedCallIds)
          budget.observeToolCalls(toolBatch.calls.length)
          const toolBatchAdmitted = budget.admitToolBatch(toolBatch.calls.length)
          const taskBatchErrors = toolBatchAdmitted
            ? input.taskRun?.prepareBatch(toolBatch.calls)
            : undefined
          metrics.toolCallCount += toolBatch.calls.length
          syncBudgetMetrics(metrics, budget)
          const assistantMessage = normalizeAssistantMessage(
            response,
            toolBatch.calls,
          )
          pendingMessages.push(assistantMessage)
          const batchExecutor = new ToolBatchExecutor({
            calls: toolBatch.calls,
            schemas: toolSchemas,
            policy: turnPolicy,
            ledger: attemptLedger,
            signal: input.signal,
            maxOutputChars: input.runtime.toolContext.maxOutputChars,
            batchValidationError: toolBatch.error,
            batchBudgetError: toolBatchAdmitted ? undefined : {
              code: 'tool_budget_exhausted',
              message: 'The complete tool-call batch exceeds the remaining run tool-attempt budget.',
            },
            createRequestId: (callId, ordinal) => createRunScopedApprovalRequestId(
              turnPolicy.createRequestId?.(callId, ordinal, fallbackTurnId),
              fallbackTurnId,
              callId,
              ordinal,
              factoryRequestIds,
            ),
            validate: (call) => {
              const taskError = taskBatchErrors?.get(call.id)
              return taskError
                ? new AgentError(taskError.code, taskError.message, true)
                : getToolValidationError(input, call, toolSchemas)
            },
            execute: async (call) => executeToolCall(input, call, toolSchemas),
            repeatPolicy: guard,
          })
          activeBatch = {
            calls: toolBatch.calls,
            schemas: toolSchemas,
            executor: batchExecutor,
            startCursor: pendingMessages.length - 1,
            observedInvocations: new Set(),
            taskFinished: false,
            taskBatchAdmitted: toolBatchAdmitted,
          }
          // The streamed deltas are a live draft. Once the response has been
          // validated and its tool calls have been canonicalized, publish one
          // stable assistant segment before any tool activity starts. The
          // active batch already exists so iterator return can close it.
          if (response.content.trim()) {
            yield { type: 'assistant_segment', content: response.content }
          }
          for await (const update of batchExecutor.run()) {
            if (update.type === 'event') {
              if (update.event.type === 'approval_required') metrics.approvalCount += 1
              yield update.event
              continue
            }
            if (update.type === 'execution_started') {
              if (observeBatchInvocation(input.taskRun, activeBatch, update.callId)) {
                sideEffectToolExecuted = true
              }
              continue
            }
            const consumed = consumeBatchResult(
              input,
              activeBatch,
              update,
              pendingMessages,
            )
            if (consumed.mutatingInvocation) sideEffectToolExecuted = true
            yield consumed.event
          }
          if (toolBatchAdmitted) input.taskRun?.finishBatch(toolBatch.calls, true)
          activeBatch.taskFinished = true
          appendClosedBatch(threadRun, activeBatch, pendingMessages)
          activeBatch = undefined
          if (guard) {
            guard.finishRound()
            // A no-progress circuit breaker has priority over the ordinary
            // budget checkpoint. Otherwise the final available request is
            // always reserved for a tool-free budget closure; recovery may
            // only consume a request while a later closure still fits.
            if (guard.shouldPause() && metrics.modelRequestCount < input.runtime.loop.maxTurns) {
              pauseReason = 'no_progress'
            } else if (
              input.runtime.loop.maxTurns >= 2 &&
              metrics.modelRequestCount === input.runtime.loop.maxTurns - 1
            ) {
              pauseReason = 'budget_exhausted'
            } else if (guard.shouldRequestRecovery()) {
              recoveryPending = true
            }
          }
          continue
        }

        if (response.stopReason === 'stop') {
          if (!response.content.trim()) {
            throw new AgentError(
              'model_protocol_error',
              'Model returned an empty response',
            )
          }
          if (input.taskRun?.completionGate() === 'missing_claim') {
            if (
              !completionCorrectionUsed &&
              metrics.modelRequestCount < input.runtime.loop.maxTurns - 2
            ) {
              completionCorrectionUsed = true
              completionCorrectionPending = true
              continue
            }
            throw new AgentError(
              'model_protocol_error',
              'The run has an explicit plan but no valid claim_run_complete for its current revision',
            )
          }
          pendingMessages.push(normalizeAssistantMessage(response, []))
          const terminal = finalizeRun(
            { type: 'done', content: response.content },
            threadRun,
            pendingMessages,
            input.taskRun,
            sideEffectToolExecuted,
          )
          finalized = true
          started = false
          if (terminal.type !== 'done') {
            yield terminal
            return
          }
          yield { type: 'assistant_message', content: response.content }
          yield terminal
          return
        }

        if (
          response.stopReason === 'length' ||
          response.stopReason === 'contentFilter'
        ) {
          throw new AgentError(
            'model_incomplete',
            response.stopReason === 'length'
              ? 'Model response was truncated by the length limit'
              : 'Model response was stopped by the content filter',
          )
        }

        throw new AgentError(
          'model_protocol_error',
          `Model returned an unsupported stop reason: ${response.stopReason}`,
        )
      }

      throw new AgentError(
        'max_turns_reached',
        `Reached max turns: ${input.runtime.loop.maxTurns}`,
      )
    } catch (error) {
      const closureEvents: AgentEvent[] = []
      if (activeBatch && threadRun) {
        try {
          const closed = closeActiveBatch(input, threadRun, activeBatch, pendingMessages)
          closureEvents.push(...closed.events)
          if (closed.mutatingInvocation) sideEffectToolExecuted = true
        } catch (closeError) {
          // An invalid/unowned batch is not repairable. Drop only that
          // Assistant draft and preserve earlier closed exchanges plus user.
          pendingMessages.length = activeBatch.startCursor
          if (!(isAbortError(error, input.signal) || error instanceof AgentError)) {
            error = closeError
          }
        } finally {
          if (activeBatch.taskBatchAdmitted && !activeBatch.taskFinished && input.taskRun?.started && !input.taskRun.settled) {
            input.taskRun.finishBatch(activeBatch.calls, false)
            activeBatch.taskFinished = true
          }
          activeBatch = undefined
        }
      }
      let terminal: AgentEvent
      if (isAbortError(error, input.signal)) {
        terminal = {
          type: 'interrupted',
          message: 'Agent turn interrupted.',
          workspaceMayHaveChanged: sideEffectToolExecuted,
        }
      } else if (error instanceof AgentError) {
        terminal = errorEvent(
          error.code,
          error.message,
          error.recoverable,
          sideEffectToolExecuted,
        )
      } else {
        terminal = errorEvent(
          'model_request_error',
          errorMessage(error),
          false,
          sideEffectToolExecuted,
        )
      }
      terminalIntent = terminal
      for (const event of closureEvents) yield event
      if (threadRun && started && !finalized) {
        terminal = finalizeRun(
          terminal,
          threadRun,
          pendingMessages,
          input.taskRun,
          sideEffectToolExecuted,
        )
        terminalIntent = terminal
        finalized = true
        started = false
      }
      yield terminal
    } finally {
      if (threadRun && started && !finalized) {
        if (activeBatch) {
          try {
            const closed = closeActiveBatch(input, threadRun, activeBatch, pendingMessages)
            if (closed.mutatingInvocation) sideEffectToolExecuted = true
          } catch {
            pendingMessages.length = activeBatch.startCursor
          } finally {
            if (activeBatch.taskBatchAdmitted && !activeBatch.taskFinished && input.taskRun?.started && !input.taskRun.settled) {
              input.taskRun.finishBatch(activeBatch.calls, false)
              activeBatch.taskFinished = true
            }
            activeBatch = undefined
          }
        }
        finalizeRun(
          terminalIntent ?? {
            type: 'interrupted',
            message: 'Agent turn interrupted.',
            workspaceMayHaveChanged: sideEffectToolExecuted,
          },
          threadRun,
          pendingMessages,
          input.taskRun,
          sideEffectToolExecuted,
        )
        finalized = true
        started = false
      }
      if (!input.taskRun?.started) {
        input.taskRun?.abandon()
      }
      budget.close()
      syncBudgetMetrics(metrics, budget)
      try {
        this.onRunMetrics?.(metrics)
      } catch {
        // Telemetry must never replace the run's terminal protocol.
      }
    }
  }
}

function canAttemptBudgetClosing(
  rejection: RunBudgetRejection,
  budget: RunBudgetLedger,
): boolean {
  if (budget.snapshot().overspent) return false
  if (rejection.reason === 'context_window' || rejection.reason === 'total_tokens') return true
  if (rejection.reason === 'absolute_requests') return budget.remainingAbsoluteRequests >= 1
  return false
}

function validateRuntimeBudget(runtime: AgentRuntime): void {
  const config = runtime.loop.runBudget
  if (!config) return
  if (config.maxCostMicrousd !== undefined) {
    throw new AgentError(
      'config_error',
      'maxCostMicrousd requires trusted pricing and retry accounting, which are not configured',
    )
  }
  if (config.maxTotalTokens === undefined && config.maxOutputTokensPerRequest === undefined) return
  const capability = runtime.modelClient.budgetCapabilities
  if (
    !runtime.modelClient.prepare ||
    !capability?.contextWindowTokens ||
    capability.inputTokenEstimation !== 'exact_tokenizer' ||
    capability.outputLimit?.covers !== 'completion_and_reasoning'
  ) {
    throw new AgentError(
      'config_error',
      'Hard token budgets require exact input measurement, context metadata, provider output limits, and prepared dispatch support',
    )
  }
}

function freezeFrameworkContext(
  port: FrameworkContextPort | undefined,
): PreparedFrameworkContext {
  if (port) return port.freezeForRequest()
  let acknowledged = false
  return Object.freeze({
    trailingMessages: Object.freeze([]),
    validate(): boolean {
      return !acknowledged
    },
    publish(): void {
      acknowledged = true
    },
    acknowledge(): boolean {
      if (acknowledged) return false
      acknowledged = true
      return true
    },
  })
}

function validateFramework(prepared: PreparedFrameworkContext): boolean {
  return prepared.validate()
}

function publishFramework(prepared: PreparedFrameworkContext): void {
  prepared.publish()
}

function finalProjection(
  requestOnlySections: readonly string[],
  framework: PreparedFrameworkContext,
): FinalRequestProjection {
  return Object.freeze({
    requestOnlySections: Object.freeze([...requestOnlySections]),
    trailingFrameworkMessages: framework.trailingMessages,
  })
}

interface ActiveToolBatch {
  readonly calls: readonly ToolCall[]
  readonly schemas: readonly ToolSchema[]
  readonly executor: ToolBatchExecutor
  readonly startCursor: number
  readonly observedInvocations: Set<string>
  readonly taskBatchAdmitted: boolean
  taskFinished: boolean
}

function observeBatchInvocation(
  taskRun: RunTaskPort | undefined,
  batch: ActiveToolBatch,
  callId: string,
): boolean {
  if (batch.observedInvocations.has(callId)) return false
  const call = batch.calls.find((candidate) => candidate.id === callId)
  const schema = call && batch.schemas.find((candidate) => candidate.name === call.name)
  if (!call) throw new AgentError('model_protocol_error', 'Invocation update has no canonical tool call')
  taskRun?.observeExecutionStarted(call, {
    mutating: schema?.readOnly !== true,
    taskManagement: schema?.taskManagement === true,
  })
  batch.observedInvocations.add(callId)
  return schema?.readOnly !== true && schema?.taskManagement !== true
}

function consumeBatchResult(
  input: RunTurnInput,
  batch: ActiveToolBatch,
  update: Extract<ToolBatchUpdate, { type: 'result' }>,
  pendingMessages: ModelMessage[],
): { readonly event: AgentEvent; readonly mutatingInvocation: boolean } {
  const schema = batch.schemas.find((candidate) => candidate.name === update.call.name)
  const mutatingInvocation = update.receipt.invoked
    ? observeBatchInvocation(input.taskRun, batch, update.call.id)
    : false
  // Construct both projections before mutating Task state or the run buffer.
  const message = toToolResultMessage(update.call, update.result)
  const event = toToolResultEvent(input.runtime.showToolResults, update.call, update.result)
  if (batch.taskBatchAdmitted) {
    input.taskRun?.observeResult(update.call, {
      content: update.result.content,
      isError: update.result.isError,
      code: update.result.code,
      mutating: schema?.readOnly !== true,
      taskManagement: schema?.taskManagement === true,
    })
  }
  pendingMessages.push(message)
  batch.executor.acknowledge(update.receipt)
  return { event, mutatingInvocation }
}

function syncCompactionMetrics(metrics: RunMetrics, compaction: CompactionPort): void {
  const ledger = compaction.ledger
  metrics.maintenanceRequestCount = ledger.calls
  metrics.compactionInstalledSummaries = ledger.installedSummaries
  if (ledger.lastFailure) metrics.compactionLastFailure = ledger.lastFailure
}

function syncBudgetMetrics(metrics: RunMetrics, budget: RunBudgetLedger): void {
  const snapshot = budget.snapshot()
  metrics.modelRequestCount = snapshot.totalModelDispatches
  metrics.segmentModelRequestCount = snapshot.segmentModelDispatches
  metrics.segmentRenewalCount = snapshot.segmentRenewalCount
  metrics.observedToolCallCount = snapshot.observedToolCalls
  metrics.admittedToolAttemptCount = snapshot.admittedToolAttempts
  assignOptionalMetric(metrics, 'reservedTokens', snapshot.reservedTokens)
  assignOptionalMetric(metrics, 'knownActualTokens', snapshot.knownActualTokens)
  assignOptionalMetric(metrics, 'unknownTokenDebt', snapshot.unknownTokenDebt)
  assignOptionalMetric(metrics, 'inputTokens', snapshot.knownInputTokens)
  assignOptionalMetric(metrics, 'outputTokens', snapshot.knownOutputTokens)
  assignOptionalMetric(metrics, 'totalTokens', snapshot.knownActualTokens)
  metrics.budgetOverspent = snapshot.overspent
  metrics.usageComplete = snapshot.usageComplete
}

function assignOptionalMetric(
  metrics: RunMetrics,
  key: 'reservedTokens' | 'knownActualTokens' | 'unknownTokenDebt' | 'inputTokens' | 'outputTokens' | 'totalTokens',
  value: number | undefined,
): void {
  if (value === undefined) delete metrics[key]
  else metrics[key] = value
}

function appendClosedBatch(
  threadRun: ThreadRunController,
  batch: ActiveToolBatch,
  pendingMessages: readonly ModelMessage[],
): void {
  threadRun.appendClosedExchange({
    expectedCursor: threadRun.savedCursor,
    messages: pendingMessages.slice(threadRun.savedCursor),
  })
}

function closeActiveBatch(
  input: RunTurnInput,
  threadRun: ThreadRunController,
  batch: ActiveToolBatch,
  pendingMessages: ModelMessage[],
): { readonly events: readonly AgentEvent[]; readonly mutatingInvocation: boolean } {
  const events: AgentEvent[] = []
  let mutatingInvocation = false
  for (const update of batch.executor.closeInterrupted()) {
    const consumed = consumeBatchResult(input, batch, update, pendingMessages)
    events.push(consumed.event)
    if (consumed.mutatingInvocation) mutatingInvocation = true
  }
  appendClosedBatch(threadRun, batch, pendingMessages)
  if (batch.taskBatchAdmitted && !batch.taskFinished && input.taskRun?.started && !input.taskRun.settled) {
    input.taskRun.finishBatch(batch.calls, false)
    batch.taskFinished = true
  }
  return { events, mutatingInvocation }
}

function finalizeRun(
  requestedTerminal: AgentEvent,
  threadRun: ThreadRunController,
  pendingMessages: readonly ModelMessage[],
  taskRun: RunTaskPort | undefined,
  workspaceMayHaveChanged: boolean,
): AgentEvent {
  let terminal = requestedTerminal
  let prepared: PreparedFinalization
  try {
    prepared = prepareFinalization(
      terminal,
      threadRun,
      pendingMessages,
      taskRun,
      workspaceMayHaveChanged,
    )
  } catch (error) {
    if (terminal.type !== 'error' && terminal.type !== 'interrupted') {
      terminal = errorEvent(
        'model_protocol_error',
        'Agent finalization failed before its terminal state was published.',
        false,
        workspaceMayHaveChanged,
      )
    }
    // The first candidate was preparation-only. Retry exactly once with the
    // fixed minimal error/interruption tail; no tool, model, or old snapshot
    // is replayed. Real owners guarantee this fallback preparation fits.
    prepared = prepareFinalization(
      terminal,
      threadRun,
      errorCloseTail(threadRun, pendingMessages),
      taskRun,
      workspaceMayHaveChanged,
      true,
    )
  }
  publishPreparedFinalization(threadRun, taskRun, prepared)
  return terminal
}

interface PreparedFinalization {
  readonly thread: ThreadCloseCandidate
  readonly task?: PreparedRunTaskSettlement
}

function prepareFinalization(
  terminal: AgentEvent,
  threadRun: ThreadRunController,
  pendingMessages: readonly ModelMessage[],
  taskRun: RunTaskPort | undefined,
  workspaceMayHaveChanged: boolean,
  fallback = false,
): PreparedFinalization {
  const threadTerminal = threadTerminalFromAgentEvent(terminal, workspaceMayHaveChanged)
  const threadCandidate = threadRun.prepareClose({
    terminal: threadTerminal,
    expectedCursor: threadRun.savedCursor,
    remainingCanonicalMessages: terminal.type === 'done' || terminal.type === 'paused'
      ? pendingMessages.slice(threadRun.savedCursor)
      : errorCloseTail(threadRun, pendingMessages),
  })
  let taskCandidate: PreparedRunTaskSettlement | undefined
  if (taskRun?.started && !taskRun.settled) {
    const settleInput = {
      terminal: taskTerminalFromAgentEvent(terminal),
      history: threadCandidate,
      workspaceMayHaveChanged,
    }
    taskCandidate = fallback
      ? taskRun.prepareFallbackSettle(settleInput)
      : taskRun.prepareSettle(settleInput)
    if (!taskCandidate) throw new Error('Task settlement did not produce a candidate')
  }

  return {
    thread: threadCandidate,
    ...(taskCandidate ? { task: taskCandidate } : {}),
  }
}

function publishPreparedFinalization(
  threadRun: ThreadRunController,
  taskRun: RunTaskPort | undefined,
  prepared: PreparedFinalization,
): void {
  // Short synchronous publish segment. Everything above may validate,
  // clone, hash, or size; everything below only installs prepared objects
  // and releases Task before Thread ownership.
  const threadReceipt = threadRun.publishClose(prepared.thread)
  if (prepared.task) taskRun!.publishSettle(prepared.task)
  threadRun.release(threadReceipt)
}

function errorCloseTail(
  threadRun: ThreadRunController,
  pendingMessages: readonly ModelMessage[],
): readonly ModelMessage[] {
  return threadRun.savedCursor === 0 && pendingMessages[0]?.role === 'user'
    ? [pendingMessages[0]]
    : []
}

function threadTerminalFromAgentEvent(
  event: AgentEvent,
  workspaceMayHaveChanged: boolean,
): ThreadTerminalFact {
  switch (event.type) {
    case 'done':
      return { type: 'done', workspaceMayHaveChanged }
    case 'paused':
      return { type: 'paused', reason: event.reason, workspaceMayHaveChanged }
    case 'interrupted':
      return { type: 'interrupted', workspaceMayHaveChanged }
    case 'error':
      return { type: 'error', code: event.code, workspaceMayHaveChanged }
    default:
      throw new Error('Only a terminal AgentEvent can finalize a Thread run')
  }
}

type NormalizedToolResult = {
  content: string
  isError: boolean
  code?: string
  /** UI-local display facts; errors and validation never carry them. */
  display?: ToolDisplayFacts
}

type ModelUpdate =
  | { type: 'assistant_delta'; content: string }
  | { type: 'response'; response: ModelResponse }

const observedModelUsageKey = Symbol('observedModelUsage')
type UsageBearingError = Error & { [observedModelUsageKey]?: ModelResponse['usage'] }

async function* consumePreparedModel(
  prepared: PreparedModelDispatch,
  signal: AbortSignal | undefined,
  limits: ModelResponseLimits,
  rejectTextControlEnvelope = true,
  emitTextDeltas = true,
): AsyncGenerator<ModelUpdate> {
  let observedUsage: ModelResponse['usage']
  try {
    const sent = prepared.send()
    if (prepared.mode === 'complete') {
      if (isAsyncIterable(sent)) throw new ModelProtocolError('Prepared complete dispatch returned a stream')
      throwIfAborted(signal)
      const rawResponse = await sent
      // A complete-only provider may ignore AbortSignal and resolve later.
      // Cancellation wins before validation, projection, history, or usage
      // settlement can observe that late response.
      throwIfAborted(signal)
      const validated = validateModelResponse(rawResponse, limits)
      observedUsage = validated.usage
      if (rejectTextControlEnvelope && detectToolControlEnvelope(validated.content)) {
        throw new ModelProtocolError('Model response returned a tool-control envelope as text content')
      }
      yield { type: 'response', response: validated }
      return
    }
    if (!isAsyncIterable(sent)) throw new ModelProtocolError('Prepared stream dispatch returned a response')
    const stream = sent
    if (!stream || typeof stream[Symbol.asyncIterator] !== 'function') {
      throw new ModelProtocolError('Model client returned an invalid stream')
    }
    const accumulator = new ModelStreamAccumulator(limits)
    const textGate = new StreamingControlEnvelopeGate()
    const iterator = stream[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await iterator.next()
        if (result.done) {
          exhausted = true
          break
        }
        throwIfAborted(signal)
        const text = accumulator.push(result.value)
        observedUsage = accumulator.usageSnapshot() ?? observedUsage
        if (text) {
          const visibleText = textGate.push(text)
          if (emitTextDeltas && visibleText) {
            yield { type: 'assistant_delta', content: visibleText }
          }
        }
      }
    } finally {
      if (!exhausted) {
        try {
          await iterator.return?.()
        } catch {
          // Preserve the protocol, cancellation, or budget error that closed the stream.
        }
      }
    }
    throwIfAborted(signal)
    const response = accumulator.finish()
    // A non-OpenAI ModelClient may not have a provider boundary validator, so
    // keep the same validation-before-presentation invariant in the Runner.
    // Do this before releasing a marker candidate buffered by the gate.
    const controlEnvelope = detectToolControlEnvelope(response.content)
    if (controlEnvelope && rejectTextControlEnvelope) {
      throw new ModelProtocolError(
        'Model response returned a tool-control envelope as text content',
      )
    }
    // Closing requests intentionally return the raw, inert response to the
    // Runner so its bounded retry/fail-closed policy can decide what to do.
    // It must still never become an assistant_delta.
    if (emitTextDeltas && (!controlEnvelope || !rejectTextControlEnvelope)) {
      const trailingText = textGate.release(response.content)
      if (trailingText) yield { type: 'assistant_delta', content: trailingText }
    }
    yield { type: 'response', response }
  } catch (error) {
    if (error instanceof AgentError || isAbortError(error, signal)) {
      throw attachModelUsage(error, observedUsage)
    }
    if (error instanceof ModelProtocolError) {
      throw attachModelUsage(new AgentError('model_protocol_error', error.message), observedUsage)
    }
    if (error instanceof ModelResponseLimitError) {
      throw attachModelUsage(new AgentError('model_incomplete', error.message), observedUsage)
    }
    throw attachModelUsage(new AgentError('model_request_error', errorMessage(error)), observedUsage)
  }
}

function attachModelUsage(error: unknown, usage: ModelResponse['usage']): unknown {
  if (!usage || !(error instanceof Error)) return error
  Object.defineProperty(error, observedModelUsageKey, {
    configurable: false,
    enumerable: false,
    value: Object.freeze({ ...usage }),
    writable: false,
  })
  return error
}

function modelUsageFromError(error: unknown): ModelResponse['usage'] {
  return error instanceof Error
    ? (error as UsageBearingError)[observedModelUsageKey]
    : undefined
}

function createModelRequest(
  input: RunTurnInput,
  messages: readonly ModelMessage[],
  tools: ModelToolSchema[],
  toolChoice?: 'auto' | 'none' | 'required',
): ModelRequest {
  const outputTokenLimit = input.runtime.loop.runBudget?.maxOutputTokensPerRequest
  const safetyMarginTokens = input.runtime.loop.runBudget?.safetyMarginTokens
  return {
    model: input.runtime.model,
    messages: [...messages],
    tools,
    toolChoice,
    signal: input.signal,
    ...(outputTokenLimit === undefined && safetyMarginTokens === undefined
      ? {}
      : { budget: { outputTokenLimit, safetyMarginTokens } }),
  }
}

function prepareModelDispatch(client: ModelClient, request: ModelRequest): PreparedModelDispatch {
  const mode = client.stream ? 'stream' : 'complete'
  try {
    if (client.prepare) return client.prepare(request, mode)
  } catch (error) {
    if (error instanceof ModelProtocolError) {
      throw new AgentError('request_budget_exhausted', error.message)
    }
    throw error
  }
  let sent = false
  return Object.freeze({
    mode,
    measurement: Object.freeze({
      providerShapeId: `generic-${mode}-degraded-v1`,
      serializedBodyBytes: Buffer.byteLength(JSON.stringify({
        model: request.model,
        messages: request.messages,
        tools: request.tools,
        toolChoice: request.toolChoice,
      }), 'utf8'),
      trust: 'unknown' as const,
    }),
    send: () => {
      if (sent) throw new ModelProtocolError('A prepared model dispatch can only be sent once')
      sent = true
      return mode === 'stream' ? client.stream!(request) : client.complete(request)
    },
  })
}

function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')
}

async function executeToolCall(
  input: RunTurnInput,
  call: ToolCall,
  schemas: readonly ToolSchema[],
): Promise<NormalizedToolResult> {
  const validationError = validateToolCall(call, schemas)
  if (validationError) {
    return {
      content: formatToolError(validationError.code, validationError.message, input.runtime.toolContext.maxOutputChars, validationError.diagnostic),
      isError: true,
      code: validationError.code,
    }
  }

  try {
    const result: ToolExecutionResult = await input.runtime.tools.execute(call, {
      ...input.runtime.toolContext,
      signal: input.signal,
      canonicalToolCallId: call.id,
      taskPlan: input.taskRun,
    })
    const isError = result?.isError === true
    return {
      content: isError
        ? boundedErrorContent(normalizeToolOutput(result?.content), result?.errorCode ?? 'tool_execution_error', input.runtime.toolContext.maxOutputChars)
        : normalizeToolOutput(result?.content),
      isError,
      code: isError ? result?.errorCode ?? 'tool_execution_error' : undefined,
      display: isError ? undefined : result?.display,
    }
  } catch (error) {
    if (isAbortError(error, input.signal)) throw error
    return {
      content: formatToolError('tool_execution_error', '', input.runtime.toolContext.maxOutputChars,
        errorDiagnostic(error) ?? toolDiagnostic('unknown_error')),
      isError: true,
      code: 'tool_execution_error',
    }
  }
}

function getToolValidationError(
  input: RunTurnInput,
  call: ToolCall,
  schemas: readonly ToolSchema[],
  batchValidationError?: string,
): AgentError | undefined {
  if (batchValidationError) {
    return new AgentError('tool_validation_error', batchValidationError, true)
  }
  const genericError = validateToolCall(call, schemas)
  if (genericError) return genericError
  const schema = schemas.find((candidate) => candidate.name === call.name)
  if (schema) {
    try {
      validateApprovalInput(schema, call.arguments)
    } catch (error) {
      if (error instanceof ToolInputError) {
        return new AgentError('tool_validation_error', error.message, true,
          error.diagnostic)
      }
      throw error
    }
  }
  try {
    input.runtime.tools.validate?.(call)
  } catch (error) {
    if (error instanceof ToolInputError) {
      return new AgentError('tool_validation_error', error.message, true,
        error.diagnostic)
    }
    throw error
  }
  return undefined
}

function toToolErrorResult(error: AgentError): NormalizedToolResult {
  return {
    content: formatToolError(error.code, error.message, undefined, error.diagnostic),
    isError: true,
    code: error.code,
  }
}

function validateToolSchemas(schemas: ToolSchema[]): ToolSchema[] {
  const names = new Set<string>()
  for (const schema of schemas) {
    if (names.has(schema.name)) {
      throw new AgentError(
        'tool_validation_error',
        `Duplicate tool name: ${schema.name}`,
      )
    }
    names.add(schema.name)
    try {
      validateApprovalSchema(schema)
    } catch (error) {
      if (error instanceof ToolInputError) {
        throw new AgentError('tool_validation_error', error.message)
      }
      throw error
    }
  }
  return schemas
}

function toModelToolSchemas(schemas: readonly ToolSchema[]): ModelToolSchema[] {
  return schemas.map(({ name, description, parameters }) => ({
    name,
    description,
    parameters,
  }))
}

interface NormalizedToolBatch {
  calls: ToolCall[]
  error?: string
}

function normalizeToolCalls(
  calls: readonly ModelToolCall[],
  usedIds: Set<string>,
): NormalizedToolBatch {
  const errors: string[] = []
  const normalizedCalls = calls.map((call, index) => {
    const sourceId = typeof call.id === 'string' ? call.id.trim() : ''
    let id = sourceId
    if (!id) {
      errors.push(`Tool call ${index + 1} is missing a call ID.`)
      id = allocateInvalidCallId(index, usedIds)
    } else if (usedIds.has(id)) {
      errors.push(`Duplicate tool call ID at call ${index + 1}.`)
      id = allocateInvalidCallId(index, usedIds)
    } else {
      usedIds.add(id)
    }

    return normalizeToolCall(call, index, id)
  })

  return {
    calls: normalizedCalls,
    error: errors.length
      ? `Tool call batch validation failed: ${errors.join(' ')}`
      : undefined,
  }
}

function validateToolCall(
  call: ToolCall,
  schemas: readonly ToolSchema[],
): AgentError | undefined {
  if (!call.id || !call.name) {
    return new AgentError(
      'tool_validation_error',
      'Tool call must include a non-empty id and name',
      true,
      toolDiagnostic('invalid_call'),
    )
  }
  if (!schemas.some((schema) => schema.name === call.name)) {
    return new AgentError('tool_not_found', `Unknown tool: ${call.name}`, true, toolDiagnostic('tool_not_found'))
  }
  if (call.argumentsError) {
    let reason = 'invalid_argument_shape'
    if (typeof call.rawArguments === 'string') {
      try { JSON.parse(call.rawArguments) } catch { reason = 'invalid_json' }
    }
    return new AgentError('tool_validation_error', call.argumentsError, true, toolDiagnostic(reason))
  }
  if (!isRecord(call.arguments)) {
    return new AgentError(
      'tool_validation_error',
      `Arguments for tool ${call.name} must be a JSON object`,
      true,
      toolDiagnostic('invalid_argument_shape'),
    )
  }
  return undefined
}

function normalizeToolCall(
  call: ModelToolCall,
  index: number,
  canonicalId?: string,
): ToolCall {
  const normalized: ToolCall = {
    id: canonicalId ?? (typeof call.id === 'string' && call.id
      ? call.id
      : `invalid-call-${index + 1}`),
    name: typeof call.name === 'string' ? call.name : '',
    arguments: isRecord(call.arguments) ? call.arguments : {},
  }
  if (call.rawArguments !== undefined) normalized.rawArguments = call.rawArguments
  if (call.argumentsError !== undefined) normalized.argumentsError = call.argumentsError
  return normalized
}

function normalizeAssistantMessage(
  response: ModelResponse,
  toolCalls: readonly ToolCall[],
): ModelAssistantMessage {
  const content: ModelContentBlock[] = []
  if (response.content) content.push(...textContent(response.content))
  for (const call of toolCalls) content.push(toToolCallBlock(call))

  return {
    role: 'assistant',
    content,
    stopReason: response.stopReason,
  }
}

function toToolCallBlock(
  call: ToolCall,
): Extract<ModelContentBlock, { type: 'toolCall' }> {
  return { type: 'toolCall', ...call }
}

function allocateInvalidCallId(index: number, usedIds: Set<string>): string {
  let suffix = 0
  let candidate = `invalid-call-${index + 1}`
  while (usedIds.has(candidate)) {
    suffix += 1
    candidate = `invalid-call-${index + 1}-${suffix}`
  }
  usedIds.add(candidate)
  return candidate
}

function toToolResultMessage(
  call: ToolCall,
  result: NormalizedToolResult,
): ModelToolResultMessage {
  return {
    role: 'toolResult',
    toolCallId: call.id,
    toolName: call.name,
    content: textContent(result.content),
    isError: result.isError,
  }
}

function toToolResultEvent(
  showToolResults: boolean,
  call: ToolCall,
  result: NormalizedToolResult,
): AgentEvent {
  return {
    type: 'tool_result',
    callId: call.id,
    name: call.name,
    output: showToolResults || result.isError ? result.content : '',
    isError: result.isError,
    ...(result.code === undefined ? {} : { code: result.code }),
    // Display facts follow the same visibility gate as the canonical output:
    // `--no-tool-results` keeps successful diffs out of the UI too.
    display: showToolResults || result.isError ? result.display : undefined,
  }
}

function textContent(text: string): ModelContentBlock[] {
  return text ? [{ type: 'text', text }] : []
}

function normalizeToolOutput(content: unknown): string {
  if (typeof content !== 'string' || !content) {
    return 'Tool completed with no output.'
  }
  return content
}

function runtimeRecoveryInstruction(summary: {
  noProgressRounds: number
  recoveryPrompts: number
  workspaceRevision: number
  errorCodes: string[]
  batchCount: number
}): string {
  const errors = summary.errorCodes.length > 0 ? summary.errorCodes.join(', ') : 'none recorded'
  return `Runtime recovery: ${summary.noProgressRounds} no-progress round(s), ${summary.recoveryPrompts} recovery prompt(s), ${summary.batchCount} recent batch(es), workspace revision ${summary.workspaceRevision}, stable error codes: ${errors}. Inspect the latest verified tool results and errors, check your assumptions, then choose a materially different next action. Do not repeat an identical call without new evidence; if the task is complete or no safe progress remains, provide a concise final response.`
}

function softProgressInstruction(): string {
  return 'Progress check: review verified evidence, avoid repeating an unchanged failed operation, and finish with an honest summary if no further safe progress is possible.'
}

function pauseCheckpointInstruction(reason: 'budget_exhausted' | 'no_progress'): string {
  return `Pause checkpoint (${reason}): do not call tools. Summarize verified results, failed attempts, unfinished items, and the safest next step. This is a checkpoint, not a claim that the task is complete.`
}

function plainTextClosingInstruction(): string {
  return 'The previous closing response was rejected: it was a tool-control envelope rather than a summary. Reply with plain natural-language prose only. Do not emit tool-call markup, control tokens, or any provider protocol envelope; no tool will run.'
}

function missingCompletionClaimInstruction(): string {
  return 'The previous final response was not accepted because this run has an explicit plan without a valid completion claim. Recheck the current trusted run-task projection. If every current stage is complete, call update_task_plan with operation claim_run_complete, the current plan revision, and every current step ID exactly once; then provide a final answer in a later response. Do not invent completion or clear the plan.'
}

function taskTerminalFromAgentEvent(event: AgentEvent): RunTaskTerminal {
  switch (event.type) {
    case 'done':
      return { type: 'done' }
    case 'paused':
      return { type: 'paused', reason: event.reason }
    case 'interrupted':
      return { type: 'interrupted' }
    case 'error':
      return { type: 'error', code: event.code }
    default:
      return { type: 'error', code: 'model_protocol_error' }
  }
}

/**
 * Approval IDs are an internal run protocol, not an authority supplied by a
 * policy factory. The default factory-compatible shape is kept for existing
 * adapters; any other factory value is retained only as a bounded hint after
 * the current run owner has been imposed.
 */
function createRunScopedApprovalRequestId(
  factoryValue: unknown,
  runOwner: string,
  callId: string,
  ordinal: number,
  factoryRequestIds: Map<string, string>,
): string {
  const fallback = `approval-${runOwner}-${ordinal + 1}-${callId}`
  if (factoryValue === undefined) return fallback
  // Check the untrusted raw value before trim, Map insertion, or Unicode
  // expansion. A policy factory is an extension point and must not be able to
  // turn one malformed return value into unbounded run-local state.
  if (
    typeof factoryValue !== 'string' ||
    factoryValue.length > MAX_APPROVAL_FACTORY_VALUE_CHARS ||
    !factoryValue.trim()
  ) return ''
  if (factoryValue === fallback) return fallback

  const normalizedFactoryValue = factoryValue.trim()
  const previousRequestId = factoryRequestIds.get(normalizedFactoryValue)
  if (previousRequestId !== undefined) return previousRequestId

  const hint = [...normalizedFactoryValue]
    .map((character) => /[\p{L}\p{N}._~-]/u.test(character) ? character : '_')
    .slice(0, 80)
    .join('')
  // The factory value is the caller's attempt to identify the request. Keep
  // one value per run so duplicate factory output still collides in the
  // ledger; ordinal/callId are included in the first identity and must not
  // silently turn a duplicate into a new approval. The run owner is the
  // namespace boundary across concurrent turns.
  const requestId = `approval-${runOwner}-${ordinal + 1}-${callId}-factory-${hint || 'custom'}`
  factoryRequestIds.set(normalizedFactoryValue, requestId)
  return requestId
}

function errorEvent(
  code: AgentErrorCode,
  message: string,
  recoverable: boolean,
  sideEffectToolExecuted = false,
): AgentEvent {
  return {
    type: 'error',
    code,
    message,
    recoverable,
    workspaceMayHaveChanged: sideEffectToolExecuted,
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

function createAbortError(): Error {
  const error = new Error('The agent turn was aborted')
  error.name = 'AbortError'
  return error
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
