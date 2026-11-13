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

// [maint-note:4096b33833c8]
// 数据契约导航：RunTurnInput；字段含义以消费端校验为准。
// 成员 thread：AgentThread。
// 成员 prompt：string。
// 成员 runtime：AgentRuntime。
// 成员 signal（可省略）：AbortSignal。
// 成员 taskRun（可省略）：RunTaskPort。
// 成员 frameworkContext（可省略）：FrameworkContextPort。
// [/maint-note:4096b33833c8]
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

// [maint-note:3296ad1146a2]
// 数据契约导航：AgentRunnerOptions；字段含义以消费端校验为准。
// 成员 modelResponseLimits（可省略）：ModelResponseLimits。
// 成员 runLoopPolicy（可省略）：Partial<RunLoopPolicy>。
// 成员 onRunMetrics（可省略）：(metrics: RunMetrics) => void。
// 成员 r6a（可省略）：{ enabled?: boolean }。
// 成员 now（可省略）：() => number。
// [/maint-note:3296ad1146a2]
export interface AgentRunnerOptions {
  modelResponseLimits?: ModelResponseLimits
  runLoopPolicy?: Partial<RunLoopPolicy>
  onRunMetrics?: (metrics: RunMetrics) => void
  /** @internal Repeat-failure guard test seam; production runs keep this enabled. */
  r6a?: { enabled?: boolean }
  /** Deterministic elapsed-admission clock for tests. */
  now?: () => number
}

// [maint-note:42f325e1faa2]
// 数据契约导航：RunMetrics；字段含义以消费端校验为准。
// 成员 modelRequestCount：number。
// 成员 toolCallCount：number。
// 成员 approvalCount：number。
// 成员 inputTokens（可省略）：number。
// 成员 outputTokens（可省略）：number。
// 成员 totalTokens（可省略）：number。
// 成员 usageComplete：boolean。
// 成员 segmentModelRequestCount（可省略）：number。
// 成员 segmentRenewalCount（可省略）：number。
// 成员 observedToolCallCount（可省略）：number。
// 成员 admittedToolAttemptCount（可省略）：number。
// 成员 reservedTokens（可省略）：number。
// 成员 knownActualTokens（可省略）：number。
// 成员 unknownTokenDebt（可省略）：number。
// 成员 budgetOverspent（可省略）：boolean。
// 成员 maintenanceRequestCount（可省略）：number。
// 成员 compactionInstalledSummaries（可省略）：number。
// 成员 compactionLastFailure（可省略）：{ readonly kind: string; readonly message: string }。
// [/maint-note:42f325e1faa2]
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

  // [maint-note:1509a756afa9]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：AgentRunnerOptions；缺省表达式 {}。
  // 写入目标：this.modelResponseLimits。
  // 写入目标：this.runLoopPolicy。
  // 写入目标：this.onRunMetrics。
  // 写入目标：this.r6aEnabled。
  // 写入目标：this.now。
  // [/maint-note:1509a756afa9]
  constructor(options: AgentRunnerOptions = {}) {
    this.modelResponseLimits = options.modelResponseLimits ?? DEFAULT_MODEL_RESPONSE_LIMITS
    this.runLoopPolicy = options.runLoopPolicy
    this.onRunMetrics = options.onRunMetrics
    this.r6aEnabled = options.r6a?.enabled ?? true
    this.now = options.now ?? Date.now
  }

  // [maint-note:f6fd500286e9]
  // 维护导航：runTurn；下列为本函数源码中的接口与路径索引。
  // 输入 input：RunTurnInput。
  // 返回契约：AsyncGenerator<AgentEvent>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!input.prompt.trim()。
  // 分支条件：compaction。
  // 分支条件：metrics.modelRequestCount >= input.runtime.loop.maxTurns。
  // 分支条件：compaction.ledger.calls > callsBeforeMaintenance。
  // 分支条件：!preparedRequest.validate()。
  // 分支条件：!validateFramework(preparedFramework)。
  // 分支条件：budgetError?.reason === 'segment_requests' && !closing && this.r6aEnabled。
  // 分支条件：!renewalError && budget.renewSegment()。
  // 分支条件：budgetError && !closing && metrics.modelRequestCount > 0 && canAttemptBudgetClosing(budgetError, budget)。
  // 分支条件：budgetError。
  // 分支条件：!dispatchReceipt。
  // 分支条件：update.type === 'assistant_delta'。
  // 分支条件：observedUsage。
  // 分支条件：!response。
  // 分支条件：closing。
  // 分支条件：response.toolCalls.length > 0 || response.stopReason !== 'stop'。
  // 分支条件：!response.content.trim()。
  // 分支条件：detectToolControlEnvelope(response.content)。
  // 分支条件：!closingRetryUsed && metrics.modelRequestCount < input.runtime.loop.maxTurns。
  // 分支条件：response.toolCalls.length > 0。
  // 分支条件：response.stopReason !== 'toolUse'。
  // 分支条件：response.stopReason === 'length' || response.stopReason === 'contentFilter'。
  // 分支条件：response.content.trim()。
  // 分支条件：update.type === 'event'。
  // 分支条件：update.event.type === 'approval_required'。
  // 分支条件：update.type === 'execution_started'。
  // 分支条件：observeBatchInvocation(input.taskRun, activeBatch, update.callId)。
  // 分支条件：consumed.mutatingInvocation。
  // 分支条件：toolBatchAdmitted。
  // 分支条件：guard。
  // 分支条件：guard.shouldPause() && metrics.modelRequestCount < input.runtime.loop.maxTurns。
  // 分支条件：input.runtime.loop.maxTurns >= 2 && metrics.modelRequestCount === input.runtime.loop.maxTurns - 1。
  // 分支条件：guard.shouldRequestRecovery()。
  // 分支条件：response.stopReason === 'stop'。
  // 分支条件：input.taskRun?.completionGate() === 'missing_claim'。
  // 分支条件：!completionCorrectionUsed && metrics.modelRequestCount < input.runtime.loop.maxTurns - 2。
  // 分支条件：terminal.type !== 'done'。
  // 分支条件：activeBatch && threadRun。
  // 分支条件：closed.mutatingInvocation。
  // 分支条件：!(isAbortError(error, input.signal) || error instanceof AgentError)。
  // 分支条件：activeBatch.taskBatchAdmitted && !activeBatch.taskFinished && input.taskRun?.started && !input.taskRun.settled。
  // 分支条件：isAbortError(error, input.signal)。
  // 分支条件：error instanceof AgentError。
  // 分支条件：threadRun && started && !finalized。
  // 分支条件：activeBatch。
  // 分支条件：!input.taskRun?.started。
  // 遍历边界：metrics.modelRequestCount < input.runtime.loop.maxTurns。
  // 遍历边界：const update ← consumePreparedModel( preparedModel, input.signal, this.modelResponseLimits, !closing && toolSchemas.length > 0, !closing && input.taskRun?.completionGate() !== 'missing_claim', )。
  // 遍历边界：const update ← batchExecutor.run()。
  // 遍历边界：const event ← closureEvents。
  // 返回路径：无返回值。
  // 显式抛出：new AgentError('model_protocol_error', 'The prepared request-only context became stale before dispatch')。
  // 显式抛出：new AgentError( 'model_protocol_error', 'The prepared framework context became stale before dispatch', )。
  // 显式抛出：new AgentError(budgetError.code, budgetError.message)。
  // 显式抛出：new AgentError('model_protocol_error', 'The prepared budget reservation became stale before dispatch')。
  // 显式抛出：error。
  // 显式抛出：new AgentError( 'model_protocol_error', 'Model stream ended without a complete response', )。
  // 显式抛出：new AgentError( response.toolCalls.length > 0 ? 'model_protocol_error' : 'model_incomplete', response.toolCalls.length > 0 ? 'The pause checkpoint request must not call tools' : 'The pause checkpoint response did not finish normally', )。
  // 显式抛出：new AgentError('model_protocol_error', 'Model returned an empty pause checkpoint')。
  // 显式抛出：new AgentError( 'model_protocol_error', 'The pause checkpoint response was a provider tool-control envelope, not a summary', )。
  // 显式抛出：new AgentError( 'model_incomplete', response.stopReason === 'length' ? 'Model response was truncated by the length limit' : 'Model response was stopped by the content filter', )。
  // 显式抛出：new AgentError( 'model_protocol_error', `Model returned tool calls with stop reason: ${response.stopReason}`, )。
  // 显式抛出：new AgentError( 'model_protocol_error', 'Model returned an empty response', )。
  // 显式抛出：new AgentError( 'model_protocol_error', 'The run has an explicit plan but no valid claim_run_complete for its current revision', )。
  // 显式抛出：new AgentError( 'model_protocol_error', `Model returned an unsupported stop reason: ${response.stopReason}`, )。
  // 显式抛出：new AgentError( 'max_turns_reached', `Reached max turns: ${input.runtime.loop.maxTurns}`, )。
  // 写入目标：threadRun。
  // 写入目标：started。
  // 写入目标：guard。
  // 写入目标：preparedFramework。
  // 写入目标：eligibleProgressBaseline。
  // 写入目标：budgetCandidate。
  // 写入目标：budgetError。
  // 写入目标：pauseReason。
  // 写入目标：response。
  // 写入目标：metrics.usageComplete。
  // 写入目标：closingRetryUsed。
  // 写入目标：closingRetryPending。
  // 写入目标：finalized。
  // 写入目标：metrics.toolCallCount。
  // 写入目标：activeBatch。
  // 写入目标：metrics.approvalCount。
  // 写入目标：sideEffectToolExecuted。
  // 写入目标：activeBatch.taskFinished。
  // 写入目标：recoveryPending。
  // 写入目标：completionCorrectionUsed。
  // 写入目标：completionCorrectionPending。
  // 写入目标：pendingMessages.length。
  // 写入目标：error。
  // 写入目标：terminal。
  // 写入目标：terminalIntent。
  // 直接调用：textContent。
  // 直接调用：Math.max。
  // 直接调用：Math.min。
  // 直接调用：input.prompt.trim。
  // 直接调用：errorEvent。
  // 直接调用：validateRuntimeBudget。
  // 直接调用：getThreadController(input.thread).begin。
  // 直接调用：getThreadController。
  // 直接调用：createToolPolicy。
  // 直接调用：turnPolicy.beginTurn。
  // 直接调用：randomUUID。
  // 直接调用：input.taskRun?.begin。
  // 直接调用：validateToolSchemas。
  // 直接调用：input.runtime.tools.schemas。
  // 直接调用：createRunLoopPolicy。
  // 直接调用：createRunCompactionCoordinator。
  // 直接调用：throwIfAborted。
  // 直接调用：freezeFrameworkContext。
  // 直接调用：threadRun.requestViewWithMetadata。
  // 直接调用：pendingMessages.slice。
  // 直接调用：finalProjection。
  // 直接调用：prepareRequestOnly。
  // 直接调用：budget.exceedsContextWindow。
  // 直接调用：prepareModelDispatch。
  // 直接调用：createModelRequest。
  // 直接调用：buildFinalRequestMessages。
  // 直接调用：toModelToolSchemas。
  // 直接调用：compaction.maintainAtBoundary。
  // 直接调用：syncCompactionMetrics。
  // 直接调用：syncBudgetMetrics。
  // 直接调用：guard?.beginRound。
  // 直接调用：budget.prepareDispatch。
  // 直接调用：preparedRequest.validate。
  // 直接调用：validateFramework。
  // 直接调用：budgetCandidate.validate。
  // 直接调用：guard?.eligibleProgressReceipt。
  // 直接调用：Boolean。
  // 直接调用：budget.previewAfterSegmentRenewal。
  // 直接调用：budget.renewSegment。
  // 直接调用：canAttemptBudgetClosing。
  // 直接调用：preparedRequest.publish。
  // 直接调用：publishFramework。
  // 直接调用：budgetCandidate.publish。
  // 直接调用：consumePreparedModel。
  // 直接调用：input.taskRun?.completionGate。
  // 直接调用：modelUsageFromError。
  // 直接调用：budget.settleDispatch。
  // 直接调用：budget.settleUnknown。
  // 直接调用：response.content.trim。
  // 直接调用：detectToolControlEnvelope。
  // 直接调用：pendingMessages.push。
  // 直接调用：normalizeAssistantMessage。
  // 直接调用：finalizeRun。
  // 直接调用：normalizeToolCalls。
  // 直接调用：budget.observeToolCalls。
  // 直接调用：budget.admitToolBatch。
  // 直接调用：input.taskRun?.prepareBatch。
  // 直接调用：batchExecutor.run。
  // 直接调用：observeBatchInvocation。
  // 直接调用：consumeBatchResult。
  // 直接调用：input.taskRun?.finishBatch。
  // 直接调用：appendClosedBatch。
  // 直接调用：guard.finishRound。
  // 直接调用：guard.shouldPause。
  // 直接调用：guard.shouldRequestRecovery。
  // 直接调用：closeActiveBatch。
  // 直接调用：closureEvents.push。
  // 直接调用：isAbortError。
  // 直接调用：input.taskRun.finishBatch。
  // 直接调用：errorMessage。
  // 直接调用：input.taskRun?.abandon。
  // 直接调用：budget.close。
  // 直接调用：this.onRunMetrics。
  // [/maint-note:f6fd500286e9]
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
      // [maint-note:06f4cc130794]
      // 维护导航：prepareRequestOnly；下列为本函数源码中的接口与路径索引。
      // 输入 requestView：ReturnType<ThreadRunController['requestViewWithMetadata']>。
      // 输入 projectedRequestCount：number。
      // 返回契约：{ readonly sections: readonly string[] readonly closing: boolean validate(): boolean publish(): void }。
      // 分支条件：requestView.frameworkContext。
      // 分支条件：input.taskRun?.started。
      // 分支条件：requestView.truncated。
      // 分支条件：pauseReason。
      // 分支条件：closingRetryPending。
      // 分支条件：completionCorrectionPending。
      // 分支条件：recoveryPending && guard。
      // 分支条件：this.r6aEnabled && !softReminderUsed && policy.softTurnThreshold > 0 && projectedRequestCount >= policy.softTurnThreshold。
      // 分支条件：!pauseReason && !higherPriority。
      // 分支条件：reminder。
      // 返回路径：{ sections: Object.freeze(sections), closing: pauseReason !== undefined, validate: () => !published && validations.every((validate) => validate()), publish() { if (published) return published = true for (const publish of publications) publi。
      // 写入目标：higherPriority。
      // 直接调用：sections.push。
      // 直接调用：input.taskRun.requestContext。
      // 直接调用：omissionRequestNotice。
      // 直接调用：pauseCheckpointInstruction。
      // 直接调用：plainTextClosingInstruction。
      // 直接调用：validations.push。
      // 直接调用：publications.push。
      // 直接调用：missingCompletionClaimInstruction。
      // 直接调用：runtimeRecoveryInstruction。
      // 直接调用：recoveryGuard.recoverySummary。
      // 直接调用：softProgressInstruction。
      // 直接调用：input.taskRun?.preparePlanReminder。
      // 直接调用：Object.freeze。
      // [/maint-note:06f4cc130794]
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
          // [maint-note:b5c20843a6af]
          // 维护导航：publish；下列为本函数源码中的接口与路径索引。
          // 分支条件：published。
          // 遍历边界：const publish ← publications。
          // 返回路径：无返回值。
          // 写入目标：published。
          // 直接调用：publish。
          // [/maint-note:b5c20843a6af]
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

// [maint-note:fc50ae988fb8]
// 维护导航：canAttemptBudgetClosing；下列为本函数源码中的接口与路径索引。
// 输入 rejection：RunBudgetRejection。
// 输入 budget：RunBudgetLedger。
// 返回契约：boolean。
// 分支条件：budget.snapshot().overspent。
// 分支条件：rejection.reason === 'context_window' || rejection.reason === 'total_tokens'。
// 分支条件：rejection.reason === 'absolute_requests'。
// 返回路径：false。
// 返回路径：true。
// 返回路径：budget.remainingAbsoluteRequests >= 1。
// 直接调用：budget.snapshot。
// [/maint-note:fc50ae988fb8]
function canAttemptBudgetClosing(
  rejection: RunBudgetRejection,
  budget: RunBudgetLedger,
): boolean {
  if (budget.snapshot().overspent) return false
  if (rejection.reason === 'context_window' || rejection.reason === 'total_tokens') return true
  if (rejection.reason === 'absolute_requests') return budget.remainingAbsoluteRequests >= 1
  return false
}

// [maint-note:ebd0cef6604a]
// 维护导航：validateRuntimeBudget；下列为本函数源码中的接口与路径索引。
// 输入 runtime：AgentRuntime。
// 返回契约：void。
// 分支条件：!config。
// 分支条件：config.maxCostMicrousd !== undefined。
// 分支条件：config.maxTotalTokens === undefined && config.maxOutputTokensPerRequest === undefined。
// 分支条件：!runtime.modelClient.prepare || !capability?.contextWindowTokens || capability.inputTokenEstimation !== 'exact_tokenizer' || capability.outputLimit?.covers !== 'completion_and_reasoning'。
// 返回路径：无返回值。
// 显式抛出：new AgentError( 'config_error', 'maxCostMicrousd requires trusted pricing and retry accounting, which are not configured', )。
// 显式抛出：new AgentError( 'config_error', 'Hard token budgets require exact input measurement, context metadata, provider output limits, and prepared dispatch support', )。
// [/maint-note:ebd0cef6604a]
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

// [maint-note:95f8c64e8c36]
// 维护导航：freezeFrameworkContext；下列为本函数源码中的接口与路径索引。
// 输入 port：FrameworkContextPort | undefined。
// 返回契约：PreparedFrameworkContext。
// 分支条件：port。
// 返回路径：port.freezeForRequest()。
// 返回路径：Object.freeze({ trailingMessages: Object.freeze([]), validate(): boolean { return !acknowledged }, publish(): void { acknowledged = true }, acknowledge(): boolean { if (acknowledged) return false acknowledged = true return true }, })。
// 直接调用：port.freezeForRequest。
// 直接调用：Object.freeze。
// [/maint-note:95f8c64e8c36]
function freezeFrameworkContext(
  port: FrameworkContextPort | undefined,
): PreparedFrameworkContext {
  if (port) return port.freezeForRequest()
  let acknowledged = false
  return Object.freeze({
    trailingMessages: Object.freeze([]),
    // [maint-note:c33641691165]
    // 维护导航：validate；下列为本函数源码中的接口与路径索引。
    // 返回契约：boolean。
    // 返回路径：!acknowledged。
    // [/maint-note:c33641691165]
    validate(): boolean {
      return !acknowledged
    },
    // [maint-note:e41a22f1b20c]
    // 维护导航：publish；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 写入目标：acknowledged。
    // [/maint-note:e41a22f1b20c]
    publish(): void {
      acknowledged = true
    },
    // [maint-note:6fc24c9fa051]
    // 维护导航：acknowledge；下列为本函数源码中的接口与路径索引。
    // 返回契约：boolean。
    // 分支条件：acknowledged。
    // 返回路径：false。
    // 返回路径：true。
    // 写入目标：acknowledged。
    // [/maint-note:6fc24c9fa051]
    acknowledge(): boolean {
      if (acknowledged) return false
      acknowledged = true
      return true
    },
  })
}

// [maint-note:92a3b68c8c7a]
// 维护导航：validateFramework；下列为本函数源码中的接口与路径索引。
// 输入 prepared：PreparedFrameworkContext。
// 返回契约：boolean。
// 返回路径：prepared.validate()。
// 直接调用：prepared.validate。
// [/maint-note:92a3b68c8c7a]
function validateFramework(prepared: PreparedFrameworkContext): boolean {
  return prepared.validate()
}

// [maint-note:2251e3c4ec29]
// 维护导航：publishFramework；下列为本函数源码中的接口与路径索引。
// 输入 prepared：PreparedFrameworkContext。
// 返回契约：void。
// 直接调用：prepared.publish。
// [/maint-note:2251e3c4ec29]
function publishFramework(prepared: PreparedFrameworkContext): void {
  prepared.publish()
}

// [maint-note:76a77a39e730]
// 维护导航：finalProjection；下列为本函数源码中的接口与路径索引。
// 输入 requestOnlySections：readonly string[]。
// 输入 framework：PreparedFrameworkContext。
// 返回契约：FinalRequestProjection。
// 返回路径：Object.freeze({ requestOnlySections: Object.freeze([...requestOnlySections]), trailingFrameworkMessages: framework.trailingMessages, })。
// 直接调用：Object.freeze。
// [/maint-note:76a77a39e730]
function finalProjection(
  requestOnlySections: readonly string[],
  framework: PreparedFrameworkContext,
): FinalRequestProjection {
  return Object.freeze({
    requestOnlySections: Object.freeze([...requestOnlySections]),
    trailingFrameworkMessages: framework.trailingMessages,
  })
}

// [maint-note:b8e2894dc488]
// 数据契约导航：ActiveToolBatch；字段含义以消费端校验为准。
// 成员 calls：readonly ToolCall[]。
// 成员 schemas：readonly ToolSchema[]。
// 成员 executor：ToolBatchExecutor。
// 成员 startCursor：number。
// 成员 observedInvocations：Set<string>。
// 成员 taskBatchAdmitted：boolean。
// 成员 taskFinished：boolean。
// [/maint-note:b8e2894dc488]
interface ActiveToolBatch {
  readonly calls: readonly ToolCall[]
  readonly schemas: readonly ToolSchema[]
  readonly executor: ToolBatchExecutor
  readonly startCursor: number
  readonly observedInvocations: Set<string>
  readonly taskBatchAdmitted: boolean
  taskFinished: boolean
}

// [maint-note:71bf66d473a9]
// 维护导航：observeBatchInvocation；下列为本函数源码中的接口与路径索引。
// 输入 taskRun：RunTaskPort | undefined。
// 输入 batch：ActiveToolBatch。
// 输入 callId：string。
// 返回契约：boolean。
// 分支条件：batch.observedInvocations.has(callId)。
// 分支条件：!call。
// 返回路径：false。
// 返回路径：schema?.readOnly !== true && schema?.taskManagement !== true。
// 显式抛出：new AgentError('model_protocol_error', 'Invocation update has no canonical tool call')。
// 直接调用：batch.observedInvocations.has。
// 直接调用：batch.calls.find。
// 直接调用：batch.schemas.find。
// 直接调用：taskRun?.observeExecutionStarted。
// 直接调用：batch.observedInvocations.add。
// [/maint-note:71bf66d473a9]
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

// [maint-note:e76472535f88]
// 维护导航：consumeBatchResult；下列为本函数源码中的接口与路径索引。
// 输入 input：RunTurnInput。
// 输入 batch：ActiveToolBatch。
// 输入 update：Extract<ToolBatchUpdate, { type: 'result' }>。
// 输入 pendingMessages：ModelMessage[]。
// 返回契约：{ readonly event: AgentEvent; readonly mutatingInvocation: boolean }。
// 分支条件：batch.taskBatchAdmitted。
// 返回路径：{ event, mutatingInvocation }。
// 直接调用：batch.schemas.find。
// 直接调用：observeBatchInvocation。
// 直接调用：toToolResultMessage。
// 直接调用：toToolResultEvent。
// 直接调用：input.taskRun?.observeResult。
// 直接调用：pendingMessages.push。
// 直接调用：batch.executor.acknowledge。
// [/maint-note:e76472535f88]
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

// [maint-note:7cc8d95132c0]
// 维护导航：syncCompactionMetrics；下列为本函数源码中的接口与路径索引。
// 输入 metrics：RunMetrics。
// 输入 compaction：CompactionPort。
// 返回契约：void。
// 分支条件：ledger.lastFailure。
// 写入目标：metrics.maintenanceRequestCount。
// 写入目标：metrics.compactionInstalledSummaries。
// 写入目标：metrics.compactionLastFailure。
// [/maint-note:7cc8d95132c0]
function syncCompactionMetrics(metrics: RunMetrics, compaction: CompactionPort): void {
  const ledger = compaction.ledger
  metrics.maintenanceRequestCount = ledger.calls
  metrics.compactionInstalledSummaries = ledger.installedSummaries
  if (ledger.lastFailure) metrics.compactionLastFailure = ledger.lastFailure
}

// [maint-note:d89893fc75e8]
// 维护导航：syncBudgetMetrics；下列为本函数源码中的接口与路径索引。
// 输入 metrics：RunMetrics。
// 输入 budget：RunBudgetLedger。
// 返回契约：void。
// 写入目标：metrics.modelRequestCount。
// 写入目标：metrics.segmentModelRequestCount。
// 写入目标：metrics.segmentRenewalCount。
// 写入目标：metrics.observedToolCallCount。
// 写入目标：metrics.admittedToolAttemptCount。
// 写入目标：metrics.budgetOverspent。
// 写入目标：metrics.usageComplete。
// 直接调用：budget.snapshot。
// 直接调用：assignOptionalMetric。
// [/maint-note:d89893fc75e8]
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

// [maint-note:3798ba6f8772]
// 维护导航：assignOptionalMetric；下列为本函数源码中的接口与路径索引。
// 输入 metrics：RunMetrics。
// 输入 key：'reservedTokens' | 'knownActualTokens' | 'unknownTokenDebt' | 'inputTokens' | 'outputTokens' | 'totalTokens'。
// 输入 value：number | undefined。
// 返回契约：void。
// 分支条件：value === undefined。
// 写入目标：metrics[key]。
// [/maint-note:3798ba6f8772]
function assignOptionalMetric(
  metrics: RunMetrics,
  key: 'reservedTokens' | 'knownActualTokens' | 'unknownTokenDebt' | 'inputTokens' | 'outputTokens' | 'totalTokens',
  value: number | undefined,
): void {
  if (value === undefined) delete metrics[key]
  else metrics[key] = value
}

// [maint-note:8c8ef344a7b7]
// 维护导航：appendClosedBatch；下列为本函数源码中的接口与路径索引。
// 输入 threadRun：ThreadRunController。
// 输入 batch：ActiveToolBatch。
// 输入 pendingMessages：readonly ModelMessage[]。
// 返回契约：void。
// 直接调用：threadRun.appendClosedExchange。
// 直接调用：pendingMessages.slice。
// [/maint-note:8c8ef344a7b7]
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

// [maint-note:caeee636d946]
// 维护导航：closeActiveBatch；下列为本函数源码中的接口与路径索引。
// 输入 input：RunTurnInput。
// 输入 threadRun：ThreadRunController。
// 输入 batch：ActiveToolBatch。
// 输入 pendingMessages：ModelMessage[]。
// 返回契约：{ readonly events: readonly AgentEvent[]; readonly mutatingInvocation: boolean }。
// 分支条件：consumed.mutatingInvocation。
// 分支条件：batch.taskBatchAdmitted && !batch.taskFinished && input.taskRun?.started && !input.taskRun.settled。
// 遍历边界：const update ← batch.executor.closeInterrupted()。
// 返回路径：{ events, mutatingInvocation }。
// 写入目标：mutatingInvocation。
// 写入目标：batch.taskFinished。
// 直接调用：batch.executor.closeInterrupted。
// 直接调用：consumeBatchResult。
// 直接调用：events.push。
// 直接调用：appendClosedBatch。
// 直接调用：input.taskRun.finishBatch。
// [/maint-note:caeee636d946]
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

// [maint-note:d5cab5198f99]
// 维护导航：finalizeRun；下列为本函数源码中的接口与路径索引。
// 输入 requestedTerminal：AgentEvent。
// 输入 threadRun：ThreadRunController。
// 输入 pendingMessages：readonly ModelMessage[]。
// 输入 taskRun：RunTaskPort | undefined。
// 输入 workspaceMayHaveChanged：boolean。
// 返回契约：AgentEvent。
// 分支条件：terminal.type !== 'error' && terminal.type !== 'interrupted'。
// 返回路径：terminal。
// 写入目标：prepared。
// 写入目标：terminal。
// 直接调用：prepareFinalization。
// 直接调用：errorEvent。
// 直接调用：errorCloseTail。
// 直接调用：publishPreparedFinalization。
// [/maint-note:d5cab5198f99]
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

// [maint-note:79ccfe47c27b]
// 数据契约导航：PreparedFinalization；字段含义以消费端校验为准。
// 成员 thread：ThreadCloseCandidate。
// 成员 task（可省略）：PreparedRunTaskSettlement。
// [/maint-note:79ccfe47c27b]
interface PreparedFinalization {
  readonly thread: ThreadCloseCandidate
  readonly task?: PreparedRunTaskSettlement
}

// [maint-note:10ac126768fb]
// 维护导航：prepareFinalization；下列为本函数源码中的接口与路径索引。
// 输入 terminal：AgentEvent。
// 输入 threadRun：ThreadRunController。
// 输入 pendingMessages：readonly ModelMessage[]。
// 输入 taskRun：RunTaskPort | undefined。
// 输入 workspaceMayHaveChanged：boolean。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 false。
// 返回契约：PreparedFinalization。
// 分支条件：taskRun?.started && !taskRun.settled。
// 分支条件：!taskCandidate。
// 返回路径：{ thread: threadCandidate, ...(taskCandidate ? { task: taskCandidate } : {}), }。
// 显式抛出：new Error('Task settlement did not produce a candidate')。
// 写入目标：taskCandidate。
// 直接调用：threadTerminalFromAgentEvent。
// 直接调用：threadRun.prepareClose。
// 直接调用：pendingMessages.slice。
// 直接调用：errorCloseTail。
// 直接调用：taskTerminalFromAgentEvent。
// 直接调用：taskRun.prepareFallbackSettle。
// 直接调用：taskRun.prepareSettle。
// [/maint-note:10ac126768fb]
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

// [maint-note:5742952c4145]
// 维护导航：publishPreparedFinalization；下列为本函数源码中的接口与路径索引。
// 输入 threadRun：ThreadRunController。
// 输入 taskRun：RunTaskPort | undefined。
// 输入 prepared：PreparedFinalization。
// 返回契约：void。
// 分支条件：prepared.task。
// 直接调用：threadRun.publishClose。
// 直接调用：taskRun!.publishSettle。
// 直接调用：threadRun.release。
// [/maint-note:5742952c4145]
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

// [maint-note:c871edd78b21]
// 维护导航：errorCloseTail；下列为本函数源码中的接口与路径索引。
// 输入 threadRun：ThreadRunController。
// 输入 pendingMessages：readonly ModelMessage[]。
// 返回契约：readonly ModelMessage[]。
// 返回路径：threadRun.savedCursor === 0 && pendingMessages[0]?.role === 'user' ? [pendingMessages[0]] : []。
// [/maint-note:c871edd78b21]
function errorCloseTail(
  threadRun: ThreadRunController,
  pendingMessages: readonly ModelMessage[],
): readonly ModelMessage[] {
  return threadRun.savedCursor === 0 && pendingMessages[0]?.role === 'user'
    ? [pendingMessages[0]]
    : []
}

// [maint-note:29f7ad372cc7]
// 维护导航：threadTerminalFromAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 输入 workspaceMayHaveChanged：boolean。
// 返回契约：ThreadTerminalFact。
// 分支条件：switch event.type。
// 分支条件：case 'done'。
// 分支条件：case 'paused'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'error'。
// 返回路径：{ type: 'done', workspaceMayHaveChanged }。
// 返回路径：{ type: 'paused', reason: event.reason, workspaceMayHaveChanged }。
// 返回路径：{ type: 'interrupted', workspaceMayHaveChanged }。
// 返回路径：{ type: 'error', code: event.code, workspaceMayHaveChanged }。
// 显式抛出：new Error('Only a terminal AgentEvent can finalize a Thread run')。
// [/maint-note:29f7ad372cc7]
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

// [maint-note:482bb9eb7bef]
// 数据契约导航：NormalizedToolResult；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 isError：boolean。
// 成员 code（可省略）：string。
// 成员 display（可省略）：ToolDisplayFacts。
// [/maint-note:482bb9eb7bef]
type NormalizedToolResult = {
  content: string
  isError: boolean
  code?: string
  /** UI-local display facts; errors and validation never carry them. */
  display?: ToolDisplayFacts
}

// [maint-note:6c871439feee]
// 数据契约导航：ModelUpdate；字段含义以消费端校验为准。
// 类型组合：| { type: 'assistant_delta'; content: string } | { type: 'response'; response: ModelResponse }。
// [/maint-note:6c871439feee]
type ModelUpdate =
  | { type: 'assistant_delta'; content: string }
  | { type: 'response'; response: ModelResponse }

const observedModelUsageKey = Symbol('observedModelUsage')
// [maint-note:81efab6b6c9a]
// 数据契约导航：UsageBearingError；字段含义以消费端校验为准。
// 类型组合：Error & { [observedModelUsageKey]?: ModelResponse['usage'] }。
// [/maint-note:81efab6b6c9a]
type UsageBearingError = Error & { [observedModelUsageKey]?: ModelResponse['usage'] }

// [maint-note:5c884c532c9f]
// 维护导航：consumePreparedModel；下列为本函数源码中的接口与路径索引。
// 输入 prepared：PreparedModelDispatch。
// 输入 signal：AbortSignal | undefined。
// 输入 limits：ModelResponseLimits。
// 输入 rejectTextControlEnvelope：类型由调用方与函数体约束；缺省表达式 true。
// 输入 emitTextDeltas：类型由调用方与函数体约束；缺省表达式 true。
// 返回契约：AsyncGenerator<ModelUpdate>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：prepared.mode === 'complete'。
// 分支条件：isAsyncIterable(sent)。
// 分支条件：rejectTextControlEnvelope && detectToolControlEnvelope(validated.content)。
// 分支条件：!isAsyncIterable(sent)。
// 分支条件：!stream || typeof stream[Symbol.asyncIterator] !== 'function'。
// 分支条件：result.done。
// 分支条件：text。
// 分支条件：emitTextDeltas && visibleText。
// 分支条件：!exhausted。
// 分支条件：controlEnvelope && rejectTextControlEnvelope。
// 分支条件：emitTextDeltas && (!controlEnvelope || !rejectTextControlEnvelope)。
// 分支条件：trailingText。
// 分支条件：error instanceof AgentError || isAbortError(error, signal)。
// 分支条件：error instanceof ModelProtocolError。
// 分支条件：error instanceof ModelResponseLimitError。
// 遍历边界：true。
// 返回路径：无返回值。
// 显式抛出：new ModelProtocolError('Prepared complete dispatch returned a stream')。
// 显式抛出：new ModelProtocolError('Model response returned a tool-control envelope as text content')。
// 显式抛出：new ModelProtocolError('Prepared stream dispatch returned a response')。
// 显式抛出：new ModelProtocolError('Model client returned an invalid stream')。
// 显式抛出：new ModelProtocolError( 'Model response returned a tool-control envelope as text content', )。
// 显式抛出：attachModelUsage(error, observedUsage)。
// 显式抛出：attachModelUsage(new AgentError('model_protocol_error', error.message), observedUsage)。
// 显式抛出：attachModelUsage(new AgentError('model_incomplete', error.message), observedUsage)。
// 显式抛出：attachModelUsage(new AgentError('model_request_error', errorMessage(error)), observedUsage)。
// 写入目标：observedUsage。
// 写入目标：exhausted。
// 直接调用：prepared.send。
// 直接调用：isAsyncIterable。
// 直接调用：throwIfAborted。
// 直接调用：validateModelResponse。
// 直接调用：detectToolControlEnvelope。
// 直接调用：stream[Symbol.asyncIterator]。
// 直接调用：iterator.next。
// 直接调用：accumulator.push。
// 直接调用：accumulator.usageSnapshot。
// 直接调用：textGate.push。
// 直接调用：iterator.return。
// 直接调用：accumulator.finish。
// 直接调用：textGate.release。
// 直接调用：isAbortError。
// 直接调用：attachModelUsage。
// 直接调用：errorMessage。
// [/maint-note:5c884c532c9f]
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

// [maint-note:9a3d2aa7e3f9]
// 维护导航：attachModelUsage；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 输入 usage：ModelResponse['usage']。
// 返回契约：unknown。
// 分支条件：!usage || !(error instanceof Error)。
// 返回路径：error。
// 直接调用：Object.defineProperty。
// 直接调用：Object.freeze。
// [/maint-note:9a3d2aa7e3f9]
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

// [maint-note:158fd25fe091]
// 维护导航：modelUsageFromError；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：ModelResponse['usage']。
// 返回路径：error instanceof Error ? (error as UsageBearingError)[observedModelUsageKey] : undefined。
// [/maint-note:158fd25fe091]
function modelUsageFromError(error: unknown): ModelResponse['usage'] {
  return error instanceof Error
    ? (error as UsageBearingError)[observedModelUsageKey]
    : undefined
}

// [maint-note:36cf40bd25ee]
// 维护导航：createModelRequest；下列为本函数源码中的接口与路径索引。
// 输入 input：RunTurnInput。
// 输入 messages：readonly ModelMessage[]。
// 输入 tools：ModelToolSchema[]。
// 输入 toolChoice：'auto' | 'none' | 'required'（可省略）。
// 返回契约：ModelRequest。
// 返回路径：{ model: input.runtime.model, messages: [...messages], tools, toolChoice, signal: input.signal, ...(outputTokenLimit === undefined && safetyMarginTokens === undefined ? {} : { budget: { outputTokenLimit, safetyMarginTokens } }), }。
// [/maint-note:36cf40bd25ee]
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

// [maint-note:4c8124bf0c1d]
// 维护导航：prepareModelDispatch；下列为本函数源码中的接口与路径索引。
// 输入 client：ModelClient。
// 输入 request：ModelRequest。
// 返回契约：PreparedModelDispatch。
// 分支条件：client.prepare。
// 分支条件：error instanceof ModelProtocolError。
// 返回路径：client.prepare(request, mode)。
// 返回路径：Object.freeze({ mode, measurement: Object.freeze({ providerShapeId: `generic-${mode}-degraded-v1`, serializedBodyBytes: Buffer.byteLength(JSON.stringify({ model: request.model, messages: request.messages, tools: request.tools, toolChoice: r。
// 显式抛出：new AgentError('request_budget_exhausted', error.message)。
// 显式抛出：error。
// 直接调用：client.prepare。
// 直接调用：Object.freeze。
// 直接调用：Buffer.byteLength。
// 直接调用：JSON.stringify。
// [/maint-note:4c8124bf0c1d]
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

// [maint-note:dcafd55fa646]
// 维护导航：isAsyncIterable；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is AsyncIterable<T>。
// 返回路径：Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')。
// 直接调用：Boolean。
// [/maint-note:dcafd55fa646]
function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')
}

// [maint-note:5166b092213a]
// 维护导航：executeToolCall；下列为本函数源码中的接口与路径索引。
// 输入 input：RunTurnInput。
// 输入 call：ToolCall。
// 输入 schemas：readonly ToolSchema[]。
// 返回契约：Promise<NormalizedToolResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：validationError。
// 分支条件：isAbortError(error, input.signal)。
// 返回路径：{ content: formatToolError(validationError.code, validationError.message, input.runtime.toolContext.maxOutputChars, validationError.diagnostic), isError: true, code: validationError.code, }。
// 返回路径：{ content: isError ? boundedErrorContent(normalizeToolOutput(result?.content), result?.errorCode ?? 'tool_execution_error', input.runtime.toolContext.maxOutputChars) : normalizeToolOutput(result?.content), isError, code: isError ? result?.e。
// 返回路径：{ content: formatToolError('tool_execution_error', '', input.runtime.toolContext.maxOutputChars, errorDiagnostic(error) ?? toolDiagnostic('unknown_error')), isError: true, code: 'tool_execution_error', }。
// 显式抛出：error。
// 直接调用：validateToolCall。
// 直接调用：formatToolError。
// 直接调用：input.runtime.tools.execute。
// 直接调用：boundedErrorContent。
// 直接调用：normalizeToolOutput。
// 直接调用：isAbortError。
// 直接调用：errorDiagnostic。
// 直接调用：toolDiagnostic。
// [/maint-note:5166b092213a]
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

// [maint-note:2805ef1688e0]
// 维护导航：getToolValidationError；下列为本函数源码中的接口与路径索引。
// 输入 input：RunTurnInput。
// 输入 call：ToolCall。
// 输入 schemas：readonly ToolSchema[]。
// 输入 batchValidationError：string（可省略）。
// 返回契约：AgentError | undefined。
// 分支条件：batchValidationError。
// 分支条件：genericError。
// 分支条件：schema。
// 分支条件：error instanceof ToolInputError。
// 返回路径：new AgentError('tool_validation_error', batchValidationError, true)。
// 返回路径：genericError。
// 返回路径：new AgentError('tool_validation_error', error.message, true, error.diagnostic)。
// 返回路径：undefined。
// 显式抛出：error。
// 直接调用：validateToolCall。
// 直接调用：schemas.find。
// 直接调用：validateApprovalInput。
// 直接调用：input.runtime.tools.validate。
// [/maint-note:2805ef1688e0]
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

// [maint-note:91d2d4084530]
// 维护导航：toToolErrorResult；下列为本函数源码中的接口与路径索引。
// 输入 error：AgentError。
// 返回契约：NormalizedToolResult。
// 返回路径：{ content: formatToolError(error.code, error.message, undefined, error.diagnostic), isError: true, code: error.code, }。
// 直接调用：formatToolError。
// [/maint-note:91d2d4084530]
function toToolErrorResult(error: AgentError): NormalizedToolResult {
  return {
    content: formatToolError(error.code, error.message, undefined, error.diagnostic),
    isError: true,
    code: error.code,
  }
}

// [maint-note:9c8a45fb4bea]
// 维护导航：validateToolSchemas；下列为本函数源码中的接口与路径索引。
// 输入 schemas：ToolSchema[]。
// 返回契约：ToolSchema[]。
// 分支条件：names.has(schema.name)。
// 分支条件：error instanceof ToolInputError。
// 遍历边界：const schema ← schemas。
// 返回路径：schemas。
// 显式抛出：new AgentError( 'tool_validation_error', `Duplicate tool name: ${schema.name}`, )。
// 显式抛出：new AgentError('tool_validation_error', error.message)。
// 显式抛出：error。
// 直接调用：names.has。
// 直接调用：names.add。
// 直接调用：validateApprovalSchema。
// [/maint-note:9c8a45fb4bea]
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

// [maint-note:cd2e6671e3c7]
// 维护导航：toModelToolSchemas；下列为本函数源码中的接口与路径索引。
// 输入 schemas：readonly ToolSchema[]。
// 返回契约：ModelToolSchema[]。
// 返回路径：schemas.map(({ name, description, parameters }) => ({ name, description, parameters, }))。
// 直接调用：schemas.map。
// [/maint-note:cd2e6671e3c7]
function toModelToolSchemas(schemas: readonly ToolSchema[]): ModelToolSchema[] {
  return schemas.map(({ name, description, parameters }) => ({
    name,
    description,
    parameters,
  }))
}

// [maint-note:6eadbd9df536]
// 数据契约导航：NormalizedToolBatch；字段含义以消费端校验为准。
// 成员 calls：ToolCall[]。
// 成员 error（可省略）：string。
// [/maint-note:6eadbd9df536]
interface NormalizedToolBatch {
  calls: ToolCall[]
  error?: string
}

// [maint-note:d0d3cc4c131e]
// 维护导航：normalizeToolCalls；下列为本函数源码中的接口与路径索引。
// 输入 calls：readonly ModelToolCall[]。
// 输入 usedIds：Set<string>。
// 返回契约：NormalizedToolBatch。
// 返回路径：{ calls: normalizedCalls, error: errors.length ? `Tool call batch validation failed: ${errors.join(' ')}` : undefined, }。
// 直接调用：calls.map。
// 直接调用：errors.join。
// [/maint-note:d0d3cc4c131e]
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

// [maint-note:5e1f88dd0a04]
// 维护导航：validateToolCall；下列为本函数源码中的接口与路径索引。
// 输入 call：ToolCall。
// 输入 schemas：readonly ToolSchema[]。
// 返回契约：AgentError | undefined。
// 分支条件：!call.id || !call.name。
// 分支条件：!schemas.some((schema) => schema.name === call.name)。
// 分支条件：call.argumentsError。
// 分支条件：typeof call.rawArguments === 'string'。
// 分支条件：!isRecord(call.arguments)。
// 返回路径：new AgentError( 'tool_validation_error', 'Tool call must include a non-empty id and name', true, toolDiagnostic('invalid_call'), )。
// 返回路径：new AgentError('tool_not_found', `Unknown tool: ${call.name}`, true, toolDiagnostic('tool_not_found'))。
// 返回路径：new AgentError('tool_validation_error', call.argumentsError, true, toolDiagnostic(reason))。
// 返回路径：new AgentError( 'tool_validation_error', `Arguments for tool ${call.name} must be a JSON object`, true, toolDiagnostic('invalid_argument_shape'), )。
// 返回路径：undefined。
// 写入目标：reason。
// 直接调用：toolDiagnostic。
// 直接调用：schemas.some。
// 直接调用：JSON.parse。
// 直接调用：isRecord。
// [/maint-note:5e1f88dd0a04]
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

// [maint-note:ec41f52e414f]
// 维护导航：normalizeToolCall；下列为本函数源码中的接口与路径索引。
// 输入 call：ModelToolCall。
// 输入 index：number。
// 输入 canonicalId：string（可省略）。
// 返回契约：ToolCall。
// 分支条件：call.rawArguments !== undefined。
// 分支条件：call.argumentsError !== undefined。
// 返回路径：normalized。
// 写入目标：normalized.rawArguments。
// 写入目标：normalized.argumentsError。
// 直接调用：isRecord。
// [/maint-note:ec41f52e414f]
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

// [maint-note:0c1efaccf3aa]
// 维护导航：normalizeAssistantMessage；下列为本函数源码中的接口与路径索引。
// 输入 response：ModelResponse。
// 输入 toolCalls：readonly ToolCall[]。
// 返回契约：ModelAssistantMessage。
// 分支条件：response.content。
// 遍历边界：const call ← toolCalls。
// 返回路径：{ role: 'assistant', content, stopReason: response.stopReason, }。
// 直接调用：content.push。
// 直接调用：textContent。
// 直接调用：toToolCallBlock。
// [/maint-note:0c1efaccf3aa]
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

// [maint-note:3786f3a97250]
// 维护导航：toToolCallBlock；下列为本函数源码中的接口与路径索引。
// 输入 call：ToolCall。
// 返回契约：Extract<ModelContentBlock, { type: 'toolCall' }>。
// 返回路径：{ type: 'toolCall', ...call }。
// [/maint-note:3786f3a97250]
function toToolCallBlock(
  call: ToolCall,
): Extract<ModelContentBlock, { type: 'toolCall' }> {
  return { type: 'toolCall', ...call }
}

// [maint-note:c2f6b63bf8bf]
// 维护导航：allocateInvalidCallId；下列为本函数源码中的接口与路径索引。
// 输入 index：number。
// 输入 usedIds：Set<string>。
// 返回契约：string。
// 遍历边界：usedIds.has(candidate)。
// 返回路径：candidate。
// 写入目标：suffix。
// 写入目标：candidate。
// 直接调用：usedIds.has。
// 直接调用：usedIds.add。
// [/maint-note:c2f6b63bf8bf]
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

// [maint-note:63f318ab68f0]
// 维护导航：toToolResultMessage；下列为本函数源码中的接口与路径索引。
// 输入 call：ToolCall。
// 输入 result：NormalizedToolResult。
// 返回契约：ModelToolResultMessage。
// 返回路径：{ role: 'toolResult', toolCallId: call.id, toolName: call.name, content: textContent(result.content), isError: result.isError, }。
// 直接调用：textContent。
// [/maint-note:63f318ab68f0]
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

// [maint-note:834a4f93d45f]
// 维护导航：toToolResultEvent；下列为本函数源码中的接口与路径索引。
// 输入 showToolResults：boolean。
// 输入 call：ToolCall。
// 输入 result：NormalizedToolResult。
// 返回契约：AgentEvent。
// 返回路径：{ type: 'tool_result', callId: call.id, name: call.name, output: showToolResults || result.isError ? result.content : '', isError: result.isError, ...(result.code === undefined ? {} : { code: result.code }), // Display facts follow the same。
// [/maint-note:834a4f93d45f]
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

// [maint-note:ebe9a5187b16]
// 维护导航：textContent；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：ModelContentBlock[]。
// 返回路径：text ? [{ type: 'text', text }] : []。
// [/maint-note:ebe9a5187b16]
function textContent(text: string): ModelContentBlock[] {
  return text ? [{ type: 'text', text }] : []
}

// [maint-note:e1bcea6e5a75]
// 维护导航：normalizeToolOutput；下列为本函数源码中的接口与路径索引。
// 输入 content：unknown。
// 返回契约：string。
// 分支条件：typeof content !== 'string' || !content。
// 返回路径：'Tool completed with no output.'。
// 返回路径：content。
// [/maint-note:e1bcea6e5a75]
function normalizeToolOutput(content: unknown): string {
  if (typeof content !== 'string' || !content) {
    return 'Tool completed with no output.'
  }
  return content
}

// [maint-note:4d022457a75e]
// 维护导航：runtimeRecoveryInstruction；下列为本函数源码中的接口与路径索引。
// 输入 summary：{ noProgressRounds: number recoveryPrompts: number workspaceRevision: number errorCodes: string[] batchCount: number }。
// 返回契约：string。
// 返回路径：`Runtime recovery: ${summary.noProgressRounds} no-progress round(s), ${summary.recoveryPrompts} recovery prompt(s), ${summary.batchCount} recent batch(es), workspace revision ${summary.workspaceRevision}, stable error codes: ${errors}. Insp。
// 直接调用：summary.errorCodes.join。
// [/maint-note:4d022457a75e]
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

// [maint-note:70922bf912d3]
// 维护导航：softProgressInstruction；下列为本函数源码中的接口与路径索引。
// 返回契约：string。
// 返回路径：'Progress check: review verified evidence, avoid repeating an unchanged failed operation, and finish with an honest summary if no further safe progress is possible.'。
// [/maint-note:70922bf912d3]
function softProgressInstruction(): string {
  return 'Progress check: review verified evidence, avoid repeating an unchanged failed operation, and finish with an honest summary if no further safe progress is possible.'
}

// [maint-note:59ea84de784c]
// 维护导航：pauseCheckpointInstruction；下列为本函数源码中的接口与路径索引。
// 输入 reason：'budget_exhausted' | 'no_progress'。
// 返回契约：string。
// 返回路径：`Pause checkpoint (${reason}): do not call tools. Summarize verified results, failed attempts, unfinished items, and the safest next step. This is a checkpoint, not a claim that the task is complete.`。
// [/maint-note:59ea84de784c]
function pauseCheckpointInstruction(reason: 'budget_exhausted' | 'no_progress'): string {
  return `Pause checkpoint (${reason}): do not call tools. Summarize verified results, failed attempts, unfinished items, and the safest next step. This is a checkpoint, not a claim that the task is complete.`
}

// [maint-note:8f4eb9045975]
// 维护导航：plainTextClosingInstruction；下列为本函数源码中的接口与路径索引。
// 返回契约：string。
// 返回路径：'The previous closing response was rejected: it was a tool-control envelope rather than a summary. Reply with plain natural-language prose only. Do not emit tool-call markup, control tokens, or any provider protocol envelope; no tool will r。
// [/maint-note:8f4eb9045975]
function plainTextClosingInstruction(): string {
  return 'The previous closing response was rejected: it was a tool-control envelope rather than a summary. Reply with plain natural-language prose only. Do not emit tool-call markup, control tokens, or any provider protocol envelope; no tool will run.'
}

// [maint-note:69e57caa752b]
// 维护导航：missingCompletionClaimInstruction；下列为本函数源码中的接口与路径索引。
// 返回契约：string。
// 返回路径：'The previous final response was not accepted because this run has an explicit plan without a valid completion claim. Recheck the current trusted run-task projection. If every current stage is complete, call update_task_plan with operation 。
// [/maint-note:69e57caa752b]
function missingCompletionClaimInstruction(): string {
  return 'The previous final response was not accepted because this run has an explicit plan without a valid completion claim. Recheck the current trusted run-task projection. If every current stage is complete, call update_task_plan with operation claim_run_complete, the current plan revision, and every current step ID exactly once; then provide a final answer in a later response. Do not invent completion or clear the plan.'
}

// [maint-note:d8402eb0e93e]
// 维护导航：taskTerminalFromAgentEvent；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent。
// 返回契约：RunTaskTerminal。
// 分支条件：switch event.type。
// 分支条件：case 'done'。
// 分支条件：case 'paused'。
// 分支条件：case 'interrupted'。
// 分支条件：case 'error'。
// 返回路径：{ type: 'done' }。
// 返回路径：{ type: 'paused', reason: event.reason }。
// 返回路径：{ type: 'interrupted' }。
// 返回路径：{ type: 'error', code: event.code }。
// 返回路径：{ type: 'error', code: 'model_protocol_error' }。
// [/maint-note:d8402eb0e93e]
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
// [maint-note:53e4564287cc]
// 维护导航：createRunScopedApprovalRequestId；下列为本函数源码中的接口与路径索引。
// 输入 factoryValue：unknown。
// 输入 runOwner：string。
// 输入 callId：string。
// 输入 ordinal：number。
// 输入 factoryRequestIds：Map<string, string>。
// 返回契约：string。
// 分支条件：factoryValue === undefined。
// 分支条件：typeof factoryValue !== 'string' || factoryValue.length > MAX_APPROVAL_FACTORY_VALUE_CHARS || !factoryValue.trim()。
// 分支条件：factoryValue === fallback。
// 分支条件：previousRequestId !== undefined。
// 返回路径：fallback。
// 返回路径：''。
// 返回路径：previousRequestId。
// 返回路径：requestId。
// 直接调用：factoryValue.trim。
// 直接调用：factoryRequestIds.get。
// 直接调用：[...normalizedFactoryValue] .map((character) => /[\p{L}\p{N}._~-]/u.test(character) ? character : '_') .slice(0, 80) .join。
// 直接调用：[...normalizedFactoryValue] .map((character) => /[\p{L}\p{N}._~-]/u.test(character) ? character : '_') .slice。
// 直接调用：[...normalizedFactoryValue] .map。
// 直接调用：factoryRequestIds.set。
// [/maint-note:53e4564287cc]
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

// [maint-note:4f5636577d7b]
// 维护导航：errorEvent；下列为本函数源码中的接口与路径索引。
// 输入 code：AgentErrorCode。
// 输入 message：string。
// 输入 recoverable：boolean。
// 输入 sideEffectToolExecuted：类型由调用方与函数体约束；缺省表达式 false。
// 返回契约：AgentEvent。
// 返回路径：{ type: 'error', code, message, recoverable, workspaceMayHaveChanged: sideEffectToolExecuted, }。
// [/maint-note:4f5636577d7b]
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

// [maint-note:fb923cd1d87a]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：createAbortError()。
// 直接调用：createAbortError。
// [/maint-note:fb923cd1d87a]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

// [maint-note:6eae570914ee]
// 维护导航：createAbortError；下列为本函数源码中的接口与路径索引。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:6eae570914ee]
function createAbortError(): Error {
  const error = new Error('The agent turn was aborted')
  error.name = 'AbortError'
  return error
}

// [maint-note:03304db641e1]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:03304db641e1]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
