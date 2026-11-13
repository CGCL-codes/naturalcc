import type {
  ThreadCloseReceipt,
  ThreadRunHistoryReceipt,
  ThreadRunIdentity,
  ThreadRunSourceRef,
} from '../agent/thread/AgentThread.js'

// [maint-note:a7e95f5313f1]
// 数据契约导航：TaskStepStatus；字段含义以消费端校验为准。
// 类型组合：'pending' | 'in_progress' | 'completed'。
// [/maint-note:a7e95f5313f1]
export type TaskStepStatus = 'pending' | 'in_progress' | 'completed'

// [maint-note:bd1dbfc1272f]
// 数据契约导航：TaskPlanStep；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 content：string。
// 成员 status：TaskStepStatus。
// [/maint-note:bd1dbfc1272f]
export interface TaskPlanStep {
  readonly id: string
  readonly content: string
  readonly status: TaskStepStatus
}

// [maint-note:6645422de73d]
// 数据契约导航：RunTaskPlan；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 revision：number。
// 成员 steps：readonly TaskPlanStep[]。
// [/maint-note:6645422de73d]
export interface RunTaskPlan {
  readonly id: string
  readonly revision: number
  readonly steps: readonly TaskPlanStep[]
}

// [maint-note:163c8037e1f3]
// 数据契约导航：TaskPlanOperation；字段含义以消费端校验为准。
// 类型组合：| { readonly operation: 'create_plan' readonly steps: readonly { readonly content: string readonly status: TaskStepStatus }[] } | { readonly operation: 'update_plan' readonly revision: number readonly reason: string readonly steps: readonly。
// [/maint-note:163c8037e1f3]
export type TaskPlanOperation =
  | {
      readonly operation: 'create_plan'
      readonly steps: readonly {
        readonly content: string
        readonly status: TaskStepStatus
      }[]
    }
  | {
      readonly operation: 'update_plan'
      readonly revision: number
      readonly reason: string
      readonly steps: readonly {
        readonly id?: string
        readonly content: string
        readonly status: TaskStepStatus
      }[]
    }
  | {
      readonly operation: 'claim_run_complete'
      readonly revision: number
      readonly stepIds: readonly string[]
    }
  | {
      readonly operation: 'list_snapshots'
      readonly limit?: number
    }
  | {
      readonly operation: 'read_snapshot'
      readonly snapshotId: string
      /** Optional paged read of the snapshot's still-available raw source. */
      readonly sourceOffset?: number
      readonly sourceLimit?: number
    }

// [maint-note:39afe156e8a6]
// 数据契约导航：RunTaskTerminal；字段含义以消费端校验为准。
// 类型组合：| { readonly type: 'done' } | { readonly type: 'paused'; readonly reason: 'budget_exhausted' | 'no_progress' } | { readonly type: 'interrupted' } | { readonly type: 'error'; readonly code: string }。
// [/maint-note:39afe156e8a6]
export type RunTaskTerminal =
  | { readonly type: 'done' }
  | { readonly type: 'paused'; readonly reason: 'budget_exhausted' | 'no_progress' }
  | { readonly type: 'interrupted' }
  | { readonly type: 'error'; readonly code: string }

// [maint-note:2d094baa23fb]
// 数据契约导航：RunTaskClaim；字段含义以消费端校验为准。
// 成员 claimRef：string。
// 成员 planRevision：number。
// 成员 stepIds：readonly string[]。
// 成员 status：'candidate' | 'committed' | 'invalidated' | 'rejected' | 'cancelled'。
// 成员 reason（可省略）：string。
// [/maint-note:2d094baa23fb]
export interface RunTaskClaim {
  /**
   * The ledger's bounded, content-derived reference to the provider
   * call (`sha256:<64 hex>` of the raw ID, 71 bytes UTF-8, one stable codec
   * for every claim/observation so raw strings cannot impersonate encoded
   * long IDs). The raw ID remains the key for pairing tool calls and results.
   */
  readonly claimRef: string
  readonly planRevision: number
  readonly stepIds: readonly string[]
  readonly status: 'candidate' | 'committed' | 'invalidated' | 'rejected' | 'cancelled'
  readonly reason?: string
}

// [maint-note:63f71653491a]
// 数据契约导航：RunToolObservation；字段含义以消费端校验为准。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 outcome：'succeeded' | 'failed' | 'not_executed' | 'unknown_effect'。
// 成员 code（可省略）：string。
// 成员 outputPreview：string。
// 成员 mutating：boolean。
// 成员 protocolDisposition：'pending' | 'retained' | 'not_retained'。
// [/maint-note:63f71653491a]
export interface RunToolObservation {
  /** Ledger reference (`sha256:<hex>` codec), NOT the raw provider call ID. */
  readonly callId: string
  readonly toolName: string
  readonly outcome: 'succeeded' | 'failed' | 'not_executed' | 'unknown_effect'
  readonly code?: string
  readonly outputPreview: string
  readonly mutating: boolean
  readonly protocolDisposition: 'pending' | 'retained' | 'not_retained'
}

// [maint-note:ade78a576b0a]
// 数据契约导航：RunTaskSnapshot；字段含义以消费端校验为准。
// 成员 schemaVersion：2。
// 成员 snapshotId：string。
// 成员 sessionId：string。
// 成员 generation：string。
// 成员 runId：string。
// 成员 source：{ readonly preview: string readonly sha256: string readonly availability: 'thread' | 'preview_only' /** Exact Thread-generated identity/range; the digest alone is not identity. */ readonly ref?: ThreadRunSourceRef /** Total source length in。
// 成员 claimStepIdsElided（可省略）：true。
// 成员 plan：RunTaskPlan | null。
// 成员 planAdjustments：readonly { readonly revision: number readonly reason: string }[]。
// 成员 claim：RunTaskClaim | null。
// 成员 terminal：RunTaskTerminal。
// 成员 history：Omit<ThreadRunHistoryReceipt, 'source'>。
// 成员 toolObservations：readonly RunToolObservation[]。
// 成员 omittedToolObservations：number。
// 成员 workspaceMayHaveChanged：boolean。
// 成员 verification：'unverified'。
// [/maint-note:ade78a576b0a]
export interface RunTaskSnapshot {
  readonly schemaVersion: 2
  readonly snapshotId: string
  readonly sessionId: string
  readonly generation: string
  readonly runId: string
  readonly source: {
    readonly preview: string
    readonly sha256: string
    readonly availability: 'thread' | 'preview_only'
    /** Exact Thread-generated identity/range; the digest alone is not identity. */
    readonly ref?: ThreadRunSourceRef
    /** Total source length in characters, frozen when resolvable. */
    readonly totalChars?: number
  }
  /**
   * Set when the single-snapshot budget forced the claim's stepIds
   * to be elided; they were validated equal to the retained plan's step IDs,
   * which remain fully present in `plan.steps`.
   */
  readonly claimStepIdsElided?: true
  readonly plan: RunTaskPlan | null
  readonly planAdjustments: readonly {
    readonly revision: number
    readonly reason: string
  }[]
  readonly claim: RunTaskClaim | null
  readonly terminal: RunTaskTerminal
  readonly history: Omit<ThreadRunHistoryReceipt, 'source'>
  readonly toolObservations: readonly RunToolObservation[]
  readonly omittedToolObservations: number
  readonly workspaceMayHaveChanged: boolean
  readonly verification: 'unverified'
}

// [maint-note:0903de2a9136]
// 数据契约导航：RunTaskSettleInput；字段含义以消费端校验为准。
// 成员 terminal：RunTaskTerminal。
// 成员 history：ThreadCloseReceipt。
// 成员 workspaceMayHaveChanged：boolean。
// [/maint-note:0903de2a9136]
export interface RunTaskSettleInput {
  readonly terminal: RunTaskTerminal
  readonly history: ThreadCloseReceipt
  readonly workspaceMayHaveChanged: boolean
}

/** Opaque-by-ownership prepared value; only its creating RunTaskPort may publish it. */
// [maint-note:fdcba544c25a]
// 数据契约导航：PreparedRunTaskSettlement；字段含义以消费端校验为准。
// 成员 snapshot：RunTaskSnapshot。
// [/maint-note:fdcba544c25a]
export interface PreparedRunTaskSettlement {
  readonly snapshot: RunTaskSnapshot
}

// [maint-note:a9b775abf5f5]
// 数据契约导航：TaskPlanToolExecution；字段含义以消费端校验为准。
// 成员 ok：boolean。
// 成员 value：Record<string, unknown>。
// 成员 errorCode（可省略）：'tool_validation_error' | 'tool_execution_error'。
// 成员 display（可省略）：{ readonly kind: 'task_plan' readonly facts: Record<string, unknown> }。
// [/maint-note:a9b775abf5f5]
export interface TaskPlanToolExecution {
  readonly ok: boolean
  readonly value: Record<string, unknown>
  readonly errorCode?: 'tool_validation_error' | 'tool_execution_error'
  readonly display?: {
    readonly kind: 'task_plan'
    readonly facts: Record<string, unknown>
  }
}

/** Narrow execution port consumed by the model-visible task tool. */
// [maint-note:023ca9462645]
// 数据契约导航：TaskPlanToolPort；字段含义以消费端校验为准。
// 成员 executeTaskPlanOperation：TaskPlanToolExecution。
// [/maint-note:023ca9462645]
export interface TaskPlanToolPort {
  executeTaskPlanOperation(
    callId: string,
    operation: TaskPlanOperation,
    signal?: AbortSignal,
  ): TaskPlanToolExecution
}

// [maint-note:aac43e0c6e0d]
// 数据契约导航：RunTaskCall；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 name：string。
// 成员 arguments：Readonly<Record<string, unknown>>。
// [/maint-note:aac43e0c6e0d]
export interface RunTaskCall {
  readonly id: string
  readonly name: string
  readonly arguments: Readonly<Record<string, unknown>>
}

// [maint-note:c58ed69a2495]
// 数据契约导航：RunTaskBatchValidationError；字段含义以消费端校验为准。
// 成员 code：'tool_validation_error'。
// 成员 message：string。
// [/maint-note:c58ed69a2495]
export interface RunTaskBatchValidationError {
  readonly code: 'tool_validation_error'
  readonly message: string
}

// [maint-note:5834ffc8cb0e]
// 数据契约导航：RunTaskResultObservation；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 isError：boolean。
// 成员 code（可省略）：string。
// 成员 mutating：boolean。
// 成员 taskManagement：boolean。
// [/maint-note:5834ffc8cb0e]
export interface RunTaskResultObservation {
  readonly content: string
  readonly isError: boolean
  readonly code?: string
  readonly mutating: boolean
  readonly taskManagement: boolean
}

/** Read-only task reminder candidate; only its creating RunTaskPort may ack it. */
// [maint-note:50911592ab19]
// 数据契约导航：PreparedPlanReminder；字段含义以消费端校验为准。
// 成员 text：string。
// [/maint-note:50911592ab19]
export interface PreparedPlanReminder {
  readonly text: string
}

/**
 * Run-scoped framework port. AgentRunner owns call ordering while the
 * Session-owned ledger remains the sole mutable task-state owner.
 */
// [maint-note:7bee73310bbb]
// 数据契约导航：RunTaskPort；字段含义以消费端校验为准。
// 成员 runId：string。
// 成员 started：boolean。
// 成员 settled：boolean。
// 成员 begin：void。
// 成员 abandon：void。
// 成员 requestContext：string。
// 成员 prepareBatch：ReadonlyMap<string, RunTaskBatchValidationError>。
// 成员 observeExecutionStarted：void。
// 成员 observeResult：void。
// 成员 finishBatch：void。
// 成员 completionGate：'no_plan' | 'valid_claim' | 'missing_claim'。
// 成员 preparePlanReminder：PreparedPlanReminder | undefined。
// 成员 validatePlanReminder：boolean。
// 成员 publishPlanReminder：void。
// 成员 acknowledgePlanReminder：boolean。
// 成员 prepareSettle：PreparedRunTaskSettlement | undefined。
// 成员 prepareFallbackSettle：PreparedRunTaskSettlement | undefined。
// 成员 publishSettle：RunTaskSnapshot | undefined。
// [/maint-note:7bee73310bbb]
export interface RunTaskPort extends TaskPlanToolPort {
  readonly runId: string
  readonly started: boolean
  readonly settled: boolean
  begin(input: { readonly threadRun: ThreadRunIdentity }): void
  abandon(): void
  requestContext(): string
  prepareBatch(calls: readonly RunTaskCall[]): ReadonlyMap<string, RunTaskBatchValidationError>
  observeExecutionStarted(call: RunTaskCall, input: {
    readonly mutating: boolean
    readonly taskManagement: boolean
  }): void
  observeResult(call: RunTaskCall, result: RunTaskResultObservation): void
  finishBatch(calls: readonly RunTaskCall[], completed: boolean): void
  completionGate(): 'no_plan' | 'valid_claim' | 'missing_claim'
  /** Build the next reminder without consuming its run-local cadence. */
  preparePlanReminder(): PreparedPlanReminder | undefined
  /** Read-only freshness check for a prepared reminder. */
  validatePlanReminder(prepared: PreparedPlanReminder): boolean
  /** Non-throwing publish after the complete dispatch transaction validates. */
  publishPlanReminder(prepared: PreparedPlanReminder): void
  /** Confirm exactly the prepared reminder at a real normal-request dispatch. */
  acknowledgePlanReminder(prepared: PreparedPlanReminder): boolean
  prepareSettle(input: RunTaskSettleInput): PreparedRunTaskSettlement | undefined
  /** Fixed core-only close path that bypasses optional normal history preparation. */
  prepareFallbackSettle(input: RunTaskSettleInput): PreparedRunTaskSettlement | undefined
  publishSettle(prepared: PreparedRunTaskSettlement): RunTaskSnapshot | undefined
}

/** Read-only quality hook reserved for future integrations; not currently invoked. */
// [maint-note:c1df72602738]
// 数据契约导航：TaskCompletionVerifier；字段含义以消费端校验为准。
// 成员 verify：Promise<{ readonly outcome: 'passed' | 'failed' | 'unknown' readonly summary: string }>。
// [/maint-note:c1df72602738]
export interface TaskCompletionVerifier {
  verify(
    input: {
      readonly runId: string
      readonly plan: RunTaskPlan
      readonly claim: RunTaskClaim
      readonly observations: readonly RunToolObservation[]
    },
    signal?: AbortSignal,
  ): Promise<{
    readonly outcome: 'passed' | 'failed' | 'unknown'
    readonly summary: string
  }>
}
