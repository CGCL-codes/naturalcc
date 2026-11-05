import type {
  ThreadCloseReceipt,
  ThreadRunHistoryReceipt,
  ThreadRunIdentity,
  ThreadRunSourceRef,
} from '../agent/thread/AgentThread.js'

export type TaskStepStatus = 'pending' | 'in_progress' | 'completed'

export interface TaskPlanStep {
  readonly id: string
  readonly content: string
  readonly status: TaskStepStatus
}

export interface RunTaskPlan {
  readonly id: string
  readonly revision: number
  readonly steps: readonly TaskPlanStep[]
}

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

export type RunTaskTerminal =
  | { readonly type: 'done' }
  | { readonly type: 'paused'; readonly reason: 'budget_exhausted' | 'no_progress' }
  | { readonly type: 'interrupted' }
  | { readonly type: 'error'; readonly code: string }

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

export interface RunTaskSettleInput {
  readonly terminal: RunTaskTerminal
  readonly history: ThreadCloseReceipt
  readonly workspaceMayHaveChanged: boolean
}

/** Opaque-by-ownership prepared value; only its creating RunTaskPort may publish it. */
export interface PreparedRunTaskSettlement {
  readonly snapshot: RunTaskSnapshot
}

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
export interface TaskPlanToolPort {
  executeTaskPlanOperation(
    callId: string,
    operation: TaskPlanOperation,
    signal?: AbortSignal,
  ): TaskPlanToolExecution
}

export interface RunTaskCall {
  readonly id: string
  readonly name: string
  readonly arguments: Readonly<Record<string, unknown>>
}

export interface RunTaskBatchValidationError {
  readonly code: 'tool_validation_error'
  readonly message: string
}

export interface RunTaskResultObservation {
  readonly content: string
  readonly isError: boolean
  readonly code?: string
  readonly mutating: boolean
  readonly taskManagement: boolean
}

/** Read-only task reminder candidate; only its creating RunTaskPort may ack it. */
export interface PreparedPlanReminder {
  readonly text: string
}

/**
 * Run-scoped framework port. AgentRunner owns call ordering while the
 * Session-owned ledger remains the sole mutable task-state owner.
 */
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
