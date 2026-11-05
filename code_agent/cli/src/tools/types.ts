export interface ToolSchema {
  name: string
  description?: string
  parameters: Record<string, unknown>
  readOnly?: boolean
  requiresApproval?: boolean
  /** Local execution metadata; never sent to the model provider. */
  repeatKind?: 'read' | 'workspaceMutation' | 'opaque'
  /** Run-task bookkeeping tool; excluded from workspace progress accounting. */
  taskManagement?: boolean
}

export interface ToolDiagnostic {
  readonly reason: string
  readonly message: string
  readonly field?: string
  readonly expected?: string
  readonly hint?: string
}

/** Metadata never determines the canonical code or execution phase. */
export class ToolDiagnosticError extends Error {
  constructor(message: string, readonly diagnostic?: ToolDiagnostic) {
    super(message)
    // Metadata is identified by instanceof, not the public error name.
    // Existing execution failures remain ordinary Error to direct consumers.
  }
}

export class ToolInputError extends ToolDiagnosticError {
  readonly code = 'tool_validation_error' as const

  constructor(message: string, diagnostic?: ToolDiagnostic) {
    super(message, diagnostic)
    this.name = 'ToolInputError'
  }
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
  rawArguments?: string
  argumentsError?: string
}

export interface ToolExecutionContext {
  projectDir: string
  signal?: AbortSignal
  timeoutMs?: number
  maxOutputChars?: number
  /** Optional subprocess environment, primarily for isolated harnesses. */
  env?: NodeJS.ProcessEnv
  /** Canonical framework identity; never accepted from model arguments. */
  canonicalToolCallId?: string
  /** Run-scoped task owner; shared registries must never capture a session. */
  taskPlan?: import('../task/contracts.js').TaskPlanToolPort
}

export interface ToolExecutionResult {
  /**
   * Canonical tool fact for the model, AgentThread, one-shot scripts and
   * future transcripts. Never carries ANSI/JSX; UI folding never rewrites
   * it.
   */
  content: string
  /** Optional canonical failure returned without throwing from a stateful tool. */
  isError?: boolean
  /** Stable failure code paired with `isError`; never inferred from display text. */
  errorCode?: string
  /**
   * additive, local, disposable UI display facts produced by the tool's
   * canonical executor on SUCCESS only. Third-party and older tools without
   * `display` keep working; the field never enters ModelToolResultMessage,
   * AgentThread requests, or the model token budget.
   */
  display?: ToolDisplayFacts
}

/** Neutral, provider-independent display facts; no JSX, no ANSI. */
export interface ToolDisplayFacts {
  /** Stable presenter kind routing this payload to a projection. */
  kind: string
  /** Bounded payload validated by the presenter that consumes it. */
  facts: Record<string, unknown>
}

export interface Tool extends ToolSchema {
  parse?(input: Record<string, unknown>): unknown
  execute(
    input: Record<string, unknown>,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}

export interface ToolRegistry {
  schemas(): ToolSchema[]
  validate?(call: ToolCall): void
  execute(
    call: ToolCall,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}
