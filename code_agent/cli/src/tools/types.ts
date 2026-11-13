// [maint-note:5e861e607bd6]
// 数据契约导航：ToolSchema；字段含义以消费端校验为准。
// 成员 name：string。
// 成员 description（可省略）：string。
// 成员 parameters：Record<string, unknown>。
// 成员 readOnly（可省略）：boolean。
// 成员 requiresApproval（可省略）：boolean。
// 成员 repeatKind（可省略）：'read' | 'workspaceMutation' | 'opaque'。
// 成员 taskManagement（可省略）：boolean。
// [/maint-note:5e861e607bd6]
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

// [maint-note:67bc3eaed6f4]
// 数据契约导航：ToolDiagnostic；字段含义以消费端校验为准。
// 成员 reason：string。
// 成员 message：string。
// 成员 field（可省略）：string。
// 成员 expected（可省略）：string。
// 成员 hint（可省略）：string。
// [/maint-note:67bc3eaed6f4]
export interface ToolDiagnostic {
  readonly reason: string
  readonly message: string
  readonly field?: string
  readonly expected?: string
  readonly hint?: string
}

/** Metadata never determines the canonical code or execution phase. */
export class ToolDiagnosticError extends Error {
  // [maint-note:2a4a610ca768]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 message：string。
  // 输入 diagnostic：ToolDiagnostic（可省略）。
  // 直接调用：super。
  // [/maint-note:2a4a610ca768]
  constructor(message: string, readonly diagnostic?: ToolDiagnostic) {
    super(message)
    // Metadata is identified by instanceof, not the public error name.
    // Existing execution failures remain ordinary Error to direct consumers.
  }
}

export class ToolInputError extends ToolDiagnosticError {
  readonly code = 'tool_validation_error' as const

  // [maint-note:b3c02dc6ffb5]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 message：string。
  // 输入 diagnostic：ToolDiagnostic（可省略）。
  // 写入目标：this.name。
  // 直接调用：super。
  // [/maint-note:b3c02dc6ffb5]
  constructor(message: string, diagnostic?: ToolDiagnostic) {
    super(message, diagnostic)
    this.name = 'ToolInputError'
  }
}

// [maint-note:8f053b5937a2]
// 数据契约导航：ToolCall；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 name：string。
// 成员 arguments：Record<string, unknown>。
// 成员 rawArguments（可省略）：string。
// 成员 argumentsError（可省略）：string。
// [/maint-note:8f053b5937a2]
export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
  rawArguments?: string
  argumentsError?: string
}

// [maint-note:f6020542ba72]
// 数据契约导航：ToolExecutionContext；字段含义以消费端校验为准。
// 成员 projectDir：string。
// 成员 signal（可省略）：AbortSignal。
// 成员 timeoutMs（可省略）：number。
// 成员 maxOutputChars（可省略）：number。
// 成员 env（可省略）：NodeJS.ProcessEnv。
// 成员 canonicalToolCallId（可省略）：string。
// 成员 taskPlan（可省略）：import('../task/contracts.js').TaskPlanToolPort。
// [/maint-note:f6020542ba72]
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

// [maint-note:82816b57602b]
// 数据契约导航：ToolExecutionResult；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 isError（可省略）：boolean。
// 成员 errorCode（可省略）：string。
// 成员 display（可省略）：ToolDisplayFacts。
// [/maint-note:82816b57602b]
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
// [maint-note:6472f1b33013]
// 数据契约导航：ToolDisplayFacts；字段含义以消费端校验为准。
// 成员 kind：string。
// 成员 facts：Record<string, unknown>。
// [/maint-note:6472f1b33013]
export interface ToolDisplayFacts {
  /** Stable presenter kind routing this payload to a projection. */
  kind: string
  /** Bounded payload validated by the presenter that consumes it. */
  facts: Record<string, unknown>
}

// [maint-note:bcba27764894]
// 数据契约导航：Tool；字段含义以消费端校验为准。
// 成员 parse（可省略）：unknown。
// 成员 execute：Promise<ToolExecutionResult>。
// [/maint-note:bcba27764894]
export interface Tool extends ToolSchema {
  parse?(input: Record<string, unknown>): unknown
  execute(
    input: Record<string, unknown>,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}

// [maint-note:43fcc1ed5fd0]
// 数据契约导航：ToolRegistry；字段含义以消费端校验为准。
// 成员 schemas：ToolSchema[]。
// 成员 validate（可省略）：void。
// 成员 execute：Promise<ToolExecutionResult>。
// [/maint-note:43fcc1ed5fd0]
export interface ToolRegistry {
  schemas(): ToolSchema[]
  validate?(call: ToolCall): void
  execute(
    call: ToolCall,
    context: ToolExecutionContext,
  ): Promise<ToolExecutionResult>
}
