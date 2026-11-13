import type { ModelClient } from '../../model/types/types.js'
import type { ToolRegistry, ToolExecutionContext } from '../../tools/types.js'
import type { ToolPolicy } from '../../tools/approval.js'

// [maint-note:3666a23a8357]
// 数据契约导航：ModelConnectionConfig；字段含义以消费端校验为准。
// 成员 model：string。
// 成员 apiKey：string。
// 成员 baseURL（可省略）：string。
// [/maint-note:3666a23a8357]
export interface ModelConnectionConfig {
  model: string
  apiKey: string
  baseURL?: string
}

// [maint-note:50f24a9b86af]
// 数据契约导航：AgentLoopConfig；字段含义以消费端校验为准。
// 成员 maxTurns：number。
// 成员 maxHistoryChars：number。
// 成员 runBudget（可省略）：RunBudgetConfig。
// [/maint-note:50f24a9b86af]
export interface AgentLoopConfig {
  maxTurns: number
  maxHistoryChars: number
  /** App/test-only budgets; maxTurns remains the absolute request cap. */
  runBudget?: RunBudgetConfig
}

// [maint-note:3f7cc9d58cee]
// 数据契约导航：RunBudgetConfig；字段含义以消费端校验为准。
// 成员 segmentMaxModelRequests（可省略）：number。
// 成员 maxToolAttempts（可省略）：number。
// 成员 maxElapsedMs（可省略）：number。
// 成员 maxTotalTokens（可省略）：number。
// 成员 maxCostMicrousd（可省略）：number。
// 成员 maxOutputTokensPerRequest（可省略）：number。
// 成员 safetyMarginTokens（可省略）：number。
// [/maint-note:3f7cc9d58cee]
export interface RunBudgetConfig {
  readonly segmentMaxModelRequests?: number
  readonly maxToolAttempts?: number
  readonly maxElapsedMs?: number
  readonly maxTotalTokens?: number
  readonly maxCostMicrousd?: number
  readonly maxOutputTokensPerRequest?: number
  readonly safetyMarginTokens?: number
}

// [maint-note:81858477b11e]
// 数据契约导航：CodingAgentConfig；字段含义以消费端校验为准。
// 成员 projectDir：string。
// 成员 toolTimeoutMs：number。
// 成员 maxToolOutputChars：number。
// 成员 maxProjectMemoryChars：number。
// 成员 maxDynamicContextChars：number。
// 成员 maxSystemPromptChars：number。
// 成员 showToolResults：boolean。
// 成员 approvalMode：import('../../tools/approval.js').ApprovalMode。
// [/maint-note:81858477b11e]
export interface CodingAgentConfig {
  projectDir: string
  toolTimeoutMs: number
  maxToolOutputChars: number
  maxProjectMemoryChars: number
  maxDynamicContextChars: number
  maxSystemPromptChars: number
  showToolResults: boolean
  approvalMode: import('../../tools/approval.js').ApprovalMode
}

// [maint-note:f977b729a5a3]
// 数据契约导航：AgentRuntime；字段含义以消费端校验为准。
// 成员 modelClient：ModelClient。
// 成员 model：string。
// 成员 tools：ToolRegistry。
// 成员 loop：AgentLoopConfig。
// 成员 toolContext：ToolExecutionContext。
// 成员 showToolResults：boolean。
// 成员 toolPolicy（可省略）：ToolPolicy。
// 成员 compactionProvider（可省略）：import('../compaction/contracts.js').CompactionProvider。
// [/maint-note:f977b729a5a3]
export interface AgentRuntime {
  modelClient: ModelClient
  model: string
  tools: ToolRegistry
  loop: AgentLoopConfig
  toolContext: ToolExecutionContext
  showToolResults: boolean
  toolPolicy?: ToolPolicy
  /** sourced context compaction; app injects the model-backed default. */
  compactionProvider?: import('../compaction/contracts.js').CompactionProvider
}

/**
 * @deprecated Kept as the legacy configuration compatibility boundary. New app
 * code should use the three split configuration types above.
 */
// [maint-note:ef9d1625afeb]
// 数据契约导航：RuntimeConfig；字段含义以消费端校验为准。
// 成员 model：string。
// 成员 baseURL（可省略）：string。
// 成员 maxTurns：number。
// 成员 projectDir：string。
// 成员 toolTimeoutMs：number。
// 成员 maxToolOutputChars：number。
// 成员 maxProjectMemoryChars：number。
// 成员 maxDynamicContextChars：number。
// 成员 maxSystemPromptChars：number。
// 成员 showToolResults：boolean。
// 成员 approvalMode（可省略）：import('../../tools/approval.js').ApprovalMode。
// [/maint-note:ef9d1625afeb]
export interface RuntimeConfig {
  model: string
  baseURL?: string
  maxTurns: number
  projectDir: string
  toolTimeoutMs: number
  maxToolOutputChars: number
  maxProjectMemoryChars: number
  maxDynamicContextChars: number
  maxSystemPromptChars: number
  showToolResults: boolean
  approvalMode?: import('../../tools/approval.js').ApprovalMode
}

// [maint-note:5fe71ace9260]
// 数据契约导航：AgentRunOptions；字段含义以消费端校验为准。
// 成员 prompt：string。
// 成员 runtime：AgentRuntime。
// 成员 thread（可省略）：import('../thread/AgentThread.js').AgentThread。
// 成员 signal（可省略）：AbortSignal。
// [/maint-note:5fe71ace9260]
export interface AgentRunOptions {
  prompt: string
  runtime: AgentRuntime
  thread?: import('../thread/AgentThread.js').AgentThread
  signal?: AbortSignal
}
