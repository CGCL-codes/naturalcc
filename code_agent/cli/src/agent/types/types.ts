import type { ModelClient } from '../../model/types/types.js'
import type { ToolRegistry, ToolExecutionContext } from '../../tools/types.js'
import type { ToolPolicy } from '../../tools/approval.js'

export interface ModelConnectionConfig {
  model: string
  apiKey: string
  baseURL?: string
}

export interface AgentLoopConfig {
  maxTurns: number
  maxHistoryChars: number
  /** App/test-only budgets; maxTurns remains the absolute request cap. */
  runBudget?: RunBudgetConfig
}

export interface RunBudgetConfig {
  readonly segmentMaxModelRequests?: number
  readonly maxToolAttempts?: number
  readonly maxElapsedMs?: number
  readonly maxTotalTokens?: number
  readonly maxCostMicrousd?: number
  readonly maxOutputTokensPerRequest?: number
  readonly safetyMarginTokens?: number
}

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

export interface AgentRunOptions {
  prompt: string
  runtime: AgentRuntime
  thread?: import('../thread/AgentThread.js').AgentThread
  signal?: AbortSignal
}
