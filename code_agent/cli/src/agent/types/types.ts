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
}

/**
 * @deprecated Kept as the M0 configuration compatibility boundary. New app
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
