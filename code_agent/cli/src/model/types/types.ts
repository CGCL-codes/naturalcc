import type { ToolCall } from '../../tools/types.js'

export type ModelStopReason =
  | 'stop'
  | 'toolUse'
  | 'length'
  | 'contentFilter'
  | 'maxTurns'
  | 'error'
  | 'unknown'

export interface ModelTextBlock {
  type: 'text'
  text: string
}

export interface ModelToolCallBlock extends ToolCall {
  type: 'toolCall'
}

export type ModelContentBlock = ModelTextBlock | ModelToolCallBlock

export type ModelMessage =
  | ModelSystemMessage
  | ModelUserMessage
  | ModelAssistantMessage
  | ModelToolResultMessage

export interface ModelSystemMessage {
  role: 'system'
  content: ModelContentBlock[]
}

export interface ModelUserMessage {
  role: 'user'
  content: ModelContentBlock[]
}

export interface ModelAssistantMessage {
  role: 'assistant'
  content: ModelContentBlock[]
  stopReason?: ModelStopReason
}

export interface ModelToolResultMessage {
  role: 'toolResult'
  toolCallId: string
  toolName: string
  content: ModelContentBlock[]
  isError: boolean
}

export interface ModelToolSchema {
  name: string
  description?: string
  parameters: Record<string, unknown>
}
export type ModelToolCall = ToolCall

export interface ModelRequest {
  model: string
  messages: ModelMessage[]
  tools?: ModelToolSchema[]
  toolChoice?: 'auto' | 'none' | 'required'
  signal?: AbortSignal
  /** Provider-enforced allowance requested by the run budget owner. */
  budget?: ModelRequestBudget
}

export interface ModelRequestBudget {
  readonly outputTokenLimit?: number
  readonly safetyMarginTokens?: number
}

export type ModelDispatchMode = 'complete' | 'stream'

export interface ModelOutputLimitCapability {
  readonly field: 'max_completion_tokens' | 'max_tokens'
  readonly covers: 'visible_output' | 'completion_and_reasoning'
}

export interface ModelBudgetCapabilities {
  readonly inputTokenEstimation: 'exact_tokenizer' | 'provider_estimate' | 'char_fallback' | 'unknown'
  readonly contextWindowTokens?: number
  readonly tokenizerId?: string
  readonly outputLimit?: ModelOutputLimitCapability
  readonly transportRetryAccounting?: 'single' | 'unknown'
}

export interface ModelRequestMeasurement {
  readonly providerShapeId: string
  readonly serializedBodyBytes: number
  readonly inputTokens?: number
  readonly contextWindowTokens?: number
  readonly trust: ModelBudgetCapabilities['inputTokenEstimation']
  readonly outputLimit?: ModelOutputLimitCapability & {
    readonly enforcement: 'provider'
    readonly tokens: number
  }
}

export interface PreparedModelDispatch {
  readonly mode: ModelDispatchMode
  readonly measurement: ModelRequestMeasurement
  send(): Promise<ModelResponse> | AsyncIterable<ModelStreamEvent>
}

export interface ModelUsage {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  /** Provider-reported cache detail; absence remains unknown, not zero. */
  cacheHitInputTokens?: number
  cacheMissInputTokens?: number
}

export interface ModelResponse {
  message: ModelAssistantMessage
  content: string
  toolCalls: ModelToolCall[]
  stopReason: ModelStopReason
  raw: unknown
  usage?: ModelUsage
}

export type ModelStreamEvent =
  | { type: 'text_delta'; text: string }
  | {
      type: 'tool_call_delta'
      index: number
      id?: string
      name?: string
      argumentsDelta?: string
  }
  | { type: 'finish'; stopReason: ModelStopReason }
  | { type: 'usage'; usage: ModelUsage }

export interface ModelClient {
  readonly budgetCapabilities?: ModelBudgetCapabilities
  /**
   * Optional provider-aware boundary. Its measurement and send closure must
   * refer to the same immutable final provider body.
   */
  prepare?(request: ModelRequest, mode: ModelDispatchMode): PreparedModelDispatch
  complete(request: ModelRequest): Promise<ModelResponse>
  /** Optional normalized stream; clients without streaming use complete(). */
  stream?(request: ModelRequest): AsyncIterable<ModelStreamEvent>
}

export interface ModelAdapter<ProviderRequest, ProviderResponse> {
  toProviderRequest(request: ModelRequest): ProviderRequest
  fromProviderResponse(response: ProviderResponse): ModelResponse
}

export type modelClient = ModelClient
