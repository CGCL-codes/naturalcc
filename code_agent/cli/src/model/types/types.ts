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
}

export interface ModelUsage {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
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
  complete(request: ModelRequest): Promise<ModelResponse>
  /** Optional normalized stream; clients without streaming use complete(). */
  stream?(request: ModelRequest): AsyncIterable<ModelStreamEvent>
}

export interface ModelAdapter<ProviderRequest, ProviderResponse> {
  toProviderRequest(request: ModelRequest): ProviderRequest
  fromProviderResponse(response: ProviderResponse): ModelResponse
}

export type modelClient = ModelClient
