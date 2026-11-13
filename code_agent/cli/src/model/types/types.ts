import type { ToolCall } from '../../tools/types.js'

// [maint-note:c5c77f3e64a1]
// 数据契约导航：ModelStopReason；字段含义以消费端校验为准。
// 类型组合：| 'stop' | 'toolUse' | 'length' | 'contentFilter' | 'maxTurns' | 'error' | 'unknown'。
// [/maint-note:c5c77f3e64a1]
export type ModelStopReason =
  | 'stop'
  | 'toolUse'
  | 'length'
  | 'contentFilter'
  | 'maxTurns'
  | 'error'
  | 'unknown'

// [maint-note:61568a31d437]
// 数据契约导航：ModelTextBlock；字段含义以消费端校验为准。
// 成员 type：'text'。
// 成员 text：string。
// [/maint-note:61568a31d437]
export interface ModelTextBlock {
  type: 'text'
  text: string
}

// [maint-note:caf6162214c5]
// 数据契约导航：ModelToolCallBlock；字段含义以消费端校验为准。
// 成员 type：'toolCall'。
// [/maint-note:caf6162214c5]
export interface ModelToolCallBlock extends ToolCall {
  type: 'toolCall'
}

// [maint-note:36c80a55f6fd]
// 数据契约导航：ModelContentBlock；字段含义以消费端校验为准。
// 类型组合：ModelTextBlock | ModelToolCallBlock。
// [/maint-note:36c80a55f6fd]
export type ModelContentBlock = ModelTextBlock | ModelToolCallBlock

// [maint-note:e008f516e125]
// 数据契约导航：ModelMessage；字段含义以消费端校验为准。
// 类型组合：| ModelSystemMessage | ModelUserMessage | ModelAssistantMessage | ModelToolResultMessage。
// [/maint-note:e008f516e125]
export type ModelMessage =
  | ModelSystemMessage
  | ModelUserMessage
  | ModelAssistantMessage
  | ModelToolResultMessage

// [maint-note:d65220e48ded]
// 数据契约导航：ModelSystemMessage；字段含义以消费端校验为准。
// 成员 role：'system'。
// 成员 content：ModelContentBlock[]。
// [/maint-note:d65220e48ded]
export interface ModelSystemMessage {
  role: 'system'
  content: ModelContentBlock[]
}

// [maint-note:34a57947e05e]
// 数据契约导航：ModelUserMessage；字段含义以消费端校验为准。
// 成员 role：'user'。
// 成员 content：ModelContentBlock[]。
// [/maint-note:34a57947e05e]
export interface ModelUserMessage {
  role: 'user'
  content: ModelContentBlock[]
}

// [maint-note:bf804c043f3e]
// 数据契约导航：ModelAssistantMessage；字段含义以消费端校验为准。
// 成员 role：'assistant'。
// 成员 content：ModelContentBlock[]。
// 成员 stopReason（可省略）：ModelStopReason。
// [/maint-note:bf804c043f3e]
export interface ModelAssistantMessage {
  role: 'assistant'
  content: ModelContentBlock[]
  stopReason?: ModelStopReason
}

// [maint-note:95a16ebb3fcc]
// 数据契约导航：ModelToolResultMessage；字段含义以消费端校验为准。
// 成员 role：'toolResult'。
// 成员 toolCallId：string。
// 成员 toolName：string。
// 成员 content：ModelContentBlock[]。
// 成员 isError：boolean。
// [/maint-note:95a16ebb3fcc]
export interface ModelToolResultMessage {
  role: 'toolResult'
  toolCallId: string
  toolName: string
  content: ModelContentBlock[]
  isError: boolean
}

// [maint-note:9debaa2dfeb0]
// 数据契约导航：ModelToolSchema；字段含义以消费端校验为准。
// 成员 name：string。
// 成员 description（可省略）：string。
// 成员 parameters：Record<string, unknown>。
// [/maint-note:9debaa2dfeb0]
export interface ModelToolSchema {
  name: string
  description?: string
  parameters: Record<string, unknown>
}
// [maint-note:6835cfa5650d]
// 数据契约导航：ModelToolCall；字段含义以消费端校验为准。
// 类型组合：ToolCall。
// [/maint-note:6835cfa5650d]
export type ModelToolCall = ToolCall

// [maint-note:b6ba52bc5546]
// 数据契约导航：ModelRequest；字段含义以消费端校验为准。
// 成员 model：string。
// 成员 messages：ModelMessage[]。
// 成员 tools（可省略）：ModelToolSchema[]。
// 成员 toolChoice（可省略）：'auto' | 'none' | 'required'。
// 成员 signal（可省略）：AbortSignal。
// 成员 budget（可省略）：ModelRequestBudget。
// [/maint-note:b6ba52bc5546]
export interface ModelRequest {
  model: string
  messages: ModelMessage[]
  tools?: ModelToolSchema[]
  toolChoice?: 'auto' | 'none' | 'required'
  signal?: AbortSignal
  /** Provider-enforced allowance requested by the run budget owner. */
  budget?: ModelRequestBudget
}

// [maint-note:767eb02354a6]
// 数据契约导航：ModelRequestBudget；字段含义以消费端校验为准。
// 成员 outputTokenLimit（可省略）：number。
// 成员 safetyMarginTokens（可省略）：number。
// [/maint-note:767eb02354a6]
export interface ModelRequestBudget {
  readonly outputTokenLimit?: number
  readonly safetyMarginTokens?: number
}

// [maint-note:864a16cc97cd]
// 数据契约导航：ModelDispatchMode；字段含义以消费端校验为准。
// 类型组合：'complete' | 'stream'。
// [/maint-note:864a16cc97cd]
export type ModelDispatchMode = 'complete' | 'stream'

// [maint-note:4816f0c1c407]
// 数据契约导航：ModelOutputLimitCapability；字段含义以消费端校验为准。
// 成员 field：'max_completion_tokens' | 'max_tokens'。
// 成员 covers：'visible_output' | 'completion_and_reasoning'。
// [/maint-note:4816f0c1c407]
export interface ModelOutputLimitCapability {
  readonly field: 'max_completion_tokens' | 'max_tokens'
  readonly covers: 'visible_output' | 'completion_and_reasoning'
}

// [maint-note:a023e0ecad1f]
// 数据契约导航：ModelBudgetCapabilities；字段含义以消费端校验为准。
// 成员 inputTokenEstimation：'exact_tokenizer' | 'provider_estimate' | 'char_fallback' | 'unknown'。
// 成员 contextWindowTokens（可省略）：number。
// 成员 tokenizerId（可省略）：string。
// 成员 outputLimit（可省略）：ModelOutputLimitCapability。
// 成员 transportRetryAccounting（可省略）：'single' | 'unknown'。
// [/maint-note:a023e0ecad1f]
export interface ModelBudgetCapabilities {
  readonly inputTokenEstimation: 'exact_tokenizer' | 'provider_estimate' | 'char_fallback' | 'unknown'
  readonly contextWindowTokens?: number
  readonly tokenizerId?: string
  readonly outputLimit?: ModelOutputLimitCapability
  readonly transportRetryAccounting?: 'single' | 'unknown'
}

// [maint-note:5df6c33d5839]
// 数据契约导航：ModelRequestMeasurement；字段含义以消费端校验为准。
// 成员 providerShapeId：string。
// 成员 serializedBodyBytes：number。
// 成员 inputTokens（可省略）：number。
// 成员 contextWindowTokens（可省略）：number。
// 成员 trust：ModelBudgetCapabilities['inputTokenEstimation']。
// 成员 outputLimit（可省略）：ModelOutputLimitCapability & { readonly enforcement: 'provider' readonly tokens: number }。
// [/maint-note:5df6c33d5839]
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

// [maint-note:437677e0fe92]
// 数据契约导航：PreparedModelDispatch；字段含义以消费端校验为准。
// 成员 mode：ModelDispatchMode。
// 成员 measurement：ModelRequestMeasurement。
// 成员 send：Promise<ModelResponse> | AsyncIterable<ModelStreamEvent>。
// [/maint-note:437677e0fe92]
export interface PreparedModelDispatch {
  readonly mode: ModelDispatchMode
  readonly measurement: ModelRequestMeasurement
  send(): Promise<ModelResponse> | AsyncIterable<ModelStreamEvent>
}

// [maint-note:9b8303d95ff9]
// 数据契约导航：ModelUsage；字段含义以消费端校验为准。
// 成员 inputTokens（可省略）：number。
// 成员 outputTokens（可省略）：number。
// 成员 totalTokens（可省略）：number。
// 成员 cacheHitInputTokens（可省略）：number。
// 成员 cacheMissInputTokens（可省略）：number。
// [/maint-note:9b8303d95ff9]
export interface ModelUsage {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  /** Provider-reported cache detail; absence remains unknown, not zero. */
  cacheHitInputTokens?: number
  cacheMissInputTokens?: number
}

// [maint-note:d5aa0c476480]
// 数据契约导航：ModelResponse；字段含义以消费端校验为准。
// 成员 message：ModelAssistantMessage。
// 成员 content：string。
// 成员 toolCalls：ModelToolCall[]。
// 成员 stopReason：ModelStopReason。
// 成员 raw：unknown。
// 成员 usage（可省略）：ModelUsage。
// [/maint-note:d5aa0c476480]
export interface ModelResponse {
  message: ModelAssistantMessage
  content: string
  toolCalls: ModelToolCall[]
  stopReason: ModelStopReason
  raw: unknown
  usage?: ModelUsage
}

// [maint-note:b4618fb21547]
// 数据契约导航：ModelStreamEvent；字段含义以消费端校验为准。
// 类型组合：| { type: 'text_delta'; text: string } | { type: 'tool_call_delta' index: number id?: string name?: string argumentsDelta?: string } | { type: 'finish'; stopReason: ModelStopReason } | { type: 'usage'; usage: ModelUsage }。
// [/maint-note:b4618fb21547]
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

// [maint-note:7edb8f4fe76d]
// 数据契约导航：ModelClient；字段含义以消费端校验为准。
// 成员 budgetCapabilities（可省略）：ModelBudgetCapabilities。
// 成员 prepare（可省略）：PreparedModelDispatch。
// 成员 complete：Promise<ModelResponse>。
// 成员 stream（可省略）：AsyncIterable<ModelStreamEvent>。
// [/maint-note:7edb8f4fe76d]
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

// [maint-note:10ac24ea800d]
// 数据契约导航：ModelAdapter；字段含义以消费端校验为准。
// 成员 toProviderRequest：ProviderRequest。
// 成员 fromProviderResponse：ModelResponse。
// [/maint-note:10ac24ea800d]
export interface ModelAdapter<ProviderRequest, ProviderResponse> {
  toProviderRequest(request: ModelRequest): ProviderRequest
  fromProviderResponse(response: ProviderResponse): ModelResponse
}

// [maint-note:a13eeddd0f9c]
// 数据契约导航：modelClient；字段含义以消费端校验为准。
// 类型组合：ModelClient。
// [/maint-note:a13eeddd0f9c]
export type modelClient = ModelClient
