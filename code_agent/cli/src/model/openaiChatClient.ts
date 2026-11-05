import OpenAI from 'openai'
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
} from 'openai/resources/chat/completions'

import type {
  ModelClient,
  ModelBudgetCapabilities,
  ModelContentBlock,
  ModelDispatchMode,
  ModelStopReason,
  ModelMessage,
  ModelRequest,
  PreparedModelDispatch,
  ModelResponse,
  ModelStreamEvent,
  ModelToolCall,
  ModelToolSchema,
  ModelUsage,
} from './types/types.js'
import { BaseModelAdapter } from './baseAdapter.js'
import {
  detectToolControlEnvelope,
} from './controlEnvelope.js'
import { ModelProtocolError } from './errors.js'

export interface OpenAIChatModelClientOptions {
  apiKey: string
  baseURL?: string
  client?: OpenAIChatTransport
  /** Trusted endpoint/model capability supplied by the application. */
  requestBudgetCapability?: ModelBudgetCapabilities & {
    readonly countInputTokens?: (
      body: Readonly<Record<string, unknown>>,
    ) => number
  }
}

export class OpenAIChatAdapter extends BaseModelAdapter<
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletion
> {
  toProviderRequest(request: ModelRequest): ChatCompletionCreateParamsNonStreaming {
    const tools = request.tools?.map(toOpenAITool)

    return {
      model: request.model,
      messages: request.messages.map(toOpenAIMessage),
      tools,
      tool_choice: request.toolChoice ?? (tools?.length ? 'auto' : undefined),
    }
  }

  fromProviderResponse(response: ChatCompletion): ModelResponse {
    const choice = response.choices[0]
    const message = choice?.message
    if (!message) {
      throw new ModelProtocolError('OpenAI response did not include a message')
    }
    const content = message.content ?? ''
    const toolCalls = (message.tool_calls ?? []).map((call) => this.fromOpenAIToolCall(call))
    const stopReason = mapOpenAIStopReason(choice.finish_reason)

    return {
      message: {
        role: 'assistant',
        content: [
          ...(content ? [{ type: 'text' as const, text: content }] : []),
          ...toolCalls.map(toToolCallBlock),
        ],
        stopReason,
      },
      content,
      toolCalls,
      stopReason,
      raw: response,
      usage: response.usage ? toModelUsage(response.usage) : undefined,
    }
  }

  private fromOpenAIToolCall(call: ChatCompletionMessageToolCall): ModelToolCall {
    if (call.type === 'function') {
      const parsed = this.parseJsonObject(call.function.arguments)
      return {
        id: call.id,
        name: call.function.name,
        arguments: parsed.value,
        rawArguments: call.function.arguments,
        argumentsError: parsed.error,
      }
    }

    const parsed = this.parseJsonObject(call.custom.input)
    return {
      id: call.id,
      name: call.custom.name,
      arguments: parsed.value,
      rawArguments: call.custom.input,
      argumentsError: parsed.error,
    }
  }
}

export interface OpenAIChatTransport {
  chat: {
    completions: {
      create(
        request: ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming,
        options?: { signal?: AbortSignal },
      ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>
    }
  }
}

export class OpenAIChatModelClient implements ModelClient {
  private readonly client: OpenAIChatTransport
  private readonly adapter: OpenAIChatAdapter
  readonly budgetCapabilities?: ModelBudgetCapabilities
  private readonly requestBudgetCapability?: OpenAIChatModelClientOptions['requestBudgetCapability']

  constructor(options: OpenAIChatModelClientOptions) {
    this.client = options.client ?? new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    })
    this.adapter = new OpenAIChatAdapter()
    this.requestBudgetCapability = options.requestBudgetCapability
    this.budgetCapabilities = options.requestBudgetCapability
      ? Object.freeze({
          // A counting callback implements the declared estimator; its mere
          // presence is not evidence that the tokenizer/framing is exact.
          inputTokenEstimation: options.requestBudgetCapability.inputTokenEstimation,
          contextWindowTokens: options.requestBudgetCapability.contextWindowTokens,
          tokenizerId: options.requestBudgetCapability.tokenizerId,
          outputLimit: options.requestBudgetCapability.outputLimit,
          transportRetryAccounting: options.requestBudgetCapability.transportRetryAccounting,
        })
      : undefined
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const sent = this.prepare(request, 'complete').send()
    if (isAsyncIterable(sent)) throw new ModelProtocolError('Prepared complete request returned a stream')
    return sent as Promise<ModelResponse>
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
    const sent = this.prepare(request, 'stream').send()
    if (!isAsyncIterable(sent)) throw new ModelProtocolError('Prepared stream request returned a response')
    yield* sent as AsyncIterable<ModelStreamEvent>
  }

  prepare(request: ModelRequest, mode: ModelDispatchMode): PreparedModelDispatch {
    const providerRequest = this.finalProviderRequest(request, mode)
    const serializedBodyBytes = Buffer.byteLength(JSON.stringify(providerRequest), 'utf8')
    const inputTokens = this.requestBudgetCapability?.countInputTokens?.(
      providerRequest as unknown as Readonly<Record<string, unknown>>,
    )
    const outputTokens = request.budget?.outputTokenLimit
    const outputCapability = this.requestBudgetCapability?.outputLimit
    const measurement = Object.freeze({
      providerShapeId: mode === 'stream' ? 'openai-chat-stream-v1' : 'openai-chat-complete-v1',
      serializedBodyBytes,
      ...(inputTokens === undefined ? {} : { inputTokens }),
      ...(this.requestBudgetCapability?.contextWindowTokens === undefined
        ? {}
        : { contextWindowTokens: this.requestBudgetCapability.contextWindowTokens }),
      // A callback implements the configured estimator. It does not prove
      // that estimator is an exact tokenizer/framing implementation.
      trust: (this.requestBudgetCapability?.inputTokenEstimation ?? 'unknown') as ModelBudgetCapabilities['inputTokenEstimation'],
      ...(outputTokens === undefined || !outputCapability
        ? {}
        : {
            outputLimit: {
              ...outputCapability,
              enforcement: 'provider' as const,
              tokens: outputTokens,
            },
          }),
    })
    let sent = false
    return Object.freeze({
      mode,
      measurement,
      send: () => {
        if (sent) throw new ModelProtocolError('A prepared model dispatch can only be sent once')
        sent = true
        return mode === 'stream'
          ? this.sendStream(request, providerRequest as ChatCompletionCreateParamsStreaming)
          : this.sendComplete(request, providerRequest as ChatCompletionCreateParamsNonStreaming)
      },
    })
  }

  private finalProviderRequest(
    request: ModelRequest,
    mode: ModelDispatchMode,
  ): ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming {
    const body = structuredClone(
      this.adapter.toProviderRequest(request),
    ) as unknown as Record<string, unknown>
    const outputTokens = request.budget?.outputTokenLimit
    const outputCapability = this.requestBudgetCapability?.outputLimit
    if (outputTokens !== undefined) {
      if (!outputCapability) {
        throw new ModelProtocolError('The configured request output token cap is not supported by this endpoint')
      }
      body[outputCapability.field] = outputTokens
    }
    if (mode === 'stream') {
      body.stream = true
      body.stream_options = { include_usage: true }
    }
    return deepFreeze(body) as unknown as ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming
  }

  private async sendComplete(
    request: ModelRequest,
    providerRequest: ChatCompletionCreateParamsNonStreaming,
  ): Promise<ModelResponse> {
    const response = await this.client.chat.completions.create(providerRequest, { signal: request.signal })
    if (isAsyncIterable(response)) throw new ModelProtocolError('OpenAI returned a stream for a non-streaming request')
    const modelResponse = this.adapter.fromProviderResponse(response)
    assertNoTextToolEnvelope(request, modelResponse.content)
    return modelResponse
  }

  private async *sendStream(
    request: ModelRequest,
    providerRequest: ChatCompletionCreateParamsStreaming,
  ): AsyncIterable<ModelStreamEvent> {
    const response = await this.client.chat.completions.create(providerRequest, { signal: request.signal })
    if (!isAsyncIterable(response)) {
      throw new ModelProtocolError('OpenAI did not return a streaming response')
    }

    let pendingUsage: ModelUsage | undefined
    let usageEmitted = false
    let finishEmitted = false
    for await (const chunk of response) {
      if (chunk.usage) {
        if (usageEmitted) {
          throw new ModelProtocolError('OpenAI stream returned usage more than once')
        }
        pendingUsage = toModelUsage(chunk.usage)
      }
      const choice = chunk.choices[0]
      if (choice) {
        const text = choice.delta.content
        if (text) {
          yield { type: 'text_delta', text }
        }
        for (const toolCall of choice.delta.tool_calls ?? []) {
          yield {
            type: 'tool_call_delta',
            index: toolCall.index,
            id: toolCall.id,
            name: toolCall.function?.name,
            argumentsDelta: toolCall.function?.arguments,
          }
        }
        if (choice.finish_reason) {
          yield {
            type: 'finish',
            stopReason: mapOpenAIStopReason(choice.finish_reason),
          }
          finishEmitted = true
        }
      }
      if (pendingUsage && finishEmitted) {
        yield { type: 'usage', usage: pendingUsage }
        pendingUsage = undefined
        usageEmitted = true
      }
    }
    if (pendingUsage) {
      // Preserve the model protocol's missing-finish diagnosis rather than
      // making provider chunk order observable to consumers.
      yield { type: 'usage', usage: pendingUsage }
    }
  }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child)
    Object.freeze(value)
  }
  return value
}

/**
 * Provider response boundary: an endpoint that answers an
 * ordinary tool-bearing request by putting its tool-call control envelope in
 * text content is reported as a protocol error here. The envelope is never
 * parsed or canonicalized into a call, so no pseudo tool call can execute.
 *
 * Tool-free closing requests are deliberately excluded: the Runner owns that
 * path, including its single in-budget plain-text retry and the stable
 * `model_protocol_error` it raises when the retry is unavailable or fails.
 */
function assertNoTextToolEnvelope(request: ModelRequest, content: string): void {
  const offeredTools = (request.tools?.length ?? 0) > 0 && request.toolChoice !== 'none'
  if (!offeredTools) return
  const diagnostic = detectToolControlEnvelope(content)
  if (!diagnostic) return
  throw new ModelProtocolError(
    `OpenAI response returned a tool-control envelope as text content (${diagnostic.id})`,
  )
}

function toModelUsage(usage: {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number | null } | null
}): ModelUsage {
  const cached = usage.prompt_tokens_details?.cached_tokens
  const cacheMiss = cached !== undefined && cached !== null && usage.prompt_tokens !== undefined
    ? Math.max(0, usage.prompt_tokens - cached)
    : undefined
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    ...(cached === undefined || cached === null ? {} : { cacheHitInputTokens: cached }),
    ...(cacheMiss === undefined ? {} : { cacheMissInputTokens: cacheMiss }),
  }
}

function isAsyncIterable(value: unknown): value is AsyncIterable<ChatCompletionChunk> {
  return Boolean(value && typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === 'function')
}

function toOpenAIMessage(message: ModelMessage): ChatCompletionMessageParam {
  switch (message.role) {
    case 'system':
      return { role: 'system', content: textFromContent(message.content) }
    case 'user':
      return { role: 'user', content: textFromContent(message.content) }
    case 'toolResult':
      return {
        role: 'tool',
        tool_call_id: message.toolCallId,
        content: textFromContent(message.content),
      }
    case 'assistant':
      const text = textFromContent(message.content)
      const toolCalls = toolCallsFromContent(message.content)
      return {
        role: 'assistant',
        content: text || null,
        tool_calls: toolCalls.length ? toolCalls.map(toOpenAIToolCall) : undefined,
      }
  }
}

function toOpenAITool(tool: ModelToolSchema): ChatCompletionTool {
  // the provider boundary validates the minimum wire-compatible
  // shape BEFORE any transport request — a function schema whose parameters
  // lack an explicit root `type: "object"` is rejected by real OpenAI-
  // compatible endpoints (observed as a 400 on every request carrying the
  // tool). Local registry schemas are trusted to stay narrow; this check
  // fails closed with a bounded developer diagnostic instead of sending a
  // known-broken declaration.
  assertRootObjectParameters(tool)
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }
}

function assertRootObjectParameters(tool: ModelToolSchema): void {
  const parameters = tool.parameters as Record<string, unknown> | undefined
  if (!parameters || typeof parameters !== 'object' || parameters.type !== 'object') {
    // Bounded developer diagnostic: the schema is developer-supplied, so the
    // message reports the offending FIELD PATH and the value's TYPE NAME —
    // never a verbatim echo of an unbounded schema value or tool name.
    const rawType = (parameters as { type?: unknown } | undefined)?.type
    const typeName = rawType === undefined || rawType === null
      ? 'missing'
      : Array.isArray(rawType) ? 'array' : typeof rawType
    const toolName = tool.name.length > 64 ? `${tool.name.slice(0, 61)}...` : tool.name
    throw new Error(
      `Outbound tool schema is not wire-compatible: tool "${toolName}" parameters must declare a root type "object" at parameters.type (found type: ${typeName}); refusing to send the request`,
    )
  }
}

function toOpenAIToolCall(call: ModelToolCall): ChatCompletionMessageToolCall {
  return {
    id: call.id,
    type: 'function',
    function: {
      name: call.name,
      arguments: call.rawArguments !== undefined
        ? call.rawArguments
        : JSON.stringify(call.arguments),
    },
  }
}

function textFromContent(content: ModelContentBlock[]): string {
  return content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('\n')
}

function toolCallsFromContent(content: ModelContentBlock[]): ModelToolCall[] {
  return content
    .filter((block) => block.type === 'toolCall')
    .map(({ type: _type, ...call }) => call)
}

function toToolCallBlock(call: ModelToolCall): ModelContentBlock {
  return {
    type: 'toolCall',
    ...call,
  }
}

function mapOpenAIStopReason(reason: ChatCompletion.Choice['finish_reason']): ModelStopReason {
  switch (reason) {
    case 'stop':
      return 'stop'
    case 'tool_calls':
    case 'function_call':
      return 'toolUse'
    case 'length':
      return 'length'
    case 'content_filter':
      return 'contentFilter'
    default:
      return 'unknown'
  }
}
