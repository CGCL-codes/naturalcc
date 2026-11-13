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

// [maint-note:1e57e40d1333]
// 数据契约导航：OpenAIChatModelClientOptions；字段含义以消费端校验为准。
// 成员 apiKey：string。
// 成员 baseURL（可省略）：string。
// 成员 client（可省略）：OpenAIChatTransport。
// 成员 requestBudgetCapability（可省略）：ModelBudgetCapabilities & { readonly countInputTokens?: ( body: Readonly<Record<string, unknown>>, ) => number }。
// [/maint-note:1e57e40d1333]
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
  // [maint-note:4ae7d0f00364]
  // 维护导航：toProviderRequest；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 返回契约：ChatCompletionCreateParamsNonStreaming。
  // 返回路径：{ model: request.model, messages: request.messages.map(toOpenAIMessage), tools, tool_choice: request.toolChoice ?? (tools?.length ? 'auto' : undefined), }。
  // 直接调用：request.tools?.map。
  // 直接调用：request.messages.map。
  // [/maint-note:4ae7d0f00364]
  toProviderRequest(request: ModelRequest): ChatCompletionCreateParamsNonStreaming {
    const tools = request.tools?.map(toOpenAITool)

    return {
      model: request.model,
      messages: request.messages.map(toOpenAIMessage),
      tools,
      tool_choice: request.toolChoice ?? (tools?.length ? 'auto' : undefined),
    }
  }

  // [maint-note:28832c391b5b]
  // 维护导航：fromProviderResponse；下列为本函数源码中的接口与路径索引。
  // 输入 response：ChatCompletion。
  // 返回契约：ModelResponse。
  // 分支条件：!message。
  // 返回路径：{ message: { role: 'assistant', content: [ ...(content ? [{ type: 'text' as const, text: content }] : []), ...toolCalls.map(toToolCallBlock), ], stopReason, }, content, toolCalls, stopReason, raw: response, usage: response.usage ? toModelUs。
  // 显式抛出：new ModelProtocolError('OpenAI response did not include a message')。
  // 直接调用：(message.tool_calls ?? []).map。
  // 直接调用：mapOpenAIStopReason。
  // 直接调用：toolCalls.map。
  // 直接调用：toModelUsage。
  // [/maint-note:28832c391b5b]
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

  // [maint-note:01d3f24ebd99]
  // 维护导航：fromOpenAIToolCall；下列为本函数源码中的接口与路径索引。
  // 输入 call：ChatCompletionMessageToolCall。
  // 返回契约：ModelToolCall。
  // 分支条件：call.type === 'function'。
  // 返回路径：{ id: call.id, name: call.function.name, arguments: parsed.value, rawArguments: call.function.arguments, argumentsError: parsed.error, }。
  // 返回路径：{ id: call.id, name: call.custom.name, arguments: parsed.value, rawArguments: call.custom.input, argumentsError: parsed.error, }。
  // 直接调用：this.parseJsonObject。
  // [/maint-note:01d3f24ebd99]
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

// [maint-note:8f2723bda22d]
// 数据契约导航：OpenAIChatTransport；字段含义以消费端校验为准。
// 成员 chat：{ completions: { create( request: ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming, options?: { signal?: AbortSignal }, ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> } }。
// [/maint-note:8f2723bda22d]
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

  // [maint-note:ebff4dd14e21]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：OpenAIChatModelClientOptions。
  // 写入目标：this.client。
  // 写入目标：this.adapter。
  // 写入目标：this.requestBudgetCapability。
  // 写入目标：this.budgetCapabilities。
  // 直接调用：Object.freeze。
  // [/maint-note:ebff4dd14e21]
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

  // [maint-note:c44c91c493ce]
  // 维护导航：complete；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 返回契约：Promise<ModelResponse>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：isAsyncIterable(sent)。
  // 返回路径：sent as Promise<ModelResponse>。
  // 显式抛出：new ModelProtocolError('Prepared complete request returned a stream')。
  // 直接调用：this.prepare(request, 'complete').send。
  // 直接调用：this.prepare。
  // 直接调用：isAsyncIterable。
  // [/maint-note:c44c91c493ce]
  async complete(request: ModelRequest): Promise<ModelResponse> {
    const sent = this.prepare(request, 'complete').send()
    if (isAsyncIterable(sent)) throw new ModelProtocolError('Prepared complete request returned a stream')
    return sent as Promise<ModelResponse>
  }

  // [maint-note:e2b412399dbe]
  // 维护导航：stream；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 返回契约：AsyncIterable<ModelStreamEvent>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!isAsyncIterable(sent)。
  // 显式抛出：new ModelProtocolError('Prepared stream request returned a response')。
  // 直接调用：this.prepare(request, 'stream').send。
  // 直接调用：this.prepare。
  // 直接调用：isAsyncIterable。
  // [/maint-note:e2b412399dbe]
  async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
    const sent = this.prepare(request, 'stream').send()
    if (!isAsyncIterable(sent)) throw new ModelProtocolError('Prepared stream request returned a response')
    yield* sent as AsyncIterable<ModelStreamEvent>
  }

  // [maint-note:14c5ec9af7ee]
  // 维护导航：prepare；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 输入 mode：ModelDispatchMode。
  // 返回契约：PreparedModelDispatch。
  // 返回路径：Object.freeze({ mode, measurement, send: () => { if (sent) throw new ModelProtocolError('A prepared model dispatch can only be sent once') sent = true return mode === 'stream' ? this.sendStream(request, providerRequest as ChatCompletionCrea。
  // 直接调用：this.finalProviderRequest。
  // 直接调用：Buffer.byteLength。
  // 直接调用：JSON.stringify。
  // 直接调用：this.requestBudgetCapability?.countInputTokens。
  // 直接调用：Object.freeze。
  // [/maint-note:14c5ec9af7ee]
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

  // [maint-note:8b62d7ea87b4]
  // 维护导航：finalProviderRequest；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 输入 mode：ModelDispatchMode。
  // 返回契约：ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming。
  // 分支条件：outputTokens !== undefined。
  // 分支条件：!outputCapability。
  // 分支条件：mode === 'stream'。
  // 返回路径：deepFreeze(body) as unknown as ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming。
  // 显式抛出：new ModelProtocolError('The configured request output token cap is not supported by this endpoint')。
  // 写入目标：body[outputCapability.field]。
  // 写入目标：body.stream。
  // 写入目标：body.stream_options。
  // 直接调用：structuredClone。
  // 直接调用：this.adapter.toProviderRequest。
  // 直接调用：deepFreeze。
  // [/maint-note:8b62d7ea87b4]
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

  // [maint-note:602bb0ad0072]
  // 维护导航：sendComplete；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 输入 providerRequest：ChatCompletionCreateParamsNonStreaming。
  // 返回契约：Promise<ModelResponse>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：isAsyncIterable(response)。
  // 返回路径：modelResponse。
  // 显式抛出：new ModelProtocolError('OpenAI returned a stream for a non-streaming request')。
  // 直接调用：this.client.chat.completions.create。
  // 直接调用：isAsyncIterable。
  // 直接调用：this.adapter.fromProviderResponse。
  // 直接调用：assertNoTextToolEnvelope。
  // [/maint-note:602bb0ad0072]
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

  // [maint-note:08c286618fe3]
  // 维护导航：sendStream；下列为本函数源码中的接口与路径索引。
  // 输入 request：ModelRequest。
  // 输入 providerRequest：ChatCompletionCreateParamsStreaming。
  // 返回契约：AsyncIterable<ModelStreamEvent>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：!isAsyncIterable(response)。
  // 分支条件：chunk.usage。
  // 分支条件：usageEmitted。
  // 分支条件：choice。
  // 分支条件：text。
  // 分支条件：choice.finish_reason。
  // 分支条件：pendingUsage && finishEmitted。
  // 分支条件：pendingUsage。
  // 遍历边界：const chunk ← response。
  // 遍历边界：const toolCall ← choice.delta.tool_calls ?? []。
  // 显式抛出：new ModelProtocolError('OpenAI did not return a streaming response')。
  // 显式抛出：new ModelProtocolError('OpenAI stream returned usage more than once')。
  // 写入目标：pendingUsage。
  // 写入目标：finishEmitted。
  // 写入目标：usageEmitted。
  // 直接调用：this.client.chat.completions.create。
  // 直接调用：isAsyncIterable。
  // 直接调用：toModelUsage。
  // 直接调用：mapOpenAIStopReason。
  // [/maint-note:08c286618fe3]
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

// [maint-note:c308096cda3c]
// 维护导航：deepFreeze；下列为本函数源码中的接口与路径索引。
// 输入 value：T。
// 返回契约：T。
// 分支条件：value && typeof value === 'object' && !Object.isFrozen(value)。
// 遍历边界：const child ← Object.values(value as Record<string, unknown>)。
// 返回路径：value。
// 直接调用：Object.isFrozen。
// 直接调用：Object.values。
// 直接调用：deepFreeze。
// 直接调用：Object.freeze。
// [/maint-note:c308096cda3c]
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
// [maint-note:8cb721940edb]
// 维护导航：assertNoTextToolEnvelope；下列为本函数源码中的接口与路径索引。
// 输入 request：ModelRequest。
// 输入 content：string。
// 返回契约：void。
// 分支条件：!offeredTools。
// 分支条件：!diagnostic。
// 返回路径：无返回值。
// 显式抛出：new ModelProtocolError( `OpenAI response returned a tool-control envelope as text content (${diagnostic.id})`, )。
// 直接调用：detectToolControlEnvelope。
// [/maint-note:8cb721940edb]
function assertNoTextToolEnvelope(request: ModelRequest, content: string): void {
  const offeredTools = (request.tools?.length ?? 0) > 0 && request.toolChoice !== 'none'
  if (!offeredTools) return
  const diagnostic = detectToolControlEnvelope(content)
  if (!diagnostic) return
  throw new ModelProtocolError(
    `OpenAI response returned a tool-control envelope as text content (${diagnostic.id})`,
  )
}

// [maint-note:6d1792f7d236]
// 维护导航：toModelUsage；下列为本函数源码中的接口与路径索引。
// 输入 usage：{ prompt_tokens?: number completion_tokens?: number total_tokens?: number prompt_tokens_details?: { cached_tokens?: number | null } | null }。
// 返回契约：ModelUsage。
// 返回路径：{ inputTokens: usage.prompt_tokens, outputTokens: usage.completion_tokens, totalTokens: usage.total_tokens, ...(cached === undefined || cached === null ? {} : { cacheHitInputTokens: cached }), ...(cacheMiss === undefined ? {} : { cacheMissI。
// 直接调用：Math.max。
// [/maint-note:6d1792f7d236]
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

// [maint-note:abbdefc99aac]
// 维护导航：isAsyncIterable；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is AsyncIterable<ChatCompletionChunk>。
// 返回路径：Boolean(value && typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === 'function')。
// 直接调用：Boolean。
// [/maint-note:abbdefc99aac]
function isAsyncIterable(value: unknown): value is AsyncIterable<ChatCompletionChunk> {
  return Boolean(value && typeof (value as AsyncIterable<unknown>)[Symbol.asyncIterator] === 'function')
}

// [maint-note:2ba0358601de]
// 维护导航：toOpenAIMessage；下列为本函数源码中的接口与路径索引。
// 输入 message：ModelMessage。
// 返回契约：ChatCompletionMessageParam。
// 分支条件：switch message.role。
// 分支条件：case 'system'。
// 分支条件：case 'user'。
// 分支条件：case 'toolResult'。
// 分支条件：case 'assistant'。
// 返回路径：{ role: 'system', content: textFromContent(message.content) }。
// 返回路径：{ role: 'user', content: textFromContent(message.content) }。
// 返回路径：{ role: 'tool', tool_call_id: message.toolCallId, content: textFromContent(message.content), }。
// 返回路径：{ role: 'assistant', content: text || null, tool_calls: toolCalls.length ? toolCalls.map(toOpenAIToolCall) : undefined, }。
// 直接调用：textFromContent。
// 直接调用：toolCallsFromContent。
// 直接调用：toolCalls.map。
// [/maint-note:2ba0358601de]
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

// [maint-note:139a658ed85c]
// 维护导航：toOpenAITool；下列为本函数源码中的接口与路径索引。
// 输入 tool：ModelToolSchema。
// 返回契约：ChatCompletionTool。
// 返回路径：{ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters, }, }。
// 直接调用：assertRootObjectParameters。
// [/maint-note:139a658ed85c]
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

// [maint-note:03022306efc3]
// 维护导航：assertRootObjectParameters；下列为本函数源码中的接口与路径索引。
// 输入 tool：ModelToolSchema。
// 返回契约：void。
// 分支条件：!parameters || typeof parameters !== 'object' || parameters.type !== 'object'。
// 显式抛出：new Error( `Outbound tool schema is not wire-compatible: tool "${toolName}" parameters must declare a root type "object" at parameters.type (found type: ${typeName}); refusing to send the request`, )。
// 直接调用：Array.isArray。
// 直接调用：tool.name.slice。
// [/maint-note:03022306efc3]
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

// [maint-note:a54c06e960da]
// 维护导航：toOpenAIToolCall；下列为本函数源码中的接口与路径索引。
// 输入 call：ModelToolCall。
// 返回契约：ChatCompletionMessageToolCall。
// 返回路径：{ id: call.id, type: 'function', function: { name: call.name, arguments: call.rawArguments !== undefined ? call.rawArguments : JSON.stringify(call.arguments), }, }。
// 直接调用：JSON.stringify。
// [/maint-note:a54c06e960da]
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

// [maint-note:98ca0137bd0f]
// 维护导航：textFromContent；下列为本函数源码中的接口与路径索引。
// 输入 content：ModelContentBlock[]。
// 返回契约：string。
// 返回路径：content .filter((block) => block.type === 'text') .map((block) => block.text) .join('\n')。
// 直接调用：content .filter((block) => block.type === 'text') .map((block) => block.text) .join。
// 直接调用：content .filter((block) => block.type === 'text') .map。
// 直接调用：content .filter。
// [/maint-note:98ca0137bd0f]
function textFromContent(content: ModelContentBlock[]): string {
  return content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('\n')
}

// [maint-note:3c848338516d]
// 维护导航：toolCallsFromContent；下列为本函数源码中的接口与路径索引。
// 输入 content：ModelContentBlock[]。
// 返回契约：ModelToolCall[]。
// 返回路径：content .filter((block) => block.type === 'toolCall') .map(({ type: _type, ...call }) => call)。
// 直接调用：content .filter((block) => block.type === 'toolCall') .map。
// 直接调用：content .filter。
// [/maint-note:3c848338516d]
function toolCallsFromContent(content: ModelContentBlock[]): ModelToolCall[] {
  return content
    .filter((block) => block.type === 'toolCall')
    .map(({ type: _type, ...call }) => call)
}

// [maint-note:255df85ce734]
// 维护导航：toToolCallBlock；下列为本函数源码中的接口与路径索引。
// 输入 call：ModelToolCall。
// 返回契约：ModelContentBlock。
// 返回路径：{ type: 'toolCall', ...call, }。
// [/maint-note:255df85ce734]
function toToolCallBlock(call: ModelToolCall): ModelContentBlock {
  return {
    type: 'toolCall',
    ...call,
  }
}

// [maint-note:9913d8ef64dc]
// 维护导航：mapOpenAIStopReason；下列为本函数源码中的接口与路径索引。
// 输入 reason：ChatCompletion.Choice['finish_reason']。
// 返回契约：ModelStopReason。
// 分支条件：switch reason。
// 分支条件：case 'stop'。
// 分支条件：case 'tool_calls'。
// 分支条件：case 'function_call'。
// 分支条件：case 'length'。
// 分支条件：case 'content_filter'。
// 返回路径：'stop'。
// 返回路径：'toolUse'。
// 返回路径：'length'。
// 返回路径：'contentFilter'。
// 返回路径：'unknown'。
// [/maint-note:9913d8ef64dc]
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
