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
  ModelContentBlock,
  ModelStopReason,
  ModelMessage,
  ModelRequest,
  ModelResponse,
  ModelStreamEvent,
  ModelToolCall,
  ModelToolSchema,
  ModelUsage,
} from './types/types.js'
import { BaseModelAdapter } from './baseAdapter.js'
import { ModelProtocolError } from './errors.js'

export interface OpenAIChatModelClientOptions {
  apiKey: string
  baseURL?: string
  client?: OpenAIChatTransport
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

  constructor(options: OpenAIChatModelClientOptions) {
    this.client = options.client ?? new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    })
    this.adapter = new OpenAIChatAdapter()
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const providerRequest = this.adapter.toProviderRequest(request)
    const response = await this.client.chat.completions.create(providerRequest, {
      signal: request.signal,
    })
    if (isAsyncIterable(response)) {
      throw new ModelProtocolError('OpenAI returned a stream for a non-streaming request')
    }
    return this.adapter.fromProviderResponse(response)
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelStreamEvent> {
    const providerRequest = {
      ...this.adapter.toProviderRequest(request),
      stream: true,
      stream_options: { include_usage: true },
    } as ChatCompletionCreateParamsStreaming
    const response = await this.client.chat.completions.create(providerRequest, {
      signal: request.signal,
    })
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
        if (text) yield { type: 'text_delta', text }
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

function toModelUsage(usage: {
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
}): ModelUsage {
  return {
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
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
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
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
