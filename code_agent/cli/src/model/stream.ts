import { ModelProtocolError, ModelResponseLimitError } from './errors.js'
import type {
  ModelAssistantMessage,
  ModelContentBlock,
  ModelResponse,
  ModelStopReason,
  ModelStreamEvent,
  ModelToolCall,
  ModelUsage,
} from './types/types.js'

export interface ModelResponseLimits {
  maxEvents: number
  maxTextChars: number
  maxToolCalls: number
  maxToolArgumentChars: number
  maxTotalToolArgumentChars: number
}

export const DEFAULT_MODEL_RESPONSE_LIMITS: ModelResponseLimits = {
  maxEvents: 10_000,
  maxTextChars: 1_000_000,
  maxToolCalls: 128,
  maxToolArgumentChars: 100_000,
  maxTotalToolArgumentChars: 1_000_000,
}

interface ToolCallAccumulator {
  id: string
  name: string
  rawArguments: string
}

type StreamState = 'open' | 'finished'

/** Provider-independent, bounded accumulator for one complete model response. */
export class ModelStreamAccumulator {
  private text = ''
  private stopReason: ModelStopReason | undefined
  private usage: ModelUsage | undefined
  private state: StreamState = 'open'
  private eventCount = 0
  private totalToolArgumentChars = 0
  private readonly toolCalls = new Map<number, ToolCallAccumulator>()

  constructor(
    private readonly limits: ModelResponseLimits = DEFAULT_MODEL_RESPONSE_LIMITS,
  ) {}

  push(event: ModelStreamEvent): string | undefined {
    if (!isRecord(event) || typeof event.type !== 'string') {
      throw new ModelProtocolError('Model stream event must be an object with a type')
    }
    const nextEventCount = projectedLimit(
      this.eventCount,
      1,
      this.limits.maxEvents,
      'Model response exceeded the event limit',
    )

    if (this.state === 'finished') {
      if (event.type !== 'usage' || this.usage !== undefined) {
        throw new ModelProtocolError(
          'Model stream content is not allowed after the finish event',
        )
      }
      const usage = parseUsage(event.usage)
      this.usage = usage
      this.eventCount = nextEventCount
      return undefined
    }

    switch (event.type) {
      case 'text_delta': {
        if (typeof event.text !== 'string') {
          throw new ModelProtocolError('Model stream text delta must be a string')
        }
        const textLength = projectedLimit(
          this.text.length,
          event.text.length,
          this.limits.maxTextChars,
          'Model response exceeded the text character limit',
        )
        const nextText = this.text + event.text
        if (nextText.length !== textLength) {
          throw new ModelResponseLimitError('Model response text length is unsafe')
        }
        this.text = nextText
        this.eventCount = nextEventCount
        return event.text
      }
      case 'tool_call_delta': {
        validateToolCallDelta(event)
        const current = this.toolCalls.get(event.index)
        if (!current) {
          projectedLimit(
            this.toolCalls.size,
            1,
            this.limits.maxToolCalls,
            'Model response exceeded the tool call limit',
          )
        }
        const next = mergeMetadata(
          current ?? { id: '', name: '', rawArguments: '' },
          event.id,
          event.name,
        )
        const delta = event.argumentsDelta ?? ''
        const nextArgumentChars = projectedLimit(
          next.rawArguments.length,
          delta.length,
          this.limits.maxToolArgumentChars,
          'Model response exceeded the single tool argument limit',
        )
        const nextTotalArgumentChars = projectedLimit(
          this.totalToolArgumentChars,
          delta.length,
          this.limits.maxTotalToolArgumentChars,
          'Model response exceeded the total tool argument limit',
        )
        next.rawArguments += delta
        if (next.rawArguments.length !== nextArgumentChars) {
          throw new ModelResponseLimitError('Model response tool argument length is unsafe')
        }
        this.toolCalls.set(event.index, next)
        this.totalToolArgumentChars = nextTotalArgumentChars
        this.eventCount = nextEventCount
        return undefined
      }
      case 'finish': {
        const stopReason = parseStopReason(event.stopReason)
        this.stopReason = stopReason
        this.state = 'finished'
        this.eventCount = nextEventCount
        return undefined
      }
      case 'usage':
        throw new ModelProtocolError('Model usage must arrive after the finish event')
      default:
        throw new ModelProtocolError(
          `Unknown model stream event: ${(event as { type: unknown }).type}`,
        )
    }
  }

  finish(): ModelResponse {
    if (this.state !== 'finished' || !this.stopReason) {
      throw new ModelProtocolError('Model stream ended without exactly one finish event')
    }
    const toolCalls = Array.from(this.toolCalls.entries())
      .sort(([left], [right]) => left - right)
      .map(([, call]) => toToolCall(call))
    const content: ModelContentBlock[] = []
    if (this.text) content.push({ type: 'text', text: this.text })
    content.push(...toolCalls.map((call) => ({ type: 'toolCall' as const, ...call })))
    const message: ModelAssistantMessage = {
      role: 'assistant',
      content,
      stopReason: this.stopReason,
    }

    const response: ModelResponse = {
      message,
      content: this.text,
      toolCalls,
      stopReason: this.stopReason,
      raw: { streamed: true },
      usage: this.usage,
    }
    return validateModelResponse(response, this.limits)
  }
}

/** The same bounded response validation used by complete() and stream(). */
export function validateModelResponse(
  response: ModelResponse,
  limits: ModelResponseLimits = DEFAULT_MODEL_RESPONSE_LIMITS,
): ModelResponse {
  if (!isRecord(response) || !isRecord(response.message)) {
    throw new ModelProtocolError('Model returned an invalid response')
  }
  if (typeof response.content !== 'string') {
    throw new ModelProtocolError('Model response content must be a string')
  }
  if (!Array.isArray(response.toolCalls)) {
    throw new ModelProtocolError('Model response toolCalls must be an array')
  }
  if (!isStopReason(response.stopReason)) {
    throw new ModelProtocolError('Model response stopReason is invalid')
  }
  if (response.toolCalls.length > limits.maxToolCalls) {
    throw new ModelResponseLimitError('Model response exceeded the tool call limit')
  }
  enforceLimit(
    response.content.length,
    limits.maxTextChars,
    'Model response exceeded the text character limit',
  )

  const canonicalCalls = response.toolCalls.map((call) => (
    canonicalizeToolCall(call, limits)
  ))
  let totalArgumentChars = 0
  for (const call of canonicalCalls) {
    totalArgumentChars = projectedLimit(
      totalArgumentChars,
      call.rawArguments?.length ?? 0,
      limits.maxTotalToolArgumentChars,
      'Model response exceeded the total tool argument limit',
    )
  }
  validateAssistantMessage(response.message, response.content, canonicalCalls, response.stopReason)
  const usage = response.usage === undefined ? undefined : parseUsage(response.usage)
  const canonicalContent = buildContent(response.content, canonicalCalls)
  return {
    message: {
      role: 'assistant',
      content: canonicalContent,
      stopReason: response.stopReason,
    },
    content: response.content,
    toolCalls: canonicalCalls,
    stopReason: response.stopReason,
    raw: response.raw,
    usage,
  }
}

function canonicalizeToolCall(
  value: unknown,
  limits: ModelResponseLimits,
): ModelToolCall {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.name !== 'string') {
    throw new ModelProtocolError('Model response contains invalid tool call metadata')
  }
  const hasRawArguments = Object.prototype.hasOwnProperty.call(value, 'rawArguments')
  if (hasRawArguments && typeof value.rawArguments !== 'string') {
    throw new ModelProtocolError('Model response raw tool arguments must be a string')
  }
  if (value.argumentsError !== undefined && typeof value.argumentsError !== 'string') {
    throw new ModelProtocolError('Model response tool argumentsError must be a string')
  }

  let rawArguments: string
  let parsed: { value: Record<string, unknown>; error?: string }
  if (hasRawArguments) {
    rawArguments = value.rawArguments as string
    enforceLimit(
      rawArguments.length,
      limits.maxToolArgumentChars,
      'Model response exceeded the single tool argument limit',
    )
    parsed = parseToolArguments(rawArguments)
  } else {
    if (!isRecord(value.arguments)) {
      throw new ModelProtocolError('Model response tool arguments must be an object')
    }
    try {
      rawArguments = JSON.stringify(value.arguments)
    } catch {
      throw new ModelProtocolError('Model response tool arguments are not serializable')
    }
    if (typeof rawArguments !== 'string') {
      throw new ModelProtocolError('Model response tool arguments are not serializable')
    }
    parsed = parseToolArguments(rawArguments)
    if (parsed.error) {
      throw new ModelProtocolError('Model response tool arguments are not serializable')
    }
    if (value.argumentsError !== undefined) {
      if (typeof value.argumentsError !== 'string') {
        throw new ModelProtocolError('Model response tool argumentsError must be a string')
      }
      parsed = { value: parsed.value, error: value.argumentsError }
    }
  }

  enforceLimit(
    rawArguments.length,
    limits.maxToolArgumentChars,
    'Model response exceeded the single tool argument limit',
  )
  const result: ModelToolCall = {
    id: value.id,
    name: value.name,
    arguments: parsed.value,
    rawArguments,
  }
  if (parsed.error !== undefined) result.argumentsError = parsed.error
  return result
}

function validateAssistantMessage(
  message: unknown,
  responseText: string,
  canonicalCalls: readonly ModelToolCall[],
  stopReason: ModelStopReason,
): void {
  if (!isRecord(message) || message.role !== 'assistant' || !Array.isArray(message.content)) {
    throw new ModelProtocolError('Model response assistant message is invalid')
  }
  if (message.stopReason !== undefined && message.stopReason !== stopReason) {
    throw new ModelProtocolError('Model response message stop reason does not match')
  }

  // Some complete-only clients provide a deliberately sparse assistant
  // message for tool turns. The top-level response fields remain canonical;
  // the returned message is rebuilt below, so this compatibility form cannot
  // hide text or arguments from validation.
  if (message.content.length === 0 && responseText === '') return

  let text = ''
  const messageCalls: ModelToolCall[] = []
  for (const block of message.content) {
    if (!isRecord(block) || typeof block.type !== 'string') {
      throw new ModelProtocolError('Model response assistant content block is invalid')
    }
    if (block.type === 'text') {
      if (typeof block.text !== 'string') {
        throw new ModelProtocolError('Model response assistant text block is invalid')
      }
      text += block.text
      continue
    }
    if (block.type === 'toolCall') {
      messageCalls.push(canonicalizeToolCall(block, {
        ...DEFAULT_MODEL_RESPONSE_LIMITS,
        maxToolArgumentChars: Number.MAX_SAFE_INTEGER,
        maxTotalToolArgumentChars: Number.MAX_SAFE_INTEGER,
      }))
      continue
    }
    throw new ModelProtocolError(`Unknown model assistant content block: ${block.type}`)
  }
  if (text !== responseText || messageCalls.length !== canonicalCalls.length) {
    throw new ModelProtocolError('Model response message content does not match response fields')
  }
  for (let index = 0; index < canonicalCalls.length; index += 1) {
    if (!sameToolCall(messageCalls[index]!, canonicalCalls[index]!)) {
      throw new ModelProtocolError('Model response message tool calls do not match response fields')
    }
  }
}

function buildContent(text: string, calls: readonly ModelToolCall[]): ModelContentBlock[] {
  return [
    ...(text ? [{ type: 'text' as const, text }] : []),
    ...calls.map((call) => ({ type: 'toolCall' as const, ...call })),
  ]
}

function sameToolCall(left: ModelToolCall, right: ModelToolCall): boolean {
  return left.id === right.id
    && left.name === right.name
    && left.rawArguments === right.rawArguments
    && left.argumentsError === right.argumentsError
    && JSON.stringify(left.arguments) === JSON.stringify(right.arguments)
}

function toToolCall(call: ToolCallAccumulator): ModelToolCall {
  if (!call.id || !call.name) {
    throw new ModelProtocolError('Model stream tool call must include id and name')
  }
  const parsed = parseToolArguments(call.rawArguments)
  const result: ModelToolCall = {
    id: call.id,
    name: call.name,
    arguments: parsed.value,
    rawArguments: call.rawArguments,
  }
  if (parsed.error !== undefined) result.argumentsError = parsed.error
  return result
}

function validateToolCallDelta(event: Extract<ModelStreamEvent, { type: 'tool_call_delta' }>): void {
  if (!Number.isSafeInteger(event.index) || event.index < 0) {
    throw new ModelProtocolError('Model stream tool call index must be a non-negative safe integer')
  }
  for (const [label, value] of [
    ['id', event.id],
    ['name', event.name],
    ['argumentsDelta', event.argumentsDelta],
  ] as const) {
    if (value !== undefined && typeof value !== 'string') {
      throw new ModelProtocolError(`Model stream tool ${label} must be a string`)
    }
  }
}

function mergeMetadata(
  target: ToolCallAccumulator,
  id?: string,
  name?: string,
): ToolCallAccumulator {
  let nextId = target.id
  let nextName = target.name
  if (id) {
    if (target.id && target.id !== id) {
      throw new ModelProtocolError('Model stream tool call id conflicts for one index')
    }
    nextId = id
  }
  if (name) {
    if (target.name && target.name !== name) {
      throw new ModelProtocolError('Model stream tool call name conflicts for one index')
    }
    nextName = name
  }
  return { ...target, id: nextId, name: nextName }
}

function parseUsage(value: unknown): ModelUsage {
  if (!isRecord(value)) throw new ModelProtocolError('Model usage must be an object')
  const usage: ModelUsage = {}
  for (const [key, field] of [
    ['inputTokens', value.inputTokens],
    ['outputTokens', value.outputTokens],
    ['totalTokens', value.totalTokens],
  ] as const) {
    if (field !== undefined) {
      if (!Number.isSafeInteger(field) || field < 0) {
        throw new ModelProtocolError(`Model usage ${key} must be a non-negative safe integer`)
      }
      usage[key] = field
    }
  }
  return usage
}

function parseStopReason(value: unknown): ModelStopReason {
  if (!isStopReason(value)) throw new ModelProtocolError('Model stream stop reason is invalid')
  return value
}

function isStopReason(value: unknown): value is ModelStopReason {
  return value === 'stop' || value === 'toolUse' || value === 'length'
    || value === 'contentFilter' || value === 'maxTurns' || value === 'error' || value === 'unknown'
}

function enforceLimit(value: number, maximum: number, message: string): void {
  if (!Number.isSafeInteger(maximum) || maximum < 0 || value > maximum) {
    throw new ModelResponseLimitError(message)
  }
}

function projectedLimit(
  current: number,
  increment: number,
  maximum: number,
  message: string,
): number {
  if (!Number.isSafeInteger(current) || !Number.isSafeInteger(increment)) {
    throw new ModelResponseLimitError(message)
  }
  if (increment < 0 || current > Number.MAX_SAFE_INTEGER - increment) {
    throw new ModelResponseLimitError(message)
  }
  const projected = current + increment
  enforceLimit(projected, maximum, message)
  return projected
}

function parseToolArguments(raw: string): {
  value: Record<string, unknown>
  error?: string
} {
  try {
    const parsed = JSON.parse(raw || '{}')
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return { value: parsed as Record<string, unknown> }
    }
    return { value: {}, error: 'Tool arguments must be a JSON object.' }
  } catch {
    return { value: {}, error: 'Tool arguments must be valid JSON.' }
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
