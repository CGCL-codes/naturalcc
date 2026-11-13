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

// [maint-note:6e3a969c0908]
// 数据契约导航：ModelResponseLimits；字段含义以消费端校验为准。
// 成员 maxEvents：number。
// 成员 maxTextChars：number。
// 成员 maxToolCalls：number。
// 成员 maxToolArgumentChars：number。
// 成员 maxTotalToolArgumentChars：number。
// [/maint-note:6e3a969c0908]
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

// [maint-note:db6aedd5ac28]
// 数据契约导航：ToolCallAccumulator；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 name：string。
// 成员 rawArguments：string。
// [/maint-note:db6aedd5ac28]
interface ToolCallAccumulator {
  id: string
  name: string
  rawArguments: string
}

// [maint-note:f5dadf0280c4]
// 数据契约导航：StreamState；字段含义以消费端校验为准。
// 类型组合：'open' | 'finished'。
// [/maint-note:f5dadf0280c4]
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

  // [maint-note:aef1bdc31298]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 limits：ModelResponseLimits；缺省表达式 DEFAULT_MODEL_RESPONSE_LIMITS。
  // [/maint-note:aef1bdc31298]
  constructor(
    private readonly limits: ModelResponseLimits = DEFAULT_MODEL_RESPONSE_LIMITS,
  ) {}

  // [maint-note:e60b675f6253]
  // 维护导航：push；下列为本函数源码中的接口与路径索引。
  // 输入 event：ModelStreamEvent。
  // 返回契约：string | undefined。
  // 分支条件：!isRecord(event) || typeof event.type !== 'string'。
  // 分支条件：this.state === 'finished'。
  // 分支条件：event.type !== 'usage' || this.usage !== undefined。
  // 分支条件：switch event.type。
  // 分支条件：case 'text_delta'。
  // 分支条件：typeof event.text !== 'string'。
  // 分支条件：nextText.length !== textLength。
  // 分支条件：case 'tool_call_delta'。
  // 分支条件：!current。
  // 分支条件：next.rawArguments.length !== nextArgumentChars。
  // 分支条件：case 'finish'。
  // 分支条件：case 'usage'。
  // 返回路径：undefined。
  // 返回路径：event.text。
  // 显式抛出：new ModelProtocolError('Model stream event must be an object with a type')。
  // 显式抛出：new ModelProtocolError( 'Model stream content is not allowed after the finish event', )。
  // 显式抛出：new ModelProtocolError('Model stream text delta must be a string')。
  // 显式抛出：new ModelResponseLimitError('Model response text length is unsafe')。
  // 显式抛出：new ModelResponseLimitError('Model response tool argument length is unsafe')。
  // 显式抛出：new ModelProtocolError('Model usage must arrive after the finish event')。
  // 显式抛出：new ModelProtocolError( `Unknown model stream event: ${(event as { type: unknown }).type}`, )。
  // 写入目标：this.usage。
  // 写入目标：this.eventCount。
  // 写入目标：this.text。
  // 写入目标：next.rawArguments。
  // 写入目标：this.totalToolArgumentChars。
  // 写入目标：this.stopReason。
  // 写入目标：this.state。
  // 直接调用：isRecord。
  // 直接调用：projectedLimit。
  // 直接调用：parseUsage。
  // 直接调用：validateToolCallDelta。
  // 直接调用：this.toolCalls.get。
  // 直接调用：mergeMetadata。
  // 直接调用：this.toolCalls.set。
  // 直接调用：parseStopReason。
  // [/maint-note:e60b675f6253]
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

  // [maint-note:39ffaca7c3ca]
  // 维护导航：finish；下列为本函数源码中的接口与路径索引。
  // 返回契约：ModelResponse。
  // 分支条件：this.state !== 'finished' || !this.stopReason。
  // 分支条件：this.text。
  // 返回路径：validateModelResponse(response, this.limits)。
  // 显式抛出：new ModelProtocolError('Model stream ended without exactly one finish event')。
  // 直接调用：Array.from(this.toolCalls.entries()) .sort(([left], [right]) => left - right) .map。
  // 直接调用：Array.from(this.toolCalls.entries()) .sort。
  // 直接调用：Array.from。
  // 直接调用：this.toolCalls.entries。
  // 直接调用：content.push。
  // 直接调用：toolCalls.map。
  // 直接调用：validateModelResponse。
  // [/maint-note:39ffaca7c3ca]
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

  /** Validated provider usage observed so far, even if the stream later fails. */
  // [maint-note:cf0ba1e8874c]
  // 维护导航：usageSnapshot；下列为本函数源码中的接口与路径索引。
  // 返回契约：ModelUsage | undefined。
  // 返回路径：this.usage === undefined ? undefined : Object.freeze({ ...this.usage })。
  // 直接调用：Object.freeze。
  // [/maint-note:cf0ba1e8874c]
  usageSnapshot(): ModelUsage | undefined {
    return this.usage === undefined ? undefined : Object.freeze({ ...this.usage })
  }
}

/** The same bounded response validation used by complete() and stream(). */
// [maint-note:286f5e351611]
// 维护导航：validateModelResponse；下列为本函数源码中的接口与路径索引。
// 输入 response：ModelResponse。
// 输入 limits：ModelResponseLimits；缺省表达式 DEFAULT_MODEL_RESPONSE_LIMITS。
// 返回契约：ModelResponse。
// 分支条件：!isRecord(response) || !isRecord(response.message)。
// 分支条件：typeof response.content !== 'string'。
// 分支条件：!Array.isArray(response.toolCalls)。
// 分支条件：!isStopReason(response.stopReason)。
// 分支条件：response.toolCalls.length > limits.maxToolCalls。
// 遍历边界：const call ← canonicalCalls。
// 返回路径：{ message: { role: 'assistant', content: canonicalContent, stopReason: response.stopReason, }, content: response.content, toolCalls: canonicalCalls, stopReason: response.stopReason, raw: response.raw, usage, }。
// 显式抛出：new ModelProtocolError('Model returned an invalid response')。
// 显式抛出：new ModelProtocolError('Model response content must be a string')。
// 显式抛出：new ModelProtocolError('Model response toolCalls must be an array')。
// 显式抛出：new ModelProtocolError('Model response stopReason is invalid')。
// 显式抛出：new ModelResponseLimitError('Model response exceeded the tool call limit')。
// 写入目标：totalArgumentChars。
// 直接调用：isRecord。
// 直接调用：Array.isArray。
// 直接调用：isStopReason。
// 直接调用：enforceLimit。
// 直接调用：response.toolCalls.map。
// 直接调用：projectedLimit。
// 直接调用：validateAssistantMessage。
// 直接调用：parseUsage。
// 直接调用：buildContent。
// [/maint-note:286f5e351611]
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

// [maint-note:a099310f2fee]
// 维护导航：canonicalizeToolCall；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 limits：ModelResponseLimits。
// 返回契约：ModelToolCall。
// 分支条件：!isRecord(value) || typeof value.id !== 'string' || typeof value.name !== 'string'。
// 分支条件：hasRawArguments && typeof value.rawArguments !== 'string'。
// 分支条件：value.argumentsError !== undefined && typeof value.argumentsError !== 'string'。
// 分支条件：hasRawArguments。
// 分支条件：!isRecord(value.arguments)。
// 分支条件：typeof rawArguments !== 'string'。
// 分支条件：parsed.error。
// 分支条件：value.argumentsError !== undefined。
// 分支条件：typeof value.argumentsError !== 'string'。
// 分支条件：parsed.error !== undefined。
// 返回路径：result。
// 显式抛出：new ModelProtocolError('Model response contains invalid tool call metadata')。
// 显式抛出：new ModelProtocolError('Model response raw tool arguments must be a string')。
// 显式抛出：new ModelProtocolError('Model response tool argumentsError must be a string')。
// 显式抛出：new ModelProtocolError('Model response tool arguments must be an object')。
// 显式抛出：new ModelProtocolError('Model response tool arguments are not serializable')。
// 写入目标：rawArguments。
// 写入目标：parsed。
// 写入目标：result.argumentsError。
// 直接调用：isRecord。
// 直接调用：Object.prototype.hasOwnProperty.call。
// 直接调用：enforceLimit。
// 直接调用：parseToolArguments。
// 直接调用：JSON.stringify。
// [/maint-note:a099310f2fee]
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

// [maint-note:37cf83914367]
// 维护导航：validateAssistantMessage；下列为本函数源码中的接口与路径索引。
// 输入 message：unknown。
// 输入 responseText：string。
// 输入 canonicalCalls：readonly ModelToolCall[]。
// 输入 stopReason：ModelStopReason。
// 返回契约：void。
// 分支条件：!isRecord(message) || message.role !== 'assistant' || !Array.isArray(message.content)。
// 分支条件：message.stopReason !== undefined && message.stopReason !== stopReason。
// 分支条件：message.content.length === 0 && responseText === ''。
// 分支条件：!isRecord(block) || typeof block.type !== 'string'。
// 分支条件：block.type === 'text'。
// 分支条件：typeof block.text !== 'string'。
// 分支条件：block.type === 'toolCall'。
// 分支条件：text !== responseText || messageCalls.length !== canonicalCalls.length。
// 分支条件：!sameToolCall(messageCalls[index]!, canonicalCalls[index]!)。
// 遍历边界：const block ← message.content。
// 遍历边界：let index = 0; index < canonicalCalls.length; index += 1。
// 返回路径：无返回值。
// 显式抛出：new ModelProtocolError('Model response assistant message is invalid')。
// 显式抛出：new ModelProtocolError('Model response message stop reason does not match')。
// 显式抛出：new ModelProtocolError('Model response assistant content block is invalid')。
// 显式抛出：new ModelProtocolError('Model response assistant text block is invalid')。
// 显式抛出：new ModelProtocolError(`Unknown model assistant content block: ${block.type}`)。
// 显式抛出：new ModelProtocolError('Model response message content does not match response fields')。
// 显式抛出：new ModelProtocolError('Model response message tool calls do not match response fields')。
// 写入目标：text。
// 写入目标：index。
// 直接调用：isRecord。
// 直接调用：Array.isArray。
// 直接调用：messageCalls.push。
// 直接调用：canonicalizeToolCall。
// 直接调用：sameToolCall。
// [/maint-note:37cf83914367]
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

// [maint-note:01130e8f1b43]
// 维护导航：buildContent；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 calls：readonly ModelToolCall[]。
// 返回契约：ModelContentBlock[]。
// 返回路径：[ ...(text ? [{ type: 'text' as const, text }] : []), ...calls.map((call) => ({ type: 'toolCall' as const, ...call })), ]。
// 直接调用：calls.map。
// [/maint-note:01130e8f1b43]
function buildContent(text: string, calls: readonly ModelToolCall[]): ModelContentBlock[] {
  return [
    ...(text ? [{ type: 'text' as const, text }] : []),
    ...calls.map((call) => ({ type: 'toolCall' as const, ...call })),
  ]
}

// [maint-note:aeeac16fd830]
// 维护导航：sameToolCall；下列为本函数源码中的接口与路径索引。
// 输入 left：ModelToolCall。
// 输入 right：ModelToolCall。
// 返回契约：boolean。
// 返回路径：left.id === right.id && left.name === right.name && left.rawArguments === right.rawArguments && left.argumentsError === right.argumentsError && JSON.stringify(left.arguments) === JSON.stringify(right.arguments)。
// 直接调用：JSON.stringify。
// [/maint-note:aeeac16fd830]
function sameToolCall(left: ModelToolCall, right: ModelToolCall): boolean {
  return left.id === right.id
    && left.name === right.name
    && left.rawArguments === right.rawArguments
    && left.argumentsError === right.argumentsError
    && JSON.stringify(left.arguments) === JSON.stringify(right.arguments)
}

// [maint-note:cad3d2497bc9]
// 维护导航：toToolCall；下列为本函数源码中的接口与路径索引。
// 输入 call：ToolCallAccumulator。
// 返回契约：ModelToolCall。
// 分支条件：!call.id || !call.name。
// 分支条件：parsed.error !== undefined。
// 返回路径：result。
// 显式抛出：new ModelProtocolError('Model stream tool call must include id and name')。
// 写入目标：result.argumentsError。
// 直接调用：parseToolArguments。
// [/maint-note:cad3d2497bc9]
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

// [maint-note:05c7877c14de]
// 维护导航：validateToolCallDelta；下列为本函数源码中的接口与路径索引。
// 输入 event：Extract<ModelStreamEvent, { type: 'tool_call_delta' }>。
// 返回契约：void。
// 分支条件：!Number.isSafeInteger(event.index) || event.index < 0。
// 分支条件：value !== undefined && typeof value !== 'string'。
// 遍历边界：const [label, value] ← [ ['id', event.id], ['name', event.name], ['argumentsDelta', event.argumentsDelta], ] as const。
// 显式抛出：new ModelProtocolError('Model stream tool call index must be a non-negative safe integer')。
// 显式抛出：new ModelProtocolError(`Model stream tool ${label} must be a string`)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:05c7877c14de]
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

// [maint-note:3c0e872e4821]
// 维护导航：mergeMetadata；下列为本函数源码中的接口与路径索引。
// 输入 target：ToolCallAccumulator。
// 输入 id：string（可省略）。
// 输入 name：string（可省略）。
// 返回契约：ToolCallAccumulator。
// 分支条件：id。
// 分支条件：target.id && target.id !== id。
// 分支条件：name。
// 分支条件：target.name && target.name !== name。
// 返回路径：{ ...target, id: nextId, name: nextName }。
// 显式抛出：new ModelProtocolError('Model stream tool call id conflicts for one index')。
// 显式抛出：new ModelProtocolError('Model stream tool call name conflicts for one index')。
// 写入目标：nextId。
// 写入目标：nextName。
// [/maint-note:3c0e872e4821]
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

// [maint-note:90a67f3082ca]
// 维护导航：parseUsage；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：ModelUsage。
// 分支条件：!isRecord(value)。
// 分支条件：field !== undefined。
// 分支条件：!Number.isSafeInteger(field) || field < 0。
// 遍历边界：const [key, field] ← [ ['inputTokens', value.inputTokens], ['outputTokens', value.outputTokens], ['totalTokens', value.totalTokens], ] as const。
// 返回路径：usage。
// 显式抛出：new ModelProtocolError('Model usage must be an object')。
// 显式抛出：new ModelProtocolError(`Model usage ${key} must be a non-negative safe integer`)。
// 写入目标：usage[key]。
// 直接调用：isRecord。
// 直接调用：Number.isSafeInteger。
// [/maint-note:90a67f3082ca]
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

// [maint-note:ebaac3cefabb]
// 维护导航：parseStopReason；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：ModelStopReason。
// 分支条件：!isStopReason(value)。
// 返回路径：value。
// 显式抛出：new ModelProtocolError('Model stream stop reason is invalid')。
// 直接调用：isStopReason。
// [/maint-note:ebaac3cefabb]
function parseStopReason(value: unknown): ModelStopReason {
  if (!isStopReason(value)) throw new ModelProtocolError('Model stream stop reason is invalid')
  return value
}

// [maint-note:998892a56f18]
// 维护导航：isStopReason；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is ModelStopReason。
// 返回路径：value === 'stop' || value === 'toolUse' || value === 'length' || value === 'contentFilter' || value === 'maxTurns' || value === 'error' || value === 'unknown'。
// [/maint-note:998892a56f18]
function isStopReason(value: unknown): value is ModelStopReason {
  return value === 'stop' || value === 'toolUse' || value === 'length'
    || value === 'contentFilter' || value === 'maxTurns' || value === 'error' || value === 'unknown'
}

// [maint-note:95357af65733]
// 维护导航：enforceLimit；下列为本函数源码中的接口与路径索引。
// 输入 value：number。
// 输入 maximum：number。
// 输入 message：string。
// 返回契约：void。
// 分支条件：!Number.isSafeInteger(maximum) || maximum < 0 || value > maximum。
// 显式抛出：new ModelResponseLimitError(message)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:95357af65733]
function enforceLimit(value: number, maximum: number, message: string): void {
  if (!Number.isSafeInteger(maximum) || maximum < 0 || value > maximum) {
    throw new ModelResponseLimitError(message)
  }
}

// [maint-note:96264b1f19c5]
// 维护导航：projectedLimit；下列为本函数源码中的接口与路径索引。
// 输入 current：number。
// 输入 increment：number。
// 输入 maximum：number。
// 输入 message：string。
// 返回契约：number。
// 分支条件：!Number.isSafeInteger(current) || !Number.isSafeInteger(increment)。
// 分支条件：increment < 0 || current > Number.MAX_SAFE_INTEGER - increment。
// 返回路径：projected。
// 显式抛出：new ModelResponseLimitError(message)。
// 直接调用：Number.isSafeInteger。
// 直接调用：enforceLimit。
// [/maint-note:96264b1f19c5]
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

// [maint-note:b4e84dbdc4e2]
// 维护导航：parseToolArguments；下列为本函数源码中的接口与路径索引。
// 输入 raw：string。
// 返回契约：{ value: Record<string, unknown> error?: string }。
// 分支条件：parsed && typeof parsed === 'object' && !Array.isArray(parsed)。
// 返回路径：{ value: parsed as Record<string, unknown> }。
// 返回路径：{ value: {}, error: 'Tool arguments must be a JSON object.' }。
// 返回路径：{ value: {}, error: 'Tool arguments must be valid JSON.' }。
// 直接调用：JSON.parse。
// 直接调用：Array.isArray。
// [/maint-note:b4e84dbdc4e2]
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

// [maint-note:dc53b7c7af2b]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, any>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:dc53b7c7af2b]
function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
