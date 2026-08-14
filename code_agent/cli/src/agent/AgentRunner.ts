import type { AgentRuntime } from './types/types.js'
import type { AgentEvent, AgentErrorCode } from './types/event.js'
import type {
  ModelAssistantMessage,
  ModelContentBlock,
  ModelResponse,
  ModelToolCall,
  ModelToolSchema,
  ModelMessage,
  ModelToolResultMessage,
} from '../model/types/types.js'
import {
  ToolInputError,
  type ToolCall,
  type ToolExecutionResult,
  type ToolSchema,
} from '../tools/types.js'
import type { AgentThread } from './thread/AgentThread.js'
import { errorMessage, isAbortError, AgentError } from './errors.js'
import { getThreadController } from './thread/AgentThread.js'
import { ModelProtocolError, ModelResponseLimitError } from '../model/errors.js'
import {
  createToolPolicy,
  validateApprovalInput,
  validateApprovalSchema,
} from '../tools/approval.js'
import {
  DEFAULT_MODEL_RESPONSE_LIMITS,
  ModelStreamAccumulator,
  type ModelResponseLimits,
  validateModelResponse,
} from '../model/stream.js'

let fallbackApprovalTurn = 0

export interface RunTurnInput {
  thread: AgentThread
  prompt: string
  runtime: AgentRuntime
  signal?: AbortSignal
}

export interface AgentRunnerOptions {
  modelResponseLimits?: ModelResponseLimits
}

export class AgentRunner {
  private readonly modelResponseLimits: ModelResponseLimits

  constructor(options: AgentRunnerOptions = {}) {
    this.modelResponseLimits = options.modelResponseLimits ?? DEFAULT_MODEL_RESPONSE_LIMITS
  }

  async *runTurn(input: RunTurnInput): AsyncGenerator<AgentEvent> {
    let committed = false
    let started = false
    let sideEffectToolExecuted = false
    let approvalOrdinal = 0
    // Provider call IDs are only meaningful within one turn. Keep a turn-wide
    // set so a later model response cannot accidentally reuse a terminal UI
    // correlation key for a new tool batch.
    const usedCallIds = new Set<string>()
    let controller: ReturnType<typeof getThreadController> | undefined

    try {
      controller = getThreadController(input.thread)
      controller.begin()
      started = true
      const turnPolicy = input.runtime.toolPolicy ?? createToolPolicy('deny')
      turnPolicy.beginTurn?.()
      const fallbackTurnId = ++fallbackApprovalTurn

      if (!input.prompt.trim()) {
        yield errorEvent(
          'config_error',
          'A non-empty prompt is required',
          false,
        )
        return
      }

      const pendingMessages: ModelMessage[] = [
        { role: 'user', content: textContent(input.prompt) },
      ]
      yield { type: 'user_message', content: input.prompt }

      const toolSchemas = validateToolSchemas(input.runtime.tools.schemas())

      for (let turn = 0; turn < input.runtime.loop.maxTurns; turn++) {
        throwIfAborted(input.signal)
        let response: ModelResponse | undefined
        for await (const update of consumeModel(
          input,
          input.thread.requestView(input.runtime.loop.maxHistoryChars, pendingMessages),
          toModelToolSchemas(toolSchemas),
          this.modelResponseLimits,
        )) {
          if (update.type === 'assistant_delta') {
            yield update
          } else {
            response = update.response
          }
        }
        if (!response) {
          throw new AgentError(
            'model_protocol_error',
            'Model stream ended without a complete response',
          )
        }

        if (response.toolCalls.length > 0) {
          if (response.stopReason !== 'toolUse') {
            if (
              response.stopReason === 'length' ||
              response.stopReason === 'contentFilter'
            ) {
              throw new AgentError(
                'model_incomplete',
                response.stopReason === 'length'
                  ? 'Model response was truncated by the length limit'
                  : 'Model response was stopped by the content filter',
              )
            }
            throw new AgentError(
              'model_protocol_error',
              `Model returned tool calls with stop reason: ${response.stopReason}`,
            )
          }

          const toolBatch = normalizeToolCalls(response.toolCalls, usedCallIds)
          const assistantMessage = normalizeAssistantMessage(
            response,
            toolBatch.calls,
          )
          pendingMessages.push(assistantMessage)

          for (const call of toolBatch.calls) {
            const batchValidationError = toolBatch.error
            const validationError = getToolValidationError(
              input,
              call,
              toolSchemas,
              batchValidationError,
            )
            yield {
              type: 'tool_start',
              callId: call.id,
              name: call.name,
              input: call.arguments,
            }
            throwIfAborted(input.signal)

            let result: NormalizedToolResult
            if (validationError) {
              result = toToolErrorResult(validationError)
            } else {
              const schema = toolSchemas.find((candidate) => candidate.name === call.name)
              const policy = turnPolicy
              const ordinal = approvalOrdinal++
              const requestId = policy.createRequestId?.(call.id, ordinal)
                ?? `approval-${fallbackTurnId}-${ordinal}-${call.id}`
              let approved = true
              if (schema && policy?.requiresApproval(schema)) {
                const reason = validateApprovalInput(schema, call.arguments)
                if (!reason) {
                  throw new AgentError(
                    'tool_validation_error',
                    `Tool ${call.name} requires an approval reason`,
                    true,
                  )
                }
                yield {
                  type: 'approval_required',
                  requestId,
                  callId: call.id,
                  name: call.name,
                  input: call.arguments,
                  reason,
                }
                try {
                  approved = await policy.approvalProvider.request({
                    requestId,
                    callId: call.id,
                    toolName: call.name,
                    input: call.arguments,
                    reason,
                  }, input.signal)
                } catch (error) {
                  if (isAbortError(error, input.signal)) throw error
                  approved = false
                }
                throwIfAborted(input.signal)
                yield { type: 'approval_resolved', requestId, approved }
                // The consumer may dispose the session while paused at the
                // approved event. Never mark side effects or call a tool
                // until the generator is resumed with a live signal.
                throwIfAborted(input.signal)
              }
              if (!approved) {
                result = {
                  content: 'Tool error (tool_denied): User denied this tool call.',
                  isError: true,
                }
              } else {
                if (schema && schema.readOnly !== true) sideEffectToolExecuted = true
                try {
                  result = await executeToolCall(input, call, toolSchemas)
                  throwIfAborted(input.signal)
                } catch (error) {
                  if (!isAbortError(error, input.signal)) throw error
                  result = {
                    content: 'Tool execution interrupted.',
                    isError: true,
                  }
                  const toolMessage = toToolResultMessage(call, result)
                  pendingMessages.push(toolMessage)
                  yield toToolResultEvent(input.runtime.showToolResults, call, result)
                  throw error
                }
              }
            }

            pendingMessages.push(toToolResultMessage(call, result))
            yield toToolResultEvent(
              input.runtime.showToolResults,
              call,
              result,
            )
          }
          continue
        }

        if (response.stopReason === 'stop') {
          if (!response.content.trim()) {
            throw new AgentError(
              'model_protocol_error',
              'Model returned an empty response',
            )
          }
          pendingMessages.push(normalizeAssistantMessage(response, []))
          controller.commit(pendingMessages)
          committed = true
          yield { type: 'assistant_message', content: response.content }
          yield { type: 'done', content: response.content }
          return
        }

        if (
          response.stopReason === 'length' ||
          response.stopReason === 'contentFilter'
        ) {
          throw new AgentError(
            'model_incomplete',
            response.stopReason === 'length'
              ? 'Model response was truncated by the length limit'
              : 'Model response was stopped by the content filter',
          )
        }

        throw new AgentError(
          'model_protocol_error',
          `Model returned an unsupported stop reason: ${response.stopReason}`,
        )
      }

      yield errorEvent(
        'max_turns_reached',
        `Reached max turns: ${input.runtime.loop.maxTurns}`,
        false,
        sideEffectToolExecuted,
      )
    } catch (error) {
      if (isAbortError(error, input.signal)) {
        yield {
          type: 'interrupted',
          message: 'Agent turn interrupted.',
          workspaceMayHaveChanged: sideEffectToolExecuted,
        }
      } else if (error instanceof AgentError) {
        yield errorEvent(
          error.code,
          error.message,
          error.recoverable,
          sideEffectToolExecuted,
        )
      } else {
        yield errorEvent(
          'model_request_error',
          errorMessage(error),
          false,
          sideEffectToolExecuted,
        )
      }
    } finally {
      if (controller && started && !committed) controller.rollback()
    }
  }
}

type NormalizedToolResult = {
  content: string
  isError: boolean
}

async function completeModel(
  input: RunTurnInput,
  messages: readonly ModelMessage[],
  tools: ModelToolSchema[],
  limits: ModelResponseLimits,
): Promise<ModelResponse> {
  try {
    const response = await input.runtime.modelClient.complete({
      model: input.runtime.model,
      messages: [...messages],
      tools,
      signal: input.signal,
    })
    throwIfAborted(input.signal)
    return validateModelResponse(response, limits)
  } catch (error) {
    if (error instanceof AgentError || isAbortError(error, input.signal)) {
      throw error
    }
    if (error instanceof ModelProtocolError) {
      throw new AgentError('model_protocol_error', error.message)
    }
    if (error instanceof ModelResponseLimitError) {
      throw new AgentError('model_incomplete', error.message)
    }
    throw new AgentError('model_request_error', errorMessage(error))
  }
}

type ModelUpdate =
  | { type: 'assistant_delta'; content: string }
  | { type: 'response'; response: ModelResponse }

async function* consumeModel(
  input: RunTurnInput,
  messages: readonly ModelMessage[],
  tools: ModelToolSchema[],
  limits: ModelResponseLimits,
): AsyncGenerator<ModelUpdate> {
  if (!input.runtime.modelClient.stream) {
    yield {
      type: 'response',
      response: await completeModel(input, messages, tools, limits),
    }
    return
  }

  const accumulator = new ModelStreamAccumulator(limits)
  try {
    const stream = input.runtime.modelClient.stream({
      model: input.runtime.model,
      messages: [...messages],
      tools,
      signal: input.signal,
    })
    if (!stream || typeof stream[Symbol.asyncIterator] !== 'function') {
      throw new ModelProtocolError('Model client returned an invalid stream')
    }
    const iterator = stream[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await iterator.next()
        if (result.done) {
          exhausted = true
          break
        }
        throwIfAborted(input.signal)
        const text = accumulator.push(result.value)
        if (text) yield { type: 'assistant_delta', content: text }
      }
    } finally {
      if (!exhausted) {
        try {
          await iterator.return?.()
        } catch {
          // Preserve the protocol, cancellation, or budget error that closed the stream.
        }
      }
    }
    throwIfAborted(input.signal)
    yield { type: 'response', response: accumulator.finish() }
  } catch (error) {
    if (error instanceof AgentError || isAbortError(error, input.signal)) {
      throw error
    }
    if (error instanceof ModelProtocolError) {
      throw new AgentError('model_protocol_error', error.message)
    }
    if (error instanceof ModelResponseLimitError) {
      throw new AgentError('model_incomplete', error.message)
    }
    throw new AgentError('model_request_error', errorMessage(error))
  }
}

async function executeToolCall(
  input: RunTurnInput,
  call: ToolCall,
  schemas: readonly ToolSchema[],
): Promise<NormalizedToolResult> {
  const validationError = validateToolCall(call, schemas)
  if (validationError) {
    return {
      content: formatToolError(validationError.code, validationError.message),
      isError: true,
    }
  }

  try {
    const result: ToolExecutionResult = await input.runtime.tools.execute(call, {
      ...input.runtime.toolContext,
      signal: input.signal,
    })
    return {
      content: normalizeToolOutput(result?.content),
      isError: false,
    }
  } catch (error) {
    if (isAbortError(error, input.signal)) throw error
    return {
      content: formatToolError('tool_execution_error', errorMessage(error)),
      isError: true,
    }
  }
}

function getToolValidationError(
  input: RunTurnInput,
  call: ToolCall,
  schemas: readonly ToolSchema[],
  batchValidationError?: string,
): AgentError | undefined {
  if (batchValidationError) {
    return new AgentError('tool_validation_error', batchValidationError, true)
  }
  const genericError = validateToolCall(call, schemas)
  if (genericError) return genericError
  const schema = schemas.find((candidate) => candidate.name === call.name)
  if (schema) {
    try {
      validateApprovalInput(schema, call.arguments)
    } catch (error) {
      if (error instanceof ToolInputError) {
        return new AgentError('tool_validation_error', error.message, true)
      }
      throw error
    }
  }
  try {
    input.runtime.tools.validate?.(call)
  } catch (error) {
    if (error instanceof ToolInputError) {
      return new AgentError('tool_validation_error', error.message, true)
    }
    throw error
  }
  return undefined
}

function toToolErrorResult(error: AgentError): NormalizedToolResult {
  return {
    content: formatToolError(error.code, error.message),
    isError: true,
  }
}

function validateToolSchemas(schemas: ToolSchema[]): ToolSchema[] {
  const names = new Set<string>()
  for (const schema of schemas) {
    if (names.has(schema.name)) {
      throw new AgentError(
        'tool_validation_error',
        `Duplicate tool name: ${schema.name}`,
      )
    }
    names.add(schema.name)
    try {
      validateApprovalSchema(schema)
    } catch (error) {
      if (error instanceof ToolInputError) {
        throw new AgentError('tool_validation_error', error.message)
      }
      throw error
    }
  }
  return schemas
}

function toModelToolSchemas(schemas: readonly ToolSchema[]): ModelToolSchema[] {
  return schemas.map(({ name, description, parameters }) => ({
    name,
    description,
    parameters,
  }))
}

interface NormalizedToolBatch {
  calls: ToolCall[]
  error?: string
}

function normalizeToolCalls(
  calls: readonly ModelToolCall[],
  usedIds: Set<string>,
): NormalizedToolBatch {
  const errors: string[] = []
  const normalizedCalls = calls.map((call, index) => {
    const sourceId = typeof call.id === 'string' ? call.id.trim() : ''
    let id = sourceId
    if (!id) {
      errors.push(`Tool call ${index + 1} is missing a call ID.`)
      id = allocateInvalidCallId(index, usedIds)
    } else if (usedIds.has(id)) {
      errors.push(`Duplicate tool call ID at call ${index + 1}.`)
      id = allocateInvalidCallId(index, usedIds)
    } else {
      usedIds.add(id)
    }

    return normalizeToolCall(call, index, id)
  })

  return {
    calls: normalizedCalls,
    error: errors.length
      ? `Tool call batch validation failed: ${errors.join(' ')}`
      : undefined,
  }
}

function validateToolCall(
  call: ToolCall,
  schemas: readonly ToolSchema[],
): AgentError | undefined {
  if (!call.id || !call.name) {
    return new AgentError(
      'tool_validation_error',
      'Tool call must include a non-empty id and name',
      true,
    )
  }
  if (!schemas.some((schema) => schema.name === call.name)) {
    return new AgentError('tool_not_found', `Unknown tool: ${call.name}`, true)
  }
  if (call.argumentsError) {
    return new AgentError('tool_validation_error', call.argumentsError, true)
  }
  if (!isRecord(call.arguments)) {
    return new AgentError(
      'tool_validation_error',
      `Arguments for tool ${call.name} must be a JSON object`,
      true,
    )
  }
  return undefined
}

function normalizeToolCall(
  call: ModelToolCall,
  index: number,
  canonicalId?: string,
): ToolCall {
  const normalized: ToolCall = {
    id: canonicalId ?? (typeof call.id === 'string' && call.id
      ? call.id
      : `invalid-call-${index + 1}`),
    name: typeof call.name === 'string' ? call.name : '',
    arguments: isRecord(call.arguments) ? call.arguments : {},
  }
  if (call.rawArguments !== undefined) normalized.rawArguments = call.rawArguments
  if (call.argumentsError !== undefined) normalized.argumentsError = call.argumentsError
  return normalized
}

function normalizeAssistantMessage(
  response: ModelResponse,
  toolCalls: readonly ToolCall[],
): ModelAssistantMessage {
  const content: ModelContentBlock[] = []
  if (response.content) content.push(...textContent(response.content))
  for (const call of toolCalls) content.push(toToolCallBlock(call))

  return {
    role: 'assistant',
    content,
    stopReason: response.stopReason,
  }
}

function toToolCallBlock(
  call: ToolCall,
): Extract<ModelContentBlock, { type: 'toolCall' }> {
  return { type: 'toolCall', ...call }
}

function allocateInvalidCallId(index: number, usedIds: Set<string>): string {
  let suffix = 0
  let candidate = `invalid-call-${index + 1}`
  while (usedIds.has(candidate)) {
    suffix += 1
    candidate = `invalid-call-${index + 1}-${suffix}`
  }
  usedIds.add(candidate)
  return candidate
}

function toToolResultMessage(
  call: ToolCall,
  result: NormalizedToolResult,
): ModelToolResultMessage {
  return {
    role: 'toolResult',
    toolCallId: call.id,
    toolName: call.name,
    content: textContent(result.content),
    isError: result.isError,
  }
}

function toToolResultEvent(
  showToolResults: boolean,
  call: ToolCall,
  result: NormalizedToolResult,
): AgentEvent {
  return {
    type: 'tool_result',
    callId: call.id,
    name: call.name,
    output: showToolResults || result.isError ? result.content : '',
    isError: result.isError,
  }
}

function textContent(text: string): ModelContentBlock[] {
  return text ? [{ type: 'text', text }] : []
}

function normalizeToolOutput(content: unknown): string {
  if (typeof content !== 'string' || !content) {
    return 'Tool completed with no output.'
  }
  return content
}

function formatToolError(code: AgentErrorCode, message: string): string {
  return `Tool error (${code}): ${message}`
}

function errorEvent(
  code: AgentErrorCode,
  message: string,
  recoverable: boolean,
  sideEffectToolExecuted = false,
): AgentEvent {
  return {
    type: 'error',
    code,
    message,
    recoverable,
    workspaceMayHaveChanged: sideEffectToolExecuted,
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

function createAbortError(): Error {
  const error = new Error('The agent turn was aborted')
  error.name = 'AbortError'
  return error
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
