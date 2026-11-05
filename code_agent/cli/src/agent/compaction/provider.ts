import type {
  ModelClient,
  ModelMessage,
  ModelRequest,
  ModelResponse,
  ModelStreamEvent,
  PreparedModelDispatch,
} from '../../model/types/types.js'
import {
  ModelStreamAccumulator,
  validateModelResponse,
  type ModelResponseLimits,
} from '../../model/stream.js'
import { ModelProtocolError, ModelResponseLimitError } from '../../model/errors.js'
import { detectToolControlEnvelope } from '../../model/controlEnvelope.js'
import {
  COMPACTION_LIMITS,
  type CompactionSource,
  type ProviderBodyResult,
} from './contracts.js'

const SUMMARY_SYSTEM_PROMPT = [
  'You are a context compaction service. Reply with ONE JSON object and nothing else.',
  'Schema: {"facts":[{"text":"...","sourceIds":["<unit id>"]}],"failures":[...],"unknowns":[...],"unfinished":[...],"suggestions":[...]}.',
  'Every entry must cite at least one sourceId from the supplied unit list; suggestions are explicitly inference and must not cite sources as confirmed facts.',
  'facts may only restate what the quoted tool results establish; put anything inferred under suggestions, and anything not derivable from the evidence under unknowns.',
  'Keep every number, path, exit code and identifier a later step could depend on.',
].join(' ')

/**
 * Maintenance is tools-disabled and has a deliberately smaller response
 * surface than a normal coding request. In particular, the first tool delta
 * is rejected instead of buffering the normal 128-call/1M-argument budget.
 */
export const COMPACTION_MODEL_RESPONSE_LIMITS: ModelResponseLimits = Object.freeze({
  maxEvents: 2_048,
  maxTextChars: COMPACTION_LIMITS.maxOutputChars,
  maxToolCalls: 0,
  maxToolArgumentChars: 0,
  maxTotalToolArgumentChars: 0,
})

/**
 * model-backed provider: one tools-disabled summarization request.
 *
 * Protocol: the SAME ModelStreamAccumulator / validateModelResponse /
 * control-envelope gate the working Agent uses — bad stop reasons, deltas
 * after finish, tool calls and textual tool-control envelopes are rejected
 * before any body parsing. The returned object is an UNTRUSTED body string
 * that the Thread core re-validates and re-envelopes.
 *
 * Resources: one AbortController per attempt with an absolute deadline; the
 * first failure kind latches BEFORE the child aborts; non-cooperative
 * iterators/promises are only awaited within the bounded cleanup grace, and
 * late resolutions/rejections are absorbed by detached handlers. The
 * complete()-only fallback bounds the returned content — never the
 * provider's own peak allocation (documented first-version boundary).
 */
/** Request shape of one maintenance call (mirrors CompactionProvider). */
export interface CompactionRequest {
  readonly source: CompactionSource
  readonly requestMessages: readonly ModelMessage[]
  readonly signal?: AbortSignal
  readonly deadlineMs: number
}

export function createModelCompactionProvider(options: {
  readonly modelClient: ModelClient
  readonly model: string
  readonly requestBudget?: ModelRequest['budget']
}) {
  return {
    budgetCapabilities: options.modelClient.budgetCapabilities,
    compact: (request: CompactionRequest) => runCompaction(options, request),
    prepare: options.modelClient.prepare
      ? (request: CompactionRequest) => prepareCompaction(options, request)
      : undefined,
  }
}




async function runCompaction(
  options: {
    readonly modelClient: ModelClient
    readonly model: string
    readonly requestBudget?: ModelRequest['budget']
  },
  request: CompactionRequest,
  preprepared?: {
    readonly controller: AbortController
    readonly model: PreparedModelDispatch
  },
): Promise<ProviderBodyResult> {
  if (request.signal?.aborted) {
    return { ok: false, kind: 'aborted', message: 'The maintenance request was aborted before dispatch' }
  }

  const controller = preprepared?.controller ?? new AbortController()
  let firstCause: string | undefined
  const externalAbort = () => {
    if (firstCause === undefined) firstCause = 'aborted'
    controller.abort()
  }
  request.signal?.addEventListener('abort', externalAbort, { once: true })

  const deadlineTimer = setTimeout(() => {
    if (firstCause === undefined) firstCause = 'timeout'
    controller.abort()
  }, Math.max(1, request.deadlineMs))

  const modelRequest: ModelRequest = {
    model: options.model,
    messages: request.requestMessages.map((message) => structuredClone(message)),
    toolChoice: 'none',
    signal: controller.signal,
    ...(options.requestBudget ? { budget: options.requestBudget } : {}),
  }

  const usage: { inputTokens?: number; outputTokens?: number; totalTokens?: number } = {}
  let body: string | undefined
  // Per-attempt accumulator: bound to THIS closure only.
  const accumulator = new ModelStreamAccumulator(COMPACTION_MODEL_RESPONSE_LIMITS)
  try {
    if (options.modelClient.stream) {
      const stream = preprepared?.model.send() ?? options.modelClient.stream(modelRequest)
      if (!isAsyncIterable<ModelStreamEvent>(stream)) {
        throw new ModelProtocolError('Prepared maintenance stream returned a response')
      }
      const iterator = stream[Symbol.asyncIterator]()
      let outputChars = 0
      try {
        while (true) {
          const step = await raceWithAbort(iterator.next(), controller.signal)
          if (step.done) break
          if (step.value.type === 'text_delta') {
            outputChars += step.value.text.length
            if (outputChars > COMPACTION_LIMITS.maxOutputChars) {
              // Overflow latches BEFORE the child aborts; the iterator is
              // then only awaited within the bounded cleanup grace.
              if (firstCause === undefined) firstCause = 'output_overflow'
              controller.abort()
              break
            }
          }
          try {
            accumulator.push(step.value)
            const validatedUsage = accumulator.usageSnapshot()
            if (validatedUsage) {
              usage.inputTokens = validatedUsage.inputTokens
              usage.outputTokens = validatedUsage.outputTokens
              usage.totalTokens = validatedUsage.totalTokens
            }
          } catch {
            if (firstCause === undefined) firstCause = 'invalid_output'
            controller.abort()
            break
          }
        }
      } finally {
        // Bounded, non-blocking tail: the result below is already determined.
        void boundedIteratorClose(iterator, 0)
      }
      if (firstCause === undefined) {
        const response = accumulator.finish()
        usage.inputTokens = response.usage?.inputTokens ?? usage.inputTokens
        usage.outputTokens = response.usage?.outputTokens ?? usage.outputTokens
        usage.totalTokens = response.usage?.totalTokens ?? usage.totalTokens
        const validated = validateMaintenanceResponse(response)
        if (!validated.ok) firstCause = validated.failure
        else body = validated.response.content
      }
    } else {
      const sent = preprepared?.model.send() ?? options.modelClient.complete(modelRequest)
      if (isAsyncIterable(sent)) throw new ModelProtocolError('Prepared maintenance completion returned a stream')
      const response = await raceWithAbort(sent, controller.signal)
      const validated = validateMaintenanceResponse(response)
      const validatedUsage = validated.response?.usage
      if (validatedUsage) {
        usage.inputTokens = validatedUsage.inputTokens
        usage.outputTokens = validatedUsage.outputTokens
        usage.totalTokens = validatedUsage.totalTokens
      }
      if (!validated.ok) firstCause = validated.failure
      else body = validated.response.content
    }
  } catch (error) {
    if (firstCause === undefined) {
      if (isAbortError(error)) {
        firstCause = request.signal?.aborted ? 'aborted' : 'timeout'
      } else if (error instanceof ModelProtocolError || error instanceof ModelResponseLimitError) {
        // a model-protocol violation (missing finish, late delta, bad
        // stop reason...) is invalid_output, NOT a transport failure — it is
        // never repairable and never reported as a network problem.
        firstCause = 'invalid_output'
      } else {
        firstCause = 'transport_error'
      }
    }
  } finally {
    clearTimeout(deadlineTimer)
    request.signal?.removeEventListener('abort', externalAbort)
    if (!controller.signal.aborted) controller.abort()
  }

  // the latched first cause and known usage are published
  // IMMEDIATELY — the non-cooperative close is only a bounded tail that can
  // never delay or rewrite the visible outcome.
  if (firstCause !== undefined) {
    return { ok: false, kind: firstCause as never, message: describeSealed(firstCause), ...usageFields(usage), ...(body !== undefined ? { body } : {}) }
  }
  if (body === undefined || !body.trim()) {
    return { ok: false, kind: 'invalid_output', message: 'The maintenance response was empty', ...usageFields(usage) }
  }
  return { ok: true, body, ...usageFields(usage) }
}

function prepareCompaction(
  options: {
    readonly modelClient: ModelClient
    readonly model: string
    readonly requestBudget?: ModelRequest['budget']
  },
  request: CompactionRequest,
) {
  const controller = new AbortController()
  const modelRequest: ModelRequest = {
    model: options.model,
    messages: request.requestMessages.map((message) => structuredClone(message)),
    toolChoice: 'none',
    signal: controller.signal,
    ...(options.requestBudget ? { budget: options.requestBudget } : {}),
  }
  const mode = options.modelClient.stream ? 'stream' : 'complete'
  const model = options.modelClient.prepare!(modelRequest, mode)
  return Object.freeze({
    measurement: model.measurement,
    dispatch: () => runCompaction(options, request, { controller, model }),
  })
}

function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')
}

/** Pre-dispatch measurement: the real outbound shell, JSON escaping included. */
export function maintenanceRequestMessages(source: CompactionSource): ModelMessage[] {
  return [
    { role: 'system', content: [{ type: 'text', text: SUMMARY_SYSTEM_PROMPT }] },
    {
      role: 'user',
      content: [{
        type: 'text',
        text: [
          `Source units (${source.units.length}):`,
          ...source.units.map((unit) => `- ${unit.exchangeId} (${unit.kind})`),
          'TRANSCRIPT (JSON; cite the unit ids above in sourceIds):',
          JSON.stringify(source.messages),
        ].join('\n'),
      }],
    },
  ]
}

/** Bounded excerpt of a rejected reply carried by the ONE schema repair. */
const MAX_REJECTED_BODY_CHARS = 2_000

/**
 * 77.4: the narrow schema repair carries the BOUNDED rejected candidate and
 * the validation diagnostics — the model sees exactly what failed and why,
 * on the SAME frozen source, within a bounded addition to the shell.
 */
export function repairRequestMessages(
  source: CompactionSource,
  rejectedBody: string,
  diagnostics: string,
): ModelMessage[] | undefined {
  const base = maintenanceRequestMessages(source)
  const sourceChars = [...rejectedBody]
  const excerptLimit = Math.min(sourceChars.length, MAX_REJECTED_BODY_CHARS)
  const build = (keep: number): ModelMessage[] => {
    const omitted = keep < sourceChars.length
    const excerpt = keep > 0
      ? `${sourceChars.slice(0, keep).join('')}${omitted ? '\n…(excerpt truncated)' : ''}`
      : '(rejected reply omitted because the maintenance input budget is exhausted)'
    return [
      ...base,
      {
        role: 'user',
        content: [{
          type: 'text',
          text: [
            'Your previous reply was rejected by schema/source validation.',
            `Diagnostics: ${diagnostics}`,
            'Fix ONLY the reported problem and reply again with ONE JSON object matching the schema, citing only the supplied unit ids. The rejected reply (bounded excerpt) was:',
            excerpt,
          ].join('\n'),
        }],
      },
    ]
  }

  // W02: the repair is a maintenance request in its own right. Select the
  // largest Unicode-safe excerpt whose ACTUAL outbound message array fits;
  // if the fixed same-source shell plus mandatory diagnostic cannot fit,
  // decline the repair instead of dispatching over the hard input cap.
  if (JSON.stringify(build(0)).length > COMPACTION_LIMITS.maxInputChars) return undefined
  let low = 0
  let high = excerptLimit
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (JSON.stringify(build(middle)).length <= COMPACTION_LIMITS.maxInputChars) low = middle
    else high = middle - 1
  }
  return build(low)
}

function validateMaintenanceResponse(response: ModelResponse):
  | { readonly ok: true; readonly response: ModelResponse }
  | {
      readonly ok: false
      readonly failure: 'invalid_output' | 'output_overflow'
      /** Present only when the base response, including usage, was validated. */
      readonly response?: ModelResponse
    } {
  let canonical: ModelResponse
  try {
    canonical = validateModelResponse(response, COMPACTION_MODEL_RESPONSE_LIMITS)
  } catch (error) {
    return {
      ok: false,
      failure: error instanceof ModelResponseLimitError && /text character limit/i.test(error.message)
        ? 'output_overflow'
        : 'invalid_output',
    }
  }
  if (canonical.stopReason !== 'stop') return { ok: false, failure: 'invalid_output', response: canonical }
  if (canonical.toolCalls.length > 0) return { ok: false, failure: 'invalid_output', response: canonical }
  if (detectToolControlEnvelope(canonical.content)) return { ok: false, failure: 'invalid_output', response: canonical }
  return { ok: true, response: canonical }
}

async function raceWithAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError())
    if (signal.aborted) {
      reject(abortError())
      return
    }
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort)
        resolve(value)
      },
      (error) => {
        signal.removeEventListener('abort', onAbort)
        // A late/parallel rejection after the signal fired does not
        // overwrite the already-latched first cause; rethrow as-is.
        reject(error)
      },
    )
  })
}

async function boundedIteratorClose(iterator: AsyncIterator<unknown>, deadlineMs: number): Promise<void> {
  // cleanup uses only the REMAINING attempt time (never a fresh
  // full grace window once the deadline already passed); late rejections are
  // absorbed and never overwrite the latched first cause.
  const remaining = Math.min(COMPACTION_LIMITS.cleanupGraceMs, Math.max(0, deadlineMs))
  try {
    const close = iterator.return?.(undefined)
    if (!close) return
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => resolve(), remaining)
      Promise.resolve(close).then(
        () => {
          clearTimeout(timer)
          resolve()
        },
        () => {
          clearTimeout(timer)
          resolve()
        },
      )
    })
  } catch {
    // A non-cooperative close rejection is absorbed.
  }
}

function usageFields(usage: { inputTokens?: number; outputTokens?: number; totalTokens?: number }):
  { usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; usageComplete: boolean } } {
  if (usage.inputTokens === undefined && usage.outputTokens === undefined && usage.totalTokens === undefined) return {}
  // completeness requires EVERY field the ledger tracks to be known;
  // a partially-known dispatch is honest usage but not complete usage.
  const complete = usage.inputTokens !== undefined && usage.outputTokens !== undefined && usage.totalTokens !== undefined
  return { usage: { ...usage, usageComplete: complete } }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function abortError(): Error {
  const error = new Error('maintenance aborted')
  error.name = 'AbortError'
  return error
}

function describeSealed(kind: string): string {
  switch (kind) {
    case 'aborted': return 'The maintenance request was aborted'
    case 'timeout': return 'The maintenance request exceeded its deadline'
    case 'output_overflow': return `The compaction output exceeded ${COMPACTION_LIMITS.maxOutputChars} characters`
    case 'invalid_output': return 'The compaction response violated the model response or control-envelope protocol'
    default: return 'The maintenance request failed at the transport boundary'
  }
}
