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
// [maint-note:29c9003b9d3e]
// 数据契约导航：CompactionRequest；字段含义以消费端校验为准。
// 成员 source：CompactionSource。
// 成员 requestMessages：readonly ModelMessage[]。
// 成员 signal（可省略）：AbortSignal。
// 成员 deadlineMs：number。
// [/maint-note:29c9003b9d3e]
export interface CompactionRequest {
  readonly source: CompactionSource
  readonly requestMessages: readonly ModelMessage[]
  readonly signal?: AbortSignal
  readonly deadlineMs: number
}

// [maint-note:374f1283324c]
// 维护导航：createModelCompactionProvider；下列为本函数源码中的接口与路径索引。
// 输入 options：{ readonly modelClient: ModelClient readonly model: string readonly requestBudget?: ModelRequest['budget'] }。
// 返回路径：{ budgetCapabilities: options.modelClient.budgetCapabilities, compact: (request: CompactionRequest) => runCompaction(options, request), prepare: options.modelClient.prepare ? (request: CompactionRequest) => prepareCompaction(options, reques。
// [/maint-note:374f1283324c]
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




// [maint-note:b60ca912113e]
// 维护导航：runCompaction；下列为本函数源码中的接口与路径索引。
// 输入 options：{ readonly modelClient: ModelClient readonly model: string readonly requestBudget?: ModelRequest['budget'] }。
// 输入 request：CompactionRequest。
// 输入 preprepared：{ readonly controller: AbortController readonly model: PreparedModelDispatch }（可省略）。
// 返回契约：Promise<ProviderBodyResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：request.signal?.aborted。
// 分支条件：options.modelClient.stream。
// 分支条件：!isAsyncIterable<ModelStreamEvent>(stream)。
// 分支条件：step.done。
// 分支条件：step.value.type === 'text_delta'。
// 分支条件：outputChars > COMPACTION_LIMITS.maxOutputChars。
// 分支条件：firstCause === undefined。
// 分支条件：validatedUsage。
// 分支条件：!validated.ok。
// 分支条件：isAsyncIterable(sent)。
// 分支条件：isAbortError(error)。
// 分支条件：error instanceof ModelProtocolError || error instanceof ModelResponseLimitError。
// 分支条件：!controller.signal.aborted。
// 分支条件：firstCause !== undefined。
// 分支条件：body === undefined || !body.trim()。
// 遍历边界：true。
// 返回路径：{ ok: false, kind: 'aborted', message: 'The maintenance request was aborted before dispatch' }。
// 返回路径：{ ok: false, kind: firstCause as never, message: describeSealed(firstCause), ...usageFields(usage), ...(body !== undefined ? { body } : {}) }。
// 返回路径：{ ok: false, kind: 'invalid_output', message: 'The maintenance response was empty', ...usageFields(usage) }。
// 返回路径：{ ok: true, body, ...usageFields(usage) }。
// 显式抛出：new ModelProtocolError('Prepared maintenance stream returned a response')。
// 显式抛出：new ModelProtocolError('Prepared maintenance completion returned a stream')。
// 写入目标：outputChars。
// 写入目标：firstCause。
// 写入目标：usage.inputTokens。
// 写入目标：usage.outputTokens。
// 写入目标：usage.totalTokens。
// 写入目标：body。
// 直接调用：request.signal?.addEventListener。
// 直接调用：setTimeout。
// 直接调用：Math.max。
// 直接调用：request.requestMessages.map。
// 直接调用：preprepared?.model.send。
// 直接调用：options.modelClient.stream。
// 直接调用：isAsyncIterable。
// 直接调用：stream[Symbol.asyncIterator]。
// 直接调用：raceWithAbort。
// 直接调用：iterator.next。
// 直接调用：controller.abort。
// 直接调用：accumulator.push。
// 直接调用：accumulator.usageSnapshot。
// 直接调用：boundedIteratorClose。
// 直接调用：accumulator.finish。
// 直接调用：validateMaintenanceResponse。
// 直接调用：options.modelClient.complete。
// 直接调用：isAbortError。
// 直接调用：clearTimeout。
// 直接调用：request.signal?.removeEventListener。
// 直接调用：describeSealed。
// 直接调用：usageFields。
// 直接调用：body.trim。
// [/maint-note:b60ca912113e]
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
  // [maint-note:e5f8ce3d349c]
  // 维护导航：externalAbort；下列为本函数源码中的接口与路径索引。
  // 分支条件：firstCause === undefined。
  // 写入目标：firstCause。
  // 直接调用：controller.abort。
  // [/maint-note:e5f8ce3d349c]
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

// [maint-note:0bcd6180d419]
// 维护导航：prepareCompaction；下列为本函数源码中的接口与路径索引。
// 输入 options：{ readonly modelClient: ModelClient readonly model: string readonly requestBudget?: ModelRequest['budget'] }。
// 输入 request：CompactionRequest。
// 返回路径：Object.freeze({ measurement: model.measurement, dispatch: () => runCompaction(options, request, { controller, model }), })。
// 直接调用：request.requestMessages.map。
// 直接调用：options.modelClient.prepare!。
// 直接调用：Object.freeze。
// [/maint-note:0bcd6180d419]
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

// [maint-note:3d77749b3537]
// 维护导航：isAsyncIterable；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is AsyncIterable<T>。
// 返回路径：Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')。
// 直接调用：Boolean。
// [/maint-note:3d77749b3537]
function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return Boolean(value && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === 'function')
}

/** Pre-dispatch measurement: the real outbound shell, JSON escaping included. */
// [maint-note:25d10f87d5dd]
// 维护导航：maintenanceRequestMessages；下列为本函数源码中的接口与路径索引。
// 输入 source：CompactionSource。
// 返回契约：ModelMessage[]。
// 返回路径：[ { role: 'system', content: [{ type: 'text', text: SUMMARY_SYSTEM_PROMPT }] }, { role: 'user', content: [{ type: 'text', text: [ `Source units (${source.units.length}):`, ...source.units.map((unit) => `- ${unit.exchangeId} (${unit.kind})`)。
// 直接调用：[ `Source units (${source.units.length}):`, ...source.units.map((unit) => `- ${unit.exchangeId} (${unit.kind})`), 'TRANSCRIPT (JSON; cite the unit ids above in sourceIds):', JSON.stringify(source.messages), ].join。
// 直接调用：source.units.map。
// 直接调用：JSON.stringify。
// [/maint-note:25d10f87d5dd]
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
// [maint-note:91f33f6a1f78]
// 维护导航：repairRequestMessages；下列为本函数源码中的接口与路径索引。
// 输入 source：CompactionSource。
// 输入 rejectedBody：string。
// 输入 diagnostics：string。
// 返回契约：ModelMessage[] | undefined。
// 分支条件：JSON.stringify(build(0)).length > COMPACTION_LIMITS.maxInputChars。
// 分支条件：JSON.stringify(build(middle)).length <= COMPACTION_LIMITS.maxInputChars。
// 遍历边界：low < high。
// 返回路径：undefined。
// 返回路径：build(low)。
// 写入目标：low。
// 写入目标：high。
// 直接调用：maintenanceRequestMessages。
// 直接调用：Math.min。
// 直接调用：JSON.stringify。
// 直接调用：build。
// 直接调用：Math.ceil。
// [/maint-note:91f33f6a1f78]
export function repairRequestMessages(
  source: CompactionSource,
  rejectedBody: string,
  diagnostics: string,
): ModelMessage[] | undefined {
  const base = maintenanceRequestMessages(source)
  const sourceChars = [...rejectedBody]
  const excerptLimit = Math.min(sourceChars.length, MAX_REJECTED_BODY_CHARS)
  // [maint-note:a5dff7787981]
  // 维护导航：build；下列为本函数源码中的接口与路径索引。
  // 输入 keep：number。
  // 返回契约：ModelMessage[]。
  // 返回路径：[ ...base, { role: 'user', content: [{ type: 'text', text: [ 'Your previous reply was rejected by schema/source validation.', `Diagnostics: ${diagnostics}`, 'Fix ONLY the reported problem and reply again with ONE JSON object matching the sc。
  // 直接调用：sourceChars.slice(0, keep).join。
  // 直接调用：sourceChars.slice。
  // 直接调用：[ 'Your previous reply was rejected by schema/source validation.', `Diagnostics: ${diagnostics}`, 'Fix ONLY the reported problem and reply again with ONE JSON object matching the schema, citing only the supplied unit ids. The rejected reply。
  // [/maint-note:a5dff7787981]
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

// [maint-note:064fe77ebf8f]
// 维护导航：validateMaintenanceResponse；下列为本函数源码中的接口与路径索引。
// 输入 response：ModelResponse。
// 返回契约：| { readonly ok: true; readonly response: ModelResponse } | { readonly ok: false readonly failure: 'invalid_output' | 'output_overflow' /** Present only when the base response, including usage, was validated. */ readonly response?: ModelRes。
// 分支条件：canonical.stopReason !== 'stop'。
// 分支条件：canonical.toolCalls.length > 0。
// 分支条件：detectToolControlEnvelope(canonical.content)。
// 返回路径：{ ok: false, failure: error instanceof ModelResponseLimitError && /text character limit/i.test(error.message) ? 'output_overflow' : 'invalid_output', }。
// 返回路径：{ ok: false, failure: 'invalid_output', response: canonical }。
// 返回路径：{ ok: true, response: canonical }。
// 写入目标：canonical。
// 直接调用：validateModelResponse。
// 直接调用：/text character limit/i.test。
// 直接调用：detectToolControlEnvelope。
// [/maint-note:064fe77ebf8f]
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

// [maint-note:0de5e9c5e7a7]
// 维护导航：raceWithAbort；下列为本函数源码中的接口与路径索引。
// 输入 promise：Promise<T>。
// 输入 signal：AbortSignal。
// 返回契约：Promise<T>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：new Promise<T>((resolve, reject) => { const onAbort = () => reject(abortError()) if (signal.aborted) { reject(abortError()) return } signal.addEventListener('abort', onAbort, { once: true }) promise.then( (value) => { signal.removeEventList。
// [/maint-note:0de5e9c5e7a7]
async function raceWithAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    // [maint-note:2a02d1f773d6]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 直接调用：reject。
    // 直接调用：abortError。
    // [/maint-note:2a02d1f773d6]
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

// [maint-note:05337da643ae]
// 维护导航：boundedIteratorClose；下列为本函数源码中的接口与路径索引。
// 输入 iterator：AsyncIterator<unknown>。
// 输入 deadlineMs：number。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!close。
// 返回路径：无返回值。
// 直接调用：Math.min。
// 直接调用：Math.max。
// 直接调用：iterator.return。
// [/maint-note:05337da643ae]
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

// [maint-note:cc83843948e7]
// 维护导航：usageFields；下列为本函数源码中的接口与路径索引。
// 输入 usage：{ inputTokens?: number; outputTokens?: number; totalTokens?: number }。
// 返回契约：{ usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; usageComplete: boolean } }。
// 分支条件：usage.inputTokens === undefined && usage.outputTokens === undefined && usage.totalTokens === undefined。
// 返回路径：{}。
// 返回路径：{ usage: { ...usage, usageComplete: complete } }。
// [/maint-note:cc83843948e7]
function usageFields(usage: { inputTokens?: number; outputTokens?: number; totalTokens?: number }):
  { usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number; usageComplete: boolean } } {
  if (usage.inputTokens === undefined && usage.outputTokens === undefined && usage.totalTokens === undefined) return {}
  // completeness requires EVERY field the ledger tracks to be known;
  // a partially-known dispatch is honest usage but not complete usage.
  const complete = usage.inputTokens !== undefined && usage.outputTokens !== undefined && usage.totalTokens !== undefined
  return { usage: { ...usage, usageComplete: complete } }
}

// [maint-note:40bc088c497d]
// 维护导航：isAbortError；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：boolean。
// 返回路径：error instanceof Error && error.name === 'AbortError'。
// [/maint-note:40bc088c497d]
function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

// [maint-note:41bbc75841dd]
// 维护导航：abortError；下列为本函数源码中的接口与路径索引。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:41bbc75841dd]
function abortError(): Error {
  const error = new Error('maintenance aborted')
  error.name = 'AbortError'
  return error
}

// [maint-note:78b444e4dc27]
// 维护导航：describeSealed；下列为本函数源码中的接口与路径索引。
// 输入 kind：string。
// 返回契约：string。
// 分支条件：switch kind。
// 分支条件：case 'aborted'。
// 分支条件：case 'timeout'。
// 分支条件：case 'output_overflow'。
// 分支条件：case 'invalid_output'。
// 返回路径：'The maintenance request was aborted'。
// 返回路径：'The maintenance request exceeded its deadline'。
// 返回路径：`The compaction output exceeded ${COMPACTION_LIMITS.maxOutputChars} characters`。
// 返回路径：'The compaction response violated the model response or control-envelope protocol'。
// 返回路径：'The maintenance request failed at the transport boundary'。
// [/maint-note:78b444e4dc27]
function describeSealed(kind: string): string {
  switch (kind) {
    case 'aborted': return 'The maintenance request was aborted'
    case 'timeout': return 'The maintenance request exceeded its deadline'
    case 'output_overflow': return `The compaction output exceeded ${COMPACTION_LIMITS.maxOutputChars} characters`
    case 'invalid_output': return 'The compaction response violated the model response or control-envelope protocol'
    default: return 'The maintenance request failed at the transport boundary'
  }
}
