import { lstat, readFile, stat } from 'node:fs/promises'

import type { ToolExecutionContext } from '../types.js'
import { operationError, toolDiagnostic, withFileDiagnostics } from '../diagnostics.js'
import { ToolInputError } from '../types.js'
import { resolveWorkspacePath } from '../workspace.js'
import {
  assertOutputBudget,
  throwIfAborted,
} from '../toolUtils.js'
import {
  DEFAULT_READ_PAGE_LINES,
  MAX_READ_PAGE_LINES,
  MAX_SUGGESTED_CONTINUATION_RANGES,
  parseReadFileInput,
  type ContinuationRange,
  type PagePlan,
} from './schema.js'

const MAX_READ_BYTES = 2_000_000
export async function executeReadFile(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
) {
  return withFileDiagnostics(() => readWorkspaceFile(input, context))
}

async function readWorkspaceFile(input: Record<string, unknown>, context: ToolExecutionContext) {
  const options = parseReadFileInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const path = await resolveWorkspacePath(context.projectDir, options.path)
  const info = await lstat(path)
  if (!info.isFile()) throw operationError(`Not a regular file: ${options.path}`, 'not_regular_file', { field: 'path' })
  const fileInfo = await stat(path)
  if (fileInfo.size > MAX_READ_BYTES) throw operationError(`File is larger than ${MAX_READ_BYTES} bytes`, 'output_limit', { field: 'path' })
  throwIfAborted(context.signal)
  const bytes = await readFile(path, { signal: context.signal })
  let raw: string
  try {
    raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    throw operationError(`File is not valid UTF-8: ${options.path}`, 'invalid_encoding', { field: 'path' })
  }
  if (raw.includes('\0')) throw operationError(`File is not a text file: ${options.path}`, 'invalid_encoding', { field: 'path' })
  throwIfAborted(context.signal)
  const { lines, endsWithNewline } = splitLines(raw)
  const totalLines = lines.length
  if (totalLines === 0 && options.startLine === 1) {
    return { content: serializePage({
      path: options.path,
      startLine: 1,
      endLine: 0,
      totalLines: 0,
      content: '',
      complete: true,
      truncated: false,
    }, budget) }
  }
  if (options.startLine > totalLines) {
    throw new ToolInputError(
      `startLine ${options.startLine} is past EOF (totalLines=${totalLines})`,
      toolDiagnostic('page_out_of_range', { field: 'startLine', message: `startLine ${options.startLine} is past EOF (totalLines=${totalLines}).` }),
    )
  }

  const requestedEnd = Math.min(
    options.endLine ?? options.startLine + (options.maxLines ?? DEFAULT_READ_PAGE_LINES) - 1,
    totalLines,
  )
  const lineParts = lines
    .slice(options.startLine - 1, requestedEnd)
    .map((line, index) => {
      const absoluteLine = options.startLine + index
      const hasFollowingLine = absoluteLine < totalLines || (absoluteLine === totalLines && endsWithNewline)
      return `${line}${hasFollowingLine ? '\n' : ''}`
    })

  if (options.includePagePlan === true) {
    // A request whose whole range serializes completely has nothing to plan:
    // complete pages never carry plan fields. Only when the complete
    // candidate cannot fit (the page will be partial) does the explicit plan
    // apply — including requestedEnd === totalLines.
    // Only a request that reaches totalLines can produce a complete page at
    // all; any other request is inherently partial and plans directly.
    let pageIsComplete = false
    if (requestedEnd === totalLines) {
      pageIsComplete = serializePage({
        path: options.path,
        startLine: options.startLine,
        endLine: totalLines,
        totalLines,
        content: lineParts.join(''),
        complete: true,
        truncated: false,
      }, budget, false) !== undefined
    }
    if (!pageIsComplete) {
      throwIfAborted(context.signal)
      const planned = planPage({
        path: options.path,
        lines,
        endsWithNewline,
        totalLines,
        startLine: options.startLine,
        requestedEnd,
        budget,
        signal: context.signal,
      })
      if (planned !== undefined) return { content: planned }
      // No legal plan could be generated: fall through to the plain page.
    }
  }

  // The completed page has a smaller envelope because it does not need a
  // continuation field. Check it before the monotonic search below; otherwise
  // an exact-budget final page could be skipped after an oversized partial
  // candidate.
  if (requestedEnd === totalLines) {
    const complete = serializePage({
      path: options.path,
      startLine: options.startLine,
      endLine: totalLines,
      totalLines,
      content: lineParts.join(''),
      complete: true,
      truncated: false,
    }, budget, false)
    if (complete) return { content: complete }
  }

  let low = 1
  let high = lineParts.length
  let best: string | undefined
  let bestCount = 0
  while (low <= high) {
    const count = Math.floor((low + high) / 2)
    const content = lineParts.slice(0, count).join('')
    const endLine = options.startLine + count - 1
    const candidate = serializePage({
      path: options.path,
      startLine: options.startLine,
      endLine,
      totalLines,
      content,
      complete: endLine === totalLines,
      truncated: endLine < totalLines,
      nextStartLine: endLine < totalLines ? endLine + 1 : undefined,
    }, budget, false)
    if (candidate) {
      best = candidate
      bestCount = count
      low = count + 1
    } else {
      high = count - 1
    }
  }

  if (best === undefined || bestCount === 0) {
    throw new ToolInputError(
      `read_file page at ${options.path}:${options.startLine} cannot fit maxOutputChars=${budget} without splitting a line; increase the output budget`,
      toolDiagnostic('output_limit', { field: 'maxOutputChars', hint: 'Request a smaller range; a line cannot be split.' }),
    )
  }
  return { content: best }
}

interface PlanPageInput {
  path: string
  lines: readonly string[]
  endsWithNewline: boolean
  totalLines: number
  startLine: number
  requestedEnd: number
  budget: number
  signal?: AbortSignal
}

/**
 * planning never measures past the horizon it can suggest: the current
 * page plus at most three follow-up pages. The seam below is the retained
 * segment count of the most recent plan, so tests can bound planner work on
 * huge files without depending on wall-clock numbers.
 */
export const PLAN_HORIZON_LINES = MAX_READ_PAGE_LINES * (1 + MAX_SUGGESTED_CONTINUATION_RANGES)
let lastPlanMeasuredSegments = 0
export function getLastPlanMeasuredSegments(): number {
  return lastPlanMeasuredSegments
}

/**
 * bounded page planner. Returns the serialized current page including
 * optional plan metadata, or undefined when no legal plan exists — the caller
 * then returns the plain page with no plan fields.
 *
 * Every suggested continuation range is proven to fit the output budget on its
 * own by serializing its complete canonical page (includePagePlan:false)
 * with the real JSON.stringify. Future content never enters the current
 * result; the plan metadata itself is paid for by shrinking the current page
 * whole lines at a time. If even "one whole line + the minimal plan" cannot
 * fit, planning is abandoned rather than returning an empty content page, an
 * empty array, or a fabricated fallback.
 */
function planPage(input: PlanPageInput): string | undefined {
  const { path, lines, endsWithNewline, totalLines, startLine, requestedEnd, budget, signal } = input
  const measurement = measureLines({
    lines,
    endsWithNewline,
    totalLines,
    startLine,
    maxLines: PLAN_HORIZON_LINES,
  })
  throwIfAborted(signal)

  // Worst-case metadata overhead is measured, not estimated from average line
  // lengths. Ranges are capped at each tier so the actual plan never exceeds
  // the reserve it was planned against.
  for (const maxRanges of [MAX_SUGGESTED_CONTINUATION_RANGES, 1]) {
    const reserve = planMetadataReserve(path, totalLines, maxRanges)
    const currentEnd = maxTruncatedPageEnd({
      path,
      measurement,
      totalLines,
      startLine,
      requestedEnd,
      budget: budget - reserve,
    })
    if (currentEnd === undefined) continue
    throwIfAborted(signal)

    const ranges: ContinuationRange[] = []
    let next = currentEnd + 1
    while (ranges.length < maxRanges && next <= totalLines) {
      const end = maxStandalonePageEnd({ path, measurement, totalLines, startLine: next, budget })
      if (end === undefined) break
      ranges.push({ startLine: next, endLine: end })
      next = end + 1
    }
    if (ranges.length === 0) continue
    throwIfAborted(signal)

    const pagePlanComplete = ranges.at(-1)!.endLine >= totalLines
    const plan: PagePlan = pagePlanComplete
      ? { suggestedContinuationRanges: ranges, pagePlanComplete: true }
      : {
          suggestedContinuationRanges: ranges,
          pagePlanComplete: false,
          nextPlanStartLine: ranges.at(-1)!.endLine + 1,
        }
    const serialized = serializePage({
      path,
      startLine,
      endLine: currentEnd,
      totalLines,
      content: measurement.content(startLine, currentEnd),
      complete: false,
      truncated: true,
      nextStartLine: currentEnd + 1,
      plan,
    }, budget, false)
    // The reserve math guarantees this fits; re-serialize defensively and
    // fall through to the next tier instead of ever emitting an over-budget
    // or empty plan.
    if (serialized !== undefined) return serialized
  }
  return undefined
}

interface LineMeasurement {
  content: (startLine: number, endLine: number) => string
  /** Exact JSON-escaped length (without surrounding quotes) of a page body. */
  escapedLength: (startLine: number, endLine: number) => number
}

/**
 * Precomputes per-line display segments and their exact JSON-escaped lengths
 * over the planning window, so candidate pages can be sized in O(1) during
 * the binary searches. The final chosen candidates are always re-serialized
 * with the real JSON.stringify before being returned.
 */
function measureLines(options: {
  lines: readonly string[]
  endsWithNewline: boolean
  totalLines: number
  startLine: number
  maxLines: number
}): LineMeasurement {
  const { lines, endsWithNewline, totalLines, startLine, maxLines } = options
  const firstLine = startLine
  const lastLine = Math.min(totalLines, startLine - 1 + maxLines)
  const segments: string[] = []
  const prefix: number[] = [0]
  for (let absoluteLine = firstLine; absoluteLine <= lastLine; absoluteLine += 1) {
    const hasFollowingLine = absoluteLine < totalLines || (absoluteLine === totalLines && endsWithNewline)
    const segment = `${lines[absoluteLine - 1]}${hasFollowingLine ? '\n' : ''}`
    segments.push(segment)
    // JSON.stringify(segment).slice(1, -1) is the exact escaped body.
    prefix.push(prefix[segments.length - 1] + JSON.stringify(segment).length - 2)
  }
  lastPlanMeasuredSegments = segments.length
  const offset = firstLine - 1
  return {
    content: (start, end) => segments.slice(start - offset - 1, end - offset).join(''),
    escapedLength: (start, end) => prefix[end - offset] - prefix[start - offset - 1],
  }
}

/** Length of the exact serialized envelope with an empty content body. */
function envelopeLength(page: {
  path: string
  startLine: number
  endLine: number
  totalLines: number
  complete: boolean
  nextStartLine?: number
  plan?: PagePlan
}): number {
  const value = Object.fromEntries([
    ['path', page.path],
    ['startLine', page.startLine],
    ['endLine', page.endLine],
    ['totalLines', page.totalLines],
    ['content', ''],
    ['complete', page.complete],
    page.nextStartLine === undefined ? [] : ['nextStartLine', page.nextStartLine],
    ...planEntries(page.plan),
    ['truncated', page.complete ? false : true],
  ])
  return JSON.stringify(value).length
}

function planEntries(plan: PagePlan | undefined): [string, unknown][] {
  if (plan === undefined) return []
  const entries: [string, unknown][] = [
    ['suggestedContinuationRanges', plan.suggestedContinuationRanges],
    ['pagePlanComplete', plan.pagePlanComplete],
  ]
  if (plan.nextPlanStartLine !== undefined) {
    entries.push(['nextPlanStartLine', plan.nextPlanStartLine])
  }
  return entries
}

/** Worst-case serialized plan metadata overhead for up to `maxRanges` ranges. */
function planMetadataReserve(path: string, totalLines: number, maxRanges: number): number {
  const worstNumber = 1_000_000_000
  const ranges = Array.from({ length: maxRanges }, () => ({
    startLine: worstNumber,
    endLine: worstNumber,
  }))
  const plan: PagePlan = {
    suggestedContinuationRanges: ranges,
    pagePlanComplete: false,
    nextPlanStartLine: worstNumber,
  }
  const base = {
    path,
    startLine: worstNumber,
    endLine: worstNumber,
    totalLines: Math.max(totalLines, worstNumber),
    content: '',
    complete: false,
    nextStartLine: worstNumber,
    truncated: true,
  }
  return JSON.stringify({ ...base, ...Object.fromEntries(planEntries(plan)) }).length -
    JSON.stringify(base).length
}

/**
 * Largest truncated page ending within requestedEnd whose full canonical
 * serialization fits `budget`, or undefined when not even one whole line
 * fits. Monotone in the line count, so a binary search is exact.
 */
function maxTruncatedPageEnd(options: {
  path: string
  measurement: LineMeasurement
  totalLines: number
  startLine: number
  requestedEnd: number
  budget: number
}): number | undefined {
  const { path, measurement, totalLines, startLine, requestedEnd, budget } = options
  const fits = (endLine: number): boolean => {
    const envelope = envelopeLength({ path, startLine, endLine, totalLines, complete: false, nextStartLine: endLine + 1 })
    return envelope + measurement.escapedLength(startLine, endLine) <= budget
  }
  let low = startLine
  let high = Math.min(requestedEnd, totalLines - 1)
  let best: number | undefined
  while (low <= high) {
    const endLine = Math.floor((low + high) / 2)
    if (fits(endLine)) {
      best = endLine
      low = endLine + 1
    } else {
      high = endLine - 1
    }
  }
  return best
}

/**
 * Largest standalone page starting at `startLine` that independently fits
 * `budget` when serialized with includePagePlan:false. The chosen range is
 * re-proven with the real JSON.stringify by the caller's final serialization
 * of the current page; this function's candidates use the same exact length
 * math, so both agree.
 */
function maxStandalonePageEnd(options: {
  path: string
  measurement: LineMeasurement
  totalLines: number
  startLine: number
  budget: number
}): number | undefined {
  const { path, measurement, totalLines, startLine, budget } = options
  const fits = (endLine: number): boolean => {
    const complete = endLine === totalLines
    const envelope = envelopeLength({
      path,
      startLine,
      endLine,
      totalLines,
      complete,
      nextStartLine: complete ? undefined : endLine + 1,
    })
    return envelope + measurement.escapedLength(startLine, endLine) <= budget
  }
  let low = startLine
  let high = Math.min(startLine + MAX_READ_PAGE_LINES - 1, totalLines)
  let best: number | undefined
  while (low <= high) {
    const endLine = Math.floor((low + high) / 2)
    if (fits(endLine)) {
      best = endLine
      low = endLine + 1
    } else {
      high = endLine - 1
    }
  }
  if (best !== undefined) {
    // Prove the chosen range with a real serialization, as required: the
    // exact-length math must agree with JSON.stringify.
    const complete = best === totalLines
    const serialized = serializePage({
      path,
      startLine,
      endLine: best,
      totalLines,
      content: measurement.content(startLine, best),
      complete,
      truncated: !complete,
      nextStartLine: complete ? undefined : best + 1,
    }, budget, false)
    if (serialized === undefined) return undefined
  }
  return best
}

function splitLines(raw: string): { lines: string[]; endsWithNewline: boolean } {
  const normalized = raw.replace(/\r\n?/g, '\n')
  const endsWithNewline = normalized.endsWith('\n')
  if (!normalized) return { lines: [], endsWithNewline: false }
  const lines = normalized.split('\n')
  if (endsWithNewline) lines.pop()
  return { lines, endsWithNewline }
}

interface ReadFilePage {
  path: string
  startLine: number
  endLine: number
  totalLines: number
  content: string
  complete: boolean
  truncated: boolean
  nextStartLine?: number
  plan?: PagePlan
}

export function serializeReadFilePage(page: ReadFilePage): string {
  return serializePage(page, Number.MAX_SAFE_INTEGER)
}

function serializePage(page: ReadFilePage, budget: number): string
function serializePage(page: ReadFilePage, budget: number, throwOnOverflow: false): string | undefined
function serializePage(
  page: ReadFilePage,
  budget: number,
  throwOnOverflow = true,
): string | undefined {
  const value = Object.fromEntries([
    ['path', page.path],
    ['startLine', page.startLine],
    ['endLine', page.endLine],
    ['totalLines', page.totalLines],
    ['content', page.content],
    ['complete', page.complete],
    page.nextStartLine === undefined ? [] : ['nextStartLine', page.nextStartLine],
    ...planEntries(page.plan),
    ['truncated', page.truncated],
  ])
  const serialized = JSON.stringify(value)
  if (serialized.length <= budget) return serialized
  if (throwOnOverflow) {
    throw new ToolInputError(
      `read_file result envelope exceeds maxOutputChars=${budget}; increase the output budget`,
      toolDiagnostic('output_limit', { field: 'maxOutputChars' }),
    )
  }
  return undefined
}
