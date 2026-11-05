import { buildApprovalPresentation } from '../../tools/approval.js'
import {
  finalizeDiffView,
  projectDiffRows,
  renderDiffLinesAsText,
  renderDiffRowsAsText,
  type DiffVisualLine,
} from '../presentation/applyPatchDiff.js'
import { UiPresentationLimits } from '../presentation/limits.js'
import { boundRetainedLines } from '../presentation/boundedLines.js'
import { wrapDisplayLines } from '../terminal/index.js'
import { ToolPresenterRegistry, type ToolPresenter } from './registry.js'
import { MAX_SUGGESTED_CONTINUATION_RANGES } from '../../tools/readFile/index.js'

/**
 * default summary budgets: every presenter folds to this by default; the
 * retained records live in the bounded detail source (see `detailSource`).
 */
const DEFAULT_MAX_PRESENTATION_CHARS = UiPresentationLimits.summaryMaxChars
const DEFAULT_MAX_PRESENTATION_LINES = UiPresentationLimits.summaryMaxLogicalLines
const GLOBAL_TRUNCATION_NOTICE = '… output truncated; details unavailable'
const VIEW_COLLAPSED_NOTICE = (hidden: number): string => `${hidden} lines hidden · Ctrl+O view details`
const SECTION_TRUNCATION_NOTICE = '… section truncated'

export type ToolDisplayBlock =
  | { kind: 'summary'; text: string }
  | { kind: 'spacer' }
  | { kind: 'field'; label: string; text: string; preformatted?: boolean; truncated?: boolean }
  | { kind: 'list'; label: string; items: string[]; truncated?: boolean }
  | { kind: 'table'; label: string; rows: string[][]; truncated?: boolean }
  | { kind: 'notice'; text: string }
  /**
   * structured diff: the FINAL visual ledger at the projection width.
   * `text` is the NO_COLOR render of `lines`; Ink renders `lines` with the
   * color projection (gray gutter, colored marker/text).
   */
  | { kind: 'diff'; text: string; lines: readonly DiffVisualLine[] }

/**
 * light detail reference stored on the stable message: counters and the
 * dual truncation state only. The heavy retained lines live exclusively in
 * the bounded detail store (see `detailSource`), so eviction actually frees.
 */
export interface ToolResultDetail {
  totalRecords: number
  omittedRecords: number
  sourceTruncated: boolean
  viewCollapsed: boolean
  canExpand: boolean
}

/**
 * Heavy, bounded detail source produced beside the summary. The message
 * commit owner transfers it into the detail store exactly once and strips it
 * from the stored message; presenters never keep it elsewhere.
 */
export interface ToolResultDetailSource {
  lines: readonly string[]
  sourceTruncated: boolean
  viewCollapsed: boolean
}

export interface ToolResultPresentation {
  toolName: string
  summary: string
  blocks: readonly ToolDisplayBlock[]
  plainText: string
  truncated: boolean
  sourceTruncated: boolean
  viewCollapsed: boolean
  canExpand: boolean
  detail: ToolResultDetail
  /** Heavy bounded detail source; the commit owner moves it to the store. */
  detailSource?: ToolResultDetailSource
}

/**
 * diff projection from the executor's display facts. The table summary
 * stays as the canonical fallback path; the diff view is additive and only
 * renders when the facts parse strictly — a malformed payload degrades to
 * the table rather than inventing a diff.
 */
function presentApplyPatchDiff(
  display: import('../../tools/types.js').ToolDisplayFacts,
  value: unknown,
  maxChars: number,
  maxLines: number,
  _reserveGlobalNotice: boolean,
  contentWidth: number,
): ToolResultPresentation | undefined {
  if (!isRecord(value) || !hasExactKeys(value, ['applied', 'truncated']) || !Array.isArray(value.applied) || typeof value.truncated !== 'boolean') return undefined
  const files = parseDiffFiles(display.facts)
  if (files === undefined) return undefined
  // Both canonical truncation and the display builder's caps are upstream
  // losses: the retained output is complete-but-partial and NOT recoverable
  // from the UI detail cache (that is viewCollapsed, a different state).
  const projection = projectDiffRows({
    files,
    displayTruncated: display.facts !== null && typeof display.facts === 'object' && (display.facts as Record<string, unknown>).displayTruncated === true,
    sourceTruncated: value.truncated === true,
  })
  const summary = `${projection.totals.files} file${projection.totals.files === 1 ? '' : 's'} changed +${projection.totals.additions}/-${projection.totals.deletions}`
  // Build the retained detail before choosing full/compact projection. The
  // compact path omits the specialized diff body, but it must preserve the
  // same detail identity and source/view truncation facts.
  const unfolded = projectDiffRows({
    files,
    displayTruncated: display.facts !== null && typeof display.facts === 'object' && (display.facts as Record<string, unknown>).displayTruncated === true,
    sourceTruncated: value.truncated === true,
  }, { collapse: false })
  const detail = boundDetailLines(renderDiffRowsAsText(unfolded.rows).split('\n'))
  const sourceTruncated = projection.sourceTruncated || detail.clipped
  const sourceNotice = sourceTruncated
    ? 'Output was truncated at the source; all retained output is shown'
    : undefined
  const physicalRows = (text: string): readonly string[] => wrapDisplayLines(text, contentWidth, {
    hard: true,
    trim: false,
    wordWrap: false,
  })

  // A useful full diff needs the complete line-number/marker gutter plus a
  // bounded amount of source/notice text. Below this threshold Ink would
  // wrap the gutter itself and invalidate every row estimate. Compact mode
  // deliberately omits line-numbered diff rows; it keeps statistics, the
  // recoverable-detail affordance and an explicit upstream-loss warning.
  const fullDiffMinWidth = projection.numberWidth + ' │ '.length + 2 + 8
  if (contentWidth < fullDiffMinWidth) {
    const compactSummary = `Δ${projection.totals.files} +${projection.totals.additions}/-${projection.totals.deletions}`
    const compactCandidates: Array<{ kind: 'summary' | 'notice'; choices: readonly string[] }> = [
      { kind: 'summary', choices: [compactSummary, `Δ${projection.totals.files}`] },
      ...(sourceTruncated
        ? [{ kind: 'notice' as const, choices: ['source cut', 'src cut', '!'] }]
        : []),
      ...(projection.totalRecords > 0
        ? [{ kind: 'notice' as const, choices: [`${projection.totalRecords} hidden Ctrl+O`, `${projection.totalRecords} hidden`, `⋮${projection.totalRecords}`] }]
        : []),
    ]
    const blocks: ToolDisplayBlock[] = []
    let usedRows = 0
    let usedChars = 0
    for (const candidate of compactCandidates) {
      const choice = candidate.choices.find((text) => {
        const lines = physicalRows(text)
        const chars = lines.reduce((sum, line) => sum + line.length + 1, 0)
        return usedRows + lines.length <= maxLines && usedChars + chars <= maxChars
      })
      if (choice === undefined) continue
      const lines = physicalRows(choice)
      usedRows += lines.length
      usedChars += lines.reduce((sum, line) => sum + line.length + 1, 0)
      blocks.push({ kind: candidate.kind, text: lines.join('\n') })
    }
    const plainText = blocks.map((block) => block.kind === 'summary' || block.kind === 'notice' ? block.text : '').filter(Boolean).join('\n')
    const viewCollapsed = detail.lines.length > 0
    return {
      toolName: 'apply_patch',
      summary: sanitizeToolText(blocks.find((block) => block.kind === 'summary')?.text ?? ''),
      blocks,
      plainText: sanitizeToolText(plainText),
      truncated: sourceTruncated,
      sourceTruncated,
      viewCollapsed,
      canExpand: viewCollapsed,
      detail: {
        totalRecords: projection.totalRecords,
        omittedRecords: projection.totalRecords,
        sourceTruncated,
        viewCollapsed,
        canExpand: viewCollapsed,
      },
      detailSource: {
        lines: detail.lines,
        sourceTruncated,
        viewCollapsed,
      },
    }
  }

  const fixedLines = [
    ...physicalRows(summary),
    ...(sourceNotice === undefined ? [] : physicalRows(sourceNotice)),
  ]
  // has one final physical ledger. Summary and upstream-loss notice
  // consume rows/chars before the diff body (whose own ledger includes file
  // headers, wrapped records and collapse notices) is admitted.
  // This presentation intentionally has no separator row between summary and
  // diff. Charge only rows that blocks/plainText/Ink actually emit.
  const fixedChars = fixedLines.reduce((sum, line) => sum + line.length + 1, 0)
  const view = finalizeDiffView(projection, contentWidth, {
    maxRows: Math.max(0, maxLines - fixedLines.length),
    maxChars: Math.max(0, maxChars - fixedChars),
  })
  const diffText = renderDiffLinesAsText(view.lines)
  const blocks: ToolDisplayBlock[] = [{ kind: 'summary', text: summary }]
  if (view.lines.length > 0) {
    blocks.push({ kind: 'diff', text: diffText, lines: view.lines })
  }
  if (sourceNotice !== undefined) blocks.push({ kind: 'notice', text: sourceNotice })
  const plainRows = [...blocks.flatMap((block) => block.kind === 'diff' ? [block.text] : block.kind === 'summary' || block.kind === 'notice' ? [block.text] : [])]
  const plainText = plainRows.filter(Boolean).join('\n')
  return {
    toolName: 'apply_patch',
    summary: sanitizeToolText(summary),
    blocks,
    plainText: sanitizeToolText(plainText),
    truncated: sourceTruncated,
    sourceTruncated,
    viewCollapsed: view.viewCollapsed,
    canExpand: view.viewCollapsed,
    detail: {
      totalRecords: projection.totalRecords,
      omittedRecords: view.hiddenRecords,
      sourceTruncated,
      viewCollapsed: view.viewCollapsed,
      canExpand: view.viewCollapsed,
    },
    detailSource: {
      lines: detail.lines,
      sourceTruncated,
      viewCollapsed: view.viewCollapsed || detail.lines.length > view.lines.filter((line) => !line.continuation && (line.kind === 'context' || line.kind === 'added' || line.kind === 'removed')).length,
    },
  }
}

/**
 * Bound the detail source: characters (separator included — an empty line
 * still costs its newline) and a record-count cap, whichever hits first.
 * Any loss is upstream-of-cache and MUST surface as `clipped` so the record
 * is marked sourceTruncated instead of silently claiming completeness.
 */
function boundDetailLines(lines: readonly string[]): { lines: string[]; clipped: boolean } {
  const bounded = boundRetainedLines(
    lines,
    UiPresentationLimits.detailRecordMaxChars,
    UiPresentationLimits.detailRecordMaxRecords,
  )
  return { lines: [...bounded.lines], clipped: bounded.clipped }
}

/** Strict parse of the neutral display payload; any drift degrades. */
function parseDiffFiles(facts: Record<string, unknown> | null | undefined): readonly import('../presentation/applyPatchDiff.js').DiffFile[] | undefined {
  if (!isRecord(facts)) return undefined
  if (!Array.isArray(facts.files)) return undefined
  const files: import('../presentation/applyPatchDiff.js').DiffFile[] = []
  for (const file of facts.files) {
    if (!isRecord(file)) return undefined
    if (typeof file.path !== 'string' || typeof file.action !== 'string') return undefined
    if (file.action !== 'add' && file.action !== 'update' && file.action !== 'delete') return undefined
    if (!isPositiveIntegerOrZero(file.additions) || !isPositiveIntegerOrZero(file.deletions)) return undefined
    if (!Array.isArray(file.records)) return undefined
    // The canonical display builder caps records per file; a payload beyond
    // that cap never came from a successful executor and is drift.
    if (file.records.length > 4_000) return undefined
    const records: import('../presentation/applyPatchDiff.js').DiffRecord[] = []
    for (const record of file.records) {
      if (!isRecord(record)) return undefined
      if (record.kind !== 'context' && record.kind !== 'added' && record.kind !== 'removed') return undefined
      if (typeof record.text !== 'string') return undefined
      const oldLine: unknown = record.oldLine
      const newLine: unknown = record.newLine
      if (oldLine !== undefined && (typeof oldLine !== 'number' || !Number.isSafeInteger(oldLine) || oldLine < 1)) return undefined
      if (newLine !== undefined && (typeof newLine !== 'number' || !Number.isSafeInteger(newLine) || newLine < 1)) return undefined
      // Each kind REQUIRES its own coordinate: removed names the old file,
      // added the new file, context both. A record without its required
      // line number is structural drift and degrades to the table.
      if (record.kind === 'removed' && oldLine === undefined) return undefined
      if (record.kind === 'added' && newLine === undefined) return undefined
      if (record.kind === 'context' && (oldLine === undefined || newLine === undefined)) return undefined
      records.push({
        kind: record.kind,
        text: record.text,
        oldLine: oldLine === undefined ? undefined : (oldLine as number),
        newLine: newLine === undefined ? undefined : (newLine as number),
      })
    }
    files.push({
      path: file.path,
      action: file.action,
      additions: file.additions,
      deletions: file.deletions,
      records,
    })
  }
  return files
}

/**
 * Denials are a terminal tool state, but not a tool execution error. Keep the
 * original request visible in the same bounded, literal presentation used by
 * the approval UI without exposing protocol IDs or routing it through the
 * result JSON parser.
 */
export function presentDeniedTool(
  toolName: string,
  input: unknown,
  reason: string,
  options: { maxChars?: number; maxLines?: number } = {},
): ToolResultPresentation {
  const requestInput = isRecord(input) ? input : {}
  const presentation = buildApprovalPresentation({
    requestId: '',
    callId: '',
    toolName,
    input: requestInput,
    reason,
  })
  const instructionLabel = presentation.instructionLabel === 'Patch'
    ? 'Patch (not executed)'
    : presentation.instructionLabel
  const safeToolName = sanitizeToolText(toolName)
  const summary = `Denied: ${safeToolName}`
  const budget = new PresentationBudget(
    Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS)),
    Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES)),
  )
  const summaryPart = budget.takePlain(summary)
  const instructionPart = budget.takeField(instructionLabel, presentation.instruction)
  const reasonPart = budget.takeField('Reason', presentation.reason)
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (instructionPart.visible) {
    appendSectionSpacer(blocks, instructionPart)
    blocks.push({ kind: 'field', label: instructionLabel, text: instructionPart.text, preformatted: true, truncated: instructionPart.truncated })
    appendSectionNotice(blocks, budget, instructionPart)
  }
  if (reasonPart.visible) {
    appendSectionSpacer(blocks, reasonPart)
    blocks.push({ kind: 'field', label: 'Reason', text: reasonPart.text, preformatted: true, truncated: reasonPart.truncated })
    appendSectionNotice(blocks, budget, reasonPart)
  }
  // the detail source keeps the FULL sanitized literal request and
  // reason — the bounded field selections are only the visible summary, so
  // the fold stays recoverable through the viewer.
  const detailLines = deniedDetailLines(summaryPart.text, presentation.instruction, presentation.reason)
  const visibleRecords = summaryPart.records.length + instructionPart.records.length + reasonPart.records.length
  return makePresentation(
    safeToolName,
    summaryPart.text,
    blocks,
    budget.output(),
    false,
    detailLines,
    visibleRecords,
    budget,
  )
}

/** Denied detail: the summary plus the full literal request and reason. */
function deniedDetailLines(summary: string, instruction: string, reason: string): string[] {
  const lines = [summary]
  for (const literal of [instruction, reason]) {
    const safe = sanitizeToolText(literal)
    if (safe) lines.push(...safe.split('\n'))
  }
  return lines
}

interface PresentationOptions {
  maxChars?: number
  maxLines?: number
  input?: unknown
  reserveGlobalNotice?: boolean
  /** display facts from the canonical executor; absent for older tools. */
  display?: import('../../tools/types.js').ToolDisplayFacts
  /** Terminal content width for physical-row projections. */
  contentWidth?: number
}

export function presentToolResult(
  toolName: string,
  output: string,
  options: PresentationOptions = {},
): ToolResultPresentation {
  const first = presentToolResultOnce(toolName, output, options)
  if ((!first.sourceTruncated && !first.viewCollapsed) || options.reserveGlobalNotice) return first
  // The first pass is exact when the result fits. If it does not, rebuild
  // from the typed records with capacity reserved for a visible marker —
  // either the source-truncation notice or the view-collapsed notice. This
  // keeps blocks, detail and plainText on one ledger.
  return presentToolResultOnce(toolName, output, { ...options, reserveGlobalNotice: true })
}

/** registry instance assembled at the UI boundary. */
function registryFallback(toolName: string, output: string, options: Parameters<ToolPresenter>[1]): ToolResultPresentation {
  return fallbackPresentation(
    toolName,
    sanitizeToolText(output),
    Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS)),
    Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES)),
    options.reserveGlobalNotice === true,
  )
}

/** A typed presenter that declines (undefined) degrades to the fallback. */
function degrade(
  toolName: string,
  typed: (output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation | undefined,
): ToolPresenter {
  return (output, options) => typed(output, options) ?? registryFallback(toolName, output, options)
}

const registryChars = (options: PresentationOptions): number =>
  Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS))
const registryLines = (options: PresentationOptions): number =>
  Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES))

export const toolPresenterRegistry: ToolPresenterRegistry = new ToolPresenterRegistry(registryFallback)
toolPresenterRegistry.register('read_file', degrade('read_file', typedViaJson((parsed, options) =>
  presentReadFile(parsed, registryChars(options), registryLines(options), options.reserveGlobalNotice === true))))
toolPresenterRegistry.register('bash', degrade('bash', typedViaJson((parsed, options) =>
  presentBash(parsed, registryChars(options), registryLines(options), options.input, options.reserveGlobalNotice === true))))
toolPresenterRegistry.register('list_files', degrade('list_files', typedViaJson((parsed, options) =>
  presentListFiles(parsed, registryChars(options), registryLines(options), options.reserveGlobalNotice === true))))
toolPresenterRegistry.register('search_text', degrade('search_text', typedViaJson((parsed, options) =>
  presentSearchText(parsed, registryChars(options), registryLines(options), options.reserveGlobalNotice === true))))
toolPresenterRegistry.register('apply_patch', degrade('apply_patch', typedViaJson((parsed, options) =>
  presentApplyPatch(parsed, registryChars(options), registryLines(options), options.reserveGlobalNotice === true, options.display, options.contentWidth ?? 80))))
toolPresenterRegistry.register('update_task_plan', degrade('update_task_plan', typedViaJson((parsed, options) =>
  presentTaskPlan(parsed, registryChars(options), registryLines(options), options.reserveGlobalNotice === true))))
toolPresenterRegistry.register('git_changes', (output, options) => {
  let parsed: unknown
  try { parsed = JSON.parse(sanitizeToolText(output)) } catch { parsed = undefined }
  return presentGitChanges(
    parsed,
    registryChars(options),
    registryLines(options),
    options.reserveGlobalNotice === true,
  ) ?? fallbackPresentation(
    'git_changes',
    'Git changes result unavailable',
    registryChars(options),
    registryLines(options),
    options.reserveGlobalNotice === true,
  )
})

/** Wrap one JSON decode around a typed presenter; decode failure declines. */
function typedViaJson(
  present: (parsed: unknown, options: PresentationOptions) => ToolResultPresentation | undefined,
): (output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation | undefined {
  return (output, options) => {
    const safeOutput = sanitizeToolText(output)
    let parsed: unknown
    try {
      // This is the one and only JSON decode in the UI projection. Assistant
      // content and preformatted tool fields are never passed through it.
      parsed = safeOutput ? JSON.parse(safeOutput) : undefined
    } catch {
      return undefined
    }
    return present(parsed, options)
  }
}

function presentToolResultOnce(
  toolName: string,
  output: string,
  options: PresentationOptions,
): ToolResultPresentation {
  const maxChars = Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS))
  const maxLines = Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES))
  const safeOutput = sanitizeToolText(output)
  return toolPresenterRegistry.present(toolName, safeOutput, { ...options, maxChars, maxLines })
}

export function sanitizeToolText(value: string): string {
  return value
    .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '')
    .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '')
    .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '')
    .replace(/\u001b/gu, '')
    .split('')
    .map((character, index, characters) => {
      const code = character.charCodeAt(0)
      if (code >= 0xD800 && code <= 0xDBFF && characters[index + 1]?.charCodeAt(0) >= 0xDC00 && characters[index + 1]?.charCodeAt(0) <= 0xDFFF) return character
      if (code >= 0xDC00 && code <= 0xDFFF && characters[index - 1]?.charCodeAt(0) >= 0xD800 && characters[index - 1]?.charCodeAt(0) <= 0xDBFF) return character
      return code >= 0xD800 && code <= 0xDFFF ? '�' : character
    })
    .join('')
    .replace(/\r\n?/g, '\n')
}

export function toolPresentationText(presentation: ToolResultPresentation): string {
  return presentation.plainText
}

/** The table renderer and character ledger must measure the same row. */
export function serializeToolTableRow(row: readonly string[]): string {
  return row.map((cell) => cell.replace(/\r\n?|\n/gu, '\\n')).join(' ')
}

/**
 * The canonical read_file page is strictly parsed before presentation. The
 * compat `truncated` flag must still equal `!complete`; a page with more
 * lines after it (`complete:false`) is a continuation, never an irrecoverable
 * source truncation, and the empty file's only legal form is 1..0 with
 * totalLines 0.
 */
export interface ContinuationRangeProjection {
  startLine: number
  endLine: number
}

export interface PagePlanProjection {
  suggestedContinuationRanges: readonly ContinuationRangeProjection[]
  pagePlanComplete: boolean
  nextPlanStartLine?: number
}

type ReadFilePageProjection =
  | {
    complete: true
    path: string
    startLine: number
    endLine: number
    totalLines: number
    content: string
  }
  | {
    complete: false
    path: string
    startLine: number
    endLine: number
    totalLines: number
    content: string
    nextStartLine: number
    plan?: PagePlanProjection
  }

function parseReadFilePage(value: unknown): ReadFilePageProjection | undefined {
  if (!isRecord(value)) return undefined
  if (!hasExactKeys(
    value,
    ['path', 'startLine', 'endLine', 'totalLines', 'content', 'complete', 'truncated'],
    ['nextStartLine', 'suggestedContinuationRanges', 'pagePlanComplete', 'nextPlanStartLine'],
  )) return undefined
  if (typeof value.path !== 'string' || !value.path) return undefined
  if (!isPositiveInteger(value.startLine)) return undefined
  if (!isNonNegativeInteger(value.endLine) || !isNonNegativeInteger(value.totalLines)) return undefined
  if (typeof value.content !== 'string' || typeof value.complete !== 'boolean' || typeof value.truncated !== 'boolean') return undefined
  if (value.truncated === value.complete) return undefined
  if (value.totalLines === 0) {
    if (value.startLine !== 1 || value.endLine !== 0 || !value.complete || value.content !== '') return undefined
    return { complete: true, path: value.path, startLine: 1, endLine: 0, totalLines: 0, content: '' }
  }
  if (value.startLine > value.endLine || value.endLine > value.totalLines) return undefined
  if (value.complete) {
    // a page that already reached EOF never carries plan fields, not
    // even an empty array.
    if ('nextStartLine' in value || 'suggestedContinuationRanges' in value || 'pagePlanComplete' in value || 'nextPlanStartLine' in value) return undefined
    if (value.endLine !== value.totalLines) return undefined
    return { complete: true, path: value.path, startLine: value.startLine, endLine: value.endLine, totalLines: value.totalLines, content: value.content }
  }
  if (value.endLine >= value.totalLines) return undefined
  if (!isPositiveInteger(value.nextStartLine) || value.nextStartLine !== value.endLine + 1) return undefined
  const plan = parsePagePlan(value, value.nextStartLine, value.totalLines)
  // The plan fields are all-or-nothing: a partially present plan is malformed
  // and must fail closed instead of being silently stripped from the page.
  if (plan === undefined && ('suggestedContinuationRanges' in value || 'pagePlanComplete' in value || 'nextPlanStartLine' in value)) {
    return undefined
  }
  return {
    complete: false,
    path: value.path,
    startLine: value.startLine,
    endLine: value.endLine,
    totalLines: value.totalLines,
    content: value.content,
    nextStartLine: value.nextStartLine,
    ...(plan === undefined ? {} : { plan }),
  }
}

/**
 * page-plan metadata. The three fields are all-or-nothing: any one of
 * them present requires all three parts. Ranges contain only unread
 * follow-up pages — the current page is never repeated — start exactly at the
 * page's nextStartLine, are contiguous without overlap or holes, stay within
 * totalLines, and never form an empty array. pagePlanComplete:true requires
 * the last range to reach totalLines and forbids nextPlanStartLine; false
 * requires nextPlanStartLine to be lastEnd + 1.
 */
function parsePagePlan(
  value: Record<string, unknown>,
  nextStartLine: number,
  totalLines: number,
): PagePlanProjection | undefined {
  const hasRanges = 'suggestedContinuationRanges' in value
  const hasComplete = 'pagePlanComplete' in value
  const hasNextPlan = 'nextPlanStartLine' in value
  if (!hasRanges && !hasComplete && !hasNextPlan) return undefined
  if (!hasRanges || !hasComplete) return undefined
  const ranges = value.suggestedContinuationRanges
  if (!Array.isArray(ranges) || ranges.length === 0 || ranges.length > MAX_SUGGESTED_CONTINUATION_RANGES) return undefined
  let expectedStart = nextStartLine
  const parsed: ContinuationRangeProjection[] = []
  for (const item of ranges) {
    if (!isRecord(item) || !hasExactKeys(item, ['startLine', 'endLine'])) return undefined
    if (!isPositiveInteger(item.startLine) || !isPositiveInteger(item.endLine)) return undefined
    if (item.startLine !== expectedStart) return undefined
    if (item.endLine < item.startLine || item.endLine > totalLines) return undefined
    parsed.push({ startLine: item.startLine, endLine: item.endLine })
    expectedStart = item.endLine + 1
  }
  if (typeof value.pagePlanComplete !== 'boolean') return undefined
  const lastEnd = parsed.at(-1)!.endLine
  if (value.pagePlanComplete) {
    if (hasNextPlan || lastEnd !== totalLines) return undefined
    return { suggestedContinuationRanges: parsed, pagePlanComplete: true }
  }
  if (!hasNextPlan || !isPositiveInteger(value.nextPlanStartLine)) return undefined
  // An incomplete plan must still have unread lines ahead: a range that
  // already covers EOF, or a next plan start past totalLines, is
  // self-contradictory.
  if (lastEnd >= totalLines) return undefined
  if (value.nextPlanStartLine > totalLines) return undefined
  if (value.nextPlanStartLine !== lastEnd + 1) return undefined
  return { suggestedContinuationRanges: parsed, pagePlanComplete: false, nextPlanStartLine: value.nextPlanStartLine }
}

function presentReadFile(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
  const page = parseReadFilePage(value)
  if (!page) return undefined
  const path = sanitizeToolText(page.path)
  const content = sanitizeToolText(page.content)
  const summary = page.totalLines === 0
    ? `${path} (empty file)`
    : page.complete
      ? `${path}:${page.startLine}-${page.endLine} of ${page.totalLines} lines (EOF)`
      : `${path}:${page.startLine}-${page.endLine} of ${page.totalLines} lines (continue at ${page.nextStartLine})`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  // A page's final newline is a terminator, not a rendered line: dropping it
  // from the projection keeps the Ink render equal to the ledger (an empty
  // trailing Text renders zero-height). The canonical content is untouched.
  const displayContent = content.endsWith('\n') ? content.slice(0, -1) : content
  const contentPart = displayContent ? budget.takeField('content', displayContent) : undefined
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (contentPart?.visible) {
    appendSectionSpacer(blocks, contentPart)
    blocks.push({ kind: 'field', label: 'content', text: contentPart.text, preformatted: true, truncated: contentPart.truncated })
    appendSectionNotice(blocks, budget, contentPart)
  }
  // the plan is advisory metadata about unread pages, never future
  // source rendered as read. It gets one bounded plain line; anything that
  // does not fit is clipped by the shared presentation budget.
  if (!page.complete && page.plan) {
    const ranges = page.plan.suggestedContinuationRanges
      .map((range) => `${range.startLine}–${range.endLine}`)
      .join(', ')
    const suffix = page.plan.pagePlanComplete ? ' (to EOF)' : ''
    const planPart = budget.takePlain(`Suggested next ranges: ${ranges}${suffix}`)
    if (planPart.visible) {
      appendSectionSpacer(blocks, planPart)
      blocks.push({ kind: 'summary', text: planPart.text })
    }
  }
  // complete:false is a canonical continuation, not source truncation. Only
  // the UI's own view budget may mark this projection truncated, and no
  // projection field is ever written back into the canonical ToolResult.
  const detailLines = displayContent ? displayContent.split('\n') : []
  const visibleRecords = contentPart?.records.length ?? 0
  return makePresentation('read_file', summaryPart.text, blocks, budget.output(), false, detailLines, visibleRecords, budget)
}

function presentBash(value: unknown, maxChars: number, maxLines: number, input: unknown, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
  if (!isRecord(value) || !hasExactKeys(value, ['stdout', 'stderr', 'exitCode', 'timedOut', 'interrupted', 'truncated'])) return undefined
  if (
    typeof value.stdout !== 'string' || typeof value.stderr !== 'string' ||
    !Number.isInteger(value.exitCode) || typeof value.timedOut !== 'boolean' ||
    typeof value.interrupted !== 'boolean' || typeof value.truncated !== 'boolean'
  ) return undefined
  const stdoutText = sanitizeToolText(value.stdout)
  const stderrText = sanitizeToolText(value.stderr)
  const flags = [
    `exit ${value.exitCode}`,
    value.timedOut ? 'timeout' : '',
    value.interrupted ? 'interrupted' : '',
    value.truncated ? 'truncated' : '',
  ].filter(Boolean).join(', ')
  const summary = `bash (${flags})`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  let command: BoundedSelection | undefined
  if (isRecord(input) && typeof input.command === 'string' && input.command.trim()) {
    command = budget.takeField('command', sanitizeToolText(input.command))
    if (command.visible) {
      appendSectionSpacer(blocks, command)
      blocks.push({ kind: 'field', label: 'command', text: command.text, preformatted: true, truncated: command.truncated })
      appendSectionNotice(blocks, budget, command)
    }
  }
  const stdout = stdoutText ? budget.takeField('stdout', stdoutText) : undefined
  const stderr = stderrText ? budget.takeField('stderr', stderrText) : undefined
  if (stdout?.visible) {
    appendSectionSpacer(blocks, stdout)
    blocks.push({ kind: 'field', label: 'stdout', text: stdout.text, preformatted: true, truncated: stdout.truncated })
    appendSectionNotice(blocks, budget, stdout)
  }
  if (stderr?.visible) {
    appendSectionSpacer(blocks, stderr)
    blocks.push({ kind: 'field', label: 'stderr', text: stderr.text, preformatted: true, truncated: stderr.truncated })
    appendSectionNotice(blocks, budget, stderr)
  }
  const detailLines: string[] = [
    ...(isRecord(input) && typeof input.command === 'string' && input.command.trim() ? [`$ ${sanitizeToolText(input.command)}`] : []),
    ...(stdoutText ? stdoutText.split('\n') : []),
    ...(stderrText ? stderrText.split('\n') : []),
  ]
  const visibleRecords = (command?.records.length ?? 0) + (stdout?.records.length ?? 0) + (stderr?.records.length ?? 0)
  return makePresentation(
    'bash', summaryPart.text, blocks,
    budget.output(), value.truncated === true,
    detailLines,
    visibleRecords,
    budget,
  )
}

function presentListFiles(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
  if (!isRecord(value) || !hasExactKeys(value, ['entries', 'truncated']) || !Array.isArray(value.entries) || typeof value.truncated !== 'boolean') return undefined
  const entries: string[] = []
  for (const item of value.entries) {
    if (!isRecord(item) || typeof item.path !== 'string' || item.type !== 'file') return undefined
    entries.push(sanitizeToolText(item.path))
  }
  const summary = `${entries.length} file${entries.length === 1 ? '' : 's'}${value.truncated ? ' (truncated)' : ''}`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const bounded = budget.takeField('files', entries.join('\n'))
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (bounded.visible) {
    appendSectionSpacer(blocks, bounded)
    blocks.push({ kind: 'list', label: 'files', items: bounded.records, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('list_files', summaryPart.text, blocks,
    budget.output(), value.truncated === true, entries, bounded.records.length, budget)
}

function presentSearchText(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
  if (!isRecord(value) || !hasExactKeys(value, ['matches', 'truncated']) || !Array.isArray(value.matches) || typeof value.truncated !== 'boolean') return undefined
  const matches: string[] = []
  for (const item of value.matches) {
    if (
      !isRecord(item) || typeof item.path !== 'string' || !isPositiveInteger(item.line) ||
      !isPositiveInteger(item.column) || typeof item.text !== 'string'
    ) return undefined
    matches.push(`${sanitizeToolText(item.path)}:${item.line}:${item.column} ${sanitizeToolText(item.text)}`)
  }
  const summary = `${matches.length} match${matches.length === 1 ? '' : 'es'}${value.truncated ? ' (truncated)' : ''}`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const bounded = budget.takeField('matches', matches.join('\n'))
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (bounded.visible) {
    appendSectionSpacer(blocks, bounded)
    blocks.push({ kind: 'list', label: 'matches', items: bounded.records, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('search_text', summaryPart.text, blocks,
    budget.output(), value.truncated === true, matches, bounded.records.length, budget)
}

function presentGitChanges(
  value: unknown,
  maxChars: number,
  maxLines: number,
  reserveGlobalNotice: boolean,
): ToolResultPresentation | undefined {
  if (!isValidGitChangesEnvelope(value)) return undefined
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const detailLines: string[] = []
  let summary: string

  if (!value.applicable) {
    if (value.reason !== 'not_git_repository' || value.complete !== true) return undefined
    summary = 'Git changes unavailable · not a Git repository'
  } else if (value.mode === 'summary') {
    if (!isValidGitSummary(value.summary)) return undefined
    summary = `${value.summary.uniqueFiles} changed file${value.summary.uniqueFiles === 1 ? '' : 's'} · ${value.summary.layerRecords} layer record${value.summary.layerRecords === 1 ? '' : 's'}`
    if (isRecord(value.summary.byLayer)) {
      detailLines.push(`staged ${String(value.summary.byLayer.staged ?? '?')} · unstaged ${String(value.summary.byLayer.unstaged ?? '?')} · unmerged ${String(value.summary.byLayer.unmerged ?? '?')}`)
    }
  } else if (value.mode === 'files') {
    if (!Array.isArray(value.items)) return undefined
    for (const item of value.items) {
      if (!isValidGitFileRecord(item)) return undefined
      const rename = typeof item.oldPath === 'string' ? `${sanitizeToolText(item.oldPath)} → ` : ''
      detailLines.push(`${sanitizeToolText(item.layer)} ${sanitizeToolText(item.action)} ${rename}${sanitizeToolText(item.path)}`)
    }
    summary = `${value.items.length} Git change${value.items.length === 1 ? '' : 's'}${value.complete ? '' : ' · more available'}`
  } else if (value.mode === 'patch') {
    if (!Array.isArray(value.patches)) return undefined
    let hunkCount = 0
    for (const patch of value.patches) {
      if (!isValidGitPatchLayer(patch)) return undefined
      detailLines.push(`${sanitizeToolText(patch.layer)} ${sanitizeToolText(String(patch.file.action ?? 'changed'))} ${sanitizeToolText(patch.file.path)}`)
      for (const hunk of patch.hunks) {
        hunkCount += 1
        detailLines.push(`@@ -${String(hunk.oldStart ?? '?')} +${String(hunk.newStart ?? '?')} @@`)
        for (const line of hunk.lines) {
          const marker = line.kind === 'add' ? '+' : line.kind === 'delete' ? '-' : line.kind === 'context' ? ' ' : '\\'
          detailLines.push(`${marker}${sanitizeToolText(typeof line.text === 'string' ? line.text : ' No newline at end of file')}`)
        }
      }
    }
    summary = `${value.patches.length} Git patch layer${value.patches.length === 1 ? '' : 's'} · ${hunkCount} hunk${hunkCount === 1 ? '' : 's'}${value.complete ? '' : ' · more available'}`
  } else return undefined

  if (isRecord(value.coverage) && value.coverage.submoduleWorktrees === 'not_observed') {
    detailLines.push(`submodule worktrees not observed · ${String(value.coverage.gitlinkCount ?? '?')} gitlink${value.coverage.gitlinkCount === 1 ? '' : 's'}`)
  }
  if (typeof value.guidance === 'string') detailLines.push(sanitizeToolText(value.guidance))

  const summaryPart = budget.takePlain(summary)
  const detailPart = detailLines.length > 0 ? budget.takeField('changes', detailLines.join('\n')) : undefined
  const blocks: ToolDisplayBlock[] = summaryPart.visible ? [{ kind: 'summary', text: summaryPart.text }] : []
  if (detailPart?.visible) {
    appendSectionSpacer(blocks, detailPart)
    blocks.push({ kind: 'list', label: 'changes', items: detailPart.records, truncated: detailPart.truncated })
    appendSectionNotice(blocks, budget, detailPart)
  }
  return makePresentation(
    'git_changes', summaryPart.text, blocks, budget.output(), false,
    detailLines, detailPart?.records.length ?? 0, budget,
  )
}

interface GitPresentationEnvelope extends Record<string, unknown> {
  version: 1
  applicable: boolean
  mode: 'summary' | 'files' | 'patch'
  scope: 'staged' | 'unstaged' | 'both'
  complete: boolean
  reason?: 'not_git_repository'
  revision?: string
  path?: string
  offset?: number
  nextCursor?: string
  guidance?: string
  coverage?: GitPresentationCoverage
  summary?: GitPresentationSummary
  items?: GitPresentationFile[]
  patches?: GitPresentationPatch[]
}

interface GitPresentationCoverage {
  parentRepositoryChanges: 'complete'
  submoduleWorktrees: 'not_present' | 'not_observed'
  gitlinkCount: number
}

interface GitPresentationSummary {
  uniqueFiles: number
  layerRecords: number
  byLayer: { staged: number; unstaged: number; unmerged: number }
  byAction: Record<string, number>
  recordsWithBinary: number
  recordsWithModeChange: number
  recordsWithTypeChange: number
  recordsUnmerged: number
}

interface GitPresentationFile extends Record<string, unknown> {
  layer: 'staged' | 'unstaged' | 'unmerged'
  action: 'added' | 'modified' | 'deleted' | 'renamed' | 'untracked' | 'type_changed' | 'unmerged'
  path: string
  oldPath?: string
  contentChanged: boolean | 'unknown'
  binary: boolean | 'unknown'
  modeChanged: boolean
  objectType: 'file' | 'symlink' | 'gitlink' | 'other'
}

interface GitPresentationPatch extends Record<string, unknown> {
  layer: 'staged' | 'unstaged'
  file: GitPresentationFile
  hunks: GitPresentationHunk[]
}

interface GitPresentationHunk extends Record<string, unknown> {
  oldStart: number
  oldLines: number
  newStart: number
  newLines: number
  section?: string
  continuedFromPrevious?: boolean
  continuesNext?: boolean
  lines: GitPresentationLine[]
}

interface GitPresentationLine extends Record<string, unknown> {
  kind: 'context' | 'add' | 'delete' | 'no_newline'
  text?: string
  oldLine?: number | null
  newLine?: number | null
  appliesTo?: 'old' | 'new'
}

function isValidGitChangesEnvelope(value: unknown): value is GitPresentationEnvelope {
  if (
    !isRecord(value) || value.version !== 1 || typeof value.applicable !== 'boolean' ||
    !isOneOf(value.mode, ['summary', 'files', 'patch']) ||
    !isOneOf(value.scope, ['staged', 'unstaged', 'both']) || typeof value.complete !== 'boolean'
  ) return false
  if (value.applicable === false) {
    return value.reason === 'not_git_repository' && value.complete === true
  }
  if (
    typeof value.revision !== 'string' || !isValidGitCoverage(value.coverage) ||
    (value.guidance !== undefined && typeof value.guidance !== 'string') ||
    (value.path !== undefined && typeof value.path !== 'string') ||
    (value.nextCursor !== undefined && typeof value.nextCursor !== 'string')
  ) return false
  if (value.mode === 'summary') return value.complete === true && isValidGitSummary(value.summary)
  if (!isNonNegativeInteger(value.offset)) return false
  if (value.complete === false && typeof value.nextCursor !== 'string') return false
  if (value.complete === true && value.nextCursor !== undefined) return false
  if (value.mode === 'files') return Array.isArray(value.items) && value.items.every(isValidGitFileRecord)
  return typeof value.path === 'string' && Array.isArray(value.patches) && value.patches.every(isValidGitPatchLayer)
}

function isValidGitCoverage(value: unknown): value is GitPresentationCoverage {
  return isRecord(value) && value.parentRepositoryChanges === 'complete' &&
    isOneOf(value.submoduleWorktrees, ['not_present', 'not_observed']) &&
    isNonNegativeInteger(value.gitlinkCount)
}

function isValidGitSummary(value: unknown): value is GitPresentationSummary {
  if (!isRecord(value) || !isNonNegativeInteger(value.uniqueFiles) || !isNonNegativeInteger(value.layerRecords)) return false
  if (
    !isRecord(value.byLayer) || !hasNonNegativeIntegerKeys(value.byLayer, ['staged', 'unstaged', 'unmerged']) ||
    !isRecord(value.byAction) || !hasNonNegativeIntegerKeys(value.byAction, [
      'added', 'modified', 'deleted', 'renamed', 'untracked', 'type_changed', 'unmerged',
    ])
  ) return false
  return [
    value.recordsWithBinary,
    value.recordsWithModeChange,
    value.recordsWithTypeChange,
    value.recordsUnmerged,
  ].every(isNonNegativeInteger)
}

function isValidGitFileRecord(value: unknown): value is GitPresentationFile {
  if (
    !isRecord(value) || !isOneOf(value.layer, ['staged', 'unstaged', 'unmerged']) ||
    !isOneOf(value.action, ['added', 'modified', 'deleted', 'renamed', 'untracked', 'type_changed', 'unmerged']) ||
    typeof value.path !== 'string' || (value.oldPath !== undefined && typeof value.oldPath !== 'string') ||
    !isBooleanOrUnknown(value.contentChanged) || !isBooleanOrUnknown(value.binary) ||
    typeof value.modeChanged !== 'boolean' || !isOneOf(value.objectType, ['file', 'symlink', 'gitlink', 'other']) ||
    (value.oldMode !== undefined && typeof value.oldMode !== 'string') ||
    (value.newMode !== undefined && typeof value.newMode !== 'string') ||
    (value.boundaryTruncated !== undefined && typeof value.boundaryTruncated !== 'boolean')
  ) return false
  if (value.conflictStages !== undefined) {
    if (!Array.isArray(value.conflictStages) || value.conflictStages.length === 0) return false
    const stages = value.conflictStages
    if (!stages.every((stage) => stage === 1 || stage === 2 || stage === 3)) return false
    if (new Set(stages).size !== stages.length) return false
  }
  return true
}

function isValidGitPatchLayer(value: unknown): value is GitPresentationPatch {
  return isRecord(value) && isOneOf(value.layer, ['staged', 'unstaged']) &&
    isValidGitFileRecord(value.file) && Array.isArray(value.hunks) && value.hunks.every(isValidGitHunk)
}

function isValidGitHunk(value: unknown): value is GitPresentationHunk {
  return isRecord(value) &&
    isNonNegativeInteger(value.oldStart) && isNonNegativeInteger(value.oldLines) &&
    isNonNegativeInteger(value.newStart) && isNonNegativeInteger(value.newLines) &&
    (value.section === undefined || typeof value.section === 'string') &&
    (value.continuedFromPrevious === undefined || typeof value.continuedFromPrevious === 'boolean') &&
    (value.continuesNext === undefined || typeof value.continuesNext === 'boolean') &&
    Array.isArray(value.lines) && value.lines.every(isValidGitLine)
}

function isValidGitLine(value: unknown): value is GitPresentationLine {
  if (!isRecord(value) || !isOneOf(value.kind, ['context', 'add', 'delete', 'no_newline'])) return false
  if (value.kind === 'no_newline') return isOneOf(value.appliesTo, ['old', 'new'])
  if (typeof value.text !== 'string') return false
  return isNullableLineNumber(value.oldLine) && isNullableLineNumber(value.newLine)
}

function isNullableLineNumber(value: unknown): boolean {
  return value === null || (Number.isSafeInteger(value) && (value as number) >= 0)
}

function isBooleanOrUnknown(value: unknown): boolean {
  return typeof value === 'boolean' || value === 'unknown'
}

function hasNonNegativeIntegerKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return hasExactKeys(value, keys) && keys.every((key) => isNonNegativeInteger(value[key]))
}

function isOneOf<const T extends readonly string[]>(value: unknown, accepted: T): value is T[number] {
  return typeof value === 'string' && accepted.includes(value)
}

function presentTaskPlan(
  value: unknown,
  maxChars: number,
  maxLines: number,
  reserveGlobalNotice: boolean,
): ToolResultPresentation | undefined {
  if (!isRecord(value) || value.ok !== true || typeof value.operation !== 'string') return undefined
  const operation = value.operation
  const historical = operation === 'read_snapshot' && isRecord(value.snapshot)
    ? value.snapshot
    : undefined
  const planValue = historical?.plan ?? value.plan
  const plan = parseTaskPlanPresentation(planValue)
  if (planValue !== undefined && planValue !== null && !plan) return undefined

  let summary: string
  if (historical) {
    const terminal = isRecord(historical.terminal) && typeof historical.terminal.type === 'string'
      ? historical.terminal.type
      : 'unknown'
    summary = `Historical run · ${terminal}`
  } else if (operation === 'list_snapshots') {
    const count = Array.isArray(value.snapshots) ? value.snapshots.length : 0
    summary = `${count} historical run snapshot${count === 1 ? '' : 's'}`
  } else if (operation === 'claim_run_complete') {
    summary = `Run completion declared · plan revision ${plan?.revision ?? '?' } · unverified`
  } else if (plan) {
    summary = `Task plan · revision ${plan.revision}`
  } else {
    summary = 'Task plan operation completed'
  }

  const items = plan?.steps.map((step) => `${taskStepSymbol(step.status)} ${sanitizeToolText(step.content)}`) ?? []
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const listPart = items.length > 0 ? budget.takeField('tasks', items.join('\n')) : undefined
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (listPart?.visible) {
    appendSectionSpacer(blocks, listPart)
    blocks.push({ kind: 'list', label: 'tasks', items: listPart.records, truncated: listPart.truncated })
    appendSectionNotice(blocks, budget, listPart)
  }
  return makePresentation(
    'update_task_plan',
    summaryPart.text,
    blocks,
    budget.output(),
    false,
    items,
    listPart?.records.length ?? 0,
    budget,
  )
}

function parseTaskPlanPresentation(value: unknown): {
  readonly revision: number
  readonly steps: readonly { readonly content: string; readonly status: 'pending' | 'in_progress' | 'completed' }[]
} | undefined {
  if (!isRecord(value) || !isPositiveInteger(value.revision) || !Array.isArray(value.steps)) return undefined
  const steps: Array<{ content: string; status: 'pending' | 'in_progress' | 'completed' }> = []
  for (const step of value.steps) {
    if (!isRecord(step) || typeof step.content !== 'string') return undefined
    if (step.status !== 'pending' && step.status !== 'in_progress' && step.status !== 'completed') return undefined
    steps.push({ content: step.content, status: step.status })
  }
  return { revision: value.revision, steps }
}

function taskStepSymbol(status: 'pending' | 'in_progress' | 'completed'): '[ ]' | '[>]' | '[✓]' {
  if (status === 'pending') return '[ ]'
  if (status === 'in_progress') return '[>]'
  return '[✓]'
}

function presentApplyPatch(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean, display: import('../../tools/types.js').ToolDisplayFacts | undefined, contentWidth: number): ToolResultPresentation | undefined {
  if (display?.kind === 'apply_patch_diff') {
    const projected = presentApplyPatchDiff(display, value, maxChars, maxLines, reserveGlobalNotice, contentWidth)
    if (projected !== undefined) return projected
  }
  if (!isRecord(value) || !hasExactKeys(value, ['applied', 'truncated']) || !Array.isArray(value.applied) || typeof value.truncated !== 'boolean') return undefined
  const rows: string[][] = []
  for (const item of value.applied) {
    if (
      !isRecord(item) || typeof item.path !== 'string' || typeof item.action !== 'string' ||
      !isPositiveIntegerOrZero(item.additions) || !isPositiveIntegerOrZero(item.deletions)
    ) return undefined
    rows.push([
      sanitizeToolText(item.action),
      sanitizeToolText(item.path),
      `+${item.additions}/-${item.deletions}`,
    ])
  }
  const summary = `${rows.length} file${rows.length === 1 ? '' : 's'} changed${value.truncated ? ' (truncated)' : ''}`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const bounded = budget.takeTable('changes', rows)
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  if (bounded.visible) {
    appendSectionSpacer(blocks, bounded)
    blocks.push({ kind: 'table', label: 'changes', rows: bounded.rows, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('apply_patch', summaryPart.text, blocks,
    budget.output(), value.truncated === true, rows.map((row) => row.join(' ')), bounded.rows.length, budget)
}

function fallbackPresentation(toolName: string, output: string, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation {
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const candidate = output || 'completed'
  const summaryPart = budget.takePlain(candidate)
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  return makePresentation(toolName, summaryPart.text, blocks, budget.output(), summaryPart.truncated, candidate.split('\n'), summaryPart.records.length, budget)
}

/**
 * Assemble the presentation and split the two truncation states. The summary
 * budget clipping records that the detail source still retains is
 * `viewCollapsed` (recoverable via the viewer); only upstream loss (canonical
 * flags or the detail bound's own clip) is `sourceTruncated`.
 */
function makePresentation(
  toolName: string,
  summary: string,
  blocks: ToolDisplayBlock[],
  plainText: string,
  upstreamTruncated: boolean,
  detailLines: readonly string[],
  visibleRecords: number,
  budget?: PresentationBudget,
): ToolResultPresentation {
  const safePlainText = sanitizeToolText(plainText)
  const detail = boundDetailLines(detailLines)
  const visible = Math.max(0, Math.min(visibleRecords, detail.lines.length))
  const sourceTruncated = upstreamTruncated || detail.clipped
  const viewCollapsed = detail.lines.length > visible
  const omittedRecords = Math.max(0, detail.lines.length - visible)
  const renderedBlocks = [...blocks]
  if (sourceTruncated && budget) {
    const notice = budget.takeNotice(GLOBAL_TRUNCATION_NOTICE)
    if (notice.visible) renderedBlocks.push({ kind: 'notice', text: notice.text })
  }
  if (viewCollapsed) {
    const notice = budget?.takeNotice(VIEW_COLLAPSED_NOTICE(omittedRecords))
    if (notice?.visible) renderedBlocks.push({ kind: 'notice', text: notice.text })
  }
  const renderedPlainText = budget?.output() ?? safePlainText
  return {
    toolName,
    summary: sanitizeToolText(summary),
    blocks: renderedBlocks,
    plainText: sanitizeToolText(renderedPlainText),
    truncated: sourceTruncated,
    sourceTruncated,
    viewCollapsed,
    canExpand: viewCollapsed,
    detail: {
      totalRecords: detail.lines.length,
      omittedRecords,
      sourceTruncated,
      viewCollapsed,
      canExpand: viewCollapsed,
    },
    detailSource: {
      lines: detail.lines,
      sourceTruncated,
      viewCollapsed,
    },
  }
}

interface BoundedSelection {
  text: string
  records: string[]
  totalRecords: number
  omittedRecords: number
  truncated: boolean
  visible: boolean
  separator: boolean
}

interface BoundedTableSelection extends BoundedSelection {
  rows: string[][]
}

class PresentationBudget {
  private readonly outputRows: string[] = []
  private usedLines = 0
  private usedChars = 0
  private noticeWritten = false
  readonly records: string[] = []
  totalRecords = 0
  omittedRecords = 0
  truncated = false

  constructor(
    private readonly maxChars: number,
    private readonly maxLines: number,
    private readonly reserveGlobalNotice = false,
  ) {}

  takePlain(value: string): BoundedSelection {
    const allRecords = value ? value.split('\n') : []
    const selected = this.selectRows(allRecords, true)
    const truncated = selected.truncated || selected.records.length < allRecords.length
    this.truncated ||= truncated
    return {
      text: selected.records.join('\n'),
      records: selected.records,
      totalRecords: allRecords.length,
      omittedRecords: Math.max(0, allRecords.length - selected.records.length),
      truncated,
      visible: selected.records.length > 0,
      separator: false,
    }
  }

  takeField(label: string, value: string): BoundedSelection {
    const allRecords = value ? value.split('\n') : []
    if (allRecords.length === 0) {
      return { text: '', records: [], totalRecords: 0, omittedRecords: 0, truncated: false, visible: false, separator: false }
    }
    const snapshot = this.snapshot()
    const separator = this.outputRows.length > 0
    if (separator && !this.appendExact('')) return this.omittedField(allRecords, false)
    if (!this.appendExact(`${label}:`)) {
      this.restore(snapshot)
      return this.omittedField(allRecords, true)
    }
    let selected = this.selectRows(allRecords, true)
    if (selected.truncated && this.reserveGlobalNotice && !this.noticeWritten) {
      this.restore(snapshot)
      if (separator && !this.appendExact('')) return this.omittedField(allRecords, false)
      if (!this.appendExact(`${label}:`)) return this.omittedField(allRecords, true)
      selected = this.selectRows(allRecords, true, { lines: 1, chars: SECTION_TRUNCATION_NOTICE.length + 1 })
    }
    if (selected.records.length === 0) {
      this.restore(snapshot)
      return this.omittedField(allRecords, true)
    }
    const omittedRecords = Math.max(0, allRecords.length - selected.records.length)
    const truncated = selected.truncated || omittedRecords > 0
    this.totalRecords += allRecords.length
    this.omittedRecords += omittedRecords
    this.records.push(...selected.records)
    this.truncated ||= truncated
    return {
      text: selected.records.join('\n'),
      records: selected.records,
      totalRecords: allRecords.length,
      omittedRecords,
      truncated,
      visible: true,
      separator,
    }
  }

  takeTable(label: string, rows: string[][]): BoundedTableSelection {
    if (rows.length === 0) {
      return { text: '', records: [], rows: [], totalRecords: 0, omittedRecords: 0, truncated: false, visible: false, separator: false }
    }
    const snapshot = this.snapshot()
    const separator = this.outputRows.length > 0
    if (separator && !this.appendExact('')) return this.omittedTable(rows, false)
    if (!this.appendExact(`${label}:`)) {
      this.restore(snapshot)
      return this.omittedTable(rows, true)
    }
    let tableSelection = this.selectTableRows(rows)
    if (tableSelection.truncated && this.reserveGlobalNotice && !this.noticeWritten) {
      this.restore(snapshot)
      if (separator && !this.appendExact('')) return this.omittedTable(rows, false)
      if (!this.appendExact(`${label}:`)) return this.omittedTable(rows, true)
      tableSelection = this.selectTableRows(rows, { lines: 1, chars: SECTION_TRUNCATION_NOTICE.length + 1 })
    }
    const { selectedRows, selectedRecords } = tableSelection
    let { truncated } = tableSelection
    if (selectedRows.length === 0) {
      this.restore(snapshot)
      return this.omittedTable(rows, true)
    }
    const omittedRecords = Math.max(0, rows.length - selectedRows.length)
    truncated ||= omittedRecords > 0
    this.totalRecords += rows.length
    this.omittedRecords += omittedRecords
    this.records.push(...selectedRecords)
    this.truncated ||= truncated
    return {
      text: selectedRecords.join('\n'),
      records: selectedRecords,
      rows: selectedRows,
      totalRecords: rows.length,
      omittedRecords,
      truncated,
      visible: true,
      separator,
    }
  }

  takeNotice(text: string): BoundedSelection {
    this.noticeWritten = true
    return this.takePlain(text)
  }

  takeSectionNotice(text: string): BoundedSelection {
    return this.takePlain(text)
  }

  output(): string {
    return this.outputRows.join('\n')
  }

  private selectRows(
    records: readonly string[],
    allowPartial: boolean,
    reserve: { lines: number; chars: number } = { lines: 0, chars: 0 },
  ): { records: string[]; truncated: boolean } {
    const selected: string[] = []
    let truncated = false
    for (const record of records) {
      if (this.remainingLines() <= reserve.lines) {
        truncated = true
        break
      }
      const available = this.remainingChars() - reserve.chars
      if (available < record.length) {
        if (allowPartial && available > 0) {
          const clipped = truncateCharacters(record, available)
          this.appendExact(clipped)
          selected.push(clipped)
        }
        truncated = true
        break
      }
      this.appendExact(record)
      selected.push(record)
    }
    return { records: selected, truncated }
  }

  private selectTableRows(
    rows: readonly string[][],
    reserve: { lines: number; chars: number } = { lines: 0, chars: 0 },
  ): { selectedRows: string[][]; selectedRecords: string[]; truncated: boolean } {
    const selectedRows: string[][] = []
    const selectedRecords: string[] = []
    let truncated = false
    for (const row of rows) {
      const record = serializeToolTableRow(row)
      // Whole table rows are retained as typed rows. Do not split a row into
      // whitespace-delimited strings when the character budget is reached.
      if (this.remainingLines() <= reserve.lines || this.remainingChars() - reserve.chars < record.length) {
        truncated = true
        break
      }
      this.appendExact(record)
      selectedRows.push(row)
      selectedRecords.push(record)
    }
    return { selectedRows, selectedRecords, truncated }
  }

  private appendExact(value: string): boolean {
    if (this.remainingLines() <= 0 || this.remainingChars() < value.length) return false
    this.outputRows.push(value)
    this.usedLines += 1
    this.usedChars += value.length + (this.outputRows.length > 1 ? 1 : 0)
    return true
  }

  private remainingChars(): number {
    const separator = this.outputRows.length > 0 ? 1 : 0
    const notice = this.reserveGlobalNotice && !this.noticeWritten
      ? Math.min(GLOBAL_TRUNCATION_NOTICE.length, Math.max(0, this.maxChars - this.usedChars)) + separator
      : 0
    return this.maxChars - this.usedChars - separator - notice
  }

  private remainingLines(): number {
    const notice = this.reserveGlobalNotice && !this.noticeWritten ? 1 : 0
    return this.maxLines - this.usedLines - notice
  }

  private snapshot(): { rows: number; usedLines: number; usedChars: number } {
    return { rows: this.outputRows.length, usedLines: this.usedLines, usedChars: this.usedChars }
  }

  private restore(snapshot: { rows: number; usedLines: number; usedChars: number }): void {
    this.outputRows.length = snapshot.rows
    this.usedLines = snapshot.usedLines
    this.usedChars = snapshot.usedChars
  }

  private omittedField(records: readonly string[], truncated: boolean): BoundedSelection {
    const omitted = records.length
    this.truncated ||= truncated || omitted > 0
    this.totalRecords += records.length
    this.omittedRecords += omitted
    return { text: '', records: [], totalRecords: records.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }
  }

  private omittedTable(rows: readonly string[][], truncated: boolean): BoundedTableSelection {
    const omitted = rows.length
    this.truncated ||= truncated || omitted > 0
    this.totalRecords += rows.length
    this.omittedRecords += omitted
    return { text: '', records: [], rows: [], totalRecords: rows.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }
  }
}

function appendSectionSpacer(blocks: ToolDisplayBlock[], selection: BoundedSelection): void {
  if (selection.separator) blocks.push({ kind: 'spacer' })
}

function appendSectionNotice(blocks: ToolDisplayBlock[], budget: PresentationBudget, selection: BoundedSelection): void {
  if (!selection.truncated) return
  const notice = budget.takeSectionNotice(SECTION_TRUNCATION_NOTICE)
  if (notice.visible) blocks.push({ kind: 'notice', text: notice.text })
}

function truncateCharacters(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value
  if (maxChars <= 1) return '…'.slice(0, maxChars)
  let prefix = ''
  for (const character of value) {
    if (prefix.length + character.length + 1 > maxChars) break
    prefix += character
  }
  return `${prefix}…`.slice(0, maxChars)
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[], optionalKeys: readonly string[] = []): boolean {
  const expected = new Set([...keys, ...optionalKeys])
  return Object.keys(value).every((key) => expected.has(key)) && keys.every((key) => key in value)
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0
}

function isPositiveIntegerOrZero(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
