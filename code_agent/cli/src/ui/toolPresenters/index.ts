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
// [maint-note:09ae81a30209]
// 维护导航：VIEW_COLLAPSED_NOTICE；下列为本函数源码中的接口与路径索引。
// 输入 hidden：number。
// 返回契约：string。
// [/maint-note:09ae81a30209]
const VIEW_COLLAPSED_NOTICE = (hidden: number): string => `${hidden} lines hidden · Ctrl+O view details`
const SECTION_TRUNCATION_NOTICE = '… section truncated'

// [maint-note:381192c731f3]
// 数据契约导航：ToolDisplayBlock；字段含义以消费端校验为准。
// 类型组合：| { kind: 'summary'; text: string } | { kind: 'spacer' } | { kind: 'field'; label: string; text: string; preformatted?: boolean; truncated?: boolean } | { kind: 'list'; label: string; items: string[]; truncated?: boolean } | { kind: 'table'。
// [/maint-note:381192c731f3]
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
// [maint-note:f5248041b43e]
// 数据契约导航：ToolResultDetail；字段含义以消费端校验为准。
// 成员 totalRecords：number。
// 成员 omittedRecords：number。
// 成员 sourceTruncated：boolean。
// 成员 viewCollapsed：boolean。
// 成员 canExpand：boolean。
// [/maint-note:f5248041b43e]
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
// [maint-note:c23b5abdc36d]
// 数据契约导航：ToolResultDetailSource；字段含义以消费端校验为准。
// 成员 lines：readonly string[]。
// 成员 sourceTruncated：boolean。
// 成员 viewCollapsed：boolean。
// [/maint-note:c23b5abdc36d]
export interface ToolResultDetailSource {
  lines: readonly string[]
  sourceTruncated: boolean
  viewCollapsed: boolean
}

// [maint-note:a86ab36e6eb9]
// 数据契约导航：ToolResultPresentation；字段含义以消费端校验为准。
// 成员 toolName：string。
// 成员 summary：string。
// 成员 blocks：readonly ToolDisplayBlock[]。
// 成员 plainText：string。
// 成员 truncated：boolean。
// 成员 sourceTruncated：boolean。
// 成员 viewCollapsed：boolean。
// 成员 canExpand：boolean。
// 成员 detail：ToolResultDetail。
// 成员 detailSource（可省略）：ToolResultDetailSource。
// [/maint-note:a86ab36e6eb9]
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
// [maint-note:d67c3b6a3419]
// 维护导航：presentApplyPatchDiff；下列为本函数源码中的接口与路径索引。
// 输入 display：import('../../tools/types.js').ToolDisplayFacts。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 _reserveGlobalNotice：boolean。
// 输入 contentWidth：number。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['applied', 'truncated']) || !Array.isArray(value.applied) || typeof value.truncated !== 'boolean'。
// 分支条件：files === undefined。
// 分支条件：contentWidth < fullDiffMinWidth。
// 分支条件：choice === undefined。
// 分支条件：view.lines.length > 0。
// 分支条件：sourceNotice !== undefined。
// 遍历边界：const candidate ← compactCandidates。
// 返回路径：undefined。
// 返回路径：{ toolName: 'apply_patch', summary: sanitizeToolText(blocks.find((block) => block.kind === 'summary')?.text ?? ''), blocks, plainText: sanitizeToolText(plainText), truncated: sourceTruncated, sourceTruncated, viewCollapsed, canExpand: viewC。
// 返回路径：{ toolName: 'apply_patch', summary: sanitizeToolText(summary), blocks, plainText: sanitizeToolText(plainText), truncated: sourceTruncated, sourceTruncated, viewCollapsed: view.viewCollapsed, canExpand: view.viewCollapsed, detail: { totalRec。
// 写入目标：usedRows。
// 写入目标：usedChars。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Array.isArray。
// 直接调用：parseDiffFiles。
// 直接调用：projectDiffRows。
// 直接调用：boundDetailLines。
// 直接调用：renderDiffRowsAsText(unfolded.rows).split。
// 直接调用：renderDiffRowsAsText。
// 直接调用：candidate.choices.find。
// 直接调用：physicalRows。
// 直接调用：lines.reduce。
// 直接调用：blocks.push。
// 直接调用：lines.join。
// 直接调用：blocks.map((block) => block.kind === 'summary' || block.kind === 'notice' ? block.text : '').filter(Boolean).join。
// 直接调用：blocks.map((block) => block.kind === 'summary' || block.kind === 'notice' ? block.text : '').filter。
// 直接调用：blocks.map。
// 直接调用：sanitizeToolText。
// 直接调用：blocks.find。
// 直接调用：fixedLines.reduce。
// 直接调用：finalizeDiffView。
// 直接调用：Math.max。
// 直接调用：renderDiffLinesAsText。
// 直接调用：blocks.flatMap。
// 直接调用：plainRows.filter(Boolean).join。
// 直接调用：plainRows.filter。
// 直接调用：view.lines.filter。
// [/maint-note:d67c3b6a3419]
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
  // [maint-note:e48d94913640]
  // 维护导航：physicalRows；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：readonly string[]。
  // 直接调用：wrapDisplayLines。
  // [/maint-note:e48d94913640]
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
// [maint-note:f91bf8f4a4da]
// 维护导航：boundDetailLines；下列为本函数源码中的接口与路径索引。
// 输入 lines：readonly string[]。
// 返回契约：{ lines: string[]; clipped: boolean }。
// 返回路径：{ lines: [...bounded.lines], clipped: bounded.clipped }。
// 直接调用：boundRetainedLines。
// [/maint-note:f91bf8f4a4da]
function boundDetailLines(lines: readonly string[]): { lines: string[]; clipped: boolean } {
  const bounded = boundRetainedLines(
    lines,
    UiPresentationLimits.detailRecordMaxChars,
    UiPresentationLimits.detailRecordMaxRecords,
  )
  return { lines: [...bounded.lines], clipped: bounded.clipped }
}

/** Strict parse of the neutral display payload; any drift degrades. */
// [maint-note:14db45444b78]
// 维护导航：parseDiffFiles；下列为本函数源码中的接口与路径索引。
// 输入 facts：Record<string, unknown> | null | undefined。
// 返回契约：readonly import('../presentation/applyPatchDiff.js').DiffFile[] | undefined。
// 分支条件：!isRecord(facts)。
// 分支条件：!Array.isArray(facts.files)。
// 分支条件：!isRecord(file)。
// 分支条件：typeof file.path !== 'string' || typeof file.action !== 'string'。
// 分支条件：file.action !== 'add' && file.action !== 'update' && file.action !== 'delete'。
// 分支条件：!isPositiveIntegerOrZero(file.additions) || !isPositiveIntegerOrZero(file.deletions)。
// 分支条件：!Array.isArray(file.records)。
// 分支条件：file.records.length > 4_000。
// 分支条件：!isRecord(record)。
// 分支条件：record.kind !== 'context' && record.kind !== 'added' && record.kind !== 'removed'。
// 分支条件：typeof record.text !== 'string'。
// 分支条件：oldLine !== undefined && (typeof oldLine !== 'number' || !Number.isSafeInteger(oldLine) || oldLine < 1)。
// 分支条件：newLine !== undefined && (typeof newLine !== 'number' || !Number.isSafeInteger(newLine) || newLine < 1)。
// 分支条件：record.kind === 'removed' && oldLine === undefined。
// 分支条件：record.kind === 'added' && newLine === undefined。
// 分支条件：record.kind === 'context' && (oldLine === undefined || newLine === undefined)。
// 遍历边界：const file ← facts.files。
// 遍历边界：const record ← file.records。
// 返回路径：undefined。
// 返回路径：files。
// 直接调用：isRecord。
// 直接调用：Array.isArray。
// 直接调用：isPositiveIntegerOrZero。
// 直接调用：Number.isSafeInteger。
// 直接调用：records.push。
// 直接调用：files.push。
// [/maint-note:14db45444b78]
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
// [maint-note:bfde4b5d0787]
// 维护导航：presentDeniedTool；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 input：unknown。
// 输入 reason：string。
// 输入 options：{ maxChars?: number; maxLines?: number }；缺省表达式 {}。
// 返回契约：ToolResultPresentation。
// 分支条件：instructionPart.visible。
// 分支条件：reasonPart.visible。
// 返回路径：makePresentation( safeToolName, summaryPart.text, blocks, budget.output(), false, detailLines, visibleRecords, budget, )。
// 直接调用：isRecord。
// 直接调用：buildApprovalPresentation。
// 直接调用：sanitizeToolText。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeField。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：deniedDetailLines。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:bfde4b5d0787]
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
// [maint-note:74ddf80ba305]
// 维护导航：deniedDetailLines；下列为本函数源码中的接口与路径索引。
// 输入 summary：string。
// 输入 instruction：string。
// 输入 reason：string。
// 返回契约：string[]。
// 分支条件：safe。
// 遍历边界：const literal ← [instruction, reason]。
// 返回路径：lines。
// 直接调用：sanitizeToolText。
// 直接调用：lines.push。
// 直接调用：safe.split。
// [/maint-note:74ddf80ba305]
function deniedDetailLines(summary: string, instruction: string, reason: string): string[] {
  const lines = [summary]
  for (const literal of [instruction, reason]) {
    const safe = sanitizeToolText(literal)
    if (safe) lines.push(...safe.split('\n'))
  }
  return lines
}

// [maint-note:f178f7139a6c]
// 数据契约导航：PresentationOptions；字段含义以消费端校验为准。
// 成员 maxChars（可省略）：number。
// 成员 maxLines（可省略）：number。
// 成员 input（可省略）：unknown。
// 成员 reserveGlobalNotice（可省略）：boolean。
// 成员 display（可省略）：import('../../tools/types.js').ToolDisplayFacts。
// 成员 contentWidth（可省略）：number。
// [/maint-note:f178f7139a6c]
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

// [maint-note:190484ccf615]
// 维护导航：presentToolResult；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 output：string。
// 输入 options：PresentationOptions；缺省表达式 {}。
// 返回契约：ToolResultPresentation。
// 分支条件：(!first.sourceTruncated && !first.viewCollapsed) || options.reserveGlobalNotice。
// 返回路径：first。
// 返回路径：presentToolResultOnce(toolName, output, { ...options, reserveGlobalNotice: true })。
// 直接调用：presentToolResultOnce。
// [/maint-note:190484ccf615]
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
// [maint-note:3e51005b1c58]
// 维护导航：registryFallback；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 output：string。
// 输入 options：Parameters<ToolPresenter>[1]。
// 返回契约：ToolResultPresentation。
// 返回路径：fallbackPresentation( toolName, sanitizeToolText(output), Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS)), Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES)), options.reserveGlobalNotice 。
// 直接调用：fallbackPresentation。
// 直接调用：sanitizeToolText。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// [/maint-note:3e51005b1c58]
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
// [maint-note:92fbbefda0c2]
// 维护导航：degrade；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 typed：(output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation | undefined。
// 返回契约：ToolPresenter。
// 返回路径：(output, options) => typed(output, options) ?? registryFallback(toolName, output, options)。
// [/maint-note:92fbbefda0c2]
function degrade(
  toolName: string,
  typed: (output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation | undefined,
): ToolPresenter {
  return (output, options) => typed(output, options) ?? registryFallback(toolName, output, options)
}

// [maint-note:207750b9f8ab]
// 维护导航：registryChars；下列为本函数源码中的接口与路径索引。
// 输入 options：PresentationOptions。
// 返回契约：number。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// [/maint-note:207750b9f8ab]
const registryChars = (options: PresentationOptions): number =>
  Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS))
// [maint-note:3a4547bad310]
// 维护导航：registryLines；下列为本函数源码中的接口与路径索引。
// 输入 options：PresentationOptions。
// 返回契约：number。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// [/maint-note:3a4547bad310]
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
// [maint-note:b55da2d5f95e]
// 维护导航：typedViaJson；下列为本函数源码中的接口与路径索引。
// 输入 present：(parsed: unknown, options: PresentationOptions) => ToolResultPresentation | undefined。
// 返回契约：(output: string, options: Parameters<ToolPresenter>[1]) => ToolResultPresentation | undefined。
// 返回路径：(output, options) => { const safeOutput = sanitizeToolText(output) let parsed: unknown try { // This is the one and only JSON decode in the UI projection. Assistant // content and preformatted tool fields are never passed through it. parsed。
// [/maint-note:b55da2d5f95e]
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

// [maint-note:a3410279fbe5]
// 维护导航：presentToolResultOnce；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 output：string。
// 输入 options：PresentationOptions。
// 返回契约：ToolResultPresentation。
// 返回路径：toolPresenterRegistry.present(toolName, safeOutput, { ...options, maxChars, maxLines })。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：sanitizeToolText。
// 直接调用：toolPresenterRegistry.present。
// [/maint-note:a3410279fbe5]
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

// [maint-note:076a645c2c48]
// 维护导航：sanitizeToolText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '') .replace(/\u001b/gu, '') .split('') .map((char。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '') .replace(/\u001b/gu, '') .split('') .map((char。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '') .replace(/\u001b/gu, '') .split('') .map。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '') .replace(/\u001b/gu, '') .split。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f-\u009f]/gu, '') .replace。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace。
// 直接调用：value .replace。
// [/maint-note:076a645c2c48]
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

// [maint-note:767fb351871c]
// 维护导航：toolPresentationText；下列为本函数源码中的接口与路径索引。
// 输入 presentation：ToolResultPresentation。
// 返回契约：string。
// 返回路径：presentation.plainText。
// [/maint-note:767fb351871c]
export function toolPresentationText(presentation: ToolResultPresentation): string {
  return presentation.plainText
}

/** The table renderer and character ledger must measure the same row. */
// [maint-note:f95a112df810]
// 维护导航：serializeToolTableRow；下列为本函数源码中的接口与路径索引。
// 输入 row：readonly string[]。
// 返回契约：string。
// 返回路径：row.map((cell) => cell.replace(/\r\n?|\n/gu, '\\n')).join(' ')。
// 直接调用：row.map((cell) => cell.replace(/\r\n?|\n/gu, '\\n')).join。
// 直接调用：row.map。
// [/maint-note:f95a112df810]
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
// [maint-note:1829b2e5e900]
// 数据契约导航：ContinuationRangeProjection；字段含义以消费端校验为准。
// 成员 startLine：number。
// 成员 endLine：number。
// [/maint-note:1829b2e5e900]
export interface ContinuationRangeProjection {
  startLine: number
  endLine: number
}

// [maint-note:e86c6d46295e]
// 数据契约导航：PagePlanProjection；字段含义以消费端校验为准。
// 成员 suggestedContinuationRanges：readonly ContinuationRangeProjection[]。
// 成员 pagePlanComplete：boolean。
// 成员 nextPlanStartLine（可省略）：number。
// [/maint-note:e86c6d46295e]
export interface PagePlanProjection {
  suggestedContinuationRanges: readonly ContinuationRangeProjection[]
  pagePlanComplete: boolean
  nextPlanStartLine?: number
}

// [maint-note:64f458c4cfaa]
// 数据契约导航：ReadFilePageProjection；字段含义以消费端校验为准。
// 类型组合：| { complete: true path: string startLine: number endLine: number totalLines: number content: string } | { complete: false path: string startLine: number endLine: number totalLines: number content: string nextStartLine: number plan?: PagePl。
// [/maint-note:64f458c4cfaa]
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

// [maint-note:0284f693f755]
// 维护导航：parseReadFilePage；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：ReadFilePageProjection | undefined。
// 分支条件：!isRecord(value)。
// 分支条件：!hasExactKeys( value, ['path', 'startLine', 'endLine', 'totalLines', 'content', 'complete', 'truncated'], ['nextStartLine', 'suggestedContinuationRanges', 'pagePlanComplete', 'nextPlanStartLine'], )。
// 分支条件：typeof value.path !== 'string' || !value.path。
// 分支条件：!isPositiveInteger(value.startLine)。
// 分支条件：!isNonNegativeInteger(value.endLine) || !isNonNegativeInteger(value.totalLines)。
// 分支条件：typeof value.content !== 'string' || typeof value.complete !== 'boolean' || typeof value.truncated !== 'boolean'。
// 分支条件：value.truncated === value.complete。
// 分支条件：value.totalLines === 0。
// 分支条件：value.startLine !== 1 || value.endLine !== 0 || !value.complete || value.content !== ''。
// 分支条件：value.startLine > value.endLine || value.endLine > value.totalLines。
// 分支条件：value.complete。
// 分支条件：'nextStartLine' in value || 'suggestedContinuationRanges' in value || 'pagePlanComplete' in value || 'nextPlanStartLine' in value。
// 分支条件：value.endLine !== value.totalLines。
// 分支条件：value.endLine >= value.totalLines。
// 分支条件：!isPositiveInteger(value.nextStartLine) || value.nextStartLine !== value.endLine + 1。
// 分支条件：plan === undefined && ('suggestedContinuationRanges' in value || 'pagePlanComplete' in value || 'nextPlanStartLine' in value)。
// 返回路径：undefined。
// 返回路径：{ complete: true, path: value.path, startLine: 1, endLine: 0, totalLines: 0, content: '' }。
// 返回路径：{ complete: true, path: value.path, startLine: value.startLine, endLine: value.endLine, totalLines: value.totalLines, content: value.content }。
// 返回路径：{ complete: false, path: value.path, startLine: value.startLine, endLine: value.endLine, totalLines: value.totalLines, content: value.content, nextStartLine: value.nextStartLine, ...(plan === undefined ? {} : { plan }), }。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：isPositiveInteger。
// 直接调用：isNonNegativeInteger。
// 直接调用：parsePagePlan。
// [/maint-note:0284f693f755]
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
// [maint-note:de3dfd8cdba1]
// 维护导航：parsePagePlan；下列为本函数源码中的接口与路径索引。
// 输入 value：Record<string, unknown>。
// 输入 nextStartLine：number。
// 输入 totalLines：number。
// 返回契约：PagePlanProjection | undefined。
// 分支条件：!hasRanges && !hasComplete && !hasNextPlan。
// 分支条件：!hasRanges || !hasComplete。
// 分支条件：!Array.isArray(ranges) || ranges.length === 0 || ranges.length > MAX_SUGGESTED_CONTINUATION_RANGES。
// 分支条件：!isRecord(item) || !hasExactKeys(item, ['startLine', 'endLine'])。
// 分支条件：!isPositiveInteger(item.startLine) || !isPositiveInteger(item.endLine)。
// 分支条件：item.startLine !== expectedStart。
// 分支条件：item.endLine < item.startLine || item.endLine > totalLines。
// 分支条件：typeof value.pagePlanComplete !== 'boolean'。
// 分支条件：value.pagePlanComplete。
// 分支条件：hasNextPlan || lastEnd !== totalLines。
// 分支条件：!hasNextPlan || !isPositiveInteger(value.nextPlanStartLine)。
// 分支条件：lastEnd >= totalLines。
// 分支条件：value.nextPlanStartLine > totalLines。
// 分支条件：value.nextPlanStartLine !== lastEnd + 1。
// 遍历边界：const item ← ranges。
// 返回路径：undefined。
// 返回路径：{ suggestedContinuationRanges: parsed, pagePlanComplete: true }。
// 返回路径：{ suggestedContinuationRanges: parsed, pagePlanComplete: false, nextPlanStartLine: value.nextPlanStartLine }。
// 写入目标：expectedStart。
// 直接调用：Array.isArray。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：isPositiveInteger。
// 直接调用：parsed.push。
// 直接调用：parsed.at。
// [/maint-note:de3dfd8cdba1]
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

// [maint-note:35f91ac39684]
// 维护导航：presentReadFile；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!page。
// 分支条件：contentPart?.visible。
// 分支条件：!page.complete && page.plan。
// 分支条件：planPart.visible。
// 返回路径：undefined。
// 返回路径：makePresentation('read_file', summaryPart.text, blocks, budget.output(), false, detailLines, visibleRecords, budget)。
// 直接调用：parseReadFilePage。
// 直接调用：sanitizeToolText。
// 直接调用：budget.takePlain。
// 直接调用：content.endsWith。
// 直接调用：content.slice。
// 直接调用：budget.takeField。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：page.plan.suggestedContinuationRanges .map((range) => `${range.startLine}–${range.endLine}`) .join。
// 直接调用：page.plan.suggestedContinuationRanges .map。
// 直接调用：displayContent.split。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:35f91ac39684]
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

// [maint-note:495e2ff7cce9]
// 维护导航：presentBash；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 input：unknown。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['stdout', 'stderr', 'exitCode', 'timedOut', 'interrupted', 'truncated'])。
// 分支条件：typeof value.stdout !== 'string' || typeof value.stderr !== 'string' || !Number.isInteger(value.exitCode) || typeof value.timedOut !== 'boolean' || typeof value.interrupted !== 'boolean' || typeof value.truncated !== 'boolean'。
// 分支条件：isRecord(input) && typeof input.command === 'string' && input.command.trim()。
// 分支条件：command.visible。
// 分支条件：stdout?.visible。
// 分支条件：stderr?.visible。
// 返回路径：undefined。
// 返回路径：makePresentation( 'bash', summaryPart.text, blocks, budget.output(), value.truncated === true, detailLines, visibleRecords, budget, )。
// 写入目标：command。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Number.isInteger。
// 直接调用：sanitizeToolText。
// 直接调用：[ `exit ${value.exitCode}`, value.timedOut ? 'timeout' : '', value.interrupted ? 'interrupted' : '', value.truncated ? 'truncated' : '', ].filter(Boolean).join。
// 直接调用：[ `exit ${value.exitCode}`, value.timedOut ? 'timeout' : '', value.interrupted ? 'interrupted' : '', value.truncated ? 'truncated' : '', ].filter。
// 直接调用：budget.takePlain。
// 直接调用：input.command.trim。
// 直接调用：budget.takeField。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：stdoutText.split。
// 直接调用：stderrText.split。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:495e2ff7cce9]
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

// [maint-note:04b9cd8fa872]
// 维护导航：presentListFiles；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['entries', 'truncated']) || !Array.isArray(value.entries) || typeof value.truncated !== 'boolean'。
// 分支条件：!isRecord(item) || typeof item.path !== 'string' || item.type !== 'file'。
// 分支条件：bounded.visible。
// 遍历边界：const item ← value.entries。
// 返回路径：undefined。
// 返回路径：makePresentation('list_files', summaryPart.text, blocks, budget.output(), value.truncated === true, entries, bounded.records.length, budget)。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Array.isArray。
// 直接调用：entries.push。
// 直接调用：sanitizeToolText。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeField。
// 直接调用：entries.join。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:04b9cd8fa872]
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

// [maint-note:6f6eb5e8eefd]
// 维护导航：presentSearchText；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['matches', 'truncated']) || !Array.isArray(value.matches) || typeof value.truncated !== 'boolean'。
// 分支条件：!isRecord(item) || typeof item.path !== 'string' || !isPositiveInteger(item.line) || !isPositiveInteger(item.column) || typeof item.text !== 'string'。
// 分支条件：bounded.visible。
// 遍历边界：const item ← value.matches。
// 返回路径：undefined。
// 返回路径：makePresentation('search_text', summaryPart.text, blocks, budget.output(), value.truncated === true, matches, bounded.records.length, budget)。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Array.isArray。
// 直接调用：isPositiveInteger。
// 直接调用：matches.push。
// 直接调用：sanitizeToolText。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeField。
// 直接调用：matches.join。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:6f6eb5e8eefd]
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

// [maint-note:783f325e91b5]
// 维护导航：presentGitChanges；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isValidGitChangesEnvelope(value)。
// 分支条件：!value.applicable。
// 分支条件：value.reason !== 'not_git_repository' || value.complete !== true。
// 分支条件：value.mode === 'summary'。
// 分支条件：!isValidGitSummary(value.summary)。
// 分支条件：isRecord(value.summary.byLayer)。
// 分支条件：value.mode === 'files'。
// 分支条件：!Array.isArray(value.items)。
// 分支条件：!isValidGitFileRecord(item)。
// 分支条件：value.mode === 'patch'。
// 分支条件：!Array.isArray(value.patches)。
// 分支条件：!isValidGitPatchLayer(patch)。
// 分支条件：isRecord(value.coverage) && value.coverage.submoduleWorktrees === 'not_observed'。
// 分支条件：typeof value.guidance === 'string'。
// 分支条件：detailPart?.visible。
// 遍历边界：const item ← value.items。
// 遍历边界：const patch ← value.patches。
// 遍历边界：const hunk ← patch.hunks。
// 遍历边界：const line ← hunk.lines。
// 返回路径：undefined。
// 返回路径：makePresentation( 'git_changes', summaryPart.text, blocks, budget.output(), false, detailLines, detailPart?.records.length ?? 0, budget, )。
// 写入目标：summary。
// 写入目标：hunkCount。
// 直接调用：isValidGitChangesEnvelope。
// 直接调用：isValidGitSummary。
// 直接调用：isRecord。
// 直接调用：detailLines.push。
// 直接调用：String。
// 直接调用：Array.isArray。
// 直接调用：isValidGitFileRecord。
// 直接调用：sanitizeToolText。
// 直接调用：isValidGitPatchLayer。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeField。
// 直接调用：detailLines.join。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:783f325e91b5]
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

// [maint-note:d524a9e2d49c]
// 数据契约导航：GitPresentationEnvelope；字段含义以消费端校验为准。
// 成员 version：1。
// 成员 applicable：boolean。
// 成员 mode：'summary' | 'files' | 'patch'。
// 成员 scope：'staged' | 'unstaged' | 'both'。
// 成员 complete：boolean。
// 成员 reason（可省略）：'not_git_repository'。
// 成员 revision（可省略）：string。
// 成员 path（可省略）：string。
// 成员 offset（可省略）：number。
// 成员 nextCursor（可省略）：string。
// 成员 guidance（可省略）：string。
// 成员 coverage（可省略）：GitPresentationCoverage。
// 成员 summary（可省略）：GitPresentationSummary。
// 成员 items（可省略）：GitPresentationFile[]。
// 成员 patches（可省略）：GitPresentationPatch[]。
// [/maint-note:d524a9e2d49c]
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

// [maint-note:469d684b8b63]
// 数据契约导航：GitPresentationCoverage；字段含义以消费端校验为准。
// 成员 parentRepositoryChanges：'complete'。
// 成员 submoduleWorktrees：'not_present' | 'not_observed'。
// 成员 gitlinkCount：number。
// [/maint-note:469d684b8b63]
interface GitPresentationCoverage {
  parentRepositoryChanges: 'complete'
  submoduleWorktrees: 'not_present' | 'not_observed'
  gitlinkCount: number
}

// [maint-note:643a43737501]
// 数据契约导航：GitPresentationSummary；字段含义以消费端校验为准。
// 成员 uniqueFiles：number。
// 成员 layerRecords：number。
// 成员 byLayer：{ staged: number; unstaged: number; unmerged: number }。
// 成员 byAction：Record<string, number>。
// 成员 recordsWithBinary：number。
// 成员 recordsWithModeChange：number。
// 成员 recordsWithTypeChange：number。
// 成员 recordsUnmerged：number。
// [/maint-note:643a43737501]
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

// [maint-note:c7582996b36d]
// 数据契约导航：GitPresentationFile；字段含义以消费端校验为准。
// 成员 layer：'staged' | 'unstaged' | 'unmerged'。
// 成员 action：'added' | 'modified' | 'deleted' | 'renamed' | 'untracked' | 'type_changed' | 'unmerged'。
// 成员 path：string。
// 成员 oldPath（可省略）：string。
// 成员 contentChanged：boolean | 'unknown'。
// 成员 binary：boolean | 'unknown'。
// 成员 modeChanged：boolean。
// 成员 objectType：'file' | 'symlink' | 'gitlink' | 'other'。
// [/maint-note:c7582996b36d]
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

// [maint-note:b862553c407e]
// 数据契约导航：GitPresentationPatch；字段含义以消费端校验为准。
// 成员 layer：'staged' | 'unstaged'。
// 成员 file：GitPresentationFile。
// 成员 hunks：GitPresentationHunk[]。
// [/maint-note:b862553c407e]
interface GitPresentationPatch extends Record<string, unknown> {
  layer: 'staged' | 'unstaged'
  file: GitPresentationFile
  hunks: GitPresentationHunk[]
}

// [maint-note:a9df1ab97819]
// 数据契约导航：GitPresentationHunk；字段含义以消费端校验为准。
// 成员 oldStart：number。
// 成员 oldLines：number。
// 成员 newStart：number。
// 成员 newLines：number。
// 成员 section（可省略）：string。
// 成员 continuedFromPrevious（可省略）：boolean。
// 成员 continuesNext（可省略）：boolean。
// 成员 lines：GitPresentationLine[]。
// [/maint-note:a9df1ab97819]
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

// [maint-note:c58685ad30e3]
// 数据契约导航：GitPresentationLine；字段含义以消费端校验为准。
// 成员 kind：'context' | 'add' | 'delete' | 'no_newline'。
// 成员 text（可省略）：string。
// 成员 oldLine（可省略）：number | null。
// 成员 newLine（可省略）：number | null。
// 成员 appliesTo（可省略）：'old' | 'new'。
// [/maint-note:c58685ad30e3]
interface GitPresentationLine extends Record<string, unknown> {
  kind: 'context' | 'add' | 'delete' | 'no_newline'
  text?: string
  oldLine?: number | null
  newLine?: number | null
  appliesTo?: 'old' | 'new'
}

// [maint-note:df486da5d927]
// 维护导航：isValidGitChangesEnvelope；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationEnvelope。
// 分支条件：!isRecord(value) || value.version !== 1 || typeof value.applicable !== 'boolean' || !isOneOf(value.mode, ['summary', 'files', 'patch']) || !isOneOf(value.scope, ['staged', 'unstaged', 'both']) || typeof value.complete !== 'boolean'。
// 分支条件：value.applicable === false。
// 分支条件：typeof value.revision !== 'string' || !isValidGitCoverage(value.coverage) || (value.guidance !== undefined && typeof value.guidance !== 'string') || (value.path !== undefined && typeof value.path !== 'string') || (value.nextCursor !== undef。
// 分支条件：value.mode === 'summary'。
// 分支条件：!isNonNegativeInteger(value.offset)。
// 分支条件：value.complete === false && typeof value.nextCursor !== 'string'。
// 分支条件：value.complete === true && value.nextCursor !== undefined。
// 分支条件：value.mode === 'files'。
// 返回路径：false。
// 返回路径：value.reason === 'not_git_repository' && value.complete === true。
// 返回路径：value.complete === true && isValidGitSummary(value.summary)。
// 返回路径：Array.isArray(value.items) && value.items.every(isValidGitFileRecord)。
// 返回路径：typeof value.path === 'string' && Array.isArray(value.patches) && value.patches.every(isValidGitPatchLayer)。
// 直接调用：isRecord。
// 直接调用：isOneOf。
// 直接调用：isValidGitCoverage。
// 直接调用：isValidGitSummary。
// 直接调用：isNonNegativeInteger。
// 直接调用：Array.isArray。
// 直接调用：value.items.every。
// 直接调用：value.patches.every。
// [/maint-note:df486da5d927]
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

// [maint-note:fbe90b85bc8a]
// 维护导航：isValidGitCoverage；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationCoverage。
// 返回路径：isRecord(value) && value.parentRepositoryChanges === 'complete' && isOneOf(value.submoduleWorktrees, ['not_present', 'not_observed']) && isNonNegativeInteger(value.gitlinkCount)。
// 直接调用：isRecord。
// 直接调用：isOneOf。
// 直接调用：isNonNegativeInteger。
// [/maint-note:fbe90b85bc8a]
function isValidGitCoverage(value: unknown): value is GitPresentationCoverage {
  return isRecord(value) && value.parentRepositoryChanges === 'complete' &&
    isOneOf(value.submoduleWorktrees, ['not_present', 'not_observed']) &&
    isNonNegativeInteger(value.gitlinkCount)
}

// [maint-note:1e95614e1afb]
// 维护导航：isValidGitSummary；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationSummary。
// 分支条件：!isRecord(value) || !isNonNegativeInteger(value.uniqueFiles) || !isNonNegativeInteger(value.layerRecords)。
// 分支条件：!isRecord(value.byLayer) || !hasNonNegativeIntegerKeys(value.byLayer, ['staged', 'unstaged', 'unmerged']) || !isRecord(value.byAction) || !hasNonNegativeIntegerKeys(value.byAction, [ 'added', 'modified', 'deleted', 'renamed', 'untracked', '。
// 返回路径：false。
// 返回路径：[ value.recordsWithBinary, value.recordsWithModeChange, value.recordsWithTypeChange, value.recordsUnmerged, ].every(isNonNegativeInteger)。
// 直接调用：isRecord。
// 直接调用：isNonNegativeInteger。
// 直接调用：hasNonNegativeIntegerKeys。
// 直接调用：[ value.recordsWithBinary, value.recordsWithModeChange, value.recordsWithTypeChange, value.recordsUnmerged, ].every。
// [/maint-note:1e95614e1afb]
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

// [maint-note:80aac81ad3ee]
// 维护导航：isValidGitFileRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationFile。
// 分支条件：!isRecord(value) || !isOneOf(value.layer, ['staged', 'unstaged', 'unmerged']) || !isOneOf(value.action, ['added', 'modified', 'deleted', 'renamed', 'untracked', 'type_changed', 'unmerged']) || typeof value.path !== 'string' || (value.oldPat。
// 分支条件：value.conflictStages !== undefined。
// 分支条件：!Array.isArray(value.conflictStages) || value.conflictStages.length === 0。
// 分支条件：!stages.every((stage) => stage === 1 || stage === 2 || stage === 3)。
// 分支条件：new Set(stages).size !== stages.length。
// 返回路径：false。
// 返回路径：true。
// 直接调用：isRecord。
// 直接调用：isOneOf。
// 直接调用：isBooleanOrUnknown。
// 直接调用：Array.isArray。
// 直接调用：stages.every。
// [/maint-note:80aac81ad3ee]
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

// [maint-note:5dc21d656aba]
// 维护导航：isValidGitPatchLayer；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationPatch。
// 返回路径：isRecord(value) && isOneOf(value.layer, ['staged', 'unstaged']) && isValidGitFileRecord(value.file) && Array.isArray(value.hunks) && value.hunks.every(isValidGitHunk)。
// 直接调用：isRecord。
// 直接调用：isOneOf。
// 直接调用：isValidGitFileRecord。
// 直接调用：Array.isArray。
// 直接调用：value.hunks.every。
// [/maint-note:5dc21d656aba]
function isValidGitPatchLayer(value: unknown): value is GitPresentationPatch {
  return isRecord(value) && isOneOf(value.layer, ['staged', 'unstaged']) &&
    isValidGitFileRecord(value.file) && Array.isArray(value.hunks) && value.hunks.every(isValidGitHunk)
}

// [maint-note:2ea781a9eae6]
// 维护导航：isValidGitHunk；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationHunk。
// 返回路径：isRecord(value) && isNonNegativeInteger(value.oldStart) && isNonNegativeInteger(value.oldLines) && isNonNegativeInteger(value.newStart) && isNonNegativeInteger(value.newLines) && (value.section === undefined || typeof value.section === 'str。
// 直接调用：isRecord。
// 直接调用：isNonNegativeInteger。
// 直接调用：Array.isArray。
// 直接调用：value.lines.every。
// [/maint-note:2ea781a9eae6]
function isValidGitHunk(value: unknown): value is GitPresentationHunk {
  return isRecord(value) &&
    isNonNegativeInteger(value.oldStart) && isNonNegativeInteger(value.oldLines) &&
    isNonNegativeInteger(value.newStart) && isNonNegativeInteger(value.newLines) &&
    (value.section === undefined || typeof value.section === 'string') &&
    (value.continuedFromPrevious === undefined || typeof value.continuedFromPrevious === 'boolean') &&
    (value.continuesNext === undefined || typeof value.continuesNext === 'boolean') &&
    Array.isArray(value.lines) && value.lines.every(isValidGitLine)
}

// [maint-note:c178d80178f9]
// 维护导航：isValidGitLine；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is GitPresentationLine。
// 分支条件：!isRecord(value) || !isOneOf(value.kind, ['context', 'add', 'delete', 'no_newline'])。
// 分支条件：value.kind === 'no_newline'。
// 分支条件：typeof value.text !== 'string'。
// 返回路径：false。
// 返回路径：isOneOf(value.appliesTo, ['old', 'new'])。
// 返回路径：isNullableLineNumber(value.oldLine) && isNullableLineNumber(value.newLine)。
// 直接调用：isRecord。
// 直接调用：isOneOf。
// 直接调用：isNullableLineNumber。
// [/maint-note:c178d80178f9]
function isValidGitLine(value: unknown): value is GitPresentationLine {
  if (!isRecord(value) || !isOneOf(value.kind, ['context', 'add', 'delete', 'no_newline'])) return false
  if (value.kind === 'no_newline') return isOneOf(value.appliesTo, ['old', 'new'])
  if (typeof value.text !== 'string') return false
  return isNullableLineNumber(value.oldLine) && isNullableLineNumber(value.newLine)
}

// [maint-note:9925f9fddea1]
// 维护导航：isNullableLineNumber；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：boolean。
// 返回路径：value === null || (Number.isSafeInteger(value) && (value as number) >= 0)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:9925f9fddea1]
function isNullableLineNumber(value: unknown): boolean {
  return value === null || (Number.isSafeInteger(value) && (value as number) >= 0)
}

// [maint-note:0a638178baf7]
// 维护导航：isBooleanOrUnknown；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：boolean。
// 返回路径：typeof value === 'boolean' || value === 'unknown'。
// [/maint-note:0a638178baf7]
function isBooleanOrUnknown(value: unknown): boolean {
  return typeof value === 'boolean' || value === 'unknown'
}

// [maint-note:734154ad5f85]
// 维护导航：hasNonNegativeIntegerKeys；下列为本函数源码中的接口与路径索引。
// 输入 value：Record<string, unknown>。
// 输入 keys：readonly string[]。
// 返回契约：boolean。
// 返回路径：hasExactKeys(value, keys) && keys.every((key) => isNonNegativeInteger(value[key]))。
// 直接调用：hasExactKeys。
// 直接调用：keys.every。
// [/maint-note:734154ad5f85]
function hasNonNegativeIntegerKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return hasExactKeys(value, keys) && keys.every((key) => isNonNegativeInteger(value[key]))
}

// [maint-note:741a6ef5f5c8]
// 维护导航：isOneOf；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 accepted：T。
// 返回契约：value is T[number]。
// 返回路径：typeof value === 'string' && accepted.includes(value)。
// 直接调用：accepted.includes。
// [/maint-note:741a6ef5f5c8]
function isOneOf<const T extends readonly string[]>(value: unknown, accepted: T): value is T[number] {
  return typeof value === 'string' && accepted.includes(value)
}

// [maint-note:f69fc7a87bd0]
// 维护导航：presentTaskPlan；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：!isRecord(value) || value.ok !== true || typeof value.operation !== 'string'。
// 分支条件：planValue !== undefined && planValue !== null && !plan。
// 分支条件：historical。
// 分支条件：operation === 'list_snapshots'。
// 分支条件：operation === 'claim_run_complete'。
// 分支条件：plan。
// 分支条件：listPart?.visible。
// 返回路径：undefined。
// 返回路径：makePresentation( 'update_task_plan', summaryPart.text, blocks, budget.output(), false, items, listPart?.records.length ?? 0, budget, )。
// 写入目标：summary。
// 直接调用：isRecord。
// 直接调用：parseTaskPlanPresentation。
// 直接调用：Array.isArray。
// 直接调用：plan?.steps.map。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeField。
// 直接调用：items.join。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// [/maint-note:f69fc7a87bd0]
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

// [maint-note:28fb4fd5fa0b]
// 维护导航：parseTaskPlanPresentation；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：{ readonly revision: number readonly steps: readonly { readonly content: string; readonly status: 'pending' | 'in_progress' | 'completed' }[] } | undefined。
// 分支条件：!isRecord(value) || !isPositiveInteger(value.revision) || !Array.isArray(value.steps)。
// 分支条件：!isRecord(step) || typeof step.content !== 'string'。
// 分支条件：step.status !== 'pending' && step.status !== 'in_progress' && step.status !== 'completed'。
// 遍历边界：const step ← value.steps。
// 返回路径：undefined。
// 返回路径：{ revision: value.revision, steps }。
// 直接调用：isRecord。
// 直接调用：isPositiveInteger。
// 直接调用：Array.isArray。
// 直接调用：steps.push。
// [/maint-note:28fb4fd5fa0b]
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

// [maint-note:2f231aad79a1]
// 维护导航：taskStepSymbol；下列为本函数源码中的接口与路径索引。
// 输入 status：'pending' | 'in_progress' | 'completed'。
// 返回契约：'[ ]' | '[>]' | '[✓]'。
// 分支条件：status === 'pending'。
// 分支条件：status === 'in_progress'。
// 返回路径：'[ ]'。
// 返回路径：'[>]'。
// 返回路径：'[✓]'。
// [/maint-note:2f231aad79a1]
function taskStepSymbol(status: 'pending' | 'in_progress' | 'completed'): '[ ]' | '[>]' | '[✓]' {
  if (status === 'pending') return '[ ]'
  if (status === 'in_progress') return '[>]'
  return '[✓]'
}

// [maint-note:ffbc97a0353d]
// 维护导航：presentApplyPatch；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 输入 display：import('../../tools/types.js').ToolDisplayFacts | undefined。
// 输入 contentWidth：number。
// 返回契约：ToolResultPresentation | undefined。
// 分支条件：display?.kind === 'apply_patch_diff'。
// 分支条件：projected !== undefined。
// 分支条件：!isRecord(value) || !hasExactKeys(value, ['applied', 'truncated']) || !Array.isArray(value.applied) || typeof value.truncated !== 'boolean'。
// 分支条件：!isRecord(item) || typeof item.path !== 'string' || typeof item.action !== 'string' || !isPositiveIntegerOrZero(item.additions) || !isPositiveIntegerOrZero(item.deletions)。
// 分支条件：bounded.visible。
// 遍历边界：const item ← value.applied。
// 返回路径：projected。
// 返回路径：undefined。
// 返回路径：makePresentation('apply_patch', summaryPart.text, blocks, budget.output(), value.truncated === true, rows.map((row) => row.join(' ')), bounded.rows.length, budget)。
// 直接调用：presentApplyPatchDiff。
// 直接调用：isRecord。
// 直接调用：hasExactKeys。
// 直接调用：Array.isArray。
// 直接调用：isPositiveIntegerOrZero。
// 直接调用：rows.push。
// 直接调用：sanitizeToolText。
// 直接调用：budget.takePlain。
// 直接调用：budget.takeTable。
// 直接调用：appendSectionSpacer。
// 直接调用：blocks.push。
// 直接调用：appendSectionNotice。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// 直接调用：rows.map。
// [/maint-note:ffbc97a0353d]
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

// [maint-note:11ecc143cbfd]
// 维护导航：fallbackPresentation；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 output：string。
// 输入 maxChars：number。
// 输入 maxLines：number。
// 输入 reserveGlobalNotice：boolean。
// 返回契约：ToolResultPresentation。
// 返回路径：makePresentation(toolName, summaryPart.text, blocks, budget.output(), summaryPart.truncated, candidate.split('\n'), summaryPart.records.length, budget)。
// 直接调用：budget.takePlain。
// 直接调用：makePresentation。
// 直接调用：budget.output。
// 直接调用：candidate.split。
// [/maint-note:11ecc143cbfd]
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
// [maint-note:d8d6cfa4601c]
// 维护导航：makePresentation；下列为本函数源码中的接口与路径索引。
// 输入 toolName：string。
// 输入 summary：string。
// 输入 blocks：ToolDisplayBlock[]。
// 输入 plainText：string。
// 输入 upstreamTruncated：boolean。
// 输入 detailLines：readonly string[]。
// 输入 visibleRecords：number。
// 输入 budget：PresentationBudget（可省略）。
// 返回契约：ToolResultPresentation。
// 分支条件：sourceTruncated && budget。
// 分支条件：notice.visible。
// 分支条件：viewCollapsed。
// 分支条件：notice?.visible。
// 返回路径：{ toolName, summary: sanitizeToolText(summary), blocks: renderedBlocks, plainText: sanitizeToolText(renderedPlainText), truncated: sourceTruncated, sourceTruncated, viewCollapsed, canExpand: viewCollapsed, detail: { totalRecords: detail.lin。
// 直接调用：sanitizeToolText。
// 直接调用：boundDetailLines。
// 直接调用：Math.max。
// 直接调用：Math.min。
// 直接调用：budget.takeNotice。
// 直接调用：renderedBlocks.push。
// 直接调用：budget?.takeNotice。
// 直接调用：VIEW_COLLAPSED_NOTICE。
// 直接调用：budget?.output。
// [/maint-note:d8d6cfa4601c]
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

// [maint-note:1722410edfe3]
// 数据契约导航：BoundedSelection；字段含义以消费端校验为准。
// 成员 text：string。
// 成员 records：string[]。
// 成员 totalRecords：number。
// 成员 omittedRecords：number。
// 成员 truncated：boolean。
// 成员 visible：boolean。
// 成员 separator：boolean。
// [/maint-note:1722410edfe3]
interface BoundedSelection {
  text: string
  records: string[]
  totalRecords: number
  omittedRecords: number
  truncated: boolean
  visible: boolean
  separator: boolean
}

// [maint-note:066d811355bb]
// 数据契约导航：BoundedTableSelection；字段含义以消费端校验为准。
// 成员 rows：string[][]。
// [/maint-note:066d811355bb]
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

  // [maint-note:31470e883b54]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 maxChars：number。
  // 输入 maxLines：number。
  // 输入 reserveGlobalNotice：类型由调用方与函数体约束；缺省表达式 false。
  // [/maint-note:31470e883b54]
  constructor(
    private readonly maxChars: number,
    private readonly maxLines: number,
    private readonly reserveGlobalNotice = false,
  ) {}

  // [maint-note:038cdfca3904]
  // 维护导航：takePlain；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 返回契约：BoundedSelection。
  // 返回路径：{ text: selected.records.join('\n'), records: selected.records, totalRecords: allRecords.length, omittedRecords: Math.max(0, allRecords.length - selected.records.length), truncated, visible: selected.records.length > 0, separator: false, }。
  // 写入目标：this.truncated。
  // 直接调用：value.split。
  // 直接调用：this.selectRows。
  // 直接调用：selected.records.join。
  // 直接调用：Math.max。
  // [/maint-note:038cdfca3904]
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

  // [maint-note:7ff0150725d8]
  // 维护导航：takeField；下列为本函数源码中的接口与路径索引。
  // 输入 label：string。
  // 输入 value：string。
  // 返回契约：BoundedSelection。
  // 分支条件：allRecords.length === 0。
  // 分支条件：separator && !this.appendExact('')。
  // 分支条件：!this.appendExact(`${label}:`)。
  // 分支条件：selected.truncated && this.reserveGlobalNotice && !this.noticeWritten。
  // 分支条件：selected.records.length === 0。
  // 返回路径：{ text: '', records: [], totalRecords: 0, omittedRecords: 0, truncated: false, visible: false, separator: false }。
  // 返回路径：this.omittedField(allRecords, false)。
  // 返回路径：this.omittedField(allRecords, true)。
  // 返回路径：{ text: selected.records.join('\n'), records: selected.records, totalRecords: allRecords.length, omittedRecords, truncated, visible: true, separator, }。
  // 写入目标：selected。
  // 写入目标：this.totalRecords。
  // 写入目标：this.omittedRecords。
  // 写入目标：this.truncated。
  // 直接调用：value.split。
  // 直接调用：this.snapshot。
  // 直接调用：this.appendExact。
  // 直接调用：this.omittedField。
  // 直接调用：this.restore。
  // 直接调用：this.selectRows。
  // 直接调用：Math.max。
  // 直接调用：this.records.push。
  // 直接调用：selected.records.join。
  // [/maint-note:7ff0150725d8]
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

  // [maint-note:1b06b5712360]
  // 维护导航：takeTable；下列为本函数源码中的接口与路径索引。
  // 输入 label：string。
  // 输入 rows：string[][]。
  // 返回契约：BoundedTableSelection。
  // 分支条件：rows.length === 0。
  // 分支条件：separator && !this.appendExact('')。
  // 分支条件：!this.appendExact(`${label}:`)。
  // 分支条件：tableSelection.truncated && this.reserveGlobalNotice && !this.noticeWritten。
  // 分支条件：selectedRows.length === 0。
  // 返回路径：{ text: '', records: [], rows: [], totalRecords: 0, omittedRecords: 0, truncated: false, visible: false, separator: false }。
  // 返回路径：this.omittedTable(rows, false)。
  // 返回路径：this.omittedTable(rows, true)。
  // 返回路径：{ text: selectedRecords.join('\n'), records: selectedRecords, rows: selectedRows, totalRecords: rows.length, omittedRecords, truncated, visible: true, separator, }。
  // 写入目标：tableSelection。
  // 写入目标：truncated。
  // 写入目标：this.totalRecords。
  // 写入目标：this.omittedRecords。
  // 写入目标：this.truncated。
  // 直接调用：this.snapshot。
  // 直接调用：this.appendExact。
  // 直接调用：this.omittedTable。
  // 直接调用：this.restore。
  // 直接调用：this.selectTableRows。
  // 直接调用：Math.max。
  // 直接调用：this.records.push。
  // 直接调用：selectedRecords.join。
  // [/maint-note:1b06b5712360]
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

  // [maint-note:c8c751a665e3]
  // 维护导航：takeNotice；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：BoundedSelection。
  // 返回路径：this.takePlain(text)。
  // 写入目标：this.noticeWritten。
  // 直接调用：this.takePlain。
  // [/maint-note:c8c751a665e3]
  takeNotice(text: string): BoundedSelection {
    this.noticeWritten = true
    return this.takePlain(text)
  }

  // [maint-note:18904fd6da09]
  // 维护导航：takeSectionNotice；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：BoundedSelection。
  // 返回路径：this.takePlain(text)。
  // 直接调用：this.takePlain。
  // [/maint-note:18904fd6da09]
  takeSectionNotice(text: string): BoundedSelection {
    return this.takePlain(text)
  }

  // [maint-note:beb42a33cbbc]
  // 维护导航：output；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 返回路径：this.outputRows.join('\n')。
  // 直接调用：this.outputRows.join。
  // [/maint-note:beb42a33cbbc]
  output(): string {
    return this.outputRows.join('\n')
  }

  // [maint-note:5c572a2bbec5]
  // 维护导航：selectRows；下列为本函数源码中的接口与路径索引。
  // 输入 records：readonly string[]。
  // 输入 allowPartial：boolean。
  // 输入 reserve：{ lines: number; chars: number }；缺省表达式 { lines: 0, chars: 0 }。
  // 返回契约：{ records: string[]; truncated: boolean }。
  // 分支条件：this.remainingLines() <= reserve.lines。
  // 分支条件：available < record.length。
  // 分支条件：allowPartial && available > 0。
  // 遍历边界：const record ← records。
  // 返回路径：{ records: selected, truncated }。
  // 写入目标：truncated。
  // 直接调用：this.remainingLines。
  // 直接调用：this.remainingChars。
  // 直接调用：truncateCharacters。
  // 直接调用：this.appendExact。
  // 直接调用：selected.push。
  // [/maint-note:5c572a2bbec5]
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

  // [maint-note:292cb345583a]
  // 维护导航：selectTableRows；下列为本函数源码中的接口与路径索引。
  // 输入 rows：readonly string[][]。
  // 输入 reserve：{ lines: number; chars: number }；缺省表达式 { lines: 0, chars: 0 }。
  // 返回契约：{ selectedRows: string[][]; selectedRecords: string[]; truncated: boolean }。
  // 分支条件：this.remainingLines() <= reserve.lines || this.remainingChars() - reserve.chars < record.length。
  // 遍历边界：const row ← rows。
  // 返回路径：{ selectedRows, selectedRecords, truncated }。
  // 写入目标：truncated。
  // 直接调用：serializeToolTableRow。
  // 直接调用：this.remainingLines。
  // 直接调用：this.remainingChars。
  // 直接调用：this.appendExact。
  // 直接调用：selectedRows.push。
  // 直接调用：selectedRecords.push。
  // [/maint-note:292cb345583a]
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

  // [maint-note:834adab110b2]
  // 维护导航：appendExact；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 返回契约：boolean。
  // 分支条件：this.remainingLines() <= 0 || this.remainingChars() < value.length。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.usedLines。
  // 写入目标：this.usedChars。
  // 直接调用：this.remainingLines。
  // 直接调用：this.remainingChars。
  // 直接调用：this.outputRows.push。
  // [/maint-note:834adab110b2]
  private appendExact(value: string): boolean {
    if (this.remainingLines() <= 0 || this.remainingChars() < value.length) return false
    this.outputRows.push(value)
    this.usedLines += 1
    this.usedChars += value.length + (this.outputRows.length > 1 ? 1 : 0)
    return true
  }

  // [maint-note:22921e538deb]
  // 维护导航：remainingChars；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.maxChars - this.usedChars - separator - notice。
  // 直接调用：Math.min。
  // 直接调用：Math.max。
  // [/maint-note:22921e538deb]
  private remainingChars(): number {
    const separator = this.outputRows.length > 0 ? 1 : 0
    const notice = this.reserveGlobalNotice && !this.noticeWritten
      ? Math.min(GLOBAL_TRUNCATION_NOTICE.length, Math.max(0, this.maxChars - this.usedChars)) + separator
      : 0
    return this.maxChars - this.usedChars - separator - notice
  }

  // [maint-note:c886000e7832]
  // 维护导航：remainingLines；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.maxLines - this.usedLines - notice。
  // [/maint-note:c886000e7832]
  private remainingLines(): number {
    const notice = this.reserveGlobalNotice && !this.noticeWritten ? 1 : 0
    return this.maxLines - this.usedLines - notice
  }

  // [maint-note:c91797da0bff]
  // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
  // 返回契约：{ rows: number; usedLines: number; usedChars: number }。
  // 返回路径：{ rows: this.outputRows.length, usedLines: this.usedLines, usedChars: this.usedChars }。
  // [/maint-note:c91797da0bff]
  private snapshot(): { rows: number; usedLines: number; usedChars: number } {
    return { rows: this.outputRows.length, usedLines: this.usedLines, usedChars: this.usedChars }
  }

  // [maint-note:6105bf4701db]
  // 维护导航：restore；下列为本函数源码中的接口与路径索引。
  // 输入 snapshot：{ rows: number; usedLines: number; usedChars: number }。
  // 返回契约：void。
  // 写入目标：this.outputRows.length。
  // 写入目标：this.usedLines。
  // 写入目标：this.usedChars。
  // [/maint-note:6105bf4701db]
  private restore(snapshot: { rows: number; usedLines: number; usedChars: number }): void {
    this.outputRows.length = snapshot.rows
    this.usedLines = snapshot.usedLines
    this.usedChars = snapshot.usedChars
  }

  // [maint-note:5918bb59f277]
  // 维护导航：omittedField；下列为本函数源码中的接口与路径索引。
  // 输入 records：readonly string[]。
  // 输入 truncated：boolean。
  // 返回契约：BoundedSelection。
  // 返回路径：{ text: '', records: [], totalRecords: records.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }。
  // 写入目标：this.truncated。
  // 写入目标：this.totalRecords。
  // 写入目标：this.omittedRecords。
  // [/maint-note:5918bb59f277]
  private omittedField(records: readonly string[], truncated: boolean): BoundedSelection {
    const omitted = records.length
    this.truncated ||= truncated || omitted > 0
    this.totalRecords += records.length
    this.omittedRecords += omitted
    return { text: '', records: [], totalRecords: records.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }
  }

  // [maint-note:682eb77e92eb]
  // 维护导航：omittedTable；下列为本函数源码中的接口与路径索引。
  // 输入 rows：readonly string[][]。
  // 输入 truncated：boolean。
  // 返回契约：BoundedTableSelection。
  // 返回路径：{ text: '', records: [], rows: [], totalRecords: rows.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }。
  // 写入目标：this.truncated。
  // 写入目标：this.totalRecords。
  // 写入目标：this.omittedRecords。
  // [/maint-note:682eb77e92eb]
  private omittedTable(rows: readonly string[][], truncated: boolean): BoundedTableSelection {
    const omitted = rows.length
    this.truncated ||= truncated || omitted > 0
    this.totalRecords += rows.length
    this.omittedRecords += omitted
    return { text: '', records: [], rows: [], totalRecords: rows.length, omittedRecords: omitted, truncated: true, visible: false, separator: false }
  }
}

// [maint-note:e8bb3cdc6807]
// 维护导航：appendSectionSpacer；下列为本函数源码中的接口与路径索引。
// 输入 blocks：ToolDisplayBlock[]。
// 输入 selection：BoundedSelection。
// 返回契约：void。
// 分支条件：selection.separator。
// 直接调用：blocks.push。
// [/maint-note:e8bb3cdc6807]
function appendSectionSpacer(blocks: ToolDisplayBlock[], selection: BoundedSelection): void {
  if (selection.separator) blocks.push({ kind: 'spacer' })
}

// [maint-note:d27a496305f9]
// 维护导航：appendSectionNotice；下列为本函数源码中的接口与路径索引。
// 输入 blocks：ToolDisplayBlock[]。
// 输入 budget：PresentationBudget。
// 输入 selection：BoundedSelection。
// 返回契约：void。
// 分支条件：!selection.truncated。
// 分支条件：notice.visible。
// 返回路径：无返回值。
// 直接调用：budget.takeSectionNotice。
// 直接调用：blocks.push。
// [/maint-note:d27a496305f9]
function appendSectionNotice(blocks: ToolDisplayBlock[], budget: PresentationBudget, selection: BoundedSelection): void {
  if (!selection.truncated) return
  const notice = budget.takeSectionNotice(SECTION_TRUNCATION_NOTICE)
  if (notice.visible) blocks.push({ kind: 'notice', text: notice.text })
}

// [maint-note:071ac8c17b1d]
// 维护导航：truncateCharacters；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxChars：number。
// 返回契约：string。
// 分支条件：value.length <= maxChars。
// 分支条件：maxChars <= 1。
// 分支条件：prefix.length + character.length + 1 > maxChars。
// 遍历边界：const character ← value。
// 返回路径：value。
// 返回路径：'…'.slice(0, maxChars)。
// 返回路径：`${prefix}…`.slice(0, maxChars)。
// 写入目标：prefix。
// 直接调用：'…'.slice。
// 直接调用：`${prefix}…`.slice。
// [/maint-note:071ac8c17b1d]
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

// [maint-note:57e67e9bd04f]
// 维护导航：hasExactKeys；下列为本函数源码中的接口与路径索引。
// 输入 value：Record<string, unknown>。
// 输入 keys：readonly string[]。
// 输入 optionalKeys：readonly string[]；缺省表达式 []。
// 返回契约：boolean。
// 返回路径：Object.keys(value).every((key) => expected.has(key)) && keys.every((key) => key in value)。
// 直接调用：Object.keys(value).every。
// 直接调用：Object.keys。
// 直接调用：keys.every。
// [/maint-note:57e67e9bd04f]
function hasExactKeys(value: Record<string, unknown>, keys: readonly string[], optionalKeys: readonly string[] = []): boolean {
  const expected = new Set([...keys, ...optionalKeys])
  return Object.keys(value).every((key) => expected.has(key)) && keys.every((key) => key in value)
}

// [maint-note:a36e9428b5d5]
// 维护导航：isPositiveInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is number。
// 返回路径：Number.isInteger(value) && (value as number) > 0。
// 直接调用：Number.isInteger。
// [/maint-note:a36e9428b5d5]
function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0
}

// [maint-note:fc7049068b6c]
// 维护导航：isNonNegativeInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is number。
// 返回路径：Number.isInteger(value) && (value as number) >= 0。
// 直接调用：Number.isInteger。
// [/maint-note:fc7049068b6c]
function isNonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0
}

// [maint-note:dd96480c90d7]
// 维护导航：isPositiveIntegerOrZero；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is number。
// 返回路径：Number.isInteger(value) && (value as number) >= 0。
// 直接调用：Number.isInteger。
// [/maint-note:dd96480c90d7]
function isPositiveIntegerOrZero(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0
}

// [maint-note:58ecf3e80aaf]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:58ecf3e80aaf]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
