import type { ApprovalPresentation, ApprovalRequest } from '../../tools/approval.js'
import { buildApprovalPresentation } from '../../tools/approval.js'
import type { Message } from '../types/message.js'
import { renderMarkdown } from '../markdown/renderer.js'
import { countMarkdownBlockRows, type MarkdownVisualRow } from '../components/MarkdownText.js'
import {
  calculateContentWidth,
  sanitizeTerminalText,
  truncateDisplayText,
  wrapDisplayLines,
} from '../terminal/index.js'
import { countDiffVisualLines } from '../presentation/applyPatchDiff.js'

// [maint-note:394a86a56ea9]
// 数据契约导航：FrameLayoutInput；字段含义以消费端校验为准。
// 成员 width：number。
// 成员 rows：number。
// 成员 footerVisible：boolean。
// 成员 inputVisible：boolean。
// 成员 showHelpHint（可省略）：boolean。
// 成员 clearing：boolean。
// 成员 exitArmed：boolean。
// 成员 dropdownOpen：boolean。
// 成员 reservedStaticRows（可省略）：number。
// 成员 rendererSafetyRows（可省略）：number。
// 成员 reservedOverlayRows（可省略）：number。
// [/maint-note:394a86a56ea9]
export interface FrameLayoutInput {
  width: number
  rows: number
  footerVisible: boolean
  inputVisible: boolean
  /** Current session generation has not accepted its first run yet. */
  showHelpHint?: boolean
  clearing: boolean
  exitArmed: boolean
  dropdownOpen: boolean
  /** Temporary physical reservation while the initial static Header is visible. */
  reservedStaticRows?: number
  /**
   * One-row slack after startup so Ink's mutable frame remains strictly
   * shorter than the terminal. Without it, appending a long Static batch
   * after an exact-height live frame makes Ink clear and replay all history.
   */
  rendererSafetyRows?: number
  /** physical rows of the detail-viewer overlay while it is open. */
  reservedOverlayRows?: number
}

// [maint-note:51b96bd8dc86]
// 数据契约导航：FrameLayout；字段含义以消费端校验为准。
// 成员 terminalRows：number。
// 成员 interactiveRows：number。
// 成员 reservedStaticRows：number。
// 成员 rendererSafetyRows：number。
// 成员 activeRows：number。
// [/maint-note:51b96bd8dc86]
export interface FrameLayout {
  readonly terminalRows: number
  readonly interactiveRows: number
  readonly reservedStaticRows: number
  readonly rendererSafetyRows: number
  /** Rows available to ActiveTurn, including its own dynamic chrome. */
  readonly activeRows: number
}

// [maint-note:86181fda9855]
// 维护导航：estimateInteractiveRows；下列为本函数源码中的接口与路径索引。
// 输入 input：Pick<FrameLayoutInput, 'width' | 'footerVisible' | 'inputVisible' | 'showHelpHint' | 'clearing' | 'exitArmed' | 'dropdownOpen'>。
// 返回契约：number。
// 分支条件：input.footerVisible。
// 分支条件：input.clearing。
// 分支条件：input.inputVisible。
// 分支条件：input.dropdownOpen。
// 返回路径：rows。
// 写入目标：rows。
// [/maint-note:86181fda9855]
export function estimateInteractiveRows(input: Pick<FrameLayoutInput, 'width' | 'footerVisible' | 'inputVisible' | 'showHelpHint' | 'clearing' | 'exitArmed' | 'dropdownOpen'>): number {
  // SessionStatusSlot is always mounted and warning is an in-place projection
  // of that same row, so exitArmed never adds another row.
  let rows = 1
  if (input.footerVisible) rows += 1
  if (input.clearing) rows += 1
  if (input.inputVisible) {
    rows += input.width <= 4
      ? 1
      : 3 + (input.showHelpHint === false ? 0 : 1) // optional help, divider, input, divider
    if (input.dropdownOpen) rows += 1 + 10 + 2 // margin, bounded options, possible arrows/divider
  }
  return rows
}

// [maint-note:cda5277dd7a4]
// 维护导航：calculateFrameLayout；下列为本函数源码中的接口与路径索引。
// 输入 input：FrameLayoutInput。
// 返回契约：FrameLayout。
// 返回路径：{ terminalRows, interactiveRows, reservedStaticRows, rendererSafetyRows, // Header is committed by StableHistory's static owner and is not part of // Ink's long-lived mutable frame ledger. This temporary reservation only // protects the ini。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：estimateInteractiveRows。
// [/maint-note:cda5277dd7a4]
export function calculateFrameLayout(input: FrameLayoutInput): FrameLayout {
  const terminalRows = Math.max(1, Math.floor(input.rows))
  const interactiveRows = estimateInteractiveRows(input)
  const reservedStaticRows = Math.max(0, Math.floor(input.reservedStaticRows ?? 0))
  const rendererSafetyRows = Math.max(0, Math.floor(input.rendererSafetyRows ?? 0))
  const reservedOverlayRows = Math.max(0, Math.floor(input.reservedOverlayRows ?? 0))
  return {
    terminalRows,
    interactiveRows,
    reservedStaticRows,
    rendererSafetyRows,
    // Header is committed by StableHistory's static owner and is not part of
    // Ink's long-lived mutable frame ledger. This temporary reservation only
    // protects the initial Header until the first non-header batch commits.
    // The detail-viewer overlay is a dynamic region panel: its physical
    // rows must leave the same ledger or long history would push it offscreen.
    activeRows: Math.max(
      1,
      terminalRows - interactiveRows - reservedStaticRows - rendererSafetyRows - reservedOverlayRows,
    ),
  }
}

// [maint-note:2f18a067768c]
// 维护导航：countWrapped；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 返回契约：number。
// 返回路径：Math.max(1, wrapDisplayLines(value, Math.max(1, width), { hard: true, trim: false, wordWrap: false }).length)。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines。
// [/maint-note:2f18a067768c]
function countWrapped(value: string, width: number): number {
  return Math.max(1, wrapDisplayLines(value, Math.max(1, width), { hard: true, trim: false, wordWrap: false }).length)
}

// [maint-note:568ad0dd0aab]
// 维护导航：messageHeader；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 返回契约：string。
// 分支条件：message.role === 'tool'。
// 分支条件：message.terminalKind === 'interrupted'。
// 分支条件：message.terminalKind === 'paused'。
// 分支条件：message.tone === 'error'。
// 返回路径：`● ${message.toolName ?? 'tool'}`。
// 返回路径：`[Interrupted] ${message.time}`。
// 返回路径：`Paused ${message.time}`。
// 返回路径：`[Error] ${message.time}`。
// 返回路径：message.time。
// [/maint-note:568ad0dd0aab]
function messageHeader(message: Message): string {
  if (message.role === 'tool') return `● ${message.toolName ?? 'tool'}`
  if (message.terminalKind === 'interrupted') return `[Interrupted] ${message.time}`
  if (message.terminalKind === 'paused') return `Paused ${message.time}`
  if (message.tone === 'error') return `[Error] ${message.time}`
  return message.time
}

// [maint-note:0eb8ec1a3f66]
// 维护导航：countMarkdownRows；下列为本函数源码中的接口与路径索引。
// 输入 content：string。
// 输入 width：number。
// 输入 draft：类型由调用方与函数体约束；缺省表达式 false。
// 返回契约：number。
// 返回路径：countMarkdownBlockRows(projection.blocks, width)。
// 直接调用：renderMarkdown。
// 直接调用：countMarkdownBlockRows。
// [/maint-note:0eb8ec1a3f66]
function countMarkdownRows(content: string, width: number, draft = false): number {
  void draft
  const projection = renderMarkdown(content, { maxWidth: width })
  return countMarkdownBlockRows(projection.blocks, width)
}

// [maint-note:1d073e39e27f]
// 维护导航：countToolRows；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 输入 width：number。
// 返回契约：number。
// 分支条件：!presentation。
// 分支条件：switch block.kind。
// 分支条件：case 'summary'。
// 分支条件：case 'notice'。
// 分支条件：case 'spacer'。
// 分支条件：case 'field'。
// 分支条件：case 'list'。
// 分支条件：case 'table'。
// 分支条件：case 'diff'。
// 遍历边界：const block ← presentation.blocks。
// 遍历边界：const item ← block.items。
// 遍历边界：const row ← block.rows。
// 返回路径：countWrapped(sanitizeTerminalText(message.content), width)。
// 返回路径：Math.max(1, rows)。
// 写入目标：rows。
// 直接调用：countWrapped。
// 直接调用：sanitizeTerminalText。
// 直接调用：row.join。
// 直接调用：countDiffVisualLines。
// 直接调用：Math.max。
// [/maint-note:1d073e39e27f]
function countToolRows(message: Message, width: number): number {
  const presentation = message.toolPresentation
  if (!presentation) return countWrapped(sanitizeTerminalText(message.content), width)
  let rows = 0
  for (const block of presentation.blocks) {
    switch (block.kind) {
      case 'summary':
      case 'notice':
        rows += countWrapped(block.text, width)
        break
      case 'spacer':
        rows += 1
        break
      case 'field':
        rows += countWrapped(`${block.label}:`, width) + countWrapped(block.text, width)
        break
      case 'list':
        rows += countWrapped(`${block.label}:`, width)
        for (const item of block.items) rows += countWrapped(item, width)
        break
      case 'table':
        rows += countWrapped(`${block.label}:`, width)
        for (const row of block.rows) rows += countWrapped(row.join(' │ '), width)
        break
      case 'diff':
        // the committed diff lines ARE the ledger; re-wrap each line
        // at the CURRENT width (the commit width may have been wider) so the
        // frame count and the ToolResultView render agree everywhere.
        rows += countDiffVisualLines(block.lines, width)
        break
    }
  }
  return Math.max(1, rows)
}

/** Count the shared Markdown visual rows, including the MessageColumn margin. */
// [maint-note:45416c3259b5]
// 维护导航：countMessageRows；下列为本函数源码中的接口与路径索引。
// 输入 message：Message。
// 输入 terminalWidth：number。
// 输入 assistantRows：readonly MarkdownVisualRow[]（可省略）。
// 返回契约：number。
// 返回路径：countWrapped(messageHeader(message), width) + body + 1。
// 直接调用：calculateContentWidth。
// 直接调用：countMarkdownRows。
// 直接调用：countToolRows。
// 直接调用：countWrapped。
// 直接调用：sanitizeTerminalText。
// 直接调用：messageHeader。
// [/maint-note:45416c3259b5]
export function countMessageRows(
  message: Message,
  terminalWidth: number,
  assistantRows?: readonly MarkdownVisualRow[],
): number {
  const width = calculateContentWidth({ terminalWidth, paddingRight: 4, minWidth: 1 })
  const body = message.role === 'assistant'
    ? assistantRows?.length ?? countMarkdownRows(message.content, width, message.isDraft === true)
    : message.role === 'tool'
      ? countToolRows(message, width)
      : countWrapped(`> ${sanitizeTerminalText(message.content)}`, width)
  return countWrapped(messageHeader(message), width) + body + 1
}

// [maint-note:ae7085115b0d]
// 维护导航：approvalRows；下列为本函数源码中的接口与路径索引。
// 输入 presentation：ApprovalPresentation。
// 输入 terminalWidth：number。
// 返回契约：number。
// 返回路径：title + 1 + countWrapped(`${presentation.instructionLabel}:`, width) + instruction + 1 + countWrapped('Reason:', width) + reason + 1 + options + help。
// 直接调用：calculateContentWidth。
// 直接调用：countWrapped。
// [/maint-note:ae7085115b0d]
export function approvalRows(presentation: ApprovalPresentation, terminalWidth: number): number {
  const width = calculateContentWidth({ terminalWidth, minWidth: 1 })
  const title = countWrapped(presentation.title, width)
  const instruction = countWrapped(presentation.instruction, width)
  const reason = countWrapped(presentation.reason, width)
  const options = countWrapped('> Yes', width) + countWrapped('  No', width)
  const help = countWrapped('Use ↑/↓ to select, Enter to confirm, Esc to interrupt.', width)
  // The section stack has one spacer before its first section, one between
  // the two sections, and one after the stack.
  return title + 1 + countWrapped(`${presentation.instructionLabel}:`, width) + instruction + 1 +
    countWrapped('Reason:', width) + reason + 1 + options + help
}

// [maint-note:319825bd6c35]
// 维护导航：fitTextToRows；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 输入 maxRows：number。
// 返回契约：string。
// 分支条件：lines.length <= maxRows。
// 分支条件：maxRows <= 1。
// 返回路径：value。
// 返回路径：truncateDisplayText(lines.join(' '), safeWidth)。
// 返回路径：[...lines.slice(0, head), marker, ...lines.slice(-tail)].join('\n')。
// 直接调用：Math.max。
// 直接调用：wrapDisplayLines。
// 直接调用：sanitizeTerminalText。
// 直接调用：truncateDisplayText。
// 直接调用：lines.join。
// 直接调用：Math.floor。
// 直接调用：[...lines.slice(0, head), ...lines.slice(-tail)].join。
// 直接调用：lines.slice。
// 直接调用：[...lines.slice(0, head), marker, ...lines.slice(-tail)].join。
// [/maint-note:319825bd6c35]
function fitTextToRows(value: string, width: number, maxRows: number): string {
  const safeWidth = Math.max(1, width)
  const lines = wrapDisplayLines(sanitizeTerminalText(value), safeWidth, { hard: true, trim: false, wordWrap: false })
  if (lines.length <= maxRows) return value
  if (maxRows <= 1) return truncateDisplayText(lines.join(' '), safeWidth)
  const head = Math.floor((maxRows - 1) / 2)
  const tail = maxRows - 1 - head
  const omittedLines = Math.max(1, lines.length - head - tail)
  const omittedChars = Math.max(1, [...value].length - [...lines.slice(0, head), ...lines.slice(-tail)].join('').length)
  const marker = truncateDisplayText(
    `… [${omittedChars} chars omitted; ${omittedLines} lines omitted]`,
    safeWidth,
  )
  return [...lines.slice(0, head), marker, ...lines.slice(-tail)].join('\n')
}

/** Keep approval details visible while making the prompt fit its active frame. */
// [maint-note:62d16ce67b3e]
// 维护导航：boundApprovalPresentation；下列为本函数源码中的接口与路径索引。
// 输入 request：ApprovalRequest。
// 输入 terminalWidth：number。
// 输入 maxRows：number。
// 返回契约：ApprovalPresentation。
// 分支条件：approvalRows(original, terminalWidth) <= maxRows。
// 分支条件：approvalRows(candidate, terminalWidth) <= maxRows || each === 0。
// 遍历边界：let attempt = 0; attempt < 32; attempt += 1。
// 返回路径：original。
// 返回路径：candidate。
// 写入目标：attempt。
// 写入目标：candidate。
// 写入目标：each。
// 直接调用：buildApprovalPresentation。
// 直接调用：approvalRows。
// 直接调用：calculateContentWidth。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：fitTextToRows。
// [/maint-note:62d16ce67b3e]
export function boundApprovalPresentation(request: ApprovalRequest, terminalWidth: number, maxRows: number): ApprovalPresentation {
  const original = buildApprovalPresentation(request)
  if (approvalRows(original, terminalWidth) <= maxRows) return original
  const width = calculateContentWidth({ terminalWidth, minWidth: 1 })
  const minimum = approvalRows({ ...original, instruction: '', reason: '' }, terminalWidth)
  let each = Math.max(0, Math.floor((maxRows - minimum) / 2))
  let candidate: ApprovalPresentation = original
  for (let attempt = 0; attempt < 32; attempt += 1) {
    candidate = {
      ...original,
      instruction: fitTextToRows(original.instruction, width, Math.max(1, each)),
      reason: fitTextToRows(original.reason, width, Math.max(1, each)),
    }
    if (approvalRows(candidate, terminalWidth) <= maxRows || each === 0) return candidate
    each -= 1
  }
  return candidate
}

// [maint-note:1c83487249a2]
// 维护导航：countThinkingRows；下列为本函数源码中的接口与路径索引。
// 输入 loading：boolean。
// 输入 streaming：boolean。
// 返回契约：number。
// 返回路径：loading ? (streaming ? 4 : 3) : 0。
// [/maint-note:1c83487249a2]
export function countThinkingRows(loading: boolean, streaming: boolean): number {
  return loading ? (streaming ? 4 : 3) : 0 // content/status/Esc + margin
}

// [maint-note:d89aa71600f3]
// 维护导航：countActiveChromeRows；下列为本函数源码中的接口与路径索引。
// 输入 input：{ loading: boolean streaming: boolean approval: ApprovalPresentation | null width: number }。
// 返回契约：number。
// 返回路径：thinking + approval。
// 直接调用：countThinkingRows。
// 直接调用：approvalRows。
// [/maint-note:d89aa71600f3]
export function countActiveChromeRows(input: {
  loading: boolean
  streaming: boolean
  approval: ApprovalPresentation | null
  width: number
}): number {
  const thinking = countThinkingRows(input.loading, input.streaming)
  const approval = input.approval ? approvalRows(input.approval, input.width) : 0
  return thinking + approval
}
