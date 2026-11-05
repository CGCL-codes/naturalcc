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

export interface FrameLayout {
  readonly terminalRows: number
  readonly interactiveRows: number
  readonly reservedStaticRows: number
  readonly rendererSafetyRows: number
  /** Rows available to ActiveTurn, including its own dynamic chrome. */
  readonly activeRows: number
}

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

function countWrapped(value: string, width: number): number {
  return Math.max(1, wrapDisplayLines(value, Math.max(1, width), { hard: true, trim: false, wordWrap: false }).length)
}

function messageHeader(message: Message): string {
  if (message.role === 'tool') return `● ${message.toolName ?? 'tool'}`
  if (message.terminalKind === 'interrupted') return `[Interrupted] ${message.time}`
  if (message.terminalKind === 'paused') return `Paused ${message.time}`
  if (message.tone === 'error') return `[Error] ${message.time}`
  return message.time
}

function countMarkdownRows(content: string, width: number, draft = false): number {
  void draft
  const projection = renderMarkdown(content, { maxWidth: width })
  return countMarkdownBlockRows(projection.blocks, width)
}

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

export function countThinkingRows(loading: boolean, streaming: boolean): number {
  return loading ? (streaming ? 4 : 3) : 0 // content/status/Esc + margin
}

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
