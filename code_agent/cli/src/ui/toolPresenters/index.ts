const DEFAULT_MAX_PRESENTATION_CHARS = 8_000
const DEFAULT_MAX_PRESENTATION_LINES = 400
const GLOBAL_TRUNCATION_NOTICE = '… output truncated; details unavailable'

export type ToolDisplayBlock =
  | { kind: 'summary'; text: string }
  | { kind: 'field'; label: string; text: string; preformatted?: boolean; truncated?: boolean }
  | { kind: 'list'; label: string; items: string[]; truncated?: boolean }
  | { kind: 'table'; label: string; rows: string[][]; truncated?: boolean }
  | { kind: 'notice'; text: string }

export interface ToolResultDetailSource {
  records: readonly string[]
  totalRecords: number
  omittedRecords: number
  sourceTruncated: boolean
  viewCollapsed: false
  canExpand: boolean
}

export interface ToolResultPresentation {
  toolName: string
  summary: string
  blocks: readonly ToolDisplayBlock[]
  plainText: string
  truncated: boolean
  sourceTruncated: boolean
  viewCollapsed: false
  canExpand: boolean
  detail: ToolResultDetailSource
}

interface PresentationOptions {
  maxChars?: number
  maxLines?: number
  input?: unknown
  reserveGlobalNotice?: boolean
}

export function presentToolResult(
  toolName: string,
  output: string,
  options: PresentationOptions = {},
): ToolResultPresentation {
  const first = presentToolResultOnce(toolName, output, options)
  if (!first.sourceTruncated || options.reserveGlobalNotice) return first
  // The first pass is exact when the result fits. If it does not, rebuild
  // from the typed records with capacity reserved for a visible global
  // marker. This keeps blocks, detail and plainText on one ledger.
  return presentToolResultOnce(toolName, output, { ...options, reserveGlobalNotice: true })
}

function presentToolResultOnce(
  toolName: string,
  output: string,
  options: PresentationOptions,
): ToolResultPresentation {
  const maxChars = Math.max(1, Math.floor(options.maxChars ?? DEFAULT_MAX_PRESENTATION_CHARS))
  const maxLines = Math.max(1, Math.floor(options.maxLines ?? DEFAULT_MAX_PRESENTATION_LINES))
  const safeOutput = sanitizeToolText(output)
  let parsed: unknown
  try {
    // This is the one and only JSON decode in the UI projection. Assistant
    // content and preformatted tool fields are never passed through it.
    parsed = safeOutput ? JSON.parse(safeOutput) : undefined
  } catch {
    return fallbackPresentation(toolName, safeOutput, maxChars, maxLines, options.reserveGlobalNotice === true)
  }

  const presentation = toolName === 'read_file'
    ? presentReadFile(parsed, maxChars, maxLines, options.reserveGlobalNotice === true)
    : toolName === 'bash'
      ? presentBash(parsed, maxChars, maxLines, options.input, options.reserveGlobalNotice === true)
      : toolName === 'list_files'
        ? presentListFiles(parsed, maxChars, maxLines, options.reserveGlobalNotice === true)
        : toolName === 'search_text'
          ? presentSearchText(parsed, maxChars, maxLines, options.reserveGlobalNotice === true)
          : toolName === 'apply_patch'
            ? presentApplyPatch(parsed, maxChars, maxLines, options.reserveGlobalNotice === true)
            : undefined
  return presentation ?? fallbackPresentation(toolName, safeOutput, maxChars, maxLines, options.reserveGlobalNotice === true)
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

function presentReadFile(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
  if (!isRecord(value) || !hasExactKeys(value, ['path', 'startLine', 'endLine', 'content', 'truncated'])) return undefined
  if (typeof value.path !== 'string' || !isPositiveInteger(value.startLine) || !isPositiveInteger(value.endLine)) return undefined
  if (typeof value.content !== 'string' || typeof value.truncated !== 'boolean') return undefined
  const path = sanitizeToolText(value.path)
  const content = sanitizeToolText(value.content)
  const summary = `${path}:${value.startLine}-${value.endLine}${value.truncated ? ' (truncated)' : ''}`
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const summaryPart = budget.takePlain(summary)
  const contentPart = content ? budget.takeField('content', content) : undefined
  const blocks: ToolDisplayBlock[] = [{ kind: 'summary', text: summaryPart.text }]
  if (contentPart?.visible) {
    blocks.push({ kind: 'field', label: 'content', text: contentPart.text, preformatted: true, truncated: contentPart.truncated })
    appendSectionNotice(blocks, budget, contentPart)
  }
  return makePresentation('read_file', summaryPart.text, [
    ...blocks,
  ], budget.output(), value.truncated || Boolean(contentPart?.truncated), budget.records, budget.totalRecords, budget.omittedRecords, budget)
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
  const blocks: ToolDisplayBlock[] = [{ kind: 'summary', text: summaryPart.text }]
  let command: BoundedSelection | undefined
  if (isRecord(input) && typeof input.command === 'string' && input.command.trim()) {
    command = budget.takeField('command', sanitizeToolText(input.command))
    if (command.visible) {
      blocks.push({ kind: 'field', label: 'command', text: command.text, preformatted: true, truncated: command.truncated })
      appendSectionNotice(blocks, budget, command)
    }
  }
  const stdout = stdoutText ? budget.takeField('stdout', stdoutText) : undefined
  const stderr = stderrText ? budget.takeField('stderr', stderrText) : undefined
  if (stdout?.visible) {
    blocks.push({ kind: 'field', label: 'stdout', text: stdout.text, preformatted: true, truncated: stdout.truncated })
    appendSectionNotice(blocks, budget, stdout)
  }
  if (stderr?.visible) {
    blocks.push({ kind: 'field', label: 'stderr', text: stderr.text, preformatted: true, truncated: stderr.truncated })
    appendSectionNotice(blocks, budget, stderr)
  }
  return makePresentation(
    'bash', summaryPart.text, blocks,
    budget.output(), value.truncated || budget.truncated,
    budget.records, budget.totalRecords, budget.omittedRecords, budget,
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
  const blocks: ToolDisplayBlock[] = [
    { kind: 'summary', text: summaryPart.text },
  ]
  if (bounded.visible) {
    blocks.push({ kind: 'list', label: 'files', items: bounded.records, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('list_files', summaryPart.text, blocks,
    budget.output(), value.truncated || budget.truncated, budget.records, budget.totalRecords, budget.omittedRecords, budget)
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
  const blocks: ToolDisplayBlock[] = [
    { kind: 'summary', text: summaryPart.text },
  ]
  if (bounded.visible) {
    blocks.push({ kind: 'list', label: 'matches', items: bounded.records, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('search_text', summaryPart.text, blocks,
    budget.output(), value.truncated || budget.truncated, budget.records, budget.totalRecords, budget.omittedRecords, budget)
}

function presentApplyPatch(value: unknown, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation | undefined {
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
  const blocks: ToolDisplayBlock[] = [
    { kind: 'summary', text: summaryPart.text },
  ]
  if (bounded.visible) {
    blocks.push({ kind: 'table', label: 'changes', rows: bounded.rows, truncated: bounded.truncated })
    appendSectionNotice(blocks, budget, bounded)
  }
  return makePresentation('apply_patch', summaryPart.text, blocks,
    budget.output(), value.truncated || budget.truncated, budget.records, budget.totalRecords, budget.omittedRecords, budget)
}

function fallbackPresentation(toolName: string, output: string, maxChars: number, maxLines: number, reserveGlobalNotice: boolean): ToolResultPresentation {
  const budget = new PresentationBudget(maxChars, maxLines, reserveGlobalNotice)
  const candidate = output || 'completed'
  const summaryPart = budget.takePlain(candidate)
  const blocks: ToolDisplayBlock[] = summaryPart.visible
    ? [{ kind: 'summary', text: summaryPart.text }]
    : []
  return makePresentation(toolName, summaryPart.text, blocks, budget.output(), summaryPart.truncated, budget.records, budget.totalRecords, budget.omittedRecords, budget)
}

function makePresentation(
  toolName: string,
  summary: string,
  blocks: ToolDisplayBlock[],
  plainText: string,
  sourceTruncated: boolean,
  records: string[],
  totalRecords: number,
  omittedRecords: number,
  budget?: PresentationBudget,
): ToolResultPresentation {
  const safePlainText = sanitizeToolText(plainText)
  const effectiveSourceTruncated = sourceTruncated || Boolean(budget?.truncated)
  const renderedBlocks = [...blocks]
  if (effectiveSourceTruncated && budget) {
    const notice = budget.takeNotice(GLOBAL_TRUNCATION_NOTICE)
    if (notice.visible) renderedBlocks.push({ kind: 'notice', text: notice.text })
  }
  const renderedPlainText = budget?.output() ?? safePlainText
  return {
    toolName,
    summary: sanitizeToolText(summary),
    blocks: renderedBlocks,
    plainText: sanitizeToolText(renderedPlainText),
    truncated: effectiveSourceTruncated,
    sourceTruncated: effectiveSourceTruncated,
    viewCollapsed: false,
    canExpand: false,
    detail: {
      records,
      totalRecords,
      omittedRecords,
      sourceTruncated: effectiveSourceTruncated,
      viewCollapsed: false,
      canExpand: false,
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
    }
  }

  takeField(label: string, value: string): BoundedSelection {
    const allRecords = value ? value.split('\n') : []
    if (!this.appendExact(`${label}:`)) {
      this.truncated ||= allRecords.length > 0
      this.totalRecords += allRecords.length
      this.omittedRecords += allRecords.length
      return { text: '', records: [], totalRecords: allRecords.length, omittedRecords: allRecords.length, truncated: allRecords.length > 0, visible: false }
    }
    const selected = this.selectRows(allRecords, true)
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
    }
  }

  takeTable(label: string, rows: string[][]): BoundedTableSelection {
    if (!this.appendExact(`${label}:`)) {
      this.truncated ||= rows.length > 0
      this.totalRecords += rows.length
      this.omittedRecords += rows.length
      return { text: '', records: [], rows: [], totalRecords: rows.length, omittedRecords: rows.length, truncated: rows.length > 0, visible: false }
    }
    const selectedRows: string[][] = []
    const selectedRecords: string[] = []
    let truncated = false
    for (const row of rows) {
      const record = serializeToolTableRow(row)
      // Whole table rows are retained as typed rows. Do not split a row into
      // whitespace-delimited strings when the character budget is reached.
      if (!this.appendExact(record)) {
        truncated = true
        break
      }
      selectedRows.push(row)
      selectedRecords.push(record)
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

  private selectRows(records: readonly string[], allowPartial: boolean): { records: string[]; truncated: boolean } {
    const selected: string[] = []
    let truncated = false
    for (const record of records) {
      if (this.remainingLines() <= 0) {
        truncated = true
        break
      }
      const available = this.remainingChars()
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
}

function appendSectionNotice(blocks: ToolDisplayBlock[], budget: PresentationBudget, selection: BoundedSelection): void {
  if (!selection.truncated) return
  const notice = budget.takeSectionNotice('… section truncated')
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

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const expected = new Set(keys)
  return Object.keys(value).every((key) => expected.has(key)) && keys.every((key) => key in value)
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0
}

function isPositiveIntegerOrZero(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) >= 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
