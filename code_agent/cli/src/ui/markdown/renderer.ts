import { lexer, type Token, type Tokens } from 'marked'
import { decodeHTML } from 'entities'
import { createHash } from 'node:crypto'
import { displayLineWidth, fillDisplayWidth, sanitizeTerminalText, truncateDisplayText } from '../terminal/index.js'
import { StreamingTerminalTextNormalizer } from '../terminal/text.js'
import { buildMarkdownTableRows } from './tableLayout.js'

export const MAX_MARKDOWN_SOURCE_CHARS = 100_000
export const MAX_MARKDOWN_CACHE_ENTRIES = 64
export const MAX_MARKDOWN_CACHE_CHARS = 300_000
const MARKDOWN_TRUNCATION_MARKER = '\n… [markdown truncated]\n'
const MARKDOWN_RETAINED_CONTENT_CHARS = MAX_MARKDOWN_SOURCE_CHARS - [...MARKDOWN_TRUNCATION_MARKER].length
const MARKDOWN_RETAINED_HEAD_CHARS = Math.ceil(MARKDOWN_RETAINED_CONTENT_CHARS / 2)
const MARKDOWN_RETAINED_TAIL_CHARS = Math.floor(MARKDOWN_RETAINED_CONTENT_CHARS / 2)

export type InlineKind = 'text' | 'strong' | 'emphasis' | 'code' | 'link' | 'deleted'

export interface InlineSegment {
  kind: InlineKind
  text: string
  href?: string
}

export interface MarkdownListItem {
  marker: string
  content: InlineSegment[]
  nested: MarkdownBlock[]
}

export type MarkdownTableAlignment = 'left' | 'center' | 'right'

export interface MarkdownTableBlock {
  kind: 'table'
  header: InlineSegment[][]
  rows: InlineSegment[][][]
  alignments: MarkdownTableAlignment[]
}

export type MarkdownBlock =
  | { kind: 'paragraph'; content: InlineSegment[] }
  | { kind: 'heading'; level: number; content: InlineSegment[] }
  | { kind: 'blockquote'; blocks: MarkdownBlock[] }
  | { kind: 'list'; ordered: boolean; items: MarkdownListItem[] }
  | { kind: 'code'; language?: string; lines: string[] }
  | MarkdownTableBlock
  | { kind: 'hr' }

export interface MarkdownProjection {
  blocks: MarkdownBlock[]
  plainText: string
  lines: string[]
  sourceChars: number
  previewOmittedChars: number
  sourceTruncated: boolean
  viewCollapsed: false
  canExpand: boolean
}

export interface MarkdownRenderOptions {
  maxWidth?: number
  maxSourceChars?: number
  cache?: MarkdownProjectionCache
}

export interface MarkdownSummaryProjection {
  /** Display-only Markdown assembled from complete lexer tokens. */
  content: string
  hiddenLines: number
}

/**
 * Build a bounded Assistant summary from the same `marked` token stream as
 * the terminal renderer. Whole tokens are admitted only when the final
 * body plus its detail notice fits both budgets. An oversized first block
 * becomes a literal notice instead of an unterminated fence/list/table.
 */
export function foldMarkdownSummary(
  source: string,
  maxLines: number,
  maxChars: number,
): MarkdownSummaryProjection | undefined {
  const safeSource = sanitizeMarkdownSource(source)
  const lineLimit = Math.max(1, Math.floor(maxLines))
  const charLimit = Math.max(1, Math.floor(maxChars))
  const totalLines = safeSource.split('\n').length
  if (totalLines <= lineLimit && safeSource.length <= charLimit) return undefined

  const tokens = lexer(safeSource, { gfm: true, breaks: true })
  let selected = ''
  let selectedLines = 0

  const build = (body: string, sourceLines: number): MarkdownSummaryProjection => {
    const hiddenLines = Math.max(1, totalLines - sourceLines)
    const notice = `… ${hiddenLines} lines hidden · Ctrl+O view details`
    const trimmed = body.replace(/\n+$/u, '')
    return {
      content: trimmed ? `${trimmed}\n\n${notice}` : notice,
      hiddenLines,
    }
  }

  for (const token of tokens) {
    const raw = token.raw ?? ''
    if (!raw) continue
    const next = selected + raw
    const normalized = next.replace(/\n+$/u, '')
    const nextLines = normalized ? normalized.split('\n').length : 0
    const candidate = build(next, nextLines)
    if (candidate.content.length > charLimit || candidate.content.split('\n').length > lineLimit) break
    selected = next
    selectedLines = nextLines
  }

  const projection = build(selected, selectedLines)
  // The fixed notice is deliberately short, but keep this helper total if a
  // caller supplies an unusually tiny budget.
  if (projection.content.length <= charLimit && projection.content.split('\n').length <= lineLimit) return projection
  return {
    content: projection.content.slice(0, charLimit),
    hiddenLines: projection.hiddenLines,
  }
}

export interface MarkdownCacheStats {
  hits: number
  misses: number
  parses: number
  lexedChars: number
  entries: number
  chars: number
  retainedSourceChars: number
  retainedBlockChars: number
}

/** A bounded cache of structured projections, never a cache of ANSI strings. */
export class MarkdownProjectionCache {
  private readonly values = new Map<string, { projection: MarkdownProjection; chars: number; source: string }>()
  private totalChars = 0
  private retainedSourceChars = 0
  private retainedBlockChars = 0
  private hits = 0
  private misses = 0
  private parses = 0
  private lexedChars = 0

  constructor(
    private readonly maxEntries = MAX_MARKDOWN_CACHE_ENTRIES,
    private readonly maxChars = MAX_MARKDOWN_CACHE_CHARS,
  ) {}

  get(key: string, source?: string): MarkdownProjection | undefined {
    const found = this.values.get(key)
    if (!found || (source !== undefined && found.source !== source)) {
      this.misses += 1
      return undefined
    }
    this.values.delete(key)
    this.values.set(key, found)
    this.hits += 1
    return found.projection
  }

  set(key: string, projection: MarkdownProjection, source = ''): void {
    const blockChars = estimateBlocks(projection.blocks)
    const chars = key.length + source.length + projection.plainText.length + blockChars
    const existing = this.values.get(key)
    if (existing) {
      this.totalChars -= existing.chars
      this.retainedSourceChars -= existing.source.length
      this.retainedBlockChars -= estimateBlocks(existing.projection.blocks)
    }
    this.values.delete(key)
    if (chars > this.maxChars) return
    this.values.set(key, { projection, chars, source })
    this.totalChars += chars
    this.retainedSourceChars += source.length
    this.retainedBlockChars += blockChars
    while (this.values.size > this.maxEntries || this.totalChars > this.maxChars) {
      const oldest = this.values.keys().next().value as string | undefined
      if (oldest === undefined) break
      const item = this.values.get(oldest)
      this.values.delete(oldest)
      if (item) {
        this.totalChars -= item.chars
        this.retainedSourceChars -= item.source.length
        this.retainedBlockChars -= estimateBlocks(item.projection.blocks)
      }
    }
  }

  markParse(lexedChars = 0): void {
    this.parses += 1
    this.lexedChars += Math.max(0, Math.floor(lexedChars))
  }

  stats(): MarkdownCacheStats {
    return {
      hits: this.hits,
      misses: this.misses,
      parses: this.parses,
      lexedChars: this.lexedChars,
      entries: this.values.size,
      chars: this.totalChars,
      retainedSourceChars: this.retainedSourceChars,
      retainedBlockChars: this.retainedBlockChars,
    }
  }

  clear(): void {
    this.values.clear()
    this.totalChars = 0
    this.retainedSourceChars = 0
    this.retainedBlockChars = 0
    this.hits = 0
    this.misses = 0
    this.parses = 0
    this.lexedChars = 0
  }
}

const defaultCache = new MarkdownProjectionCache()

export function renderMarkdown(
  source: string,
  options: MarkdownRenderOptions = {},
): MarkdownProjection {
  const maxWidth = clampWidth(options.maxWidth ?? 80)
  const maxSourceChars = Math.min(MAX_MARKDOWN_SOURCE_CHARS, Math.max(1, Math.floor(options.maxSourceChars ?? MAX_MARKDOWN_SOURCE_CHARS)))
  const safeSource = sanitizeMarkdownSource(source)
  const sourceChars = [...safeSource].length
  const bounded = boundSource(safeSource, maxSourceChars)
  const cache = options.cache ?? defaultCache
  // Keep the sanitized canonical source intact. Reference-style Markdown is
  // downgraded at the token adapter layer so code/fence contents remain byte
  // for byte faithful to the model response.
  const parserSource = bounded.text
  const key = `md-v4\u0000${maxWidth}\u0000${maxSourceChars}\u0000${sourceChars}\u0000plain\u0000${contentDigest(parserSource)}`
  const cached = cache.get(key, bounded.text)
  if (cached) return cached

  cache.markParse(parserSource.length)
  let blocks: MarkdownBlock[]
  try {
    blocks = parseBlocks(parserSource, maxWidth)
  } catch {
    blocks = [{ kind: 'paragraph', content: [{ kind: 'text', text: bounded.text }] }]
  }
  const projection = createProjection(blocks, sourceChars, maxWidth, bounded.truncated)
  cache.set(key, projection, bounded.text)
  return projection
}

/**
 * Incremental owner for one Assistant response. The terminal normalizer sees
 * each appended chunk exactly once, while every visible snapshot is parsed
 * from the complete bounded source retained here. A snapshot never calls
 * `finish()`: unfinished terminal/control state remains pending until the
 * real response boundary.
 */
export class StreamingMarkdownRenderer {
  private readonly sourceChunks: string[] = []
  private sourceChars = 0
  private retainedSourceChars = 0
  private canonicalTruncated = false
  private retainedHead = ''
  private retainedTail = ''
  private readonly sourceNormalizer = new StreamingTerminalTextNormalizer()
  private finalized = false
  private currentProjection: MarkdownProjection | null = null

  constructor(
    private readonly options: MarkdownRenderOptions = {},
  ) {}

  /** Append a canonical delta without forcing a display refresh/parse. */
  append(delta: string): void {
    if (this.finalized) throw new Error('StreamingMarkdownRenderer cannot push after complete()')
    const safeDelta = normalizeMarkdownCodePoints(this.sourceNormalizer.push(delta))
    if (safeDelta) {
      this.appendNormalized(safeDelta)
      this.currentProjection = null
    }
  }

  /** Parse the current bounded source without treating it as EOF. */
  snapshot(): MarkdownProjection {
    if (this.currentProjection === null) this.currentProjection = this.renderCurrentSource()
    return this.currentProjection
  }

  push(delta: string): MarkdownProjection {
    this.append(delta)
    return this.snapshot()
  }

  complete(): MarkdownProjection {
    if (!this.finalized) {
      const normalizedTail = normalizeMarkdownCodePoints(this.sourceNormalizer.finish())
      if (normalizedTail) {
        this.appendNormalized(normalizedTail)
        this.currentProjection = null
      }
      this.finalized = true
    }
    if (this.currentProjection === null) this.currentProjection = this.renderCurrentSource()
    return this.currentProjection
  }

  reset(): void {
    this.sourceChunks.length = 0
    this.sourceChars = 0
    this.retainedSourceChars = 0
    this.canonicalTruncated = false
    this.retainedHead = ''
    this.retainedTail = ''
    this.currentProjection = null
    this.sourceNormalizer.reset()
    this.finalized = false
  }

  get sourceText(): string {
    if (this.canonicalTruncated) {
      return `${this.retainedHead}${MARKDOWN_TRUNCATION_MARKER}${this.retainedTail}`
    }
    return this.sourceChunks.join('')
  }

  get stablePrefix(): MarkdownProjection {
    return emptyMarkdownProjection()
  }

  private appendNormalized(safeDelta: string): void {
    const deltaChars = [...safeDelta]
    this.sourceChars += deltaChars.length
    if (!this.canonicalTruncated && this.sourceChars <= MAX_MARKDOWN_SOURCE_CHARS) {
      this.sourceChunks.push(safeDelta)
      this.retainedSourceChars += deltaChars.length
    } else if (!this.canonicalTruncated) {
      const previous = this.sourceChunks.join('')
      const bounded = boundSource(`${previous}${safeDelta}`, MAX_MARKDOWN_SOURCE_CHARS)
      const boundedChars = [...bounded.text]
      this.retainedHead = boundedChars.slice(0, MARKDOWN_RETAINED_HEAD_CHARS).join('')
      this.retainedTail = boundedChars.slice(-MARKDOWN_RETAINED_TAIL_CHARS).join('')
      this.sourceChunks.length = 0
      this.retainedSourceChars = MAX_MARKDOWN_SOURCE_CHARS
      this.canonicalTruncated = true
    } else {
      this.retainedTail = takeLastCodePoints(this.retainedTail, deltaChars, MARKDOWN_RETAINED_TAIL_CHARS)
    }
  }

  private renderCurrentSource(): MarkdownProjection {
    const projection = renderMarkdown(this.sourceText, this.options)
    if (!this.canonicalTruncated && this.sourceChars === projection.sourceChars) return projection
    return {
      ...projection,
      sourceChars: this.sourceChars,
      previewOmittedChars: 0,
      sourceTruncated: projection.sourceTruncated || this.canonicalTruncated,
      canExpand: false,
    }
  }
}

function createProjection(
  blocks: MarkdownBlock[],
  sourceChars: number,
  maxWidth: number,
  sourceTruncated: boolean,
  previewOmittedChars = 0,
): MarkdownProjection {
  const plainText = boundProjectionText(blocks, maxWidth)
  return {
    blocks,
    plainText,
    lines: plainText.split('\n'),
    sourceChars,
    previewOmittedChars,
    sourceTruncated,
    viewCollapsed: false,
    canExpand: false,
  }
}

function parseBlocks(source: string, maxWidth: number): MarkdownBlock[] {
  const tokens = lexer(source, { gfm: true, breaks: true })
  return tokens.flatMap((token) => parseBlock(token, maxWidth))
}

function parseBlock(token: Token, maxWidth: number): MarkdownBlock[] {
  switch (token.type) {
    case 'space':
      return []
    case 'heading':
      return [{ kind: 'heading', level: clampLevel(token.depth), content: inlineTokens(nestedInlineTokens(token), maxWidth) }]
    case 'paragraph':
    case 'text':
      return [{ kind: 'paragraph', content: inlineTokens('tokens' in token && token.tokens ? token.tokens : [token], maxWidth) }]
    case 'code':
      return [{
        kind: 'code',
        language: safeLanguage(token.lang),
        // Keep source lines independent of the current terminal width. Width
        // is a projection concern; truncating here would make the structured
        // block look complete while silently losing code that could be
        // re-rendered after a resize.
        // marked exposes fenced/indented code as raw source. Entity decoding
        // belongs to the inline token adapter only; decoding here would turn
        // literal code such as `&amp;` into a different program.
        lines: safeRawDisplayText(token.text).split('\n'),
      }]
    case 'blockquote':
      return [{ kind: 'blockquote', blocks: (token.tokens ?? []).flatMap((item: Token) => parseBlock(item, maxWidth)) }]
    case 'list':
      return [{
        kind: 'list',
        ordered: token.ordered,
        items: token.items.map((item: Tokens.ListItem, index: number) => parseListItem(item, token.ordered, token.start, index, maxWidth)),
      }]
    case 'table':
      return [{
        kind: 'table',
        header: token.header.map((cell: Tokens.TableCell) => inlineTokens(cell.tokens, maxWidth)),
        rows: token.rows.map((row: Tokens.TableCell[]) => row.map((cell: Tokens.TableCell) => inlineTokens(cell.tokens, maxWidth))),
        alignments: token.align.map((alignment: 'center' | 'left' | 'right' | null) => alignment ?? 'left'),
      }]
    case 'hr':
      return [{ kind: 'hr' }]
    case 'html':
      {
        const visible = safeDisplayText(stripHtml(token.text))
        return visible ? [{ kind: 'paragraph', content: [{ kind: 'text', text: visible }] }] : []
      }
    default:
      return []
  }
}

function parseListItem(
  item: Tokens.ListItem,
  ordered: boolean,
  start: number | '',
  index: number,
  maxWidth: number,
): MarkdownListItem {
  const marker = ordered ? `${typeof start === 'number' ? start + index : index + 1}.` : '•'
  const nestedTokens = item.tokens.filter((token) => token.type === 'list')
  const contentTokens = item.tokens.filter((token) => token.type !== 'list')
  return {
    marker,
    content: inlineTokens(contentTokens, maxWidth),
    nested: nestedTokens.flatMap((token) => parseBlock(token, maxWidth)),
  }
}

function inlineTokens(tokens: readonly Token[], maxWidth: number): InlineSegment[] {
  const result: InlineSegment[] = []
  for (const token of tokens) {
    switch (token.type) {
      case 'text':
      case 'escape':
        if ('tokens' in token && Array.isArray(token.tokens)) appendPreserving(result, inlineTokens(token.tokens, maxWidth))
        else appendSegment(result, { kind: 'text', text: safeDisplayText(token.text) })
        break
      case 'strong':
        appendSegments(result, inlineTokens(nestedInlineTokens(token), maxWidth), 'strong')
        break
      case 'em':
        appendSegments(result, inlineTokens(nestedInlineTokens(token), maxWidth), 'emphasis')
        break
      case 'codespan':
        appendSegment(result, { kind: 'code', text: safeDisplayText(token.text) })
        break
      case 'del':
        appendSegments(result, inlineTokens(nestedInlineTokens(token), maxWidth), 'deleted')
        break
      case 'link': {
        if (!isExplicitInlineLink(token.raw)) {
          appendSegment(result, { kind: 'text', text: safeDisplayText(token.raw || token.text) })
          break
        }
        const label = inlineTokens(nestedInlineTokens(token), maxWidth)
        // marked classifies the destination before this adapter sees it, but
        // leaves character references in href/title. Decode exactly once,
        // then apply the scheme policy to the decoded destination.
        const href = decodeMarkdownEntities(token.href)
        const title = typeof token.title === 'string' ? decodeMarkdownEntities(token.title) : undefined
        if (isSafeLink(href)) {
          appendSegments(result, label, 'link', safeRawDisplayText(href))
          const titleSuffix = title ? ` "${safeRawDisplayText(title)}"` : ''
          appendSegment(result, { kind: 'text', text: ` (${safeRawDisplayText(href)}${titleSuffix})` })
        } else {
          appendSegments(result, label, 'text')
        }
        break
      }
      case 'image':
        if (!isExplicitInlineLink(token.raw)) {
          appendSegment(result, { kind: 'text', text: safeDisplayText(token.raw || token.text) })
        } else {
          appendSegment(result, { kind: 'text', text: `[image: ${safeDisplayText(token.text)}]` })
        }
        break
      case 'br':
        appendSegment(result, { kind: 'text', text: '\n' })
        break
      case 'html':
        appendSegment(result, { kind: 'text', text: safeDisplayText(stripHtml(token.text)) })
        break
      default:
        if ('text' in token && typeof token.text === 'string') {
          appendSegment(result, { kind: 'text', text: safeDisplayText(token.text) })
        }
    }
  }
  return result
}

function nestedInlineTokens(token: Token): readonly Token[] {
  if ('tokens' in token && Array.isArray(token.tokens)) return token.tokens
  const text = 'text' in token && typeof token.text === 'string' ? token.text : ''
  return [{ type: 'text', raw: text, text }]
}

function appendSegments(target: InlineSegment[], segments: InlineSegment[], kind: InlineKind, href?: string): void {
  for (const segment of segments) {
    appendSegment(target, {
      ...segment,
      kind,
      href: kind === 'link' ? href : undefined,
    })
  }
}

function appendPreserving(target: InlineSegment[], segments: InlineSegment[]): void {
  for (const segment of segments) appendSegment(target, segment)
}

function appendSegment(target: InlineSegment[], segment: InlineSegment): void {
  if (!segment.text) return
  const previous = target.at(-1)
  if (previous && previous.kind === segment.kind && previous.href === segment.href) {
    previous.text += segment.text
  } else {
    target.push(segment)
  }
}

function boundProjectionText(blocks: readonly MarkdownBlock[], maxWidth: number): string {
  const lines: string[] = []
  for (const block of blocks) appendBlockText(lines, block, maxWidth, '')
  return lines.map((line) => boundLine(line, maxWidth)).join('\n')
}

function appendBlockText(lines: string[], block: MarkdownBlock, maxWidth: number, prefix: string): void {
  switch (block.kind) {
    case 'paragraph':
    case 'heading':
      appendMultiline(lines, `${prefix}${inlineText(block.content)}`)
      break
    case 'blockquote':
      for (const nested of block.blocks) appendBlockText(lines, nested, maxWidth, `${prefix}│ `)
      break
    case 'list':
      for (const item of block.items) {
        appendMultiline(lines, `${prefix}${item.marker} ${inlineText(item.content)}`)
        for (const nested of item.nested) appendBlockText(lines, nested, maxWidth, `${prefix}  `)
      }
      break
    case 'code':
      if (block.language) lines.push(`${prefix}[${block.language}]`)
      for (const line of block.lines) lines.push(`${prefix}${line}`)
      break
    case 'table':
      for (const row of buildMarkdownTableRows(block, Math.max(1, maxWidth - displayLineWidth(prefix)))) {
        lines.push(`${prefix}${inlineText(row.segments)}`)
      }
      break
    case 'hr':
      lines.push(`${prefix}${fillDisplayWidth('─', maxWidth)}`)
      break
  }
}

function appendMultiline(lines: string[], text: string): void {
  const parts = text.split('\n')
  if (parts.length === 0) lines.push('')
  else lines.push(...parts)
}

function inlineText(segments: readonly InlineSegment[]): string {
  return segments.map((segment) => segment.text).join('')
}

function boundLine(value: string, maxWidth: number): string {
  return truncateDisplayText(value, maxWidth)
}

function emptyMarkdownProjection(): MarkdownProjection {
  return {
    blocks: [],
    plainText: '',
    lines: [''],
    sourceChars: 0,
    previewOmittedChars: 0,
    sourceTruncated: false,
    viewCollapsed: false,
    canExpand: false,
  }
}

function estimateBlocks(blocks: readonly MarkdownBlock[]): number {
  let total = 0
  for (const block of blocks) {
    switch (block.kind) {
      case 'paragraph':
      case 'heading':
        total += block.content.reduce((sum, segment) => sum + segment.text.length + (segment.href?.length ?? 0), 0)
        break
      case 'blockquote':
        total += estimateBlocks(block.blocks)
        break
      case 'list':
        total += block.items.reduce((sum, item) => sum + item.marker.length + item.content.reduce((inner, segment) => inner + segment.text.length, 0) + estimateBlocks(item.nested), 0)
        break
      case 'code':
        total += (block.language?.length ?? 0) + block.lines.reduce((sum, line) => sum + line.length, 0)
        break
      case 'table':
        total += block.header.flat().reduce((sum, segment) => sum + segment.text.length, 0)
        total += block.rows.flat(2).reduce((sum, segment) => sum + segment.text.length, 0)
        break
      case 'hr':
        total += 1
        break
    }
  }
  return total
}

function boundSource(value: string, maxChars: number): { text: string; truncated: boolean } {
  const chars = [...value]
  if (chars.length <= maxChars) return { text: value, truncated: false }
  const marker = MARKDOWN_TRUNCATION_MARKER
  const budget = Math.max(0, maxChars - [...marker].length)
  const head = Math.ceil(budget / 2)
  const tail = Math.floor(budget / 2)
  return {
    text: `${chars.slice(0, head).join('')}${marker}${chars.slice(-tail).join('')}`,
    truncated: true,
  }
}

function takeLastCodePoints(previous: string, delta: readonly string[], limit: number): string {
  if (delta.length >= limit) return delta.slice(-limit).join('')
  return [...previous, ...delta].slice(-limit).join('')
}

function sanitizeMarkdownSource(value: string): string {
  return normalizeMarkdownCodePoints(sanitizeTerminalText(value))
}

function normalizeMarkdownCodePoints(value: string): string {
  return value
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

function contentDigest(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex')
}

function safeDisplayText(value: string): string {
  return sanitizeMarkdownSource(decodeMarkdownEntities(value))
}

function safeRawDisplayText(value: string): string {
  return sanitizeMarkdownSource(value)
}

/** Decode marked's HTML-renderer-facing token text once, after tokenization. */
function decodeMarkdownEntities(value: string): string {
  return decodeHTML(value)
}

function stripHtml(value: string): string {
  const visible: string[] = []
  let index = 0
  let rawTextTag: string | undefined
  while (index < value.length) {
    if (rawTextTag) {
      const closingTag = findRawTextClosingTag(value, index, rawTextTag)
      if (closingTag < 0) {
        visible.push(value.slice(index))
        break
      }
      visible.push(value.slice(index, closingTag))
      const closingMarkup = findHtmlMarkupAt(value, closingTag)
      if (closingMarkup?.closing && closingMarkup.tagName === rawTextTag) {
        index = closingMarkup.end + 1
        rawTextTag = undefined
        continue
      }
      visible.push(value[closingTag] ?? '')
      index = closingTag + 1
      continue
    }

    const markup = value[index] === '<' ? findHtmlMarkupAt(value, index) : undefined
    if (markup) {
      if (!markup.closing && !markup.selfClosing && (markup.tagName === 'script' || markup.tagName === 'style')) {
        rawTextTag = markup.tagName
      }
      index = markup.end + 1
      continue
    }

    visible.push(value[index] ?? '')
    index += 1
  }
  return visible.join('')
}

function isSafeLink(value: string): boolean {
  return /^(?:https?:|mailto:)/iu.test(value.trim())
}

function isExplicitInlineLink(raw: string): boolean {
  const opening = raw.startsWith('![') ? 1 : raw.startsWith('[') ? 0 : -1
  if (opening < 0) return false
  let nestedBrackets = 0
  let escaped = false
  for (let index = opening + 1; index < raw.length; index += 1) {
    const character = raw[index]
    if (escaped) {
      escaped = false
      continue
    }
    if (character === '\\') {
      escaped = true
      continue
    }
    if (character === '`') {
      const runLength = countRun(raw, index, '`')
      const codeEnd = findCodeSpanEnd(raw, index + runLength, runLength)
      if (codeEnd >= 0) {
        index = codeEnd - 1
        continue
      }
    }
    if (character === '<') {
      const htmlEnd = findHtmlTagEnd(raw, index + 1)
      if (htmlEnd >= 0) {
        index = htmlEnd
        continue
      }
    }
    if (character === '[') {
      nestedBrackets += 1
      continue
    }
    if (character !== ']') continue
    if (nestedBrackets > 0) {
      nestedBrackets -= 1
      continue
    }
    return raw[index + 1] === '('
  }
  return false
}

function countRun(value: string, start: number, character: string): number {
  let end = start
  while (value[end] === character) end += 1
  return end - start
}

function findCodeSpanEnd(value: string, start: number, runLength: number): number {
  let index = start
  while (index < value.length) {
    if (value[index] !== '`') {
      index += 1
      continue
    }
    const length = countRun(value, index, '`')
    if (length === runLength) return index + length
    index += length
  }
  return -1
}

function findHtmlTagEnd(value: string, start: number): number {
  let quote = ''
  for (let index = start; index < value.length; index += 1) {
    const character = value[index]
    if (quote) {
      if (character === quote) quote = ''
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '<') return -1
    if (character === '>') return index
  }
  return -1
}

function skipHtmlWhitespace(value: string, start: number): number {
  let index = start
  while (index < value.length && /\s/u.test(value[index] ?? '')) index += 1
  return index
}

function findHtmlClosingTagEnd(value: string, start: number): number {
  const end = skipHtmlWhitespace(value, start)
  return value[end] === '>' ? end : -1
}

function findHtmlOpeningTagEnd(
  value: string,
  start: number,
): { end: number; selfClosing: boolean } | undefined {
  let index = start

  while (index < value.length) {
    index = skipHtmlWhitespace(value, index)
    if (value[index] === '>') return { end: index, selfClosing: false }
    if (value[index] === '/' && value[index + 1] === '>') {
      return { end: index + 1, selfClosing: true }
    }

    const attribute = /^[A-Za-z_:][A-Za-z0-9_.:-]*/u.exec(value.slice(index))
    if (!attribute) return undefined
    index += attribute[0].length
    index = skipHtmlWhitespace(value, index)

    if (value[index] !== '=') continue
    index = skipHtmlWhitespace(value, index + 1)

    const quote = value[index]
    if (quote === '"' || quote === "'") {
      index += 1
      while (index < value.length && value[index] !== quote) index += 1
      if (value[index] !== quote) return undefined
      index += 1
      continue
    }

    const valueStart = index
    while (
      index < value.length &&
      !/\s|>/u.test(value[index] ?? '') &&
      value[index] !== '<' &&
      value[index] !== '"' &&
      value[index] !== "'" &&
      value[index] !== '`'
    ) {
      index += 1
    }
    if (index === valueStart) return undefined
  }

  return undefined
}

interface HtmlMarkup {
  end: number
  tagName?: string
  closing: boolean
  selfClosing: boolean
}

/**
 * Recognize markup syntax before dropping it. A bare '<' is ordinary model
 * text (often a comparison operator), so it must not enter the tag scanner.
 */
function findHtmlMarkupAt(value: string, start: number): HtmlMarkup | undefined {
  if (value[start] !== '<') return undefined

  if (value.startsWith('<!--', start)) {
    const commentEnd = value.indexOf('-->', start + 4)
    return commentEnd >= 0
      ? { end: commentEnd + 2, closing: false, selfClosing: false }
      : undefined
  }

  const next = value[start + 1]
  if (next === '!') {
    if (/^<!doctype(?:\s|>)/iu.test(value.slice(start))) {
      const declarationEnd = findHtmlTagEnd(value, start + 2)
      return declarationEnd >= 0
        ? { end: declarationEnd, closing: false, selfClosing: false }
        : undefined
    }
    if (value.startsWith('<![CDATA[', start)) {
      const cdataEnd = value.indexOf(']]>', start + '<![CDATA['.length)
      return cdataEnd >= 0
        ? { end: cdataEnd + 2, closing: false, selfClosing: false }
        : undefined
    }
    return undefined
  }
  if (next === '?') {
    const instructionEnd = value.indexOf('?>', start + 2)
    return instructionEnd >= 0
      ? { end: instructionEnd + 1, closing: false, selfClosing: false }
      : undefined
  }

  const closing = next === '/'
  const nameStart = start + (closing ? 2 : 1)
  const nameMatch = /^[A-Za-z][A-Za-z0-9:-]*/u.exec(value.slice(nameStart))
  if (!nameMatch) return undefined
  const tagName = nameMatch[0]?.toLowerCase()
  if (!tagName) return undefined

  const bodyStart = nameStart + nameMatch[0].length
  if (closing) {
    const end = findHtmlClosingTagEnd(value, bodyStart)
    if (end < 0) return undefined
    return { end, tagName, closing: true, selfClosing: false }
  }

  const opening = findHtmlOpeningTagEnd(value, bodyStart)
  if (!opening) return undefined
  return {
    end: opening.end,
    tagName,
    closing: false,
    selfClosing: opening.selfClosing,
  }
}

function findRawTextClosingTag(value: string, start: number, tagName: string): number {
  let candidate = value.indexOf('</', start)
  while (candidate >= 0) {
    const markup = findHtmlMarkupAt(value, candidate)
    if (markup?.closing && markup.tagName === tagName) return candidate
    candidate = value.indexOf('</', candidate + 2)
  }
  return -1
}

function safeLanguage(value: string | undefined): string | undefined {
  if (!value) return undefined
  const safe = value.replace(/[^a-z0-9_+.#-]/giu, '').slice(0, 24)
  return safe || undefined
}

function clampLevel(value: number): number {
  return Math.max(1, Math.min(6, value))
}

function clampWidth(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}
