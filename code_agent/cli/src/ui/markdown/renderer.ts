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

// [maint-note:c40cd25933ea]
// 数据契约导航：InlineKind；字段含义以消费端校验为准。
// 类型组合：'text' | 'strong' | 'emphasis' | 'code' | 'link' | 'deleted'。
// [/maint-note:c40cd25933ea]
export type InlineKind = 'text' | 'strong' | 'emphasis' | 'code' | 'link' | 'deleted'

// [maint-note:d2153d0a8699]
// 数据契约导航：InlineSegment；字段含义以消费端校验为准。
// 成员 kind：InlineKind。
// 成员 text：string。
// 成员 href（可省略）：string。
// [/maint-note:d2153d0a8699]
export interface InlineSegment {
  kind: InlineKind
  text: string
  href?: string
}

// [maint-note:d313b40ff4c8]
// 数据契约导航：MarkdownListItem；字段含义以消费端校验为准。
// 成员 marker：string。
// 成员 content：InlineSegment[]。
// 成员 nested：MarkdownBlock[]。
// [/maint-note:d313b40ff4c8]
export interface MarkdownListItem {
  marker: string
  content: InlineSegment[]
  nested: MarkdownBlock[]
}

// [maint-note:924e95594614]
// 数据契约导航：MarkdownTableAlignment；字段含义以消费端校验为准。
// 类型组合：'left' | 'center' | 'right'。
// [/maint-note:924e95594614]
export type MarkdownTableAlignment = 'left' | 'center' | 'right'

// [maint-note:4c3eddec52b9]
// 数据契约导航：MarkdownTableBlock；字段含义以消费端校验为准。
// 成员 kind：'table'。
// 成员 header：InlineSegment[][]。
// 成员 rows：InlineSegment[][][]。
// 成员 alignments：MarkdownTableAlignment[]。
// [/maint-note:4c3eddec52b9]
export interface MarkdownTableBlock {
  kind: 'table'
  header: InlineSegment[][]
  rows: InlineSegment[][][]
  alignments: MarkdownTableAlignment[]
}

// [maint-note:b085d95bfa4c]
// 数据契约导航：MarkdownBlock；字段含义以消费端校验为准。
// 类型组合：| { kind: 'paragraph'; content: InlineSegment[] } | { kind: 'heading'; level: number; content: InlineSegment[] } | { kind: 'blockquote'; blocks: MarkdownBlock[] } | { kind: 'list'; ordered: boolean; items: MarkdownListItem[] } | { kind: 'co。
// [/maint-note:b085d95bfa4c]
export type MarkdownBlock =
  | { kind: 'paragraph'; content: InlineSegment[] }
  | { kind: 'heading'; level: number; content: InlineSegment[] }
  | { kind: 'blockquote'; blocks: MarkdownBlock[] }
  | { kind: 'list'; ordered: boolean; items: MarkdownListItem[] }
  | { kind: 'code'; language?: string; lines: string[] }
  | MarkdownTableBlock
  | { kind: 'hr' }

// [maint-note:7093c2f2eeb0]
// 数据契约导航：MarkdownProjection；字段含义以消费端校验为准。
// 成员 blocks：MarkdownBlock[]。
// 成员 plainText：string。
// 成员 lines：string[]。
// 成员 sourceChars：number。
// 成员 previewOmittedChars：number。
// 成员 sourceTruncated：boolean。
// 成员 viewCollapsed：false。
// 成员 canExpand：boolean。
// [/maint-note:7093c2f2eeb0]
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

// [maint-note:b4ed9a385b83]
// 数据契约导航：MarkdownRenderOptions；字段含义以消费端校验为准。
// 成员 maxWidth（可省略）：number。
// 成员 maxSourceChars（可省略）：number。
// 成员 cache（可省略）：MarkdownProjectionCache。
// [/maint-note:b4ed9a385b83]
export interface MarkdownRenderOptions {
  maxWidth?: number
  maxSourceChars?: number
  cache?: MarkdownProjectionCache
}

// [maint-note:9003aacd7eec]
// 数据契约导航：MarkdownSummaryProjection；字段含义以消费端校验为准。
// 成员 content：string。
// 成员 hiddenLines：number。
// [/maint-note:9003aacd7eec]
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
// [maint-note:519cbb39e6ae]
// 维护导航：foldMarkdownSummary；下列为本函数源码中的接口与路径索引。
// 输入 source：string。
// 输入 maxLines：number。
// 输入 maxChars：number。
// 返回契约：MarkdownSummaryProjection | undefined。
// 分支条件：totalLines <= lineLimit && safeSource.length <= charLimit。
// 分支条件：!raw。
// 分支条件：candidate.content.length > charLimit || candidate.content.split('\n').length > lineLimit。
// 分支条件：projection.content.length <= charLimit && projection.content.split('\n').length <= lineLimit。
// 遍历边界：const token ← tokens。
// 返回路径：undefined。
// 返回路径：projection。
// 返回路径：{ content: projection.content.slice(0, charLimit), hiddenLines: projection.hiddenLines, }。
// 写入目标：selected。
// 写入目标：selectedLines。
// 直接调用：sanitizeMarkdownSource。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：safeSource.split。
// 直接调用：lexer。
// 直接调用：next.replace。
// 直接调用：normalized.split。
// 直接调用：build。
// 直接调用：candidate.content.split。
// 直接调用：projection.content.split。
// 直接调用：projection.content.slice。
// [/maint-note:519cbb39e6ae]
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

  // [maint-note:457f130a4899]
  // 维护导航：build；下列为本函数源码中的接口与路径索引。
  // 输入 body：string。
  // 输入 sourceLines：number。
  // 返回契约：MarkdownSummaryProjection。
  // 返回路径：{ content: trimmed ? `${trimmed}\n\n${notice}` : notice, hiddenLines, }。
  // 直接调用：Math.max。
  // 直接调用：body.replace。
  // [/maint-note:457f130a4899]
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

// [maint-note:4af6f84df99c]
// 数据契约导航：MarkdownCacheStats；字段含义以消费端校验为准。
// 成员 hits：number。
// 成员 misses：number。
// 成员 parses：number。
// 成员 lexedChars：number。
// 成员 entries：number。
// 成员 chars：number。
// 成员 retainedSourceChars：number。
// 成员 retainedBlockChars：number。
// [/maint-note:4af6f84df99c]
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

  // [maint-note:14b6b0452284]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 maxEntries：类型由调用方与函数体约束；缺省表达式 MAX_MARKDOWN_CACHE_ENTRIES。
  // 输入 maxChars：类型由调用方与函数体约束；缺省表达式 MAX_MARKDOWN_CACHE_CHARS。
  // [/maint-note:14b6b0452284]
  constructor(
    private readonly maxEntries = MAX_MARKDOWN_CACHE_ENTRIES,
    private readonly maxChars = MAX_MARKDOWN_CACHE_CHARS,
  ) {}

  // [maint-note:4cad1914dab3]
  // 维护导航：get；下列为本函数源码中的接口与路径索引。
  // 输入 key：string。
  // 输入 source：string（可省略）。
  // 返回契约：MarkdownProjection | undefined。
  // 分支条件：!found || (source !== undefined && found.source !== source)。
  // 返回路径：undefined。
  // 返回路径：found.projection。
  // 写入目标：this.misses。
  // 写入目标：this.hits。
  // 直接调用：this.values.get。
  // 直接调用：this.values.delete。
  // 直接调用：this.values.set。
  // [/maint-note:4cad1914dab3]
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

  // [maint-note:a8f055c27444]
  // 维护导航：set；下列为本函数源码中的接口与路径索引。
  // 输入 key：string。
  // 输入 projection：MarkdownProjection。
  // 输入 source：类型由调用方与函数体约束；缺省表达式 ''。
  // 返回契约：void。
  // 分支条件：existing。
  // 分支条件：chars > this.maxChars。
  // 分支条件：oldest === undefined。
  // 分支条件：item。
  // 遍历边界：this.values.size > this.maxEntries || this.totalChars > this.maxChars。
  // 返回路径：无返回值。
  // 写入目标：this.totalChars。
  // 写入目标：this.retainedSourceChars。
  // 写入目标：this.retainedBlockChars。
  // 直接调用：estimateBlocks。
  // 直接调用：this.values.get。
  // 直接调用：this.values.delete。
  // 直接调用：this.values.set。
  // 直接调用：this.values.keys().next。
  // 直接调用：this.values.keys。
  // [/maint-note:a8f055c27444]
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

  // [maint-note:c7d30523d8ed]
  // 维护导航：markParse；下列为本函数源码中的接口与路径索引。
  // 输入 lexedChars：类型由调用方与函数体约束；缺省表达式 0。
  // 返回契约：void。
  // 写入目标：this.parses。
  // 写入目标：this.lexedChars。
  // 直接调用：Math.max。
  // 直接调用：Math.floor。
  // [/maint-note:c7d30523d8ed]
  markParse(lexedChars = 0): void {
    this.parses += 1
    this.lexedChars += Math.max(0, Math.floor(lexedChars))
  }

  // [maint-note:8d8cade75220]
  // 维护导航：stats；下列为本函数源码中的接口与路径索引。
  // 返回契约：MarkdownCacheStats。
  // 返回路径：{ hits: this.hits, misses: this.misses, parses: this.parses, lexedChars: this.lexedChars, entries: this.values.size, chars: this.totalChars, retainedSourceChars: this.retainedSourceChars, retainedBlockChars: this.retainedBlockChars, }。
  // [/maint-note:8d8cade75220]
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

  // [maint-note:e801a9b9795c]
  // 维护导航：clear；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.totalChars。
  // 写入目标：this.retainedSourceChars。
  // 写入目标：this.retainedBlockChars。
  // 写入目标：this.hits。
  // 写入目标：this.misses。
  // 写入目标：this.parses。
  // 写入目标：this.lexedChars。
  // 直接调用：this.values.clear。
  // [/maint-note:e801a9b9795c]
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

// [maint-note:96c013773bc7]
// 维护导航：renderMarkdown；下列为本函数源码中的接口与路径索引。
// 输入 source：string。
// 输入 options：MarkdownRenderOptions；缺省表达式 {}。
// 返回契约：MarkdownProjection。
// 分支条件：cached。
// 返回路径：cached。
// 返回路径：projection。
// 写入目标：blocks。
// 直接调用：clampWidth。
// 直接调用：Math.min。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：sanitizeMarkdownSource。
// 直接调用：boundSource。
// 直接调用：contentDigest。
// 直接调用：cache.get。
// 直接调用：cache.markParse。
// 直接调用：parseBlocks。
// 直接调用：createProjection。
// 直接调用：cache.set。
// [/maint-note:96c013773bc7]
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

  // [maint-note:732fc0594a8e]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：MarkdownRenderOptions；缺省表达式 {}。
  // [/maint-note:732fc0594a8e]
  constructor(
    private readonly options: MarkdownRenderOptions = {},
  ) {}

  /** Append a canonical delta without forcing a display refresh/parse. */
  // [maint-note:b6b2a7c03682]
  // 维护导航：append；下列为本函数源码中的接口与路径索引。
  // 输入 delta：string。
  // 返回契约：void。
  // 分支条件：this.finalized。
  // 分支条件：safeDelta。
  // 显式抛出：new Error('StreamingMarkdownRenderer cannot push after complete()')。
  // 写入目标：this.currentProjection。
  // 直接调用：normalizeMarkdownCodePoints。
  // 直接调用：this.sourceNormalizer.push。
  // 直接调用：this.appendNormalized。
  // [/maint-note:b6b2a7c03682]
  append(delta: string): void {
    if (this.finalized) throw new Error('StreamingMarkdownRenderer cannot push after complete()')
    const safeDelta = normalizeMarkdownCodePoints(this.sourceNormalizer.push(delta))
    if (safeDelta) {
      this.appendNormalized(safeDelta)
      this.currentProjection = null
    }
  }

  /** Parse the current bounded source without treating it as EOF. */
  // [maint-note:022fc34cda7f]
  // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
  // 返回契约：MarkdownProjection。
  // 分支条件：this.currentProjection === null。
  // 返回路径：this.currentProjection。
  // 写入目标：this.currentProjection。
  // 直接调用：this.renderCurrentSource。
  // [/maint-note:022fc34cda7f]
  snapshot(): MarkdownProjection {
    if (this.currentProjection === null) this.currentProjection = this.renderCurrentSource()
    return this.currentProjection
  }

  // [maint-note:5700374dd708]
  // 维护导航：push；下列为本函数源码中的接口与路径索引。
  // 输入 delta：string。
  // 返回契约：MarkdownProjection。
  // 返回路径：this.snapshot()。
  // 直接调用：this.append。
  // 直接调用：this.snapshot。
  // [/maint-note:5700374dd708]
  push(delta: string): MarkdownProjection {
    this.append(delta)
    return this.snapshot()
  }

  // [maint-note:e1cf441ca5cc]
  // 维护导航：complete；下列为本函数源码中的接口与路径索引。
  // 返回契约：MarkdownProjection。
  // 分支条件：!this.finalized。
  // 分支条件：normalizedTail。
  // 分支条件：this.currentProjection === null。
  // 返回路径：this.currentProjection。
  // 写入目标：this.currentProjection。
  // 写入目标：this.finalized。
  // 直接调用：normalizeMarkdownCodePoints。
  // 直接调用：this.sourceNormalizer.finish。
  // 直接调用：this.appendNormalized。
  // 直接调用：this.renderCurrentSource。
  // [/maint-note:e1cf441ca5cc]
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

  // [maint-note:da2c4b4452cb]
  // 维护导航：reset；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.sourceChunks.length。
  // 写入目标：this.sourceChars。
  // 写入目标：this.retainedSourceChars。
  // 写入目标：this.canonicalTruncated。
  // 写入目标：this.retainedHead。
  // 写入目标：this.retainedTail。
  // 写入目标：this.currentProjection。
  // 写入目标：this.finalized。
  // 直接调用：this.sourceNormalizer.reset。
  // [/maint-note:da2c4b4452cb]
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

  // [maint-note:203e49945416]
  // 维护导航：sourceText；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 分支条件：this.canonicalTruncated。
  // 返回路径：`${this.retainedHead}${MARKDOWN_TRUNCATION_MARKER}${this.retainedTail}`。
  // 返回路径：this.sourceChunks.join('')。
  // 直接调用：this.sourceChunks.join。
  // [/maint-note:203e49945416]
  get sourceText(): string {
    if (this.canonicalTruncated) {
      return `${this.retainedHead}${MARKDOWN_TRUNCATION_MARKER}${this.retainedTail}`
    }
    return this.sourceChunks.join('')
  }

  // [maint-note:bd175b63f1f2]
  // 维护导航：stablePrefix；下列为本函数源码中的接口与路径索引。
  // 返回契约：MarkdownProjection。
  // 返回路径：emptyMarkdownProjection()。
  // 直接调用：emptyMarkdownProjection。
  // [/maint-note:bd175b63f1f2]
  get stablePrefix(): MarkdownProjection {
    return emptyMarkdownProjection()
  }

  // [maint-note:62fa6ee13961]
  // 维护导航：appendNormalized；下列为本函数源码中的接口与路径索引。
  // 输入 safeDelta：string。
  // 返回契约：void。
  // 分支条件：!this.canonicalTruncated && this.sourceChars <= MAX_MARKDOWN_SOURCE_CHARS。
  // 分支条件：!this.canonicalTruncated。
  // 写入目标：this.sourceChars。
  // 写入目标：this.retainedSourceChars。
  // 写入目标：this.retainedHead。
  // 写入目标：this.retainedTail。
  // 写入目标：this.sourceChunks.length。
  // 写入目标：this.canonicalTruncated。
  // 直接调用：this.sourceChunks.push。
  // 直接调用：this.sourceChunks.join。
  // 直接调用：boundSource。
  // 直接调用：boundedChars.slice(0, MARKDOWN_RETAINED_HEAD_CHARS).join。
  // 直接调用：boundedChars.slice。
  // 直接调用：boundedChars.slice(-MARKDOWN_RETAINED_TAIL_CHARS).join。
  // 直接调用：takeLastCodePoints。
  // [/maint-note:62fa6ee13961]
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

  // [maint-note:3e1d2c845fef]
  // 维护导航：renderCurrentSource；下列为本函数源码中的接口与路径索引。
  // 返回契约：MarkdownProjection。
  // 分支条件：!this.canonicalTruncated && this.sourceChars === projection.sourceChars。
  // 返回路径：projection。
  // 返回路径：{ ...projection, sourceChars: this.sourceChars, previewOmittedChars: 0, sourceTruncated: projection.sourceTruncated || this.canonicalTruncated, canExpand: false, }。
  // 直接调用：renderMarkdown。
  // [/maint-note:3e1d2c845fef]
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

// [maint-note:54a69e8ef43b]
// 维护导航：createProjection；下列为本函数源码中的接口与路径索引。
// 输入 blocks：MarkdownBlock[]。
// 输入 sourceChars：number。
// 输入 maxWidth：number。
// 输入 sourceTruncated：boolean。
// 输入 previewOmittedChars：类型由调用方与函数体约束；缺省表达式 0。
// 返回契约：MarkdownProjection。
// 返回路径：{ blocks, plainText, lines: plainText.split('\n'), sourceChars, previewOmittedChars, sourceTruncated, viewCollapsed: false, canExpand: false, }。
// 直接调用：boundProjectionText。
// 直接调用：plainText.split。
// [/maint-note:54a69e8ef43b]
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

// [maint-note:3d2b5c24787a]
// 维护导航：parseBlocks；下列为本函数源码中的接口与路径索引。
// 输入 source：string。
// 输入 maxWidth：number。
// 返回契约：MarkdownBlock[]。
// 返回路径：tokens.flatMap((token) => parseBlock(token, maxWidth))。
// 直接调用：lexer。
// 直接调用：tokens.flatMap。
// [/maint-note:3d2b5c24787a]
function parseBlocks(source: string, maxWidth: number): MarkdownBlock[] {
  const tokens = lexer(source, { gfm: true, breaks: true })
  return tokens.flatMap((token) => parseBlock(token, maxWidth))
}

// [maint-note:5b06c6a535ce]
// 维护导航：parseBlock；下列为本函数源码中的接口与路径索引。
// 输入 token：Token。
// 输入 maxWidth：number。
// 返回契约：MarkdownBlock[]。
// 分支条件：switch token.type。
// 分支条件：case 'space'。
// 分支条件：case 'heading'。
// 分支条件：case 'paragraph'。
// 分支条件：case 'text'。
// 分支条件：case 'code'。
// 分支条件：case 'blockquote'。
// 分支条件：case 'list'。
// 分支条件：case 'table'。
// 分支条件：case 'hr'。
// 分支条件：case 'html'。
// 返回路径：[]。
// 返回路径：[{ kind: 'heading', level: clampLevel(token.depth), content: inlineTokens(nestedInlineTokens(token), maxWidth) }]。
// 返回路径：[{ kind: 'paragraph', content: inlineTokens('tokens' in token && token.tokens ? token.tokens : [token], maxWidth) }]。
// 返回路径：[{ kind: 'code', language: safeLanguage(token.lang), // Keep source lines independent of the current terminal width. Width // is a projection concern; truncating here would make the structured // block look complete while silently losing co。
// 返回路径：[{ kind: 'blockquote', blocks: (token.tokens ?? []).flatMap((item: Token) => parseBlock(item, maxWidth)) }]。
// 返回路径：[{ kind: 'list', ordered: token.ordered, items: token.items.map((item: Tokens.ListItem, index: number) => parseListItem(item, token.ordered, token.start, index, maxWidth)), }]。
// 返回路径：[{ kind: 'table', header: token.header.map((cell: Tokens.TableCell) => inlineTokens(cell.tokens, maxWidth)), rows: token.rows.map((row: Tokens.TableCell[]) => row.map((cell: Tokens.TableCell) => inlineTokens(cell.tokens, maxWidth))), alignm。
// 返回路径：[{ kind: 'hr' }]。
// 返回路径：visible ? [{ kind: 'paragraph', content: [{ kind: 'text', text: visible }] }] : []。
// 直接调用：clampLevel。
// 直接调用：inlineTokens。
// 直接调用：nestedInlineTokens。
// 直接调用：safeLanguage。
// 直接调用：safeRawDisplayText(token.text).split。
// 直接调用：safeRawDisplayText。
// 直接调用：(token.tokens ?? []).flatMap。
// 直接调用：token.items.map。
// 直接调用：token.header.map。
// 直接调用：token.rows.map。
// 直接调用：token.align.map。
// 直接调用：safeDisplayText。
// 直接调用：stripHtml。
// [/maint-note:5b06c6a535ce]
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

// [maint-note:67ba4353c446]
// 维护导航：parseListItem；下列为本函数源码中的接口与路径索引。
// 输入 item：Tokens.ListItem。
// 输入 ordered：boolean。
// 输入 start：number | ''。
// 输入 index：number。
// 输入 maxWidth：number。
// 返回契约：MarkdownListItem。
// 返回路径：{ marker, content: inlineTokens(contentTokens, maxWidth), nested: nestedTokens.flatMap((token) => parseBlock(token, maxWidth)), }。
// 直接调用：item.tokens.filter。
// 直接调用：inlineTokens。
// 直接调用：nestedTokens.flatMap。
// [/maint-note:67ba4353c446]
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

// [maint-note:f6c353d0af45]
// 维护导航：inlineTokens；下列为本函数源码中的接口与路径索引。
// 输入 tokens：readonly Token[]。
// 输入 maxWidth：number。
// 返回契约：InlineSegment[]。
// 分支条件：switch token.type。
// 分支条件：case 'text'。
// 分支条件：case 'escape'。
// 分支条件：'tokens' in token && Array.isArray(token.tokens)。
// 分支条件：case 'strong'。
// 分支条件：case 'em'。
// 分支条件：case 'codespan'。
// 分支条件：case 'del'。
// 分支条件：case 'link'。
// 分支条件：!isExplicitInlineLink(token.raw)。
// 分支条件：isSafeLink(href)。
// 分支条件：case 'image'。
// 分支条件：case 'br'。
// 分支条件：case 'html'。
// 分支条件：'text' in token && typeof token.text === 'string'。
// 遍历边界：const token ← tokens。
// 返回路径：result。
// 直接调用：Array.isArray。
// 直接调用：appendPreserving。
// 直接调用：inlineTokens。
// 直接调用：appendSegment。
// 直接调用：safeDisplayText。
// 直接调用：appendSegments。
// 直接调用：nestedInlineTokens。
// 直接调用：isExplicitInlineLink。
// 直接调用：decodeMarkdownEntities。
// 直接调用：isSafeLink。
// 直接调用：safeRawDisplayText。
// 直接调用：stripHtml。
// [/maint-note:f6c353d0af45]
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

// [maint-note:9796902c0d89]
// 维护导航：nestedInlineTokens；下列为本函数源码中的接口与路径索引。
// 输入 token：Token。
// 返回契约：readonly Token[]。
// 分支条件：'tokens' in token && Array.isArray(token.tokens)。
// 返回路径：token.tokens。
// 返回路径：[{ type: 'text', raw: text, text }]。
// 直接调用：Array.isArray。
// [/maint-note:9796902c0d89]
function nestedInlineTokens(token: Token): readonly Token[] {
  if ('tokens' in token && Array.isArray(token.tokens)) return token.tokens
  const text = 'text' in token && typeof token.text === 'string' ? token.text : ''
  return [{ type: 'text', raw: text, text }]
}

// [maint-note:240332e6dd57]
// 维护导航：appendSegments；下列为本函数源码中的接口与路径索引。
// 输入 target：InlineSegment[]。
// 输入 segments：InlineSegment[]。
// 输入 kind：InlineKind。
// 输入 href：string（可省略）。
// 返回契约：void。
// 遍历边界：const segment ← segments。
// 直接调用：appendSegment。
// [/maint-note:240332e6dd57]
function appendSegments(target: InlineSegment[], segments: InlineSegment[], kind: InlineKind, href?: string): void {
  for (const segment of segments) {
    appendSegment(target, {
      ...segment,
      kind,
      href: kind === 'link' ? href : undefined,
    })
  }
}

// [maint-note:8ad98c995ffb]
// 维护导航：appendPreserving；下列为本函数源码中的接口与路径索引。
// 输入 target：InlineSegment[]。
// 输入 segments：InlineSegment[]。
// 返回契约：void。
// 遍历边界：const segment ← segments。
// 直接调用：appendSegment。
// [/maint-note:8ad98c995ffb]
function appendPreserving(target: InlineSegment[], segments: InlineSegment[]): void {
  for (const segment of segments) appendSegment(target, segment)
}

// [maint-note:19717c0d0c88]
// 维护导航：appendSegment；下列为本函数源码中的接口与路径索引。
// 输入 target：InlineSegment[]。
// 输入 segment：InlineSegment。
// 返回契约：void。
// 分支条件：!segment.text。
// 分支条件：previous && previous.kind === segment.kind && previous.href === segment.href。
// 返回路径：无返回值。
// 写入目标：previous.text。
// 直接调用：target.at。
// 直接调用：target.push。
// [/maint-note:19717c0d0c88]
function appendSegment(target: InlineSegment[], segment: InlineSegment): void {
  if (!segment.text) return
  const previous = target.at(-1)
  if (previous && previous.kind === segment.kind && previous.href === segment.href) {
    previous.text += segment.text
  } else {
    target.push(segment)
  }
}

// [maint-note:bb0886e0e27b]
// 维护导航：boundProjectionText；下列为本函数源码中的接口与路径索引。
// 输入 blocks：readonly MarkdownBlock[]。
// 输入 maxWidth：number。
// 返回契约：string。
// 遍历边界：const block ← blocks。
// 返回路径：lines.map((line) => boundLine(line, maxWidth)).join('\n')。
// 直接调用：appendBlockText。
// 直接调用：lines.map((line) => boundLine(line, maxWidth)).join。
// 直接调用：lines.map。
// [/maint-note:bb0886e0e27b]
function boundProjectionText(blocks: readonly MarkdownBlock[], maxWidth: number): string {
  const lines: string[] = []
  for (const block of blocks) appendBlockText(lines, block, maxWidth, '')
  return lines.map((line) => boundLine(line, maxWidth)).join('\n')
}

// [maint-note:557dc82a8087]
// 维护导航：appendBlockText；下列为本函数源码中的接口与路径索引。
// 输入 lines：string[]。
// 输入 block：MarkdownBlock。
// 输入 maxWidth：number。
// 输入 prefix：string。
// 返回契约：void。
// 分支条件：switch block.kind。
// 分支条件：case 'paragraph'。
// 分支条件：case 'heading'。
// 分支条件：case 'blockquote'。
// 分支条件：case 'list'。
// 分支条件：case 'code'。
// 分支条件：block.language。
// 分支条件：case 'table'。
// 分支条件：case 'hr'。
// 遍历边界：const nested ← block.blocks。
// 遍历边界：const item ← block.items。
// 遍历边界：const nested ← item.nested。
// 遍历边界：const line ← block.lines。
// 遍历边界：const row ← buildMarkdownTableRows(block, Math.max(1, maxWidth - displayLineWidth(prefix)))。
// 直接调用：appendMultiline。
// 直接调用：inlineText。
// 直接调用：appendBlockText。
// 直接调用：lines.push。
// 直接调用：buildMarkdownTableRows。
// 直接调用：Math.max。
// 直接调用：displayLineWidth。
// 直接调用：fillDisplayWidth。
// [/maint-note:557dc82a8087]
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

// [maint-note:934492219ecd]
// 维护导航：appendMultiline；下列为本函数源码中的接口与路径索引。
// 输入 lines：string[]。
// 输入 text：string。
// 返回契约：void。
// 分支条件：parts.length === 0。
// 直接调用：text.split。
// 直接调用：lines.push。
// [/maint-note:934492219ecd]
function appendMultiline(lines: string[], text: string): void {
  const parts = text.split('\n')
  if (parts.length === 0) lines.push('')
  else lines.push(...parts)
}

// [maint-note:dc4f5e4cbd4e]
// 维护导航：inlineText；下列为本函数源码中的接口与路径索引。
// 输入 segments：readonly InlineSegment[]。
// 返回契约：string。
// 返回路径：segments.map((segment) => segment.text).join('')。
// 直接调用：segments.map((segment) => segment.text).join。
// 直接调用：segments.map。
// [/maint-note:dc4f5e4cbd4e]
function inlineText(segments: readonly InlineSegment[]): string {
  return segments.map((segment) => segment.text).join('')
}

// [maint-note:6da8b5948c0d]
// 维护导航：boundLine；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxWidth：number。
// 返回契约：string。
// 返回路径：truncateDisplayText(value, maxWidth)。
// 直接调用：truncateDisplayText。
// [/maint-note:6da8b5948c0d]
function boundLine(value: string, maxWidth: number): string {
  return truncateDisplayText(value, maxWidth)
}

// [maint-note:acbffeef4188]
// 维护导航：emptyMarkdownProjection；下列为本函数源码中的接口与路径索引。
// 返回契约：MarkdownProjection。
// 返回路径：{ blocks: [], plainText: '', lines: [''], sourceChars: 0, previewOmittedChars: 0, sourceTruncated: false, viewCollapsed: false, canExpand: false, }。
// [/maint-note:acbffeef4188]
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

// [maint-note:665149eba35d]
// 维护导航：estimateBlocks；下列为本函数源码中的接口与路径索引。
// 输入 blocks：readonly MarkdownBlock[]。
// 返回契约：number。
// 分支条件：switch block.kind。
// 分支条件：case 'paragraph'。
// 分支条件：case 'heading'。
// 分支条件：case 'blockquote'。
// 分支条件：case 'list'。
// 分支条件：case 'code'。
// 分支条件：case 'table'。
// 分支条件：case 'hr'。
// 遍历边界：const block ← blocks。
// 返回路径：total。
// 写入目标：total。
// 直接调用：block.content.reduce。
// 直接调用：estimateBlocks。
// 直接调用：block.items.reduce。
// 直接调用：block.lines.reduce。
// 直接调用：block.header.flat().reduce。
// 直接调用：block.header.flat。
// 直接调用：block.rows.flat(2).reduce。
// 直接调用：block.rows.flat。
// [/maint-note:665149eba35d]
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

// [maint-note:19b2c2e097f1]
// 维护导航：boundSource；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxChars：number。
// 返回契约：{ text: string; truncated: boolean }。
// 分支条件：chars.length <= maxChars。
// 返回路径：{ text: value, truncated: false }。
// 返回路径：{ text: `${chars.slice(0, head).join('')}${marker}${chars.slice(-tail).join('')}`, truncated: true, }。
// 直接调用：Math.max。
// 直接调用：Math.ceil。
// 直接调用：Math.floor。
// 直接调用：chars.slice(0, head).join。
// 直接调用：chars.slice。
// 直接调用：chars.slice(-tail).join。
// [/maint-note:19b2c2e097f1]
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

// [maint-note:4d8e2ec368d2]
// 维护导航：takeLastCodePoints；下列为本函数源码中的接口与路径索引。
// 输入 previous：string。
// 输入 delta：readonly string[]。
// 输入 limit：number。
// 返回契约：string。
// 分支条件：delta.length >= limit。
// 返回路径：delta.slice(-limit).join('')。
// 返回路径：[...previous, ...delta].slice(-limit).join('')。
// 直接调用：delta.slice(-limit).join。
// 直接调用：delta.slice。
// 直接调用：[...previous, ...delta].slice(-limit).join。
// 直接调用：[...previous, ...delta].slice。
// [/maint-note:4d8e2ec368d2]
function takeLastCodePoints(previous: string, delta: readonly string[], limit: number): string {
  if (delta.length >= limit) return delta.slice(-limit).join('')
  return [...previous, ...delta].slice(-limit).join('')
}

// [maint-note:5906b9304fdb]
// 维护导航：sanitizeMarkdownSource；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：normalizeMarkdownCodePoints(sanitizeTerminalText(value))。
// 直接调用：normalizeMarkdownCodePoints。
// 直接调用：sanitizeTerminalText。
// [/maint-note:5906b9304fdb]
function sanitizeMarkdownSource(value: string): string {
  return normalizeMarkdownCodePoints(sanitizeTerminalText(value))
}

// [maint-note:ea5db175fb9c]
// 维护导航：normalizeMarkdownCodePoints；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：value .split('') .map((character, index, characters) => { const code = character.charCodeAt(0) if (code >= 0xD800 && code <= 0xDBFF && characters[index + 1]?.charCodeAt(0) >= 0xDC00 && characters[index + 1]?.charCodeAt(0) <= 0xDFFF) return 。
// 直接调用：value .split('') .map((character, index, characters) => { const code = character.charCodeAt(0) if (code >= 0xD800 && code <= 0xDBFF && characters[index + 1]?.charCodeAt(0) >= 0xDC00 && characters[index + 1]?.charCodeAt(0) <= 0xDFFF) return 。
// 直接调用：value .split('') .map。
// 直接调用：value .split。
// [/maint-note:ea5db175fb9c]
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

// [maint-note:d2b76ceb7cb1]
// 维护导航：contentDigest；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：createHash('sha256').update(value, 'utf8').digest('hex')。
// 直接调用：createHash('sha256').update(value, 'utf8').digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// [/maint-note:d2b76ceb7cb1]
function contentDigest(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex')
}

// [maint-note:bc4faa369145]
// 维护导航：safeDisplayText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：sanitizeMarkdownSource(decodeMarkdownEntities(value))。
// 直接调用：sanitizeMarkdownSource。
// 直接调用：decodeMarkdownEntities。
// [/maint-note:bc4faa369145]
function safeDisplayText(value: string): string {
  return sanitizeMarkdownSource(decodeMarkdownEntities(value))
}

// [maint-note:a14d4a376551]
// 维护导航：safeRawDisplayText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：sanitizeMarkdownSource(value)。
// 直接调用：sanitizeMarkdownSource。
// [/maint-note:a14d4a376551]
function safeRawDisplayText(value: string): string {
  return sanitizeMarkdownSource(value)
}

/** Decode marked's HTML-renderer-facing token text once, after tokenization. */
// [maint-note:67a98e0e4761]
// 维护导航：decodeMarkdownEntities；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：decodeHTML(value)。
// 直接调用：decodeHTML。
// [/maint-note:67a98e0e4761]
function decodeMarkdownEntities(value: string): string {
  return decodeHTML(value)
}

// [maint-note:b5943f0de2fb]
// 维护导航：stripHtml；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 分支条件：rawTextTag。
// 分支条件：closingTag < 0。
// 分支条件：closingMarkup?.closing && closingMarkup.tagName === rawTextTag。
// 分支条件：markup。
// 分支条件：!markup.closing && !markup.selfClosing && (markup.tagName === 'script' || markup.tagName === 'style')。
// 遍历边界：index < value.length。
// 返回路径：visible.join('')。
// 写入目标：index。
// 写入目标：rawTextTag。
// 直接调用：findRawTextClosingTag。
// 直接调用：visible.push。
// 直接调用：value.slice。
// 直接调用：findHtmlMarkupAt。
// 直接调用：visible.join。
// [/maint-note:b5943f0de2fb]
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

// [maint-note:bd3a60c2a3ab]
// 维护导航：isSafeLink；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：boolean。
// 返回路径：/^(?:https?:|mailto:)/iu.test(value.trim())。
// 直接调用：/^(?:https?:|mailto:)/iu.test。
// 直接调用：value.trim。
// [/maint-note:bd3a60c2a3ab]
function isSafeLink(value: string): boolean {
  return /^(?:https?:|mailto:)/iu.test(value.trim())
}

// [maint-note:3f099a09e627]
// 维护导航：isExplicitInlineLink；下列为本函数源码中的接口与路径索引。
// 输入 raw：string。
// 返回契约：boolean。
// 分支条件：opening < 0。
// 分支条件：escaped。
// 分支条件：character === '\\'。
// 分支条件：character === '`'。
// 分支条件：codeEnd >= 0。
// 分支条件：character === '<'。
// 分支条件：htmlEnd >= 0。
// 分支条件：character === '['。
// 分支条件：character !== ']'。
// 分支条件：nestedBrackets > 0。
// 遍历边界：let index = opening + 1; index < raw.length; index += 1。
// 返回路径：false。
// 返回路径：raw[index + 1] === '('。
// 写入目标：index。
// 写入目标：escaped。
// 写入目标：nestedBrackets。
// 直接调用：raw.startsWith。
// 直接调用：countRun。
// 直接调用：findCodeSpanEnd。
// 直接调用：findHtmlTagEnd。
// [/maint-note:3f099a09e627]
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

// [maint-note:f9c6c7bc5186]
// 维护导航：countRun；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 输入 character：string。
// 返回契约：number。
// 遍历边界：value[end] === character。
// 返回路径：end - start。
// 写入目标：end。
// [/maint-note:f9c6c7bc5186]
function countRun(value: string, start: number, character: string): number {
  let end = start
  while (value[end] === character) end += 1
  return end - start
}

// [maint-note:2b40dc4e9772]
// 维护导航：findCodeSpanEnd；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 输入 runLength：number。
// 返回契约：number。
// 分支条件：value[index] !== '`'。
// 分支条件：length === runLength。
// 遍历边界：index < value.length。
// 返回路径：index + length。
// 返回路径：-1。
// 写入目标：index。
// 直接调用：countRun。
// [/maint-note:2b40dc4e9772]
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

// [maint-note:f3891017ef62]
// 维护导航：findHtmlTagEnd；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 返回契约：number。
// 分支条件：quote。
// 分支条件：character === quote。
// 分支条件：character === '"' || character === "'"。
// 分支条件：character === '<'。
// 分支条件：character === '>'。
// 遍历边界：let index = start; index < value.length; index += 1。
// 返回路径：-1。
// 返回路径：index。
// 写入目标：index。
// 写入目标：quote。
// [/maint-note:f3891017ef62]
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

// [maint-note:15a1914f6c79]
// 维护导航：skipHtmlWhitespace；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 返回契约：number。
// 遍历边界：index < value.length && /\s/u.test(value[index] ?? '')。
// 返回路径：index。
// 写入目标：index。
// 直接调用：/\s/u.test。
// [/maint-note:15a1914f6c79]
function skipHtmlWhitespace(value: string, start: number): number {
  let index = start
  while (index < value.length && /\s/u.test(value[index] ?? '')) index += 1
  return index
}

// [maint-note:4934d8d39408]
// 维护导航：findHtmlClosingTagEnd；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 返回契约：number。
// 返回路径：value[end] === '>' ? end : -1。
// 直接调用：skipHtmlWhitespace。
// [/maint-note:4934d8d39408]
function findHtmlClosingTagEnd(value: string, start: number): number {
  const end = skipHtmlWhitespace(value, start)
  return value[end] === '>' ? end : -1
}

// [maint-note:7bef93866481]
// 维护导航：findHtmlOpeningTagEnd；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 返回契约：{ end: number; selfClosing: boolean } | undefined。
// 分支条件：value[index] === '>'。
// 分支条件：value[index] === '/' && value[index + 1] === '>'。
// 分支条件：!attribute。
// 分支条件：value[index] !== '='。
// 分支条件：quote === '"' || quote === "'"。
// 分支条件：value[index] !== quote。
// 分支条件：index === valueStart。
// 遍历边界：index < value.length。
// 遍历边界：index < value.length && value[index] !== quote。
// 遍历边界：index < value.length && !/\s|>/u.test(value[index] ?? '') && value[index] !== '<' && value[index] !== '"' && value[index] !== "'" && value[index] !== '`'。
// 返回路径：{ end: index, selfClosing: false }。
// 返回路径：{ end: index + 1, selfClosing: true }。
// 返回路径：undefined。
// 写入目标：index。
// 直接调用：skipHtmlWhitespace。
// 直接调用：/^[A-Za-z_:][A-Za-z0-9_.:-]*/u.exec。
// 直接调用：value.slice。
// 直接调用：/\s|>/u.test。
// [/maint-note:7bef93866481]
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

// [maint-note:786334ca8415]
// 数据契约导航：HtmlMarkup；字段含义以消费端校验为准。
// 成员 end：number。
// 成员 tagName（可省略）：string。
// 成员 closing：boolean。
// 成员 selfClosing：boolean。
// [/maint-note:786334ca8415]
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
// [maint-note:2a18cc495292]
// 维护导航：findHtmlMarkupAt；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 返回契约：HtmlMarkup | undefined。
// 分支条件：value[start] !== '<'。
// 分支条件：value.startsWith('<!--', start)。
// 分支条件：next === '!'。
// 分支条件：/^<!doctype(?:\s|>)/iu.test(value.slice(start))。
// 分支条件：value.startsWith('<![CDATA[', start)。
// 分支条件：next === '?'。
// 分支条件：!nameMatch。
// 分支条件：!tagName。
// 分支条件：closing。
// 分支条件：end < 0。
// 分支条件：!opening。
// 返回路径：undefined。
// 返回路径：commentEnd >= 0 ? { end: commentEnd + 2, closing: false, selfClosing: false } : undefined。
// 返回路径：declarationEnd >= 0 ? { end: declarationEnd, closing: false, selfClosing: false } : undefined。
// 返回路径：cdataEnd >= 0 ? { end: cdataEnd + 2, closing: false, selfClosing: false } : undefined。
// 返回路径：instructionEnd >= 0 ? { end: instructionEnd + 1, closing: false, selfClosing: false } : undefined。
// 返回路径：{ end, tagName, closing: true, selfClosing: false }。
// 返回路径：{ end: opening.end, tagName, closing: false, selfClosing: opening.selfClosing, }。
// 直接调用：value.startsWith。
// 直接调用：value.indexOf。
// 直接调用：/^<!doctype(?:\s|>)/iu.test。
// 直接调用：value.slice。
// 直接调用：findHtmlTagEnd。
// 直接调用：/^[A-Za-z][A-Za-z0-9:-]*/u.exec。
// 直接调用：nameMatch[0]?.toLowerCase。
// 直接调用：findHtmlClosingTagEnd。
// 直接调用：findHtmlOpeningTagEnd。
// [/maint-note:2a18cc495292]
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

// [maint-note:0073ba338afe]
// 维护导航：findRawTextClosingTag；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 start：number。
// 输入 tagName：string。
// 返回契约：number。
// 分支条件：markup?.closing && markup.tagName === tagName。
// 遍历边界：candidate >= 0。
// 返回路径：candidate。
// 返回路径：-1。
// 写入目标：candidate。
// 直接调用：value.indexOf。
// 直接调用：findHtmlMarkupAt。
// [/maint-note:0073ba338afe]
function findRawTextClosingTag(value: string, start: number, tagName: string): number {
  let candidate = value.indexOf('</', start)
  while (candidate >= 0) {
    const markup = findHtmlMarkupAt(value, candidate)
    if (markup?.closing && markup.tagName === tagName) return candidate
    candidate = value.indexOf('</', candidate + 2)
  }
  return -1
}

// [maint-note:abad4d9afe33]
// 维护导航：safeLanguage；下列为本函数源码中的接口与路径索引。
// 输入 value：string | undefined。
// 返回契约：string | undefined。
// 分支条件：!value。
// 返回路径：undefined。
// 返回路径：safe || undefined。
// 直接调用：value.replace(/[^a-z0-9_+.#-]/giu, '').slice。
// 直接调用：value.replace。
// [/maint-note:abad4d9afe33]
function safeLanguage(value: string | undefined): string | undefined {
  if (!value) return undefined
  const safe = value.replace(/[^a-z0-9_+.#-]/giu, '').slice(0, 24)
  return safe || undefined
}

// [maint-note:b68fae56e57b]
// 维护导航：clampLevel；下列为本函数源码中的接口与路径索引。
// 输入 value：number。
// 返回契约：number。
// 返回路径：Math.max(1, Math.min(6, value))。
// 直接调用：Math.max。
// 直接调用：Math.min。
// [/maint-note:b68fae56e57b]
function clampLevel(value: number): number {
  return Math.max(1, Math.min(6, value))
}

// [maint-note:875767ccb61d]
// 维护导航：clampWidth；下列为本函数源码中的接口与路径索引。
// 输入 value：number。
// 返回契约：number。
// 返回路径：Number.isFinite(value) && value > 0 ? Math.floor(value) : 1。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// [/maint-note:875767ccb61d]
function clampWidth(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}
