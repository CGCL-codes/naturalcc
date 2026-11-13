import stripAnsi from 'strip-ansi'

import { splitDisplayGraphemes } from './text.js'
import { displayLineWidth } from './width.js'

// [maint-note:3a20184270dd]
// 数据契约导航：TerminalEventName；字段含义以消费端校验为准。
// 类型组合：string | symbol。
// [/maint-note:3a20184270dd]
type TerminalEventName = string | symbol
// [maint-note:cd95c1245b10]
// 数据契约导航：TerminalListener；字段含义以消费端校验为准。
// 类型组合：(this: TerminalResizeOwnerStream, ...args: any[]) => void。
// [/maint-note:cd95c1245b10]
type TerminalListener = (this: TerminalResizeOwnerStream, ...args: any[]) => void
// [maint-note:3dcffad6c676]
// 数据契约导航：ResizeListener；字段含义以消费端校验为准。
// 成员 listener：TerminalListener。
// 成员 once：boolean。
// [/maint-note:3dcffad6c676]
type ResizeListener = {
  listener: TerminalListener
  once: boolean
}
// [maint-note:fa70657772a8]
// 数据契约导航：TerminalWriteCallback；字段含义以消费端校验为准。
// 类型组合：(error?: Error | null) => void。
// [/maint-note:fa70657772a8]
type TerminalWriteCallback = (error?: Error | null) => void
// [maint-note:d2e2c41a1dec]
// 数据契约导航：PendingResizeCleanup；字段含义以消费端校验为准。
// 成员 chunk：string | Uint8Array。
// 成员 encoding（可省略）：BufferEncoding。
// 成员 callback（可省略）：TerminalWriteCallback。
// 成员 byteLength：number。
// [/maint-note:d2e2c41a1dec]
type PendingResizeCleanup = {
  chunk: string | Uint8Array
  encoding?: BufferEncoding
  callback?: TerminalWriteCallback
  byteLength: number
}
// [maint-note:7aa4da7e2bc2]
// 数据契约导航：PendingLayoutRepaint；字段含义以消费端校验为准。
// 成员 id：number。
// 成员 writes：PendingResizeCleanup[]。
// 成员 originAligned：boolean。
// 成员 nativeCleanupLines：number。
// 成员 rendererCacheLines（可省略）：number。
// 成员 rendererPhysicalLines（可省略）：number。
// 成员 rendererBody（可省略）：string。
// 成员 physicalLinesBeforeCleanup（可省略）：number。
// 成员 settling（可省略）：Promise<void>。
// [/maint-note:7aa4da7e2bc2]
type PendingLayoutRepaint = {
  id: number
  writes: PendingResizeCleanup[]
  originAligned: boolean
  nativeCleanupLines: number
  rendererCacheLines?: number
  rendererPhysicalLines?: number
  rendererBody?: string
  physicalLinesBeforeCleanup?: number
  settling?: Promise<void>
}
// [maint-note:4c2389a750f4]
// 数据契约导航：PendingLayoutRepaintDebt；字段含义以消费端校验为准。
// 类型组合：Pick< PendingLayoutRepaint, 'rendererBody' | 'rendererCacheLines' | 'rendererPhysicalLines' >。
// [/maint-note:4c2389a750f4]
type PendingLayoutRepaintDebt = Pick<
  PendingLayoutRepaint,
  'rendererBody' | 'rendererCacheLines' | 'rendererPhysicalLines'
>
// [maint-note:e70558ae6378]
// 数据契约导航：PendingRendererTransaction；字段含义以消费端校验为准。
// 成员 repaintId：number。
// 成员 writes：PendingResizeCleanup[]。
// 成员 commitStarts：number[]。
// 成员 byteLength：number。
// [/maint-note:e70558ae6378]
type PendingRendererTransaction = {
  repaintId: number
  writes: PendingResizeCleanup[]
  commitStarts: number[]
  byteLength: number
}
// [maint-note:01abd28deaec]
// 数据契约导航：RendererCommitClassification；字段含义以消费端校验为准。
// 成员 frameIndex：number。
// 成员 staticCommit（可省略）：{ cleanupIndex: number; bodyIndex: number }。
// 成员 body：string。
// [/maint-note:01abd28deaec]
type RendererCommitClassification = {
  frameIndex: number
  staticCommit?: { cleanupIndex: number; bodyIndex: number }
  body: string
}
// [maint-note:ecb00f5caef6]
// 数据契约导航：RendererTransactionClassification；字段含义以消费端校验为准。
// 成员 commits：RendererCommitClassification[]。
// 成员 finalBody：string。
// [/maint-note:ecb00f5caef6]
type RendererTransactionClassification = {
  commits: RendererCommitClassification[]
  finalBody: string
}

// [maint-note:2ff7c2021bd9]
// 数据契约导航：TerminalResizeOwnerStream；字段含义以消费端校验为准。
// 成员 columns：unknown。
// 成员 rows：unknown。
// 成员 isTTY：boolean | undefined。
// 成员 destroyed：boolean。
// 成员 writableEnded：boolean。
// 成员 writable：boolean。
// 成员 writableLength：number。
// 成员 write：boolean。
// 成员 on：this。
// 成员 off：this。
// 成员 subscribeBeforeResize：() => void。
// 成员 attachRendererFlush：() => void。
// 成员 beginRendererCommit：void。
// 成员 completeResizeLayout：Promise<void>。
// 成员 disposeResizeOwner：void。
// [/maint-note:2ff7c2021bd9]
export interface TerminalResizeOwnerStream {
  readonly columns: unknown
  readonly rows: unknown
  readonly isTTY: boolean | undefined
  readonly destroyed: boolean
  readonly writableEnded: boolean
  readonly writable: boolean
  readonly writableLength: number
  write(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): boolean
  on(event: TerminalEventName, listener: (...args: any[]) => void): this
  off(event: TerminalEventName, listener: (...args: any[]) => void): this
  subscribeBeforeResize(listener: () => void): () => void
  attachRendererFlush(flush: () => Promise<void>): () => void
  beginRendererCommit(): void
  completeResizeLayout(): Promise<void>
  disposeResizeOwner(): void
}

// Ink 7.0.3's ansi-escapes.eraseLines(n) writes one erase-line command, then
// zero or more cursor-up/erase-line pairs, and finally cursor-left. Keep this
// grammar exact: a generic CSI matcher could accidentally pass clear-screen,
// SGR, cursor movement, or other control sequences through the shrink gate.
const resizeCleanupPattern = /^\u001b\[2K(?:\u001b\[1A\u001b\[2K)*\u001b\[G$/u
const resizeFramePattern = /^(\u001b\[2K(?:\u001b\[1A\u001b\[2K)*\u001b\[G)([\s\S]+)$/u
const synchronizedOutputBegin = '\u001b[?2026h'
const synchronizedOutputEnd = '\u001b[?2026l'
const MAX_RENDERER_TRANSACTION_WRITES = 64
const MAX_RENDERER_TRANSACTION_BYTES = 2 * 1024 * 1024

// [maint-note:81f6764f8975]
// 维护导航：isResizeCleanup；下列为本函数源码中的接口与路径索引。
// 输入 chunk：string | Uint8Array。
// 返回契约：boolean。
// 返回路径：resizeCleanupPattern.test(text)。
// 直接调用：Buffer.from(chunk).toString。
// 直接调用：Buffer.from。
// 直接调用：resizeCleanupPattern.test。
// [/maint-note:81f6764f8975]
function isResizeCleanup(chunk: string | Uint8Array): boolean {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
  return resizeCleanupPattern.test(text)
}

// [maint-note:78c7a1454d12]
// 维护导航：resizeCleanupByteLength；下列为本函数源码中的接口与路径索引。
// 输入 chunk：string | Uint8Array。
// 返回契约：number。
// 返回路径：typeof chunk === 'string' ? Buffer.byteLength(chunk, 'utf8') : chunk.byteLength。
// 直接调用：Buffer.byteLength。
// [/maint-note:78c7a1454d12]
function resizeCleanupByteLength(chunk: string | Uint8Array): number {
  return typeof chunk === 'string' ? Buffer.byteLength(chunk, 'utf8') : chunk.byteLength
}

// [maint-note:b848a4cad22c]
// 维护导航：resizeCleanupLineCount；下列为本函数源码中的接口与路径索引。
// 输入 chunk：string | Uint8Array。
// 返回契约：number | undefined。
// 分支条件：!resizeCleanupPattern.test(text)。
// 返回路径：undefined。
// 返回路径：text.match(/\u001b\[2K/gu)?.length。
// 直接调用：Buffer.from(chunk).toString。
// 直接调用：Buffer.from。
// 直接调用：resizeCleanupPattern.test。
// 直接调用：text.match。
// [/maint-note:b848a4cad22c]
function resizeCleanupLineCount(chunk: string | Uint8Array): number | undefined {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
  if (!resizeCleanupPattern.test(text)) return undefined
  return text.match(/\u001b\[2K/gu)?.length
}

// [maint-note:55bd2b488495]
// 维护导航：resizeRendererFrame；下列为本函数源码中的接口与路径索引。
// 输入 chunk：string | Uint8Array。
// 返回契约：{ body: string; cleanupLines: number } | undefined。
// 分支条件：!match?.[2]?.endsWith('\n')。
// 返回路径：undefined。
// 返回路径：cleanupLines ? { body: match[2], cleanupLines } : undefined。
// 直接调用：Buffer.from(chunk).toString。
// 直接调用：Buffer.from。
// 直接调用：resizeFramePattern.exec。
// 直接调用：match?.[2]?.endsWith。
// 直接调用：match[1].match。
// [/maint-note:55bd2b488495]
function resizeRendererFrame(
  chunk: string | Uint8Array,
): { body: string; cleanupLines: number } | undefined {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
  const match = resizeFramePattern.exec(text)
  // The locked standard renderer appends a newline to every non-fullscreen
  // interactive frame. Requiring it keeps a cleanup followed only by cursor
  // controls from impersonating a complete renderer frame.
  if (!match?.[2]?.endsWith('\n')) return undefined
  const cleanupLines = match[1].match(/\u001b\[2K/gu)?.length
  return cleanupLines ? { body: match[2], cleanupLines } : undefined
}

// [maint-note:b4593ba62ca9]
// 维护导航：rendererFrameMetrics；下列为本函数源码中的接口与路径索引。
// 输入 writes：readonly PendingResizeCleanup[]。
// 输入 columns：unknown。
// 返回契约：{ body: string; cacheLines: number; physicalLines: number } | undefined。
// 分支条件：text.includes('\n')。
// 分支条件：physicalLines === undefined。
// 遍历边界：const write ← writes。
// 返回路径：undefined。
// 返回路径：{ body: text, cacheLines: lines.length, physicalLines, }。
// 直接调用：Buffer.from(write.chunk).toString。
// 直接调用：Buffer.from。
// 直接调用：text.includes。
// 直接调用：text.split。
// 直接调用：rendererPhysicalLineCount。
// [/maint-note:b4593ba62ca9]
function rendererFrameMetrics(
  writes: readonly PendingResizeCleanup[],
  columns: unknown,
): { body: string; cacheLines: number; physicalLines: number } | undefined {
  for (const write of writes) {
    const text = typeof write.chunk === 'string'
      ? write.chunk
      : Buffer.from(write.chunk).toString('utf8')
    if (text.includes('\n')) {
      const lines = text.split('\n')
      const physicalLines = rendererPhysicalLineCount(text, columns)
      if (physicalLines === undefined) return undefined
      return {
        body: text,
        cacheLines: lines.length,
        physicalLines,
      }
    }
  }
  return undefined
}

// [maint-note:0291a27e6aee]
// 维护导航：rendererPhysicalLineCount；下列为本函数源码中的接口与路径索引。
// 输入 body：string。
// 输入 columns：unknown。
// 返回契约：number | undefined。
// 分支条件：typeof columns !== 'number' || !Number.isFinite(columns) || columns <= 0。
// 分支条件：graphemeWidth <= 0。
// 分支条件：occupiedColumns > 0 && occupiedColumns + graphemeWidth > width。
// 分支条件：occupiedColumns > width。
// 遍历边界：const hardLine ← body.split('\n')。
// 遍历边界：const grapheme ← splitDisplayGraphemes(stripAnsi(hardLine))。
// 返回路径：undefined。
// 返回路径：total。
// 写入目标：physicalLines。
// 写入目标：occupiedColumns。
// 写入目标：total。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// 直接调用：body.split。
// 直接调用：splitDisplayGraphemes。
// 直接调用：stripAnsi。
// 直接调用：displayLineWidth。
// [/maint-note:0291a27e6aee]
function rendererPhysicalLineCount(body: string, columns: unknown): number | undefined {
  if (typeof columns !== 'number' || !Number.isFinite(columns) || columns <= 0) return undefined
  const width = Math.floor(columns)
  let total = 0
  for (const hardLine of body.split('\n')) {
    let physicalLines = 1
    let occupiedColumns = 0
    for (const grapheme of splitDisplayGraphemes(stripAnsi(hardLine))) {
      const graphemeWidth = displayLineWidth(grapheme)
      if (graphemeWidth <= 0) continue
      if (occupiedColumns > 0 && occupiedColumns + graphemeWidth > width) {
        physicalLines += 1
        occupiedColumns = 0
      }
      occupiedColumns += graphemeWidth
      if (occupiedColumns > width) {
        physicalLines += Math.floor((occupiedColumns - 1) / width)
        occupiedColumns = ((occupiedColumns - 1) % width) + 1
      }
    }
    total += physicalLines
  }
  return total
}

// [maint-note:dcc1a408ca51]
// 维护导航：rendererWriteText；下列为本函数源码中的接口与路径索引。
// 输入 write：PendingResizeCleanup。
// 返回契约：string。
// 返回路径：typeof write.chunk === 'string' ? write.chunk : Buffer.from(write.chunk).toString('utf8')。
// 直接调用：Buffer.from(write.chunk).toString。
// 直接调用：Buffer.from。
// [/maint-note:dcc1a408ca51]
function rendererWriteText(write: PendingResizeCleanup): string {
  return typeof write.chunk === 'string'
    ? write.chunk
    : Buffer.from(write.chunk).toString('utf8')
}

// [maint-note:a2dccaaff4e3]
// 维护导航：normalizeCombinedResizeFrame；下列为本函数源码中的接口与路径索引。
// 输入 writes：readonly PendingResizeCleanup[]。
// 返回契约：PendingResizeCleanup[] | undefined。
// 分支条件：rendererWriteText(writes[0]!) === synchronizedOutputBegin。
// 分支条件：end > start && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd。
// 分支条件：end - start !== 1。
// 分支条件：!frame。
// 返回路径：undefined。
// 返回路径：writes.map((write, index) => index === start ? { ...write, chunk: frame.body, byteLength: resizeCleanupByteLength(frame.body) } : write)。
// 写入目标：start。
// 写入目标：end。
// 直接调用：rendererWriteText。
// 直接调用：resizeRendererFrame。
// 直接调用：writes.map。
// [/maint-note:a2dccaaff4e3]
function normalizeCombinedResizeFrame(
  writes: readonly PendingResizeCleanup[],
): PendingResizeCleanup[] | undefined {
  let start = 0
  let end = writes.length
  if (rendererWriteText(writes[0]!) === synchronizedOutputBegin) start += 1
  if (end > start && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd) end -= 1
  if (end - start !== 1) return undefined
  const frame = resizeRendererFrame(writes[start]!.chunk)
  if (!frame) return undefined
  return writes.map((write, index) => index === start
    ? { ...write, chunk: frame.body, byteLength: resizeCleanupByteLength(frame.body) }
    : write)
}

// [maint-note:913b74d26c48]
// 维护导航：splitStaticRendererCommit；下列为本函数源码中的接口与路径索引。
// 输入 writes：readonly PendingResizeCleanup[]。
// 输入 rendererCacheLines：number | undefined。
// 返回契约：{ cleanupIndex: number; bodyIndex: number } | undefined。
// 分支条件：rendererCacheLines === undefined || writes.length < 3。
// 分支条件：rendererWriteText(writes[0]!) === synchronizedOutputBegin。
// 分支条件：end > start && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd。
// 分支条件：end - start !== 3。
// 分支条件：cleanupLines !== rendererCacheLines || staticOutput === '\n' || !staticOutput.endsWith('\n') || !dynamicBody.endsWith('\n')。
// 返回路径：undefined。
// 返回路径：{ cleanupIndex: start, bodyIndex: start + 2 }。
// 写入目标：start。
// 写入目标：end。
// 直接调用：rendererWriteText。
// 直接调用：resizeCleanupLineCount。
// 直接调用：staticOutput.endsWith。
// 直接调用：dynamicBody.endsWith。
// [/maint-note:913b74d26c48]
function splitStaticRendererCommit(
  writes: readonly PendingResizeCleanup[],
  rendererCacheLines: number | undefined,
): { cleanupIndex: number; bodyIndex: number } | undefined {
  if (rendererCacheLines === undefined || writes.length < 3) return undefined
  let start = 0
  let end = writes.length
  if (rendererWriteText(writes[0]!) === synchronizedOutputBegin) start += 1
  if (end > start && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd) end -= 1
  if (end - start !== 3) return undefined

  const cleanupLines = resizeCleanupLineCount(writes[start]!.chunk)
  const staticOutput = rendererWriteText(writes[start + 1]!)
  const dynamicBody = rendererWriteText(writes[start + 2]!)
  if (
    cleanupLines !== rendererCacheLines
    || staticOutput === '\n'
    || !staticOutput.endsWith('\n')
    || !dynamicBody.endsWith('\n')
  ) return undefined
  return { cleanupIndex: start, bodyIndex: start + 2 }
}

// [maint-note:636d7854879c]
// 维护导航：classifyRendererCommit；下列为本函数源码中的接口与路径索引。
// 输入 writes：readonly PendingResizeCleanup[]。
// 输入 rendererCacheLines：number | undefined。
// 返回契约：RendererCommitClassification | undefined。
// 分支条件：rendererWriteText(writes[0]!) === synchronizedOutputBegin。
// 分支条件：end > frameIndex && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd。
// 分支条件：!isDynamicCommit && !staticCommit。
// 返回路径：undefined。
// 返回路径：{ frameIndex: isDynamicCommit ? frameIndex : -1, staticCommit, body: staticCommit ? rendererWriteText(writes[staticCommit.bodyIndex]!) : frame!.body, }。
// 写入目标：frameIndex。
// 写入目标：end。
// 直接调用：rendererWriteText。
// 直接调用：resizeRendererFrame。
// 直接调用：splitStaticRendererCommit。
// [/maint-note:636d7854879c]
function classifyRendererCommit(
  writes: readonly PendingResizeCleanup[],
  rendererCacheLines: number | undefined,
): RendererCommitClassification | undefined {
  let frameIndex = 0
  let end = writes.length
  if (rendererWriteText(writes[0]!) === synchronizedOutputBegin) frameIndex += 1
  if (end > frameIndex && rendererWriteText(writes[end - 1]!) === synchronizedOutputEnd) end -= 1
  const frame = end - frameIndex === 1
    ? resizeRendererFrame(writes[frameIndex]!.chunk)
    : undefined
  const isDynamicCommit = frame !== undefined && frame.cleanupLines === rendererCacheLines
  const staticCommit = isDynamicCommit
    ? undefined
    : splitStaticRendererCommit(writes, rendererCacheLines)
  if (!isDynamicCommit && !staticCommit) return undefined
  return {
    frameIndex: isDynamicCommit ? frameIndex : -1,
    staticCommit,
    body: staticCommit
      ? rendererWriteText(writes[staticCommit.bodyIndex]!)
      : frame!.body,
  }
}

// [maint-note:eb5d317ea0f8]
// 维护导航：classifyRendererTransaction；下列为本函数源码中的接口与路径索引。
// 输入 transaction：PendingRendererTransaction。
// 输入 rendererCacheLines：number | undefined。
// 返回契约：RendererTransactionClassification | undefined。
// 分支条件：start === end。
// 分支条件：!classification。
// 遍历边界：const [commitIndex, start] ← transaction.commitStarts.entries()。
// 返回路径：undefined。
// 返回路径：finalBody === undefined ? undefined : { commits, finalBody }。
// 写入目标：expectedCacheLines。
// 直接调用：transaction.commitStarts.entries。
// 直接调用：classifyRendererCommit。
// 直接调用：transaction.writes.slice。
// 直接调用：commits.push。
// 直接调用：classification.body.split。
// 直接调用：commits.at。
// [/maint-note:eb5d317ea0f8]
function classifyRendererTransaction(
  transaction: PendingRendererTransaction,
  rendererCacheLines: number | undefined,
): RendererTransactionClassification | undefined {
  const commits: RendererCommitClassification[] = []
  let expectedCacheLines = rendererCacheLines
  for (const [commitIndex, start] of transaction.commitStarts.entries()) {
    const end = transaction.commitStarts[commitIndex + 1] ?? transaction.writes.length
    if (start === end) continue
    const classification = classifyRendererCommit(
      transaction.writes.slice(start, end),
      expectedCacheLines,
    )
    if (!classification) return undefined
    const absolute = {
      ...classification,
      frameIndex: classification.frameIndex < 0
        ? classification.frameIndex
        : start + classification.frameIndex,
      staticCommit: classification.staticCommit
        ? {
            cleanupIndex: start + classification.staticCommit.cleanupIndex,
            bodyIndex: start + classification.staticCommit.bodyIndex,
          }
        : undefined,
    }
    commits.push(absolute)
    expectedCacheLines = classification.body.split('\n').length
  }
  const finalBody = commits.at(-1)?.body
  return finalBody === undefined ? undefined : { commits, finalBody }
}

// [maint-note:0683b78c90e1]
// 维护导航：eraseLines；下列为本函数源码中的接口与路径索引。
// 输入 lineCount：number。
// 返回契约：string。
// 分支条件：lineCount <= 0。
// 返回路径：''。
// 返回路径：`\u001b[2K${'\u001b[1A\u001b[2K'.repeat(lineCount - 1)}\u001b[G`。
// 直接调用：'\u001b[1A\u001b[2K'.repeat。
// [/maint-note:0683b78c90e1]
function eraseLines(lineCount: number): string {
  if (lineCount <= 0) return ''
  return `\u001b[2K${'\u001b[1A\u001b[2K'.repeat(lineCount - 1)}\u001b[G`
}

/**
 * Owns the native resize subscription used by Ink and the width hook.
 *
 * The wrapper has its own event state. It delegates non-resize stream events
 * to the source, while every resize registration (including EventEmitter
 * aliases) stays in this owner's listener table. Ink subscribes through the
 * public stdout event API. On a width shrink, the owner forwards Ink's exact
 * old-frame cleanup and retains the matching synchronous repaint until the
 * application layout has committed and Ink has flushed. If that commit wrote
 * a new frame, the retained repaint is discarded. If Ink deduplicated an
 * unchanged projection, the retained repaint completes the already-visible
 * cleanup so the terminal and renderer cache cannot diverge.
 */
class TerminalResizeOwner {
  private readonly resizeListeners: ResizeListener[] = []
  private readonly beforeResizeListeners = new Set<() => void>()
  private readonly delegatedListeners = new Map<TerminalEventName, TerminalListener[]>()
  private disposed = false
  private previousColumns: unknown
  private resizeWriteMode: 'normal' | 'clearOnly' | 'drop' = 'normal'
  private pendingResizeCleanups = new Map<ResizeListener | undefined, PendingResizeCleanup>()
  private pendingResizeRepaints = new Map<ResizeListener | undefined, PendingResizeCleanup[]>()
  private pendingLayoutRepaint: PendingLayoutRepaint | undefined
  private pendingRendererTransaction: PendingRendererTransaction | undefined
  private nextLayoutRepaintId = 1
  private rendererFlush: (() => Promise<void>) | undefined
  private activeResizeListener: ResizeListener | undefined
  private rendererObservationOpen = false
  private lastRendererFrameBody: string | undefined
  private physicalLinesBeforeCleanup: number | undefined

  // [maint-note:2286413c3fbe]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 source：NodeJS.WriteStream。
  // 写入目标：this.previousColumns。
  // 直接调用：source.on。
  // [/maint-note:2286413c3fbe]
  public constructor(private readonly source: NodeJS.WriteStream) {
    this.previousColumns = source.columns
    source.on('resize', this.onNativeResize)
  }

  // [maint-note:698e6b2debf3]
  // 维护导航：columns；下列为本函数源码中的接口与路径索引。
  // 返回契约：unknown。
  // 返回路径：this.source.columns。
  // [/maint-note:698e6b2debf3]
  public get columns(): unknown {
    return this.source.columns
  }

  // [maint-note:1c1f64763f89]
  // 维护导航：rows；下列为本函数源码中的接口与路径索引。
  // 返回契约：unknown。
  // 返回路径：this.source.rows。
  // [/maint-note:1c1f64763f89]
  public get rows(): unknown {
    return this.source.rows
  }

  // [maint-note:72a0f6a8c290]
  // 维护导航：isTTY；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean | undefined。
  // 返回路径：this.source.isTTY。
  // [/maint-note:72a0f6a8c290]
  public get isTTY(): boolean | undefined {
    return this.source.isTTY
  }

  // [maint-note:66a54bd1636c]
  // 维护导航：destroyed；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.source.destroyed。
  // [/maint-note:66a54bd1636c]
  public get destroyed(): boolean {
    return this.source.destroyed
  }

  // [maint-note:01268dab14d1]
  // 维护导航：writableEnded；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.source.writableEnded。
  // [/maint-note:01268dab14d1]
  public get writableEnded(): boolean {
    return this.source.writableEnded
  }

  // [maint-note:458381f9644c]
  // 维护导航：writable；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.source.writable。
  // [/maint-note:458381f9644c]
  public get writable(): boolean {
    return this.source.writable
  }

  // [maint-note:c5bff6f5e04d]
  // 维护导航：writableLength；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.source.writableLength。
  // [/maint-note:c5bff6f5e04d]
  public get writableLength(): number {
    return this.source.writableLength
  }

  // [maint-note:7e74b1fa519a]
  // 维护导航：write；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 输入 encoding：BufferEncoding | ((error?: Error | null) => void)（可省略）。
  // 输入 callback：(error?: Error | null) => void（可省略）。
  // 返回契约：boolean。
  // 分支条件：this.resizeWriteMode === 'drop'。
  // 分支条件：this.resizeWriteMode === 'clearOnly'。
  // 分支条件：!isResizeCleanup(chunk)。
  // 分支条件：this.pendingRendererTransaction && resizeCleanupByteLength(chunk) > 0。
  // 分支条件：this.rendererObservationOpen && resizeCleanupByteLength(chunk) > 0。
  // 返回路径：true。
  // 返回路径：this.writeToSource(chunk, encoding, callback)。
  // 直接调用：done。
  // 直接调用：isResizeCleanup。
  // 直接调用：this.captureResizeRepaint。
  // 直接调用：this.captureResizeCleanup。
  // 直接调用：resizeCleanupByteLength。
  // 直接调用：this.captureRendererWrite。
  // 直接调用：this.observeRendererWrite。
  // 直接调用：this.writeToSource。
  // [/maint-note:7e74b1fa519a]
  public write(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | ((error?: Error | null) => void),
    callback?: (error?: Error | null) => void,
  ): boolean {
    const done = typeof encoding === 'function' ? encoding : callback
    if (this.resizeWriteMode === 'drop') {
      done?.()
      return true
    }

    if (this.resizeWriteMode === 'clearOnly') {
      if (!isResizeCleanup(chunk)) {
        this.captureResizeRepaint(chunk, encoding, callback)
        return true
      }
      this.captureResizeCleanup(chunk, encoding, callback)
      return true
    }

    if (this.pendingRendererTransaction && resizeCleanupByteLength(chunk) > 0) {
      this.captureRendererWrite(chunk, encoding, callback)
      return true
    }

    if (this.rendererObservationOpen && resizeCleanupByteLength(chunk) > 0) {
      this.observeRendererWrite(chunk)
    }

    return this.writeToSource(chunk, encoding, callback)
  }

  // [maint-note:ddc8bfabfeab]
  // 维护导航：on；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 分支条件：event === 'resize'。
  // 返回路径：this。
  // 直接调用：this.addResizeListener。
  // 直接调用：this.source.on。
  // 直接调用：this.trackDelegatedListener。
  // [/maint-note:ddc8bfabfeab]
  public on(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, false, false)
    else {
      this.source.on(event, listener)
      this.trackDelegatedListener(event, listener)
    }
    return this
  }

  // [maint-note:7cc54e908bf1]
  // 维护导航：addListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 返回路径：this.on(event, listener)。
  // 直接调用：this.on。
  // [/maint-note:7cc54e908bf1]
  public addListener(event: TerminalEventName, listener: TerminalListener): this {
    return this.on(event, listener)
  }

  // [maint-note:4e6a2e6ddc2f]
  // 维护导航：prependListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 分支条件：event === 'resize'。
  // 返回路径：this。
  // 直接调用：this.addResizeListener。
  // 直接调用：this.source.prependListener。
  // 直接调用：this.trackDelegatedListener。
  // [/maint-note:4e6a2e6ddc2f]
  public prependListener(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, true, false)
    else {
      this.source.prependListener(event, listener)
      this.trackDelegatedListener(event, listener, true)
    }
    return this
  }

  // [maint-note:20a0b1160eea]
  // 维护导航：once；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 分支条件：event === 'resize'。
  // 返回路径：this。
  // 直接调用：this.addResizeListener。
  // 直接调用：this.source.once。
  // 直接调用：this.trackDelegatedListener。
  // [/maint-note:20a0b1160eea]
  public once(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, false, true)
    else {
      this.source.once(event, listener)
      this.trackDelegatedListener(event, listener)
    }
    return this
  }

  // [maint-note:b7e8e8637944]
  // 维护导航：prependOnceListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 分支条件：event === 'resize'。
  // 返回路径：this。
  // 直接调用：this.addResizeListener。
  // 直接调用：this.source.prependOnceListener。
  // 直接调用：this.trackDelegatedListener。
  // [/maint-note:b7e8e8637944]
  public prependOnceListener(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, true, true)
    else {
      this.source.prependOnceListener(event, listener)
      this.trackDelegatedListener(event, listener, true)
    }
    return this
  }

  // [maint-note:8ada038883ad]
  // 维护导航：off；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 分支条件：event === 'resize'。
  // 返回路径：this。
  // 直接调用：this.removeResizeListener。
  // 直接调用：this.source.off。
  // 直接调用：this.untrackDelegatedListener。
  // [/maint-note:8ada038883ad]
  public off(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.removeResizeListener(listener)
    else {
      this.source.off(event, listener)
      this.untrackDelegatedListener(event, listener)
    }
    return this
  }

  // [maint-note:25b649386ea3]
  // 维护导航：removeListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：this。
  // 返回路径：this.off(event, listener)。
  // 直接调用：this.off。
  // [/maint-note:25b649386ea3]
  public removeListener(event: TerminalEventName, listener: TerminalListener): this {
    return this.off(event, listener)
  }

  // [maint-note:ce7586df6b8a]
  // 维护导航：removeAllListeners；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName（可省略）。
  // 返回契约：this。
  // 分支条件：event === undefined || event === 'resize'。
  // 分支条件：event === undefined。
  // 分支条件：event !== 'resize'。
  // 返回路径：this。
  // 写入目标：this.resizeListeners.length。
  // 直接调用：this.beforeResizeListeners.clear。
  // 直接调用：this.removeAllDelegatedListeners。
  // [/maint-note:ce7586df6b8a]
  public removeAllListeners(event?: TerminalEventName): this {
    if (event === undefined || event === 'resize') this.resizeListeners.length = 0
    if (event === undefined) {
      this.beforeResizeListeners.clear()
      this.removeAllDelegatedListeners()
    } else if (event !== 'resize') {
      this.removeAllDelegatedListeners(event)
    }
    return this
  }

  // [maint-note:4e0f363ee420]
  // 维护导航：listeners；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 返回契约：TerminalListener[]。
  // 分支条件：event === 'resize'。
  // 返回路径：this.resizeListeners.map(entry => entry.listener)。
  // 返回路径：this.source.listeners(event)。
  // 直接调用：this.resizeListeners.map。
  // 直接调用：this.source.listeners。
  // [/maint-note:4e0f363ee420]
  public listeners(event: TerminalEventName): TerminalListener[] {
    if (event === 'resize') return this.resizeListeners.map(entry => entry.listener)
    return this.source.listeners(event)
  }

  // [maint-note:d7239b471255]
  // 维护导航：rawListeners；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 返回契约：TerminalListener[]。
  // 分支条件：event === 'resize'。
  // 返回路径：this.resizeListeners.map(entry => entry.listener)。
  // 返回路径：this.source.rawListeners(event)。
  // 直接调用：this.resizeListeners.map。
  // 直接调用：this.source.rawListeners。
  // [/maint-note:d7239b471255]
  public rawListeners(event: TerminalEventName): TerminalListener[] {
    if (event === 'resize') return this.resizeListeners.map(entry => entry.listener)
    return this.source.rawListeners(event)
  }

  // [maint-note:1f985868dc91]
  // 维护导航：listenerCount；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 返回契约：number。
  // 分支条件：event === 'resize'。
  // 返回路径：this.resizeListeners.length。
  // 返回路径：this.source.listenerCount(event)。
  // 直接调用：this.source.listenerCount。
  // [/maint-note:1f985868dc91]
  public listenerCount(event: TerminalEventName): number {
    if (event === 'resize') return this.resizeListeners.length
    return this.source.listenerCount(event)
  }

  // [maint-note:f46fd131b7f3]
  // 维护导航：eventNames；下列为本函数源码中的接口与路径索引。
  // 返回契约：Array<string | symbol>。
  // 分支条件：this.resizeListeners.length > 0。
  // 返回路径：names。
  // 直接调用：this.source.eventNames().filter。
  // 直接调用：this.source.eventNames。
  // 直接调用：names.push。
  // [/maint-note:f46fd131b7f3]
  public eventNames(): Array<string | symbol> {
    const names = this.source.eventNames().filter(name => name !== 'resize')
    if (this.resizeListeners.length > 0) names.push('resize')
    return names
  }

  // [maint-note:e64e33357de2]
  // 维护导航：emit；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 args：any[]。
  // 返回契约：boolean。
  // 分支条件：event === 'resize'。
  // 返回路径：true。
  // 返回路径：this.source.emit(event, ...args)。
  // 直接调用：this.onNativeResize。
  // 直接调用：this.source.emit。
  // [/maint-note:e64e33357de2]
  public emit(event: TerminalEventName, ...args: any[]): boolean {
    if (event === 'resize') {
      this.onNativeResize()
      return true
    }
    return this.source.emit(event, ...args)
  }

  // [maint-note:ca5a82ebf06e]
  // 维护导航：subscribeBeforeResize；下列为本函数源码中的接口与路径索引。
  // 输入 listener：() => void。
  // 返回契约：() => void。
  // 分支条件：this.disposed。
  // 返回路径：() => {}。
  // 返回路径：() => { this.beforeResizeListeners.delete(listener) }。
  // 直接调用：this.beforeResizeListeners.add。
  // [/maint-note:ca5a82ebf06e]
  public subscribeBeforeResize(listener: () => void): () => void {
    if (this.disposed) return () => {}
    this.beforeResizeListeners.add(listener)
    return () => { this.beforeResizeListeners.delete(listener) }
  }

  // [maint-note:62f303343455]
  // 维护导航：attachRendererFlush；下列为本函数源码中的接口与路径索引。
  // 输入 flush：() => Promise<void>。
  // 返回契约：() => void。
  // 分支条件：this.disposed。
  // 返回路径：() => {}。
  // 返回路径：() => { if (this.rendererFlush === flush) this.rendererFlush = undefined }。
  // 写入目标：this.rendererFlush。
  // [/maint-note:62f303343455]
  public attachRendererFlush(flush: () => Promise<void>): () => void {
    if (this.disposed) return () => {}
    this.rendererFlush = flush
    return () => {
      if (this.rendererFlush === flush) this.rendererFlush = undefined
    }
  }

  public beginRendererCommit = (): void => {
    const pending = this.pendingLayoutRepaint
    if (this.disposed || this.resizeWriteMode !== 'normal') return
    if (!pending) {
      this.rendererObservationOpen = true
      return
    }
    if (this.pendingRendererTransaction?.repaintId === pending.id) {
      const transaction = this.pendingRendererTransaction
      if (transaction.commitStarts.at(-1) !== transaction.writes.length) {
        transaction.commitStarts.push(transaction.writes.length)
      }
      this.rendererObservationOpen = true
      return
    }
    this.discardRendererTransaction()
    this.rendererObservationOpen = true
    this.pendingRendererTransaction = {
      repaintId: pending.id,
      writes: [],
      commitStarts: [0],
      byteLength: 0,
    }
  }

  // [maint-note:34b020b6066a]
  // 维护导航：completeResizeLayout；下列为本函数源码中的接口与路径索引。
  // 返回契约：Promise<void>。
  // 分支条件：!pending || this.disposed。
  // 分支条件：pending.settling。
  // 返回路径：Promise.resolve()。
  // 返回路径：pending.settling。
  // 返回路径：settling。
  // 写入目标：pending.settling。
  // 直接调用：Promise.resolve。
  // 直接调用：this.settleLayoutRepaint。
  // [/maint-note:34b020b6066a]
  public completeResizeLayout(): Promise<void> {
    const pending = this.pendingLayoutRepaint
    if (!pending || this.disposed) return Promise.resolve()
    if (pending.settling) return pending.settling

    const settling = this.settleLayoutRepaint(pending)
    pending.settling = settling
    return settling
  }

  // [maint-note:2265b0f4a75e]
  // 维护导航：disposeResizeOwner；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：this.disposed。
  // 返回路径：无返回值。
  // 写入目标：this.disposed。
  // 写入目标：this.resizeWriteMode。
  // 写入目标：this.rendererFlush。
  // 写入目标：this.rendererObservationOpen。
  // 写入目标：this.lastRendererFrameBody。
  // 写入目标：this.physicalLinesBeforeCleanup。
  // 写入目标：this.resizeListeners.length。
  // 直接调用：this.pendingResizeCleanups.clear。
  // 直接调用：this.discardResizeRepaints。
  // 直接调用：this.discardRendererTransaction。
  // 直接调用：this.discardPendingLayoutRepaint。
  // 直接调用：this.source.off。
  // 直接调用：this.beforeResizeListeners.clear。
  // 直接调用：this.removeAllDelegatedListeners。
  // [/maint-note:2265b0f4a75e]
  public disposeResizeOwner(): void {
    if (this.disposed) return
    this.disposed = true
    this.resizeWriteMode = 'normal'
    this.pendingResizeCleanups.clear()
    this.discardResizeRepaints()
    this.discardRendererTransaction()
    this.discardPendingLayoutRepaint()
    this.rendererFlush = undefined
    this.rendererObservationOpen = false
    this.lastRendererFrameBody = undefined
    this.physicalLinesBeforeCleanup = undefined
    this.source.off('resize', this.onNativeResize)
    this.beforeResizeListeners.clear()
    this.resizeListeners.length = 0
    this.removeAllDelegatedListeners()
  }

  // [maint-note:3bd30f4f1f84]
  // 维护导航：captureResizeCleanup；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 输入 encoding：BufferEncoding | TerminalWriteCallback（可省略）。
  // 输入 callback：TerminalWriteCallback（可省略）。
  // 返回契约：void。
  // 分支条件：previous && previous.byteLength >= next.byteLength。
  // 返回路径：无返回值。
  // 直接调用：resizeCleanupByteLength。
  // 直接调用：this.pendingResizeCleanups.get。
  // 直接调用：next.callback。
  // 直接调用：previous?.callback。
  // 直接调用：this.pendingResizeCleanups.set。
  // [/maint-note:3bd30f4f1f84]
  private captureResizeCleanup(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): void {
    const next: PendingResizeCleanup = {
      chunk,
      encoding: typeof encoding === 'function' ? undefined : encoding,
      callback: typeof encoding === 'function' ? encoding : callback,
      byteLength: resizeCleanupByteLength(chunk),
    }
    const previous = this.pendingResizeCleanups.get(this.activeResizeListener)
    if (previous && previous.byteLength >= next.byteLength) {
      next.callback?.()
      return
    }
    previous?.callback?.()
    this.pendingResizeCleanups.set(this.activeResizeListener, next)
  }

  // [maint-note:9cec78b2185a]
  // 维护导航：captureResizeRepaint；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 输入 encoding：BufferEncoding | TerminalWriteCallback（可省略）。
  // 输入 callback：TerminalWriteCallback（可省略）。
  // 返回契约：void。
  // 直接调用：this.pendingResizeRepaints.get。
  // 直接调用：writes.push。
  // 直接调用：resizeCleanupByteLength。
  // 直接调用：this.pendingResizeRepaints.set。
  // [/maint-note:9cec78b2185a]
  private captureResizeRepaint(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): void {
    const writes = this.pendingResizeRepaints.get(this.activeResizeListener) ?? []
    writes.push({
      chunk,
      encoding: typeof encoding === 'function' ? undefined : encoding,
      callback: typeof encoding === 'function' ? encoding : callback,
      byteLength: resizeCleanupByteLength(chunk),
    })
    this.pendingResizeRepaints.set(this.activeResizeListener, writes)
  }

  // [maint-note:22575e5bff43]
  // 维护导航：captureRendererWrite；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 输入 encoding：BufferEncoding | TerminalWriteCallback（可省略）。
  // 输入 callback：TerminalWriteCallback（可省略）。
  // 返回契约：void。
  // 分支条件：!pending。
  // 分支条件：pending.writes.length >= MAX_RENDERER_TRANSACTION_WRITES || pending.byteLength + byteLength > MAX_RENDERER_TRANSACTION_BYTES。
  // 返回路径：无返回值。
  // 显式抛出：error。
  // 写入目标：pending.byteLength。
  // 直接调用：resizeCleanupByteLength。
  // 直接调用：this.discardRendererTransaction。
  // 直接调用：this.discardPendingLayoutRepaint。
  // 直接调用：done。
  // 直接调用：pending.writes.push。
  // [/maint-note:22575e5bff43]
  private captureRendererWrite(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): void {
    const pending = this.pendingRendererTransaction
    if (!pending) return
    const byteLength = resizeCleanupByteLength(chunk)
    if (
      pending.writes.length >= MAX_RENDERER_TRANSACTION_WRITES
      || pending.byteLength + byteLength > MAX_RENDERER_TRANSACTION_BYTES
    ) {
      const error = new Error('terminal resize renderer transaction exceeded its bounded buffer')
      const done = typeof encoding === 'function' ? encoding : callback
      this.discardRendererTransaction(error)
      this.discardPendingLayoutRepaint(error)
      done?.(error)
      throw error
    }
    pending.byteLength += byteLength
    pending.writes.push({
      chunk,
      encoding: typeof encoding === 'function' ? undefined : encoding,
      callback: typeof encoding === 'function' ? encoding : callback,
      byteLength,
    })
  }

  // [maint-note:d57c804ca75d]
  // 维护导航：observeRendererWrite；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 返回契约：void。
  // 分支条件：frame?.[2]?.includes('\n')。
  // 分支条件：text.includes('\n')。
  // 分支条件：text === '\u001b[?2026l'。
  // 写入目标：this.lastRendererFrameBody。
  // 写入目标：this.rendererObservationOpen。
  // 直接调用：Buffer.from(chunk).toString。
  // 直接调用：Buffer.from。
  // 直接调用：resizeFramePattern.exec。
  // 直接调用：frame?.[2]?.includes。
  // 直接调用：text.includes。
  // [/maint-note:d57c804ca75d]
  private observeRendererWrite(chunk: string | Uint8Array): void {
    const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
    const frame = resizeFramePattern.exec(text)
    if (frame?.[2]?.includes('\n')) this.lastRendererFrameBody = frame[2]
    else if (text.includes('\n')) this.lastRendererFrameBody = text
    if (text === '\u001b[?2026l') this.rendererObservationOpen = false
  }

  // [maint-note:5c6a2ede0c56]
  // 维护导航：writeToSource；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：string | Uint8Array。
  // 输入 encoding：BufferEncoding | TerminalWriteCallback（可省略）。
  // 输入 callback：TerminalWriteCallback（可省略）。
  // 返回契约：boolean。
  // 返回路径：typeof encoding === 'function' ? this.source.write(chunk, encoding) : this.source.write(chunk, encoding, callback)。
  // 直接调用：this.source.write。
  // [/maint-note:5c6a2ede0c56]
  private writeToSource(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): boolean {
    return typeof encoding === 'function'
      ? this.source.write(chunk, encoding)
      : this.source.write(chunk, encoding, callback)
  }

  // [maint-note:bf4c0518d65e]
  // 维护导航：flushResizeCleanup；下列为本函数源码中的接口与路径索引。
  // 输入 originAlreadyAligned：类型由调用方与函数体约束；缺省表达式 false。
  // 输入 inheritedDebt：PendingLayoutRepaintDebt（可省略）。
  // 返回契约：void。
  // 分支条件：cleanups.length === 1。
  // 分支条件：otherListener === listener。
  // 分支条件：originAlreadyAligned。
  // 分支条件：originAlreadyAligned && cleanups.length === 0 && repaintEntries.length === 1。
  // 分支条件：repaints.length === 0。
  // 分支条件：writes === retainedRepaintSource。
  // 分支条件：!originAlreadyAligned。
  // 分支条件：repaints.length === 0 && originAlreadyAligned && inheritedDebt?.rendererBody。
  // 分支条件：repaints.length > 0。
  // 分支条件：!pending.originAligned。
  // 分支条件：correction。
  // 遍历边界：const [otherListener, writes] ← repaintEntries。
  // 遍历边界：const write ← writes。
  // 遍历边界：const [, cleanup] ← cleanups。
  // 遍历边界：const [, writes] ← repaintEntries。
  // 返回路径：无返回值。
  // 写入目标：retainedRepaintSource。
  // 写入目标：repaints。
  // 写入目标：nativeCleanupLines。
  // 写入目标：rendererMetrics。
  // 写入目标：this.pendingLayoutRepaint。
  // 写入目标：this.nextLayoutRepaintId。
  // 写入目标：pending.originAligned。
  // 直接调用：this.pendingResizeCleanups.entries。
  // 直接调用：this.pendingResizeCleanups.clear。
  // 直接调用：this.pendingResizeRepaints.entries。
  // 直接调用：this.pendingResizeRepaints.clear。
  // 直接调用：repaintEntries.find。
  // 直接调用：write.callback。
  // 直接调用：resizeCleanupLineCount。
  // 直接调用：cleanup.callback。
  // 直接调用：this.writeToSource。
  // 直接调用：normalizeCombinedResizeFrame。
  // 直接调用：rendererFrameMetrics。
  // 直接调用：resizeCleanupByteLength。
  // 直接调用：this.discardPendingLayoutRepaint。
  // 直接调用：this.pendingOriginCorrection。
  // [/maint-note:bf4c0518d65e]
  private flushResizeCleanup(
    originAlreadyAligned = false,
    inheritedDebt?: PendingLayoutRepaintDebt,
  ): void {
    const cleanups = [...this.pendingResizeCleanups.entries()]
    this.pendingResizeCleanups.clear()
    const repaintEntries = [...this.pendingResizeRepaints.entries()]
    this.pendingResizeRepaints.clear()
    let repaints: PendingResizeCleanup[] = []
    let retainedRepaintSource: PendingResizeCleanup[] | undefined
    let nativeCleanupLines = 0

    if (cleanups.length === 1) {
      const [[listener, cleanup]] = cleanups
      retainedRepaintSource = repaintEntries.find(([entry]) => entry === listener)?.[1]
      repaints = retainedRepaintSource ?? []
      for (const [otherListener, writes] of repaintEntries) {
        if (otherListener === listener) continue
        for (const write of writes) write.callback?.()
      }
      nativeCleanupLines = resizeCleanupLineCount(cleanup.chunk) ?? 0
      if (originAlreadyAligned) cleanup.callback?.()
      else this.writeToSource(cleanup.chunk, cleanup.encoding, cleanup.callback)
    } else {
      for (const [, cleanup] of cleanups) cleanup.callback?.()
      if (originAlreadyAligned && cleanups.length === 0 && repaintEntries.length === 1) {
        retainedRepaintSource = repaintEntries[0]![1]
        repaints = normalizeCombinedResizeFrame(retainedRepaintSource) ?? []
        if (repaints.length === 0) retainedRepaintSource = undefined
      }
      for (const [, writes] of repaintEntries) {
        if (writes === retainedRepaintSource) continue
        for (const write of writes) write.callback?.()
      }
      if (!originAlreadyAligned) return
    }

    let rendererMetrics = rendererFrameMetrics(repaints, this.source.columns)
    if (repaints.length === 0 && originAlreadyAligned && inheritedDebt?.rendererBody) {
      const body = inheritedDebt.rendererBody
      repaints = [{ chunk: body, byteLength: resizeCleanupByteLength(body) }]
      rendererMetrics = rendererFrameMetrics(repaints, this.source.columns)
    }
    if (repaints.length > 0) {
      this.discardPendingLayoutRepaint()
      const pending: PendingLayoutRepaint = {
        id: this.nextLayoutRepaintId,
        writes: repaints,
        originAligned: originAlreadyAligned,
        nativeCleanupLines,
        rendererCacheLines: rendererMetrics?.cacheLines ?? inheritedDebt?.rendererCacheLines,
        rendererPhysicalLines: rendererMetrics?.physicalLines ?? inheritedDebt?.rendererPhysicalLines,
        rendererBody: rendererMetrics?.body ?? inheritedDebt?.rendererBody,
        physicalLinesBeforeCleanup: this.physicalLinesBeforeCleanup,
      }
      this.pendingLayoutRepaint = pending
      this.nextLayoutRepaintId += 1
      if (!pending.originAligned) {
        const correction = this.pendingOriginCorrection(pending)
        if (correction) this.writeToSource(correction)
        pending.originAligned = true
      }
    }
  }

  // [maint-note:ab225f87f45b]
  // 维护导航：discardResizeRepaints；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 遍历边界：const writes ← this.pendingResizeRepaints.values()。
  // 遍历边界：const write ← writes。
  // 直接调用：this.pendingResizeRepaints.values。
  // 直接调用：write.callback。
  // 直接调用：this.pendingResizeRepaints.clear。
  // [/maint-note:ab225f87f45b]
  private discardResizeRepaints(): void {
    for (const writes of this.pendingResizeRepaints.values()) {
      for (const write of writes) write.callback?.()
    }
    this.pendingResizeRepaints.clear()
  }

  // [maint-note:32f38f1c1cad]
  // 维护导航：discardPendingLayoutRepaint；下列为本函数源码中的接口与路径索引。
  // 输入 error：Error（可省略）。
  // 返回契约：void。
  // 分支条件：!pending。
  // 遍历边界：const write ← pending.writes。
  // 返回路径：无返回值。
  // 写入目标：this.pendingLayoutRepaint。
  // 直接调用：write.callback。
  // [/maint-note:32f38f1c1cad]
  private discardPendingLayoutRepaint(error?: Error): void {
    const pending = this.pendingLayoutRepaint
    this.pendingLayoutRepaint = undefined
    if (!pending) return
    for (const write of pending.writes) write.callback?.(error)
  }

  // [maint-note:4f2b988ef605]
  // 维护导航：restorePendingLayoutRepaint；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：!pending。
  // 分支条件：this.disposed。
  // 分支条件：pending.rendererBody。
  // 分支条件：correction。
  // 遍历边界：const write ← pending.writes。
  // 返回路径：无返回值。
  // 写入目标：this.pendingLayoutRepaint。
  // 写入目标：this.lastRendererFrameBody。
  // 直接调用：write.callback。
  // 直接调用：this.pendingOriginCorrection。
  // 直接调用：this.writeToSource。
  // [/maint-note:4f2b988ef605]
  private restorePendingLayoutRepaint(): void {
    const pending = this.pendingLayoutRepaint
    this.pendingLayoutRepaint = undefined
    if (!pending) return
    if (this.disposed) {
      for (const write of pending.writes) write.callback?.()
      return
    }
    if (pending.rendererBody) this.lastRendererFrameBody = pending.rendererBody
    const correction = this.pendingOriginCorrection(pending)
    if (correction) this.writeToSource(correction)
    for (const write of pending.writes) {
      this.writeToSource(write.chunk, write.encoding, write.callback)
    }
  }

  // [maint-note:4b11dcb8da71]
  // 维护导航：pendingOriginCorrection；下列为本函数源码中的接口与路径索引。
  // 输入 pending：PendingLayoutRepaint。
  // 返回契约：string。
  // 分支条件：pending.originAligned。
  // 返回路径：''。
  // 返回路径：reflowLines > 0 ? eraseLines(reflowLines + 1) : ''。
  // 直接调用：Math.max。
  // 直接调用：eraseLines。
  // [/maint-note:4b11dcb8da71]
  private pendingOriginCorrection(pending: PendingLayoutRepaint): string {
    if (pending.originAligned) return ''
    const reflowLines = Math.max(
      0,
      (
        pending.physicalLinesBeforeCleanup
        ?? pending.rendererPhysicalLines
        ?? pending.nativeCleanupLines
      ) - pending.nativeCleanupLines,
    )
    // Native cleanup already erased its current row. Erasing that row plus
    // every additional wrapped row lands at the original dynamic origin.
    return reflowLines > 0 ? eraseLines(reflowLines + 1) : ''
  }

  // [maint-note:9ed1b4b0fe47]
  // 维护导航：discardRendererTransaction；下列为本函数源码中的接口与路径索引。
  // 输入 error：Error（可省略）。
  // 返回契约：void。
  // 分支条件：!pending。
  // 遍历边界：const write ← pending.writes。
  // 返回路径：无返回值。
  // 写入目标：this.rendererObservationOpen。
  // 写入目标：this.pendingRendererTransaction。
  // 直接调用：write.callback。
  // [/maint-note:9ed1b4b0fe47]
  private discardRendererTransaction(error?: Error): void {
    this.rendererObservationOpen = false
    const pending = this.pendingRendererTransaction
    this.pendingRendererTransaction = undefined
    if (!pending) return
    for (const write of pending.writes) write.callback?.(error)
  }

  // [maint-note:5a5dc2271542]
  // 维护导航：closeRendererTransaction；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：!transaction。
  // 分支条件：!pending || pending.id !== transaction.repaintId || this.disposed。
  // 分支条件：!classification。
  // 分支条件：!firstCommit。
  // 分支条件：commit?.frameIndex === index。
  // 分支条件：commit?.staticCommit?.cleanupIndex === index。
  // 分支条件：chunk.length === 0。
  // 遍历边界：const write ← transaction.writes。
  // 遍历边界：const write ← pending.writes。
  // 遍历边界：const [index, write] ← transaction.writes.entries()。
  // 返回路径：无返回值。
  // 写入目标：this.rendererObservationOpen。
  // 写入目标：this.pendingRendererTransaction。
  // 写入目标：this.pendingLayoutRepaint。
  // 写入目标：this.lastRendererFrameBody。
  // 写入目标：chunk。
  // 写入目标：previousBody。
  // 直接调用：this.restorePendingLayoutRepaint。
  // 直接调用：write.callback。
  // 直接调用：classifyRendererTransaction。
  // 直接调用：this.writeToSource。
  // 直接调用：transaction.writes.entries。
  // 直接调用：classification.commits.find。
  // 直接调用：resizeRendererFrame。
  // 直接调用：this.pendingOriginCorrection。
  // 直接调用：eraseLines。
  // 直接调用：rendererPhysicalLineCount。
  // 直接调用：resizeCleanupLineCount。
  // [/maint-note:5a5dc2271542]
  private closeRendererTransaction(): void {
    this.rendererObservationOpen = false
    const transaction = this.pendingRendererTransaction
    this.pendingRendererTransaction = undefined
    if (!transaction) {
      this.restorePendingLayoutRepaint()
      return
    }
    const pending = this.pendingLayoutRepaint
    if (!pending || pending.id !== transaction.repaintId || this.disposed) {
      for (const write of transaction.writes) write.callback?.()
      return
    }

    const classification = classifyRendererTransaction(
      transaction,
      pending.rendererCacheLines,
    )
    if (!classification) {
      // Ink acknowledged no complete frame. First restore the physical frame
      // that its cache already accepted, then apply cursor/control writes.
      this.restorePendingLayoutRepaint()
      for (const write of transaction.writes) {
        this.writeToSource(write.chunk, write.encoding, write.callback)
      }
      return
    }
    const [firstCommit] = classification.commits
    if (!firstCommit) return

    // The native cleanup already left the cursor at the dynamic origin. Ink's
    // first post-resize frame was diffed against its suppressed sync repaint,
    // so its leading cleanup would erase Static a second time. Suppress only
    // that exact cleanup and preserve every other renderer byte in order.
    this.pendingLayoutRepaint = undefined
    for (const write of pending.writes) write.callback?.()
    this.lastRendererFrameBody = classification.finalBody
    let previousBody: string | undefined
    for (const [index, write] of transaction.writes.entries()) {
      let chunk = write.chunk
      const commit = classification.commits.find(candidate => (
        candidate.frameIndex === index || candidate.staticCommit?.cleanupIndex === index
      ))
      if (commit?.frameIndex === index) {
        const frame = resizeRendererFrame(write.chunk)!
        const cleanup = previousBody === undefined
          ? this.pendingOriginCorrection(pending)
          : eraseLines(rendererPhysicalLineCount(previousBody, this.source.columns) ?? frame.cleanupLines)
        chunk = cleanup + frame.body
        previousBody = commit.body
      } else if (commit?.staticCommit?.cleanupIndex === index) {
        const cleanupLines = resizeCleanupLineCount(write.chunk) ?? 0
        chunk = previousBody === undefined
          ? this.pendingOriginCorrection(pending)
          : eraseLines(rendererPhysicalLineCount(previousBody, this.source.columns) ?? cleanupLines)
        previousBody = commit.body
        if (chunk.length === 0) {
          write.callback?.()
          continue
        }
      }
      this.writeToSource(chunk, write.encoding, write.callback)
    }
  }

  // [maint-note:b490d16f1269]
  // 维护导航：settleLayoutRepaint；下列为本函数源码中的接口与路径索引。
  // 输入 pending：PendingLayoutRepaint。
  // 返回契约：Promise<void>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：this.pendingLayoutRepaint?.id === pending.id。
  // 分支条件：this.pendingLayoutRepaint?.id !== pending.id。
  // 返回路径：无返回值。
  // 直接调用：this.rendererFlush。
  // 直接调用：this.discardRendererTransaction。
  // 直接调用：this.discardPendingLayoutRepaint。
  // 直接调用：this.closeRendererTransaction。
  // [/maint-note:b490d16f1269]
  private async settleLayoutRepaint(pending: PendingLayoutRepaint): Promise<void> {
    try {
      await this.rendererFlush?.()
    } catch {
      if (this.pendingLayoutRepaint?.id === pending.id) {
        this.discardRendererTransaction()
        this.discardPendingLayoutRepaint()
      }
      return
    }
    if (this.pendingLayoutRepaint?.id !== pending.id) return
    this.closeRendererTransaction()
  }

  // [maint-note:84e1782d4f03]
  // 维护导航：addResizeListener；下列为本函数源码中的接口与路径索引。
  // 输入 listener：TerminalListener。
  // 输入 prepend：boolean。
  // 输入 once：boolean。
  // 返回契约：void。
  // 分支条件：prepend。
  // 直接调用：this.resizeListeners.unshift。
  // 直接调用：this.resizeListeners.push。
  // [/maint-note:84e1782d4f03]
  private addResizeListener(listener: TerminalListener, prepend: boolean, once: boolean): void {
    const entry = { listener, once }
    if (prepend) this.resizeListeners.unshift(entry)
    else this.resizeListeners.push(entry)
  }

  // [maint-note:7699f4a557cc]
  // 维护导航：removeResizeListener；下列为本函数源码中的接口与路径索引。
  // 输入 listener：TerminalListener。
  // 返回契约：void。
  // 分支条件：this.resizeListeners[index]?.listener === listener。
  // 遍历边界：let index = this.resizeListeners.length - 1; index >= 0; index -= 1。
  // 返回路径：无返回值。
  // 写入目标：index。
  // 直接调用：this.resizeListeners.splice。
  // [/maint-note:7699f4a557cc]
  private removeResizeListener(listener: TerminalListener): void {
    for (let index = this.resizeListeners.length - 1; index >= 0; index -= 1) {
      if (this.resizeListeners[index]?.listener === listener) {
        this.resizeListeners.splice(index, 1)
        return
      }
    }
  }

  private onNativeResize = (): void => {
    if (this.disposed) return

    // Once a shrink has erased the dynamic frame and landed at its origin, a
    // later native event owns that empty origin. Retire the older generation
    // instead of restoring its body with geometry that is already stale.
    const inheritedPending = this.pendingLayoutRepaint
    let inheritsAlignedOrigin = inheritedPending?.originAligned === true
    let inheritedDebt: PendingLayoutRepaintDebt | undefined = inheritedPending?.originAligned === true
      ? {
          rendererBody: inheritedPending.rendererBody,
          rendererCacheLines: inheritedPending.rendererCacheLines,
          rendererPhysicalLines: inheritedPending.rendererPhysicalLines,
        }
      : undefined
    if (inheritsAlignedOrigin && inheritedPending) {
      const transaction = this.pendingRendererTransaction
      const hasCompleteRendererAdvance = transaction?.repaintId === inheritedPending.id
        && classifyRendererTransaction(
          transaction,
          inheritedPending.rendererCacheLines,
        ) !== undefined
      if (hasCompleteRendererAdvance) {
        // This transaction is newer than inheritedPending: it may contain
        // Static bytes that the application has already released and Ink will
        // never regenerate. Deliver the exact classified transaction before
        // the new resize listener advances its cache. Its body becomes the
        // physical/cache baseline for the new generation.
        this.closeRendererTransaction()
        inheritsAlignedOrigin = false
        inheritedDebt = undefined
      } else {
        this.discardRendererTransaction()
        this.discardPendingLayoutRepaint()
      }
    } else {
      this.closeRendererTransaction()
    }

    const currentColumns = this.source.columns
    const previousColumns = this.previousColumns
    this.previousColumns = currentColumns
    const isWidthShrink = typeof currentColumns === 'number' &&
      typeof previousColumns === 'number' && currentColumns < previousColumns
    this.physicalLinesBeforeCleanup = isWidthShrink
      ? this.lastRendererFrameBody === undefined
        ? undefined
        : rendererPhysicalLineCount(this.lastRendererFrameBody, currentColumns)
      : undefined
    this.resizeWriteMode = isWidthShrink || inheritsAlignedOrigin ? 'clearOnly' : 'drop'
    this.pendingResizeCleanups.clear()
    this.discardResizeRepaints()

    let hasError = false
    let firstError: unknown
    // [maint-note:6dc6ab759320]
    // 维护导航：remember；下列为本函数源码中的接口与路径索引。
    // 输入 error：unknown。
    // 返回契约：void。
    // 分支条件：!hasError。
    // 写入目标：hasError。
    // 写入目标：firstError。
    // [/maint-note:6dc6ab759320]
    const remember = (error: unknown): void => {
      if (!hasError) {
        hasError = true
        firstError = error
      }
    }

    try {
      for (const entry of [...this.resizeListeners]) {
        if (entry.once) this.removeResizeEntry(entry)
        this.activeResizeListener = entry
        try {
          entry.listener.call(this)
        } catch (error) {
          remember(error)
        } finally {
          this.activeResizeListener = undefined
        }
      }
    } finally {
      this.resizeWriteMode = 'normal'
    }

    try {
      this.flushResizeCleanup(inheritsAlignedOrigin, inheritedDebt)
    } catch (error) {
      remember(error)
    }

    for (const listener of [...this.beforeResizeListeners]) {
      try {
        listener()
      } catch (error) {
        remember(error)
      }
    }

    if (hasError) throw firstError
  }

  // [maint-note:ebe2688cd5fa]
  // 维护导航：trackDelegatedListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 输入 prepend：类型由调用方与函数体约束；缺省表达式 false。
  // 返回契约：void。
  // 分支条件：prepend。
  // 直接调用：this.delegatedListeners.get。
  // 直接调用：listeners.unshift。
  // 直接调用：listeners.push。
  // 直接调用：this.delegatedListeners.set。
  // [/maint-note:ebe2688cd5fa]
  private trackDelegatedListener(
    event: TerminalEventName,
    listener: TerminalListener,
    prepend = false,
  ): void {
    const listeners = this.delegatedListeners.get(event) ?? []
    if (prepend) listeners.unshift(listener)
    else listeners.push(listener)
    this.delegatedListeners.set(event, listeners)
  }

  // [maint-note:4258f62bddb2]
  // 维护导航：untrackDelegatedListener；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName。
  // 输入 listener：TerminalListener。
  // 返回契约：void。
  // 分支条件：!listeners。
  // 分支条件：index >= 0。
  // 分支条件：listeners.length === 0。
  // 返回路径：无返回值。
  // 直接调用：this.delegatedListeners.get。
  // 直接调用：listeners.lastIndexOf。
  // 直接调用：listeners.splice。
  // 直接调用：this.delegatedListeners.delete。
  // [/maint-note:4258f62bddb2]
  private untrackDelegatedListener(event: TerminalEventName, listener: TerminalListener): void {
    const listeners = this.delegatedListeners.get(event)
    if (!listeners) return
    const index = listeners.lastIndexOf(listener)
    if (index >= 0) listeners.splice(index, 1)
    if (listeners.length === 0) this.delegatedListeners.delete(event)
  }

  // [maint-note:7082e1f28007]
  // 维护导航：removeAllDelegatedListeners；下列为本函数源码中的接口与路径索引。
  // 输入 event：TerminalEventName（可省略）。
  // 返回契约：void。
  // 遍历边界：const [eventName, listeners] ← entries。
  // 遍历边界：const listener ← listeners。
  // 直接调用：this.delegatedListeners.entries。
  // 直接调用：this.delegatedListeners.get。
  // 直接调用：this.source.off。
  // 直接调用：this.delegatedListeners.delete。
  // [/maint-note:7082e1f28007]
  private removeAllDelegatedListeners(event?: TerminalEventName): void {
    const entries = event === undefined
      ? [...this.delegatedListeners.entries()]
      : [[event, this.delegatedListeners.get(event) ?? []] as const]
    for (const [eventName, listeners] of entries) {
      for (const listener of listeners) this.source.off(eventName, listener)
      this.delegatedListeners.delete(eventName)
    }
  }

  // [maint-note:3022f0b9177e]
  // 维护导航：removeResizeEntry；下列为本函数源码中的接口与路径索引。
  // 输入 entry：ResizeListener。
  // 返回契约：void。
  // 分支条件：index >= 0。
  // 直接调用：this.resizeListeners.indexOf。
  // 直接调用：this.resizeListeners.splice。
  // [/maint-note:3022f0b9177e]
  private removeResizeEntry(entry: ResizeListener): void {
    const index = this.resizeListeners.indexOf(entry)
    if (index >= 0) this.resizeListeners.splice(index, 1)
  }
}

// [maint-note:b68a6cf45a6a]
// 维护导航：createTerminalResizeOwner；下列为本函数源码中的接口与路径索引。
// 输入 source：NodeJS.WriteStream。
// 返回契约：TerminalResizeOwnerStream。
// 返回路径：new TerminalResizeOwner(source) as unknown as TerminalResizeOwnerStream。
// [/maint-note:b68a6cf45a6a]
export function createTerminalResizeOwner(source: NodeJS.WriteStream): TerminalResizeOwnerStream {
  return new TerminalResizeOwner(source) as unknown as TerminalResizeOwnerStream
}
