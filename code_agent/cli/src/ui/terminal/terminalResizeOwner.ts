import stripAnsi from 'strip-ansi'

import { splitDisplayGraphemes } from './text.js'
import { displayLineWidth } from './width.js'

type TerminalEventName = string | symbol
type TerminalListener = (this: TerminalResizeOwnerStream, ...args: any[]) => void
type ResizeListener = {
  listener: TerminalListener
  once: boolean
}
type TerminalWriteCallback = (error?: Error | null) => void
type PendingResizeCleanup = {
  chunk: string | Uint8Array
  encoding?: BufferEncoding
  callback?: TerminalWriteCallback
  byteLength: number
}
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
type PendingLayoutRepaintDebt = Pick<
  PendingLayoutRepaint,
  'rendererBody' | 'rendererCacheLines' | 'rendererPhysicalLines'
>
type PendingRendererTransaction = {
  repaintId: number
  writes: PendingResizeCleanup[]
  commitStarts: number[]
  byteLength: number
}
type RendererCommitClassification = {
  frameIndex: number
  staticCommit?: { cleanupIndex: number; bodyIndex: number }
  body: string
}
type RendererTransactionClassification = {
  commits: RendererCommitClassification[]
  finalBody: string
}

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

function isResizeCleanup(chunk: string | Uint8Array): boolean {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
  return resizeCleanupPattern.test(text)
}

function resizeCleanupByteLength(chunk: string | Uint8Array): number {
  return typeof chunk === 'string' ? Buffer.byteLength(chunk, 'utf8') : chunk.byteLength
}

function resizeCleanupLineCount(chunk: string | Uint8Array): number | undefined {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
  if (!resizeCleanupPattern.test(text)) return undefined
  return text.match(/\u001b\[2K/gu)?.length
}

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

function rendererWriteText(write: PendingResizeCleanup): string {
  return typeof write.chunk === 'string'
    ? write.chunk
    : Buffer.from(write.chunk).toString('utf8')
}

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

  public constructor(private readonly source: NodeJS.WriteStream) {
    this.previousColumns = source.columns
    source.on('resize', this.onNativeResize)
  }

  public get columns(): unknown {
    return this.source.columns
  }

  public get rows(): unknown {
    return this.source.rows
  }

  public get isTTY(): boolean | undefined {
    return this.source.isTTY
  }

  public get destroyed(): boolean {
    return this.source.destroyed
  }

  public get writableEnded(): boolean {
    return this.source.writableEnded
  }

  public get writable(): boolean {
    return this.source.writable
  }

  public get writableLength(): number {
    return this.source.writableLength
  }

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

  public on(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, false, false)
    else {
      this.source.on(event, listener)
      this.trackDelegatedListener(event, listener)
    }
    return this
  }

  public addListener(event: TerminalEventName, listener: TerminalListener): this {
    return this.on(event, listener)
  }

  public prependListener(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, true, false)
    else {
      this.source.prependListener(event, listener)
      this.trackDelegatedListener(event, listener, true)
    }
    return this
  }

  public once(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, false, true)
    else {
      this.source.once(event, listener)
      this.trackDelegatedListener(event, listener)
    }
    return this
  }

  public prependOnceListener(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.addResizeListener(listener, true, true)
    else {
      this.source.prependOnceListener(event, listener)
      this.trackDelegatedListener(event, listener, true)
    }
    return this
  }

  public off(event: TerminalEventName, listener: TerminalListener): this {
    if (event === 'resize') this.removeResizeListener(listener)
    else {
      this.source.off(event, listener)
      this.untrackDelegatedListener(event, listener)
    }
    return this
  }

  public removeListener(event: TerminalEventName, listener: TerminalListener): this {
    return this.off(event, listener)
  }

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

  public listeners(event: TerminalEventName): TerminalListener[] {
    if (event === 'resize') return this.resizeListeners.map(entry => entry.listener)
    return this.source.listeners(event)
  }

  public rawListeners(event: TerminalEventName): TerminalListener[] {
    if (event === 'resize') return this.resizeListeners.map(entry => entry.listener)
    return this.source.rawListeners(event)
  }

  public listenerCount(event: TerminalEventName): number {
    if (event === 'resize') return this.resizeListeners.length
    return this.source.listenerCount(event)
  }

  public eventNames(): Array<string | symbol> {
    const names = this.source.eventNames().filter(name => name !== 'resize')
    if (this.resizeListeners.length > 0) names.push('resize')
    return names
  }

  public emit(event: TerminalEventName, ...args: any[]): boolean {
    if (event === 'resize') {
      this.onNativeResize()
      return true
    }
    return this.source.emit(event, ...args)
  }

  public subscribeBeforeResize(listener: () => void): () => void {
    if (this.disposed) return () => {}
    this.beforeResizeListeners.add(listener)
    return () => { this.beforeResizeListeners.delete(listener) }
  }

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

  public completeResizeLayout(): Promise<void> {
    const pending = this.pendingLayoutRepaint
    if (!pending || this.disposed) return Promise.resolve()
    if (pending.settling) return pending.settling

    const settling = this.settleLayoutRepaint(pending)
    pending.settling = settling
    return settling
  }

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

  private observeRendererWrite(chunk: string | Uint8Array): void {
    const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8')
    const frame = resizeFramePattern.exec(text)
    if (frame?.[2]?.includes('\n')) this.lastRendererFrameBody = frame[2]
    else if (text.includes('\n')) this.lastRendererFrameBody = text
    if (text === '\u001b[?2026l') this.rendererObservationOpen = false
  }

  private writeToSource(
    chunk: string | Uint8Array,
    encoding?: BufferEncoding | TerminalWriteCallback,
    callback?: TerminalWriteCallback,
  ): boolean {
    return typeof encoding === 'function'
      ? this.source.write(chunk, encoding)
      : this.source.write(chunk, encoding, callback)
  }

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

  private discardResizeRepaints(): void {
    for (const writes of this.pendingResizeRepaints.values()) {
      for (const write of writes) write.callback?.()
    }
    this.pendingResizeRepaints.clear()
  }

  private discardPendingLayoutRepaint(error?: Error): void {
    const pending = this.pendingLayoutRepaint
    this.pendingLayoutRepaint = undefined
    if (!pending) return
    for (const write of pending.writes) write.callback?.(error)
  }

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

  private discardRendererTransaction(error?: Error): void {
    this.rendererObservationOpen = false
    const pending = this.pendingRendererTransaction
    this.pendingRendererTransaction = undefined
    if (!pending) return
    for (const write of pending.writes) write.callback?.(error)
  }

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

  private addResizeListener(listener: TerminalListener, prepend: boolean, once: boolean): void {
    const entry = { listener, once }
    if (prepend) this.resizeListeners.unshift(entry)
    else this.resizeListeners.push(entry)
  }

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

  private untrackDelegatedListener(event: TerminalEventName, listener: TerminalListener): void {
    const listeners = this.delegatedListeners.get(event)
    if (!listeners) return
    const index = listeners.lastIndexOf(listener)
    if (index >= 0) listeners.splice(index, 1)
    if (listeners.length === 0) this.delegatedListeners.delete(event)
  }

  private removeAllDelegatedListeners(event?: TerminalEventName): void {
    const entries = event === undefined
      ? [...this.delegatedListeners.entries()]
      : [[event, this.delegatedListeners.get(event) ?? []] as const]
    for (const [eventName, listeners] of entries) {
      for (const listener of listeners) this.source.off(eventName, listener)
      this.delegatedListeners.delete(eventName)
    }
  }

  private removeResizeEntry(entry: ResizeListener): void {
    const index = this.resizeListeners.indexOf(entry)
    if (index >= 0) this.resizeListeners.splice(index, 1)
  }
}

export function createTerminalResizeOwner(source: NodeJS.WriteStream): TerminalResizeOwnerStream {
  return new TerminalResizeOwner(source) as unknown as TerminalResizeOwnerStream
}
