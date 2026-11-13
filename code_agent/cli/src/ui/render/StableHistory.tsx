import { Static } from 'ink'
import { useLayoutEffect, useRef } from 'react'
import type { StableBatch } from '../state/messageController.js'
import { Header } from '../components/Header.js'
import { TurnRecords } from './TurnRecords.js'
import { useStableBatchObservation } from './frameLayoutObservation.js'

// [maint-note:2890b1f4e610]
// 数据契约导航：StableHeaderItem；字段含义以消费端校验为准。
// 成员 kind：'header'。
// 成员 id：'repl-header'。
// 成员 width：number。
// [/maint-note:2890b1f4e610]
export interface StableHeaderItem {
  readonly kind: 'header'
  readonly id: 'repl-header'
  readonly width: number
}

// [maint-note:ac81a9bd11ef]
// 数据契约导航：StableBatchItem；字段含义以消费端校验为准。
// 成员 kind：'batch'。
// 成员 id：string。
// [/maint-note:ac81a9bd11ef]
export interface StableBatchItem {
  readonly kind: 'batch'
  readonly id: string
}

// [maint-note:36204f2ecf51]
// 数据契约导航：StableHistoryItem；字段含义以消费端校验为准。
// 类型组合：StableHeaderItem | StableBatchItem。
// [/maint-note:36204f2ecf51]
export type StableHistoryItem = StableHeaderItem | StableBatchItem

// [maint-note:dab02bff0bd1]
// 数据契约导航：StableHistoryProps；字段含义以消费端校验为准。
// 成员 batches：readonly StableBatch[]。
// 成员 width：number。
// 成员 generation：number。
// 成员 headerWidth（可省略）：number。
// 成员 onBatchCommit（可省略）：(batchIds: readonly string[], generation: number) => void。
// [/maint-note:dab02bff0bd1]
interface StableHistoryProps {
  batches: readonly StableBatch[]
  width: number
  /** Bumps on /clear: canonical batches start a new logical generation. */
  generation: number
  /** Only the first session generation owns the one static Header sentinel. */
  headerWidth?: number
  /** Product owner acknowledgement after Ink has consumed new Static items. */
  onBatchCommit?: (batchIds: readonly string[], generation: number) => void
}

// [maint-note:742fb2d72f5a]
// 数据契约导航：StableHistoryLedgerInput；字段含义以消费端校验为准。
// 成员 headerWidth（可省略）：number。
// 成员 batches：readonly StableBatch[]。
// [/maint-note:742fb2d72f5a]
export interface StableHistoryLedgerInput {
  readonly headerWidth?: number
  readonly batches: readonly StableBatch[]
}

// [maint-note:2ac473a0a68b]
// 数据契约导航：StableHistoryLedger；字段含义以消费端校验为准。
// 成员 items：readonly StableHistoryItem[]。
// 成员 retainedPayloadBytes：number。
// 成员 append：void。
// 成员 markRendered：void。
// 成员 releaseRendered：void。
// 成员 getBatch：StableBatch | undefined。
// [/maint-note:2ac473a0a68b]
export interface StableHistoryLedger {
  readonly items: readonly StableHistoryItem[]
  /** Bytes retained only between React render and the Static commit. */
  readonly retainedPayloadBytes: number
  append(input: StableHistoryLedgerInput): void
  markRendered(ids: readonly string[]): void
  releaseRendered(): void
  getBatch(id: string): StableBatch | undefined
}

/**
 * Static needs stable item identities, but it only needs the complete batch
 * while Ink invokes the render callback for that newly appended item. Keep
 * that transient payload outside the Static item list and release it after
 * the layout commit; otherwise every clear/turn would leave a full transcript
 * strongly reachable from a presentation ref.
 */
// [maint-note:c2f0c9f23c5c]
// 维护导航：createStableHistoryLedger；下列为本函数源码中的接口与路径索引。
// 返回契约：StableHistoryLedger。
// 返回路径：{ get items() { return items }, get retainedPayloadBytes() { return retainedPayloadBytes }, append(input) { if (input.headerWidth !== undefined && !ids.has('repl-header')) { items.push({ kind: 'header', id: 'repl-header', width: input.heade。
// [/maint-note:c2f0c9f23c5c]
export function createStableHistoryLedger(): StableHistoryLedger {
  const items: StableHistoryItem[] = []
  const ids = new Set<string>()
  const payloads = new Map<string, StableBatch>()
  const rendered = new Set<string>()
  let retainedPayloadBytes = 0

  return {
    // [maint-note:7fcac1ea1da9]
    // 维护导航：items；下列为本函数源码中的接口与路径索引。
    // 返回路径：items。
    // [/maint-note:7fcac1ea1da9]
    get items() {
      return items
    },
    // [maint-note:61d188b29370]
    // 维护导航：retainedPayloadBytes；下列为本函数源码中的接口与路径索引。
    // 返回路径：retainedPayloadBytes。
    // [/maint-note:61d188b29370]
    get retainedPayloadBytes() {
      return retainedPayloadBytes
    },
    // [maint-note:e9cc5f9fb8ec]
    // 维护导航：append；下列为本函数源码中的接口与路径索引。
    // 输入 input：类型由调用方与函数体约束。
    // 分支条件：input.headerWidth !== undefined && !ids.has('repl-header')。
    // 分支条件：ids.has(batch.batchId)。
    // 遍历边界：const batch ← input.batches。
    // 写入目标：retainedPayloadBytes。
    // 直接调用：ids.has。
    // 直接调用：items.push。
    // 直接调用：ids.add。
    // 直接调用：payloads.set。
    // 直接调用：estimateBatchBytes。
    // [/maint-note:e9cc5f9fb8ec]
    append(input) {
      if (input.headerWidth !== undefined && !ids.has('repl-header')) {
        items.push({ kind: 'header', id: 'repl-header', width: input.headerWidth })
        ids.add('repl-header')
      }
      for (const batch of input.batches) {
        if (ids.has(batch.batchId)) continue
        items.push({ kind: 'batch', id: batch.batchId })
        ids.add(batch.batchId)
        payloads.set(batch.batchId, batch)
        retainedPayloadBytes += estimateBatchBytes(batch)
      }
    },
    // [maint-note:1d4337799d6b]
    // 维护导航：markRendered；下列为本函数源码中的接口与路径索引。
    // 输入 idsToMark：类型由调用方与函数体约束。
    // 遍历边界：const id ← idsToMark。
    // 直接调用：rendered.add。
    // [/maint-note:1d4337799d6b]
    markRendered(idsToMark) {
      for (const id of idsToMark) rendered.add(id)
    },
    // [maint-note:2d4532f6ce42]
    // 维护导航：releaseRendered；下列为本函数源码中的接口与路径索引。
    // 分支条件：!batch。
    // 遍历边界：const id ← rendered。
    // 写入目标：retainedPayloadBytes。
    // 直接调用：payloads.get。
    // 直接调用：estimateBatchBytes。
    // 直接调用：payloads.delete。
    // 直接调用：rendered.clear。
    // [/maint-note:2d4532f6ce42]
    releaseRendered() {
      for (const id of rendered) {
        const batch = payloads.get(id)
        if (!batch) continue
        retainedPayloadBytes -= estimateBatchBytes(batch)
        payloads.delete(id)
      }
      rendered.clear()
    },
    // [maint-note:10b3e8096738]
    // 维护导航：getBatch；下列为本函数源码中的接口与路径索引。
    // 输入 id：类型由调用方与函数体约束。
    // 返回路径：payloads.get(id)。
    // 直接调用：payloads.get。
    // [/maint-note:10b3e8096738]
    getBatch(id) {
      return payloads.get(id)
    },
  }
}

/**
 * stable region: frozen turn batches are committed to the terminal once
 * via Ink's <Static>. Items are append-only with a stable batch ID; a batch
 * is never mutated after it appears here, so already-printed history is
 * never erased or rewritten by later updates.
 */
// [maint-note:af655962f5b5]
// 维护导航：StableHistory；下列为本函数源码中的接口与路径索引。
// 输入 { batches, width, generation, headerWidth, onBatchCommit }：StableHistoryProps。
// 分支条件：ledgerRef.current === null。
// 返回路径：( <Static items={[...ledger.items]}> {(item) => { if (item.kind === 'header') return <Header key={item.id} width={item.width} /> const batch = ledger.getBatch(item.id) if (!batch) return null renderedIdsRef.current.push(item.id) ledger.mark。
// 写入目标：ledgerRef.current。
// 直接调用：useRef。
// 直接调用：createStableHistoryLedger。
// 直接调用：useStableBatchObservation。
// 直接调用：ledger.append。
// 直接调用：useLayoutEffect。
// [/maint-note:af655962f5b5]
export function StableHistory({ batches, width, generation, headerWidth, onBatchCommit }: StableHistoryProps) {
  const ledgerRef = useRef<StableHistoryLedger | null>(null)
  if (ledgerRef.current === null) ledgerRef.current = createStableHistoryLedger()
  const ledger = ledgerRef.current
  const renderedIdsRef = useRef<string[]>([])
  const onStableBatchCommit = useStableBatchObservation()
  // `generation` resets the app's canonical message view, but it must not
  // remount this Static owner: Ink would otherwise be allowed to replay the
  // already-printed Header when a second /clear creates another generation.
  void generation
  ledger.append({ headerWidth, batches })
  // Ink's Static render callback has consumed these immutable records by the
  // time the component's layout effect runs. Releasing here leaves only the
  // bounded ID list in the long-lived presentation owner.
  useLayoutEffect(() => {
    const renderedIds = renderedIdsRef.current.slice()
    if (renderedIds.length > 0) {
      onStableBatchCommit?.(renderedIds)
      onBatchCommit?.(renderedIds, generation)
    }
    ledger.releaseRendered()
    renderedIdsRef.current = []
  }, [generation, ledger, ledger.items.length, onBatchCommit, onStableBatchCommit])
  return (
    <Static items={[...ledger.items]}>
      {(item) => {
        if (item.kind === 'header') return <Header key={item.id} width={item.width} />
        const batch = ledger.getBatch(item.id)
        if (!batch) return null
        renderedIdsRef.current.push(item.id)
        ledger.markRendered([item.id])
        return <StableBatchView key={item.id} batch={batch} width={batch.terminalWidth ?? width} />
      }}
    </Static>
  )
}

// [maint-note:b14428b34835]
// 维护导航：estimateBatchBytes；下列为本函数源码中的接口与路径索引。
// 输入 batch：StableBatch。
// 返回契约：number。
// 返回路径：JSON.stringify(batch).length。
// 直接调用：JSON.stringify。
// [/maint-note:b14428b34835]
function estimateBatchBytes(batch: StableBatch): number {
  return JSON.stringify(batch).length
}

// [maint-note:006dda54a724]
// 维护导航：StableBatchView；下列为本函数源码中的接口与路径索引。
// 输入 { batch, width }：{ batch: StableBatch; width: number }。
// 返回路径：<TurnRecords messages={batch.messages} width={width} separator={batch.endsTurn ? 'after' : 'none'} />。
// [/maint-note:006dda54a724]
function StableBatchView({ batch, width }: { batch: StableBatch; width: number }) {
  return <TurnRecords messages={batch.messages} width={width} separator={batch.endsTurn ? 'after' : 'none'} />
}
