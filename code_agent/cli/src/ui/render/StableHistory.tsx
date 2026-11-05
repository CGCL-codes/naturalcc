import { Static } from 'ink'
import { useLayoutEffect, useRef } from 'react'
import type { StableBatch } from '../state/messageController.js'
import { Header } from '../components/Header.js'
import { TurnRecords } from './TurnRecords.js'
import { useStableBatchObservation } from './frameLayoutObservation.js'

export interface StableHeaderItem {
  readonly kind: 'header'
  readonly id: 'repl-header'
  readonly width: number
}

export interface StableBatchItem {
  readonly kind: 'batch'
  readonly id: string
}

export type StableHistoryItem = StableHeaderItem | StableBatchItem

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

export interface StableHistoryLedgerInput {
  readonly headerWidth?: number
  readonly batches: readonly StableBatch[]
}

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
export function createStableHistoryLedger(): StableHistoryLedger {
  const items: StableHistoryItem[] = []
  const ids = new Set<string>()
  const payloads = new Map<string, StableBatch>()
  const rendered = new Set<string>()
  let retainedPayloadBytes = 0

  return {
    get items() {
      return items
    },
    get retainedPayloadBytes() {
      return retainedPayloadBytes
    },
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
    markRendered(idsToMark) {
      for (const id of idsToMark) rendered.add(id)
    },
    releaseRendered() {
      for (const id of rendered) {
        const batch = payloads.get(id)
        if (!batch) continue
        retainedPayloadBytes -= estimateBatchBytes(batch)
        payloads.delete(id)
      }
      rendered.clear()
    },
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

function estimateBatchBytes(batch: StableBatch): number {
  return JSON.stringify(batch).length
}

function StableBatchView({ batch, width }: { batch: StableBatch; width: number }) {
  return <TurnRecords messages={batch.messages} width={width} separator={batch.endsTurn ? 'after' : 'none'} />
}
