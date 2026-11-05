import { createContext, createElement, useContext, type ReactNode } from 'react'
import type { FrameLayout } from './frameLayout.js'

type FrameLayoutObserver = (layout: FrameLayout) => void
type StableBatchObserver = (batchIds: readonly string[]) => void

interface FrameLayoutObservation {
  readonly onChange: FrameLayoutObserver
  readonly onStableBatchCommit?: StableBatchObserver
}

/**
 * Internal evidence seam for production-root PTY tests. Repl has no
 * observer prop in its product-facing API; without a provider this context is
 * a no-op and cannot affect rendering.
 */
const FrameLayoutObservationContext = createContext<FrameLayoutObservation | null>(null)

export function FrameLayoutObservationProvider({
  onChange,
  onStableBatchCommit,
  children,
}: {
  onChange: FrameLayoutObserver
  onStableBatchCommit?: StableBatchObserver
  children?: ReactNode
}) {
  return createElement(
    FrameLayoutObservationContext.Provider,
    { value: { onChange, onStableBatchCommit } },
    children,
  )
}

export function useFrameLayoutObservation(): FrameLayoutObserver | null {
  return useContext(FrameLayoutObservationContext)?.onChange ?? null
}

export function useStableBatchObservation(): StableBatchObserver | null {
  return useContext(FrameLayoutObservationContext)?.onStableBatchCommit ?? null
}
