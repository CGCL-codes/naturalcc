import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useStdout } from 'ink'
import { normalizeTerminalWidth } from './width.js'

function finitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

export interface TerminalWidthState {
  width: number
  rows: number
  generation: number
}

export interface TerminalWidthSnapshot extends TerminalWidthState {
  /**
   * Synchronously read the latest native-resize snapshot. React state may
   * still be waiting to render that snapshot; commit owners use this method
   * so a terminal event in the same JS turn cannot freeze an older width.
   */
  readCurrent(): TerminalWidthState
}

export interface TerminalStdoutLike {
  columns?: unknown
  rows?: unknown
  on(event: 'resize', listener: () => void): unknown
  off(event: 'resize', listener: () => void): unknown
  subscribeBeforeResize?: (listener: () => void) => () => void
  completeResizeLayout?: () => Promise<void>
}

export interface TerminalWidthSubscriptionOptions {
  emitInitial?: boolean
  initialGeneration?: number
  fallbackRows?: number
}

export function normalizeTerminalRows(rows: unknown, fallback = 24): number {
  const safeFallback = finitePositive(fallback) ? Math.floor(fallback) : 24
  return finitePositive(rows) ? Math.floor(rows as number) : safeFallback
}

export function subscribeTerminalWidth(
  stdout: TerminalStdoutLike | undefined,
  fallback: number,
  onChange: (state: TerminalWidthState) => void,
  options: TerminalWidthSubscriptionOptions = {},
): () => void {
  let current = normalizeTerminalWidth(stdout?.columns, fallback)
  let currentRows = normalizeTerminalRows(stdout?.rows, options.fallbackRows)
  let generation = Math.max(0, Math.floor(options.initialGeneration ?? 0))
  const sync = (): boolean => {
    const next = normalizeTerminalWidth(stdout?.columns, fallback)
    const nextRows = normalizeTerminalRows(stdout?.rows, options.fallbackRows)
    if (next === current && nextRows === currentRows) return false
    current = next
    currentRows = nextRows
    generation += 1
    onChange({ width: current, rows: currentRows, generation })
    return true
  }
  const unsubscribeBeforeResize = stdout?.subscribeBeforeResize?.(sync)
  if (!unsubscribeBeforeResize) stdout?.on('resize', sync)
  if (options.emitInitial) onChange({ width: current, rows: currentRows, generation })
  sync()
  return () => {
    unsubscribeBeforeResize?.()
    if (!unsubscribeBeforeResize) stdout?.off('resize', sync)
  }
}

export function useTerminalWidth(fallback = 80): TerminalWidthSnapshot {
  const { stdout } = useStdout()
  const generationRef = useRef(0)
  const initialRef = useRef<TerminalWidthState | null>(null)
  if (initialRef.current === null) initialRef.current = {
    width: normalizeTerminalWidth(stdout?.columns, fallback),
    rows: normalizeTerminalRows(stdout?.rows),
    generation: 0,
  }
  const currentRef = useRef<TerminalWidthState>(initialRef.current)
  const [state, setState] = useState<TerminalWidthState>(initialRef.current)

  useEffect(() => {
    let active = true
    const cleanup = subscribeTerminalWidth(stdout, fallback, (next) => {
      if (!active) return
      const previous = currentRef.current
      if (next.width === previous.width && next.rows === previous.rows && next.generation <= previous.generation) return
      const generation = Math.max(
        previous.generation,
        generationRef.current,
        next.generation,
        next.width !== previous.width || next.rows !== previous.rows ? previous.generation + 1 : previous.generation,
      )
      const snapshot = { width: next.width, rows: next.rows, generation }
      // Publish to the synchronous commit seam before asking React to render.
      // A model terminal may arrive before that render/effect flush.
      generationRef.current = generation
      currentRef.current = snapshot
      setState(snapshot)
    }, {
      emitInitial: true,
      initialGeneration: generationRef.current,
    })
    return () => {
      active = false
      cleanup()
    }
  }, [fallback, stdout])

  const readCurrent = useCallback(() => currentRef.current, [])
  useLayoutEffect(() => {
    if (state.generation === 0) return
    void (stdout as TerminalStdoutLike | undefined)?.completeResizeLayout?.()
  }, [state.generation, stdout])
  return {
    ...state,
    readCurrent,
  }
}
