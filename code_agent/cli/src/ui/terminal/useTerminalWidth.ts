import { useEffect, useRef, useState } from 'react'
import { useStdout } from 'ink'
import { normalizeTerminalWidth } from './width.js'

export interface TerminalWidthState {
  width: number
  generation: number
}

export interface TerminalStdoutLike {
  columns?: unknown
  on(event: 'resize', listener: () => void): unknown
  off(event: 'resize', listener: () => void): unknown
}

export interface TerminalWidthSubscriptionOptions {
  emitInitial?: boolean
  initialGeneration?: number
}

export function subscribeTerminalWidth(
  stdout: TerminalStdoutLike | undefined,
  fallback: number,
  onChange: (state: TerminalWidthState) => void,
  options: TerminalWidthSubscriptionOptions = {},
): () => void {
  let current = normalizeTerminalWidth(stdout?.columns, fallback)
  let generation = Math.max(0, Math.floor(options.initialGeneration ?? 0))
  const sync = () => {
    const next = normalizeTerminalWidth(stdout?.columns, fallback)
    if (next === current) return
    current = next
    generation += 1
    onChange({ width: current, generation })
  }
  stdout?.on('resize', sync)
  if (options.emitInitial) onChange({ width: current, generation })
  sync()
  return () => { stdout?.off('resize', sync) }
}

export function useTerminalWidth(fallback = 80): TerminalWidthState {
  const { stdout } = useStdout()
  const generationRef = useRef(0)
  const [state, setState] = useState<TerminalWidthState>(() => ({
    width: normalizeTerminalWidth(stdout?.columns, fallback),
    generation: 0,
  }))

  useEffect(() => {
    let active = true
    const cleanup = subscribeTerminalWidth(stdout, fallback, (next) => {
      if (!active) return
      setState((previous) => {
        if (next.width === previous.width && next.generation <= previous.generation) return previous
        const generation = Math.max(
          previous.generation,
          generationRef.current,
          next.generation,
          next.width === previous.width ? previous.generation : previous.generation + 1,
        )
        generationRef.current = generation
        return { width: next.width, generation }
      })
    }, { emitInitial: true, initialGeneration: generationRef.current })
    return () => {
      active = false
      cleanup()
    }
  }, [fallback, stdout])

  return state
}
