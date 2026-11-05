import { useCallback, useEffect, useReducer, useRef } from 'react'
import type { AgentEvent } from '../../agent/types/event.js'
import {
  SessionTimingController,
  type MonotonicClock,
  type SessionGeneration,
  type SessionTimingSnapshot,
} from '../state/sessionTiming.js'

// the live timer drives a human-readable status line only. A 500 ms tick
// keeps whole-tree updates out of the high-frequency delta path without
// changing the timing semantics (durations still come from the injected
// controller clock in tests).
const TICK_MS = 500

/** React projection for the UI-only SessionTimingController. */
export function useSessionTiming(generation: SessionGeneration = 'initial', clock?: MonotonicClock) {
  const controllerRef = useRef<SessionTimingController | null>(null)
  if (!controllerRef.current) controllerRef.current = new SessionTimingController(clock, generation)
  const controller = controllerRef.current
  const [, refresh] = useReducer((value: number) => value + 1, 0)
  const invalidate = useCallback(() => refresh(), [])

  const snapshot = controller.snapshot()
  useEffect(() => {
    if (!snapshot.live) return
    const timer = setInterval(invalidate, TICK_MS)
    return () => clearInterval(timer)
  }, [invalidate, snapshot.live])

  const beginRun = useCallback((runId: number) => {
    controller.beginRun(runId)
    invalidate()
  }, [controller, invalidate])

  const handleEvent = useCallback((runId: number, event: AgentEvent) => {
    // Most AgentEvents (especially assistant_delta) do not change timing.
    // Keep event consumption immediate, but only invalidate the React timing
    // projection when the controller actually paused/resumed approval.
    if (controller.handleEvent(runId, event)) invalidate()
  }, [controller, invalidate])

  const finishRun = useCallback((runId: number) => {
    controller.finishRun(runId)
    invalidate()
  }, [controller, invalidate])

  const clearSession = useCallback((nextGeneration?: SessionGeneration) => {
    controller.clearSession(nextGeneration)
    invalidate()
  }, [controller, invalidate])

  const finalizeSession = useCallback(() => {
    controller.finalizeSession()
    invalidate()
  }, [controller, invalidate])

  return {
    controller,
    snapshot,
    beginRun,
    handleEvent,
    finishRun,
    clearSession,
    finalizeSession,
  } as const
}

export type SessionTimingView = SessionTimingSnapshot
