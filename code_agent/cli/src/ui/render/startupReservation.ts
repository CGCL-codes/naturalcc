import type { HeaderProjection } from '../components/Header.js'
import { measureFrozenHeaderRows } from '../components/Header.js'

export interface StartupReservationState {
  readonly ownerGeneration: number
  readonly headerProjection: HeaderProjection
  readonly phase: 'reserved' | 'released'
}

export type StartupReservationEvent =
  | { readonly type: 'stable_batch_committed'; readonly generation: number }
  | { readonly type: 'clear'; readonly generation: number }
  | { readonly type: 'resize'; readonly generation: number }

export interface StartupReservationProjection {
  readonly active: boolean
  readonly reservedRows: number
}

export type StartupReservationProjector = (
  state: StartupReservationState,
  current: { readonly generation: number; readonly width: number },
) => StartupReservationProjection

export function createStartupReservationState(
  ownerGeneration: number,
  headerProjection: HeaderProjection,
): StartupReservationState {
  return { ownerGeneration, headerProjection, phase: 'reserved' }
}

/**
 * The startup reservation is a monotonic lifecycle. A resize only changes
 * the physical wrapping of the frozen source rows; it never creates another
 * Header owner. A clear creates a new canonical message generation but cannot
 * re-arm the already-submitted Header.
 */
export function reduceStartupReservationState(
  state: StartupReservationState,
  event: StartupReservationEvent,
): StartupReservationState {
  if (state.phase === 'released') return state
  if (event.type === 'stable_batch_committed' && event.generation === state.ownerGeneration) {
    return { ...state, phase: 'released' }
  }
  return state
}

export function projectStartupReservation(
  state: StartupReservationState,
  current: { readonly generation: number; readonly width: number },
): StartupReservationProjection {
  const active = state.phase === 'reserved' && current.generation === state.ownerGeneration
  return {
    active,
    reservedRows: active ? measureFrozenHeaderRows(state.headerProjection, current.width) : 0,
  }
}

/**
 * Keep exactly one derived startup reservation. Repl input, backspace and
 * timing renders do not change this key, while every fact that can change
 * the frozen Header geometry or ownership does. The source projection is
 * immutable, so reference identity is its bounded content identity.
 */
export function createStartupReservationProjector(
  derive: StartupReservationProjector = projectStartupReservation,
): StartupReservationProjector {
  let previous: {
    readonly headerProjection: HeaderProjection
    readonly ownerGeneration: number
    readonly phase: StartupReservationState['phase']
    readonly currentGeneration: number
    readonly width: number
    readonly projection: StartupReservationProjection
  } | undefined

  return (state, current) => {
    if (
      previous?.headerProjection === state.headerProjection
      && previous.ownerGeneration === state.ownerGeneration
      && previous.phase === state.phase
      && previous.currentGeneration === current.generation
      && previous.width === current.width
    ) return previous.projection

    const projection = derive(state, current)
    previous = {
      headerProjection: state.headerProjection,
      ownerGeneration: state.ownerGeneration,
      phase: state.phase,
      currentGeneration: current.generation,
      width: current.width,
      projection,
    }
    return projection
  }
}
