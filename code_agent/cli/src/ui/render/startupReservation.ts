import type { HeaderProjection } from '../components/Header.js'
import { measureFrozenHeaderRows } from '../components/Header.js'

// [maint-note:50860c7fa04a]
// 数据契约导航：StartupReservationState；字段含义以消费端校验为准。
// 成员 ownerGeneration：number。
// 成员 headerProjection：HeaderProjection。
// 成员 phase：'reserved' | 'released'。
// [/maint-note:50860c7fa04a]
export interface StartupReservationState {
  readonly ownerGeneration: number
  readonly headerProjection: HeaderProjection
  readonly phase: 'reserved' | 'released'
}

// [maint-note:ded7633a14ff]
// 数据契约导航：StartupReservationEvent；字段含义以消费端校验为准。
// 类型组合：| { readonly type: 'stable_batch_committed'; readonly generation: number } | { readonly type: 'clear'; readonly generation: number } | { readonly type: 'resize'; readonly generation: number }。
// [/maint-note:ded7633a14ff]
export type StartupReservationEvent =
  | { readonly type: 'stable_batch_committed'; readonly generation: number }
  | { readonly type: 'clear'; readonly generation: number }
  | { readonly type: 'resize'; readonly generation: number }

// [maint-note:73d42b0b1131]
// 数据契约导航：StartupReservationProjection；字段含义以消费端校验为准。
// 成员 active：boolean。
// 成员 reservedRows：number。
// [/maint-note:73d42b0b1131]
export interface StartupReservationProjection {
  readonly active: boolean
  readonly reservedRows: number
}

// [maint-note:d81de6bdbef1]
// 数据契约导航：StartupReservationProjector；字段含义以消费端校验为准。
// 类型组合：( state: StartupReservationState, current: { readonly generation: number; readonly width: number }, ) => StartupReservationProjection。
// [/maint-note:d81de6bdbef1]
export type StartupReservationProjector = (
  state: StartupReservationState,
  current: { readonly generation: number; readonly width: number },
) => StartupReservationProjection

// [maint-note:e6c99af995b7]
// 维护导航：createStartupReservationState；下列为本函数源码中的接口与路径索引。
// 输入 ownerGeneration：number。
// 输入 headerProjection：HeaderProjection。
// 返回契约：StartupReservationState。
// 返回路径：{ ownerGeneration, headerProjection, phase: 'reserved' }。
// [/maint-note:e6c99af995b7]
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
// [maint-note:d3fe35f0b833]
// 维护导航：reduceStartupReservationState；下列为本函数源码中的接口与路径索引。
// 输入 state：StartupReservationState。
// 输入 event：StartupReservationEvent。
// 返回契约：StartupReservationState。
// 分支条件：state.phase === 'released'。
// 分支条件：event.type === 'stable_batch_committed' && event.generation === state.ownerGeneration。
// 返回路径：state。
// 返回路径：{ ...state, phase: 'released' }。
// [/maint-note:d3fe35f0b833]
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

// [maint-note:449dc1473d59]
// 维护导航：projectStartupReservation；下列为本函数源码中的接口与路径索引。
// 输入 state：StartupReservationState。
// 输入 current：{ readonly generation: number; readonly width: number }。
// 返回契约：StartupReservationProjection。
// 返回路径：{ active, reservedRows: active ? measureFrozenHeaderRows(state.headerProjection, current.width) : 0, }。
// 直接调用：measureFrozenHeaderRows。
// [/maint-note:449dc1473d59]
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
// [maint-note:9b95a1dffd8a]
// 维护导航：createStartupReservationProjector；下列为本函数源码中的接口与路径索引。
// 输入 derive：StartupReservationProjector；缺省表达式 projectStartupReservation。
// 返回契约：StartupReservationProjector。
// 返回路径：(state, current) => { if ( previous?.headerProjection === state.headerProjection && previous.ownerGeneration === state.ownerGeneration && previous.phase === state.phase && previous.currentGeneration === current.generation && previous.width 。
// [/maint-note:9b95a1dffd8a]
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
