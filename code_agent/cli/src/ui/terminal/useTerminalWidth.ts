import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useStdout } from 'ink'
import { normalizeTerminalWidth } from './width.js'

// [maint-note:49bb6157cb7a]
// 维护导航：finitePositive；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is number。
// 返回路径：typeof value === 'number' && Number.isFinite(value) && value > 0。
// 直接调用：Number.isFinite。
// [/maint-note:49bb6157cb7a]
function finitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

// [maint-note:e591890b7b89]
// 数据契约导航：TerminalWidthState；字段含义以消费端校验为准。
// 成员 width：number。
// 成员 rows：number。
// 成员 generation：number。
// [/maint-note:e591890b7b89]
export interface TerminalWidthState {
  width: number
  rows: number
  generation: number
}

// [maint-note:12a528cb9984]
// 数据契约导航：TerminalWidthSnapshot；字段含义以消费端校验为准。
// 成员 readCurrent：TerminalWidthState。
// [/maint-note:12a528cb9984]
export interface TerminalWidthSnapshot extends TerminalWidthState {
  /**
   * Synchronously read the latest native-resize snapshot. React state may
   * still be waiting to render that snapshot; commit owners use this method
   * so a terminal event in the same JS turn cannot freeze an older width.
   */
  readCurrent(): TerminalWidthState
}

// [maint-note:5d3e93eb68b7]
// 数据契约导航：TerminalStdoutLike；字段含义以消费端校验为准。
// 成员 columns（可省略）：unknown。
// 成员 rows（可省略）：unknown。
// 成员 on：unknown。
// 成员 off：unknown。
// 成员 subscribeBeforeResize（可省略）：(listener: () => void) => () => void。
// 成员 completeResizeLayout（可省略）：() => Promise<void>。
// [/maint-note:5d3e93eb68b7]
export interface TerminalStdoutLike {
  columns?: unknown
  rows?: unknown
  on(event: 'resize', listener: () => void): unknown
  off(event: 'resize', listener: () => void): unknown
  subscribeBeforeResize?: (listener: () => void) => () => void
  completeResizeLayout?: () => Promise<void>
}

// [maint-note:db07977593b5]
// 数据契约导航：TerminalWidthSubscriptionOptions；字段含义以消费端校验为准。
// 成员 emitInitial（可省略）：boolean。
// 成员 initialGeneration（可省略）：number。
// 成员 fallbackRows（可省略）：number。
// [/maint-note:db07977593b5]
export interface TerminalWidthSubscriptionOptions {
  emitInitial?: boolean
  initialGeneration?: number
  fallbackRows?: number
}

// [maint-note:e256e8b9ebf5]
// 维护导航：normalizeTerminalRows；下列为本函数源码中的接口与路径索引。
// 输入 rows：unknown。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 24。
// 返回契约：number。
// 返回路径：finitePositive(rows) ? Math.floor(rows as number) : safeFallback。
// 直接调用：finitePositive。
// 直接调用：Math.floor。
// [/maint-note:e256e8b9ebf5]
export function normalizeTerminalRows(rows: unknown, fallback = 24): number {
  const safeFallback = finitePositive(fallback) ? Math.floor(fallback) : 24
  return finitePositive(rows) ? Math.floor(rows as number) : safeFallback
}

// [maint-note:7b2496825ec9]
// 维护导航：subscribeTerminalWidth；下列为本函数源码中的接口与路径索引。
// 输入 stdout：TerminalStdoutLike | undefined。
// 输入 fallback：number。
// 输入 onChange：(state: TerminalWidthState) => void。
// 输入 options：TerminalWidthSubscriptionOptions；缺省表达式 {}。
// 返回契约：() => void。
// 分支条件：!unsubscribeBeforeResize。
// 分支条件：options.emitInitial。
// 返回路径：() => { unsubscribeBeforeResize?.() if (!unsubscribeBeforeResize) stdout?.off('resize', sync) }。
// 直接调用：normalizeTerminalWidth。
// 直接调用：normalizeTerminalRows。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：stdout?.subscribeBeforeResize。
// 直接调用：stdout?.on。
// 直接调用：onChange。
// 直接调用：sync。
// [/maint-note:7b2496825ec9]
export function subscribeTerminalWidth(
  stdout: TerminalStdoutLike | undefined,
  fallback: number,
  onChange: (state: TerminalWidthState) => void,
  options: TerminalWidthSubscriptionOptions = {},
): () => void {
  let current = normalizeTerminalWidth(stdout?.columns, fallback)
  let currentRows = normalizeTerminalRows(stdout?.rows, options.fallbackRows)
  let generation = Math.max(0, Math.floor(options.initialGeneration ?? 0))
  // [maint-note:fab7faf7e9ec]
  // 维护导航：sync；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：next === current && nextRows === currentRows。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：current。
  // 写入目标：currentRows。
  // 写入目标：generation。
  // 直接调用：normalizeTerminalWidth。
  // 直接调用：normalizeTerminalRows。
  // 直接调用：onChange。
  // [/maint-note:fab7faf7e9ec]
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

// [maint-note:4fc32ff1e3bd]
// 维护导航：useTerminalWidth；下列为本函数源码中的接口与路径索引。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 80。
// 返回契约：TerminalWidthSnapshot。
// 分支条件：initialRef.current === null。
// 返回路径：{ ...state, readCurrent, }。
// 写入目标：initialRef.current。
// 直接调用：useStdout。
// 直接调用：useRef。
// 直接调用：normalizeTerminalWidth。
// 直接调用：normalizeTerminalRows。
// 直接调用：useState。
// 直接调用：useEffect。
// 直接调用：useCallback。
// 直接调用：useLayoutEffect。
// [/maint-note:4fc32ff1e3bd]
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
