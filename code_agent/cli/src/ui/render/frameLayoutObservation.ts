import { createContext, createElement, useContext, type ReactNode } from 'react'
import type { FrameLayout } from './frameLayout.js'

// [maint-note:be4186b9f72b]
// 数据契约导航：FrameLayoutObserver；字段含义以消费端校验为准。
// 类型组合：(layout: FrameLayout) => void。
// [/maint-note:be4186b9f72b]
type FrameLayoutObserver = (layout: FrameLayout) => void
// [maint-note:e7b882d30f10]
// 数据契约导航：StableBatchObserver；字段含义以消费端校验为准。
// 类型组合：(batchIds: readonly string[]) => void。
// [/maint-note:e7b882d30f10]
type StableBatchObserver = (batchIds: readonly string[]) => void

// [maint-note:3c9949be5477]
// 数据契约导航：FrameLayoutObservation；字段含义以消费端校验为准。
// 成员 onChange：FrameLayoutObserver。
// 成员 onStableBatchCommit（可省略）：StableBatchObserver。
// [/maint-note:3c9949be5477]
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

// [maint-note:f611176c127d]
// 维护导航：FrameLayoutObservationProvider；下列为本函数源码中的接口与路径索引。
// 输入 { onChange, onStableBatchCommit, children, }：{ onChange: FrameLayoutObserver onStableBatchCommit?: StableBatchObserver children?: ReactNode }。
// 返回路径：createElement( FrameLayoutObservationContext.Provider, { value: { onChange, onStableBatchCommit } }, children, )。
// 直接调用：createElement。
// [/maint-note:f611176c127d]
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

// [maint-note:c808ba4c1774]
// 维护导航：useFrameLayoutObservation；下列为本函数源码中的接口与路径索引。
// 返回契约：FrameLayoutObserver | null。
// 返回路径：useContext(FrameLayoutObservationContext)?.onChange ?? null。
// 直接调用：useContext。
// [/maint-note:c808ba4c1774]
export function useFrameLayoutObservation(): FrameLayoutObserver | null {
  return useContext(FrameLayoutObservationContext)?.onChange ?? null
}

// [maint-note:1dc1568009a8]
// 维护导航：useStableBatchObservation；下列为本函数源码中的接口与路径索引。
// 返回契约：StableBatchObserver | null。
// 返回路径：useContext(FrameLayoutObservationContext)?.onStableBatchCommit ?? null。
// 直接调用：useContext。
// [/maint-note:1dc1568009a8]
export function useStableBatchObservation(): StableBatchObserver | null {
  return useContext(FrameLayoutObservationContext)?.onStableBatchCommit ?? null
}
