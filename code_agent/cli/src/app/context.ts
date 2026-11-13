import {
  createDefaultContextProvider,
  composeContextProviders,
} from '../agent/content/content.js'
import type { ContextProvider } from '../agent/types/content.js'
import type { GitStatusSampler } from '../agent/types/content.js'

/** Coding Agent application boundary for selecting built-in and future providers. */
// [maint-note:a198e7b59730]
// 维护导航：createCodingAgentContextProvider；下列为本函数源码中的接口与路径索引。
// 输入 providers：readonly ContextProvider[]；缺省表达式 []。
// 输入 options：{ readonly gitStatusSampler?: GitStatusSampler }；缺省表达式 {}。
// 返回契约：ContextProvider。
// 返回路径：providers.length === 0 ? createDefaultContextProvider([], options) : composeContextProviders([ createDefaultContextProvider([], options), ...providers, ])。
// 直接调用：createDefaultContextProvider。
// 直接调用：composeContextProviders。
// [/maint-note:a198e7b59730]
export function createCodingAgentContextProvider(
  providers: readonly ContextProvider[] = [],
  options: { readonly gitStatusSampler?: GitStatusSampler } = {},
): ContextProvider {
  return providers.length === 0
    ? createDefaultContextProvider([], options)
    : composeContextProviders([
        createDefaultContextProvider([], options),
        ...providers,
      ])
}
