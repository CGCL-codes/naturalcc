import {
  createDefaultContextProvider,
  composeContextProviders,
} from '../agent/content/content.js'
import type { ContextProvider } from '../agent/types/content.js'
import type { GitStatusSampler } from '../agent/types/content.js'

/** Coding Agent application boundary for selecting built-in and future providers. */
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
