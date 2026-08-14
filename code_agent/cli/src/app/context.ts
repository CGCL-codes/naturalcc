import {
  createDefaultContextProvider,
  composeContextProviders,
} from '../agent/content/content.js'
import type { ContextProvider } from '../agent/types/content.js'

/** Coding Agent application boundary for selecting built-in and future providers. */
export function createCodingAgentContextProvider(
  providers: readonly ContextProvider[] = [],
): ContextProvider {
  return providers.length === 0
    ? createDefaultContextProvider()
    : composeContextProviders([
        createDefaultContextProvider(),
        ...providers,
      ])
}
