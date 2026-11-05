import type { ModelSystemMessage } from '../../model/types/types.js'

/** The complete non-history inputs consumed by the one final request builder. */
export interface FinalRequestProjection {
  readonly requestOnlySections: readonly string[]
  readonly trailingFrameworkMessages: readonly ModelSystemMessage[]
}

/**
 * Session-owned environment projection frozen for one ordinary model request.
 * Acknowledgement is valid only at the logical dispatch boundary.
 */
export interface PreparedFrameworkContext {
  readonly trailingMessages: readonly ModelSystemMessage[]
  /** Read-only freshness check; never consumes the candidate. */
  validate(): boolean
  /** Non-throwing publish after all dispatch candidates validated. */
  publish(): void
  /** @deprecated Compatibility shim for direct callers. */
  acknowledge(): boolean
}

/** Narrow core port: the Runner neither reads a clock nor knows about Git. */
export interface FrameworkContextPort {
  freezeForRequest(): PreparedFrameworkContext
  dispose(): void
}
