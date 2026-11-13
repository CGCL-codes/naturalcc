import type { ModelSystemMessage } from '../../model/types/types.js'

/** The complete non-history inputs consumed by the one final request builder. */
// [maint-note:0696f24e7e51]
// 数据契约导航：FinalRequestProjection；字段含义以消费端校验为准。
// 成员 requestOnlySections：readonly string[]。
// 成员 trailingFrameworkMessages：readonly ModelSystemMessage[]。
// [/maint-note:0696f24e7e51]
export interface FinalRequestProjection {
  readonly requestOnlySections: readonly string[]
  readonly trailingFrameworkMessages: readonly ModelSystemMessage[]
}

/**
 * Session-owned environment projection frozen for one ordinary model request.
 * Acknowledgement is valid only at the logical dispatch boundary.
 */
// [maint-note:7bf3a01ded7c]
// 数据契约导航：PreparedFrameworkContext；字段含义以消费端校验为准。
// 成员 trailingMessages：readonly ModelSystemMessage[]。
// 成员 validate：boolean。
// 成员 publish：void。
// 成员 acknowledge：boolean。
// [/maint-note:7bf3a01ded7c]
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
// [maint-note:e69e90716613]
// 数据契约导航：FrameworkContextPort；字段含义以消费端校验为准。
// 成员 freezeForRequest：PreparedFrameworkContext。
// 成员 dispose：void。
// [/maint-note:e69e90716613]
export interface FrameworkContextPort {
  freezeForRequest(): PreparedFrameworkContext
  dispose(): void
}
