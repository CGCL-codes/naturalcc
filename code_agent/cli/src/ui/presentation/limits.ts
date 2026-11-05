/**
 * centralized presentation/detail budgets. All default-view folding and
 * detail-cache limits live here as testable constants — no scattered magic
 * numbers. The first limit reached wins.
 */
export const UiPresentationLimits = Object.freeze({
  /** Default presenter summary: at most this many logical lines. */
  summaryMaxLogicalLines: 40,
  /** Default presenter summary: at most this many characters. */
  summaryMaxChars: 6_000,
  /**
   * hard contract: the apply_patch presentation body may occupy at
   * most this many PHYSICAL visible rows at the current terminal width
   * (titles, wrapped diff rows, ⋮ and notices included). Stable tool
   * message chrome outside the body is not counted here, but stays under
   * the whole-frame ledger.
   */
  applyPatchDefaultPhysicalRows: 40,
  /** apply_patch body shares the summary character ceiling. */
  applyPatchMaxChars: 6_000,
  /** One detail record retains at most this many characters. */
  detailRecordMaxChars: 100_000,
  /** One detail record retains at most this many lines (dual cap). */
  detailRecordMaxRecords: 4_000,
  /** The whole REPL detail cache retains at most this many characters. */
  detailCacheMaxChars: 1_000_000,
  /** Detail viewer: rendered viewport rows plus this much overscan. */
  detailViewportOverscanRows: 10,
})

export type UiPresentationLimitsType = typeof UiPresentationLimits
