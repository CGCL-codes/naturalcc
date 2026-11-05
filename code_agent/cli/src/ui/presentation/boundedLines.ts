export interface BoundedLines {
  readonly lines: readonly string[]
  readonly clipped: boolean
  /** UTF-16 code units plus one separator unit per retained line. */
  readonly chars: number
}

/**
 * Return the longest UTF-16 prefix no larger than `maxUnits` without
 * leaving a high surrogate at the end. The surrounding UI budgets are
 * defined in JavaScript string units, so slicing and accounting must use
 * the same unit.
 */
export function utf16SafePrefix(value: string, maxUnits: number): string {
  const limit = Math.max(0, Math.min(value.length, Math.floor(maxUnits)))
  if (limit === value.length) return value
  let end = limit
  const final = value.charCodeAt(end - 1)
  if (end > 0 && final >= 0xd800 && final <= 0xdbff) end -= 1
  return value.slice(0, end)
}

/**
 * Bound retained logical records once for both presenter output and the
 * reducer-owned detail store. Empty lines still consume their separator;
 * clipping is explicit and never exceeds either hard cap.
 */
export function boundRetainedLines(
  lines: readonly string[],
  maxChars: number,
  maxRecords: number,
): BoundedLines {
  const safeChars = Math.max(0, Math.floor(maxChars))
  const safeRecords = Math.max(0, Math.floor(maxRecords))
  const kept: string[] = []
  let used = 0
  let clipped = false

  for (const line of lines) {
    if (kept.length >= safeRecords) {
      clipped = true
      break
    }
    const completeCost = line.length + 1
    if (used + completeCost <= safeChars) {
      kept.push(line)
      used += completeCost
      continue
    }

    const availableTextUnits = safeChars - used - 1
    if (availableTextUnits >= 0) {
      const prefix = utf16SafePrefix(line, availableTextUnits)
      if (prefix.length > 0 || line.length === 0) {
        kept.push(prefix)
        used += prefix.length + 1
      }
    }
    clipped = true
    break
  }

  return { lines: kept, clipped, chars: used }
}
