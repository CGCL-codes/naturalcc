// [maint-note:a42c35e448a5]
// 数据契约导航：BoundedLines；字段含义以消费端校验为准。
// 成员 lines：readonly string[]。
// 成员 clipped：boolean。
// 成员 chars：number。
// [/maint-note:a42c35e448a5]
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
// [maint-note:6f28d9bd404f]
// 维护导航：utf16SafePrefix；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxUnits：number。
// 返回契约：string。
// 分支条件：limit === value.length。
// 分支条件：end > 0 && final >= 0xd800 && final <= 0xdbff。
// 返回路径：value。
// 返回路径：value.slice(0, end)。
// 写入目标：end。
// 直接调用：Math.max。
// 直接调用：Math.min。
// 直接调用：Math.floor。
// 直接调用：value.charCodeAt。
// 直接调用：value.slice。
// [/maint-note:6f28d9bd404f]
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
// [maint-note:f56106349f58]
// 维护导航：boundRetainedLines；下列为本函数源码中的接口与路径索引。
// 输入 lines：readonly string[]。
// 输入 maxChars：number。
// 输入 maxRecords：number。
// 返回契约：BoundedLines。
// 分支条件：kept.length >= safeRecords。
// 分支条件：used + completeCost <= safeChars。
// 分支条件：availableTextUnits >= 0。
// 分支条件：prefix.length > 0 || line.length === 0。
// 遍历边界：const line ← lines。
// 返回路径：{ lines: kept, clipped, chars: used }。
// 写入目标：clipped。
// 写入目标：used。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：kept.push。
// 直接调用：utf16SafePrefix。
// [/maint-note:f56106349f58]
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
