// [maint-note:6abe23c4b86c]
// 数据契约导航：LiveViewportLimits；字段含义以消费端校验为准。
// 成员 maxLines：number。
// 成员 maxChars：number。
// [/maint-note:6abe23c4b86c]
export interface LiveViewportLimits {
  /** Maximum logical lines kept in the live viewport. */
  maxLines: number
  /** Maximum characters kept in the live viewport. */
  maxChars: number
}

export const DEFAULT_LIVE_VIEWPORT_LIMITS: LiveViewportLimits = {
  maxLines: 24,
  maxChars: 4_000,
}

// [maint-note:50e258671ebc]
// 数据契约导航：LiveViewportProjection；字段含义以消费端校验为准。
// 成员 text：string。
// 成员 hiddenLines：number。
// 成员 hiddenChars：number。
// 成员 notice：string | null。
// [/maint-note:50e258671ebc]
export interface LiveViewportProjection {
  /** Bounded tail of the live draft, safe to hand to the renderer. */
  text: string
  /** Lines dropped from the front (display only; canonical draft is intact). */
  hiddenLines: number
  hiddenChars: number
  /** Stable notice line when anything was dropped. */
  notice: string | null
}

/**
 * live viewport: the streaming draft is projected to a fixed, bounded
 * window of recent logical lines. This bounds only what the dynamic region
 * renders - the canonical assistant content in the message reducer and the
 * model stream are untouched.
 */
// [maint-note:d48454eff4ec]
// 维护导航：projectLiveViewport；下列为本函数源码中的接口与路径索引。
// 输入 content：string。
// 输入 limits：LiveViewportLimits；缺省表达式 DEFAULT_LIVE_VIEWPORT_LIMITS。
// 返回契约：LiveViewportProjection。
// 分支条件：content.length === 0。
// 分支条件：characters.length > limits.maxChars。
// 分支条件：lines.length > limits.maxLines。
// 返回路径：{ text: '', hiddenLines: 0, hiddenChars: 0, notice: null }。
// 返回路径：{ text: kept.join('\n'), hiddenLines, hiddenChars, notice }。
// 写入目标：hiddenChars。
// 写入目标：tail。
// 写入目标：hiddenLines。
// 写入目标：kept。
// 直接调用：content.replace。
// 直接调用：Array.from。
// 直接调用：characters.slice。
// 直接调用：tail.join。
// 直接调用：text.split。
// 直接调用：lines.slice。
// 直接调用：kept.join。
// [/maint-note:d48454eff4ec]
export function projectLiveViewport(
  content: string,
  limits: LiveViewportLimits = DEFAULT_LIVE_VIEWPORT_LIMITS,
): LiveViewportProjection {
  if (content.length === 0) {
    return { text: '', hiddenLines: 0, hiddenChars: 0, notice: null }
  }
  // Char tail first: scalar-safe via Array.from so a surrogate pair is never
  // split, and control characters are scrubbed before the renderer sees them.
  // Scrub control characters EXCEPT newline (0x0a): lines are the viewport's
  // unit, so collapsing newlines would turn the whole draft into one row.
  const sanitized = content.replace(/[\u0000-\u0009\u000b-\u001f\u007f]/g, ' ')
  const characters = Array.from(sanitized)
  let hiddenChars = 0
  let tail = characters
  if (characters.length > limits.maxChars) {
    hiddenChars = characters.length - limits.maxChars
    tail = characters.slice(characters.length - limits.maxChars)
  }
  const text = tail.join('')
  const lines = text.split('\n')
  let hiddenLines = 0
  let kept = lines
  if (lines.length > limits.maxLines) {
    hiddenLines = lines.length - limits.maxLines
    kept = lines.slice(lines.length - limits.maxLines)
  }
  const notice = hiddenLines > 0 || hiddenChars > 0
    ? `... earlier live output hidden (${hiddenLines} lines, ${hiddenChars} chars)`
    : null
  return { text: kept.join('\n'), hiddenLines, hiddenChars, notice }
}
