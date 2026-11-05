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
