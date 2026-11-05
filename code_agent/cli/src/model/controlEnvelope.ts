/**
 * Narrow provider tool-control envelope diagnostic.
 *
 * Some OpenAI-compatible endpoints answer a request by emitting their internal
 * tool-call control envelope as ordinary assistant text instead of structured
 * `tool_calls`. The tokens below are provider control sentinels, not natural
 * language.
 *
 * This module only reports that the response as a whole has that shape. It
 * never parses, canonicalizes, or executes anything it finds: a pseudo tool
 * call in text stays inert, and no caller may use this as an execution path.
 *
 * Detection is deliberately conservative. A response is a control envelope
 * only when it *opens* with one, or when a complete open..close envelope is
 * the dominant half of it. Prose that mentions `tool_calls`, quotes a token
 * mid-sentence, or shows the markup inside a code fence or inline code span is
 * ordinary assistant text and is never rejected.
 */

export interface ToolControlEnvelopeDiagnostic {
  /** Stable internal id. Never rendered in the default UI or a paused header. */
  readonly id: string
}

/**
 * Bounded presentation gate for a streamed text response.
 *
 * Every supported control marker starts with `<`.  Keep only the suffix from
 * the most recent possible marker while it can still become one.  Ordinary
 * text is released as soon as that possibility is disproved.  If the marker
 * is completed, the gate stops releasing the remainder until the complete
 * response can be validated. The bounded model accumulator owns that
 * remainder; callers release it from the accumulator's canonical text only
 * after `detectToolControlEnvelope()` has returned no diagnostic.
 */
export class StreamingControlEnvelopeGate {
  private pending = ''
  private releasedChars = 0
  private holding = false

  push(text: string): string {
    if (!text) return ''
    let visible = ''
    for (const character of text) {
      if (this.holding) continue
      if (this.pending === '') {
        if (character !== '<') {
          visible += character
          this.releasedChars += character.length
          continue
        }
        this.pending = character
        continue
      }

      const candidate = this.pending + character
      const suffix = longestMarkerPrefixSuffix(candidate)
      const emitted = candidate.slice(0, candidate.length - suffix.length)
      if (emitted) {
        visible += emitted
        this.releasedChars += emitted.length
      }
      this.pending = suffix
      if (isCompleteMarker(suffix)) {
        // The remainder is deliberately not retained here. The bounded model
        // accumulator owns the raw response and release() recovers this
        // suffix from that canonical, already-limited text after validation.
        this.pending = ''
        this.holding = true
      }
    }
    return visible
  }

  /**
   * Release the part not presented by push() after the whole response has
   * been accepted. The caller must pass the accumulator's complete text; the
   * gate itself never becomes a second response buffer.
   */
  release(completeText: string): string {
    const trailing = completeText.slice(this.releasedChars)
    this.pending = ''
    this.releasedChars = 0
    this.holding = false
    return trailing
  }

  /** Release only a still-ambiguous suffix for direct gate tests/callers. */
  flush(): string {
    if (this.holding) return ''
    const visible = this.pending
    this.releasedChars += visible.length
    this.pending = ''
    return visible
  }
}

function longestMarkerPrefixSuffix(candidate: string): string {
  let best = ''
  for (const marker of CONTROL_MARKERS) {
    for (let length = 1; length <= Math.min(candidate.length, marker.open.length); length += 1) {
      const suffix = candidate.slice(candidate.length - length)
      if (marker.open.startsWith(suffix) && suffix.length > best.length) {
        best = suffix
      }
    }
  }
  return best
}

function isCompleteMarker(candidate: string): boolean {
  return CONTROL_MARKERS.some((marker) => marker.open === candidate)
}

interface ControlMarker {
  readonly id: string
  readonly open: string
  /** Present only for families whose envelope has a matching terminator. */
  readonly close?: string
}

/**
 * First batch: the observed real DSML sample plus the equivalent control
 * tokens of the OpenAI-compatible provider families this CLI can be pointed
 * at. New providers extend this table; callers never pass their own patterns.
 */
const CONTROL_MARKERS: readonly ControlMarker[] = [
  // Observed real sample: <｜｜DSML｜｜tool_calls>...bash...</｜｜DSML｜｜tool_calls>
  { id: 'dsml_tool_calls', open: '<｜｜DSML｜｜tool_calls>', close: '</｜｜DSML｜｜tool_calls>' },
  // DeepSeek native tool-call sentinels (U+FF5C plus U+2581).
  { id: 'deepseek_tool_calls', open: '<｜tool▁calls▁begin｜>', close: '<｜tool▁calls▁end｜>' },
  { id: 'deepseek_tool_call', open: '<｜tool▁call▁begin｜>', close: '<｜tool▁call▁end｜>' },
  // Hermes / Qwen style.
  { id: 'hermes_tool_call', open: '<tool_call>', close: '</tool_call>' },
  // Llama 3 built-in tool prefix; it has no terminator.
  { id: 'llama_python_tag', open: '<|python_tag|>' },
  // OpenAI Harmony commentary channel carrying a tool recipient.
  { id: 'harmony_commentary_call', open: '<|channel|>commentary to=' },
]

/** Minimum share of the response an open..close envelope must occupy. */
const DOMINANT_SHARE = 0.5

/**
 * Reports a control envelope only when it is the complete or dominant shape of
 * the response. Returns undefined for every other text.
 */
export function detectToolControlEnvelope(text: string): ToolControlEnvelopeDiagnostic | undefined {
  if (!text) return undefined
  const masked = maskCodeSpans(text)
  const visible = masked.trim()
  if (!visible) return undefined
  const leading = masked.length - masked.trimStart().length

  for (const marker of CONTROL_MARKERS) {
    const openIndex = masked.indexOf(marker.open)
    if (openIndex === -1) continue
    if (openIndex === leading) return { id: marker.id }
    if (!marker.close) continue
    const closeIndex = masked.indexOf(marker.close, openIndex + marker.open.length)
    if (closeIndex === -1) continue
    const envelopeLength = closeIndex + marker.close.length - openIndex
    if (envelopeLength >= visible.length * DOMINANT_SHARE) return { id: marker.id }
  }
  return undefined
}

/**
 * Blanks fenced blocks and inline code spans while preserving offsets, so a
 * documented or quoted control token is never mistaken for a live envelope.
 */
function maskCodeSpans(text: string): string {
  return text
    .replace(/```[\s\S]*?(?:```|$)/gu, blank)
    .replace(/~~~[\s\S]*?(?:~~~|$)/gu, blank)
    .replace(/`[^`\n]*`/gu, blank)
}

function blank(span: string): string {
  return ' '.repeat(span.length)
}
