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

// [maint-note:8b1ff8b6f0a4]
// 数据契约导航：ToolControlEnvelopeDiagnostic；字段含义以消费端校验为准。
// 成员 id：string。
// [/maint-note:8b1ff8b6f0a4]
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

  // [maint-note:80147642f8e7]
  // 维护导航：push；下列为本函数源码中的接口与路径索引。
  // 输入 text：string。
  // 返回契约：string。
  // 分支条件：!text。
  // 分支条件：this.holding。
  // 分支条件：this.pending === ''。
  // 分支条件：character !== '<'。
  // 分支条件：emitted。
  // 分支条件：isCompleteMarker(suffix)。
  // 遍历边界：const character ← text。
  // 返回路径：''。
  // 返回路径：visible。
  // 写入目标：visible。
  // 写入目标：this.releasedChars。
  // 写入目标：this.pending。
  // 写入目标：this.holding。
  // 直接调用：longestMarkerPrefixSuffix。
  // 直接调用：candidate.slice。
  // 直接调用：isCompleteMarker。
  // [/maint-note:80147642f8e7]
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
  // [maint-note:b1e4e42c61d7]
  // 维护导航：release；下列为本函数源码中的接口与路径索引。
  // 输入 completeText：string。
  // 返回契约：string。
  // 返回路径：trailing。
  // 写入目标：this.pending。
  // 写入目标：this.releasedChars。
  // 写入目标：this.holding。
  // 直接调用：completeText.slice。
  // [/maint-note:b1e4e42c61d7]
  release(completeText: string): string {
    const trailing = completeText.slice(this.releasedChars)
    this.pending = ''
    this.releasedChars = 0
    this.holding = false
    return trailing
  }

  /** Release only a still-ambiguous suffix for direct gate tests/callers. */
  // [maint-note:8c07a824ba28]
  // 维护导航：flush；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 分支条件：this.holding。
  // 返回路径：''。
  // 返回路径：visible。
  // 写入目标：this.releasedChars。
  // 写入目标：this.pending。
  // [/maint-note:8c07a824ba28]
  flush(): string {
    if (this.holding) return ''
    const visible = this.pending
    this.releasedChars += visible.length
    this.pending = ''
    return visible
  }
}

// [maint-note:ebbf8d124a93]
// 维护导航：longestMarkerPrefixSuffix；下列为本函数源码中的接口与路径索引。
// 输入 candidate：string。
// 返回契约：string。
// 分支条件：marker.open.startsWith(suffix) && suffix.length > best.length。
// 遍历边界：const marker ← CONTROL_MARKERS。
// 遍历边界：let length = 1; length <= Math.min(candidate.length, marker.open.length); length += 1。
// 返回路径：best。
// 写入目标：length。
// 写入目标：best。
// 直接调用：Math.min。
// 直接调用：candidate.slice。
// 直接调用：marker.open.startsWith。
// [/maint-note:ebbf8d124a93]
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

// [maint-note:b69514aa1336]
// 维护导航：isCompleteMarker；下列为本函数源码中的接口与路径索引。
// 输入 candidate：string。
// 返回契约：boolean。
// 返回路径：CONTROL_MARKERS.some((marker) => marker.open === candidate)。
// 直接调用：CONTROL_MARKERS.some。
// [/maint-note:b69514aa1336]
function isCompleteMarker(candidate: string): boolean {
  return CONTROL_MARKERS.some((marker) => marker.open === candidate)
}

// [maint-note:8e06d6919fc8]
// 数据契约导航：ControlMarker；字段含义以消费端校验为准。
// 成员 id：string。
// 成员 open：string。
// 成员 close（可省略）：string。
// [/maint-note:8e06d6919fc8]
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
// [maint-note:25159d4d1d7c]
// 维护导航：detectToolControlEnvelope；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：ToolControlEnvelopeDiagnostic | undefined。
// 分支条件：!text。
// 分支条件：!visible。
// 分支条件：openIndex === -1。
// 分支条件：openIndex === leading。
// 分支条件：!marker.close。
// 分支条件：closeIndex === -1。
// 分支条件：envelopeLength >= visible.length * DOMINANT_SHARE。
// 遍历边界：const marker ← CONTROL_MARKERS。
// 返回路径：undefined。
// 返回路径：{ id: marker.id }。
// 直接调用：maskCodeSpans。
// 直接调用：masked.trim。
// 直接调用：masked.trimStart。
// 直接调用：masked.indexOf。
// [/maint-note:25159d4d1d7c]
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
// [maint-note:5a094992b979]
// 维护导航：maskCodeSpans；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：string。
// 返回路径：text .replace(/```[\s\S]*?(?:```|$)/gu, blank) .replace(/~~~[\s\S]*?(?:~~~|$)/gu, blank) .replace(/`[^`\n]*`/gu, blank)。
// 直接调用：text .replace(/```[\s\S]*?(?:```|$)/gu, blank) .replace(/~~~[\s\S]*?(?:~~~|$)/gu, blank) .replace。
// 直接调用：text .replace(/```[\s\S]*?(?:```|$)/gu, blank) .replace。
// 直接调用：text .replace。
// [/maint-note:5a094992b979]
function maskCodeSpans(text: string): string {
  return text
    .replace(/```[\s\S]*?(?:```|$)/gu, blank)
    .replace(/~~~[\s\S]*?(?:~~~|$)/gu, blank)
    .replace(/`[^`\n]*`/gu, blank)
}

// [maint-note:fa3780be8a6c]
// 维护导航：blank；下列为本函数源码中的接口与路径索引。
// 输入 span：string。
// 返回契约：string。
// 返回路径：' '.repeat(span.length)。
// 直接调用：' '.repeat。
// [/maint-note:fa3780be8a6c]
function blank(span: string): string {
  return ' '.repeat(span.length)
}
