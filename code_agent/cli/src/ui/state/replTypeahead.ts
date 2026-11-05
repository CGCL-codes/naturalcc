export const MAX_TYPEAHEAD_CHARS = 256

/**
 * Extract the plain printable text of one input chunk.
 *
 * Control bytes (C0, DEL, C1) are never queued: they keep their live meaning
 * while the input owner is inactive, and a carriage return cannot be replayed
 * as a submit — the user confirms once the line is interactive again. Lone
 * UTF-16 surrogates are not scalar values and are dropped too, so the queue
 * only ever holds well-formed Unicode text.
 */
export function extractTypeaheadText(content: string): string {
  let result = ''
  for (const character of content) {
    const code = character.codePointAt(0) ?? 0
    if (code >= 0xd800 && code <= 0xdfff) continue
    if ((code >= 0x20 && code < 0x7f) || code >= 0xa0) result += character
  }
  return result
}

/**
 * Append one chunk to the pending type-ahead queue. The queue is bounded by
 * Unicode scalar values: a UTF-16 slice could split a surrogate pair at the
 * limit and leave an unpaired half, so the cap counts characters, not code
 * units. Overflow drops the excess instead of growing without limit.
 */
export function appendTypeahead(queue: string, chunk: string): string {
  if (!chunk) return queue
  const next = queue + chunk
  const characters = Array.from(next)
  if (characters.length <= MAX_TYPEAHEAD_CHARS) return next
  return characters.slice(0, MAX_TYPEAHEAD_CHARS).join('')
}
