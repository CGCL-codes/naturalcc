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
// [maint-note:1c2fafe1871b]
// 维护导航：extractTypeaheadText；下列为本函数源码中的接口与路径索引。
// 输入 content：string。
// 返回契约：string。
// 分支条件：code >= 0xd800 && code <= 0xdfff。
// 分支条件：(code >= 0x20 && code < 0x7f) || code >= 0xa0。
// 遍历边界：const character ← content。
// 返回路径：result。
// 写入目标：result。
// 直接调用：character.codePointAt。
// [/maint-note:1c2fafe1871b]
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
// [maint-note:dfe186f931ec]
// 维护导航：appendTypeahead；下列为本函数源码中的接口与路径索引。
// 输入 queue：string。
// 输入 chunk：string。
// 返回契约：string。
// 分支条件：!chunk。
// 分支条件：characters.length <= MAX_TYPEAHEAD_CHARS。
// 返回路径：queue。
// 返回路径：next。
// 返回路径：characters.slice(0, MAX_TYPEAHEAD_CHARS).join('')。
// 直接调用：Array.from。
// 直接调用：characters.slice(0, MAX_TYPEAHEAD_CHARS).join。
// 直接调用：characters.slice。
// [/maint-note:dfe186f931ec]
export function appendTypeahead(queue: string, chunk: string): string {
  if (!chunk) return queue
  const next = queue + chunk
  const characters = Array.from(next)
  if (characters.length <= MAX_TYPEAHEAD_CHARS) return next
  return characters.slice(0, MAX_TYPEAHEAD_CHARS).join('')
}
