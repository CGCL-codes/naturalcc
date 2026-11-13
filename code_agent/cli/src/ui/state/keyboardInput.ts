import { StreamingTerminalTextNormalizer } from '../terminal/text.js'

/** The small, provider-independent key shape used by the single REPL owner. */
// [maint-note:4c8a5537a1c8]
// 数据契约导航：KeyboardKey；字段含义以消费端校验为准。
// 成员 upArrow：boolean。
// 成员 downArrow：boolean。
// 成员 leftArrow：boolean。
// 成员 rightArrow：boolean。
// 成员 pageDown：boolean。
// 成员 pageUp：boolean。
// 成员 home：boolean。
// 成员 end：boolean。
// 成员 return：boolean。
// 成员 escape：boolean。
// 成员 ctrl：boolean。
// 成员 shift：boolean。
// 成员 tab：boolean。
// 成员 backspace：boolean。
// 成员 delete：boolean。
// 成员 meta：boolean。
// [/maint-note:4c8a5537a1c8]
export interface KeyboardKey {
  readonly upArrow: boolean
  readonly downArrow: boolean
  readonly leftArrow: boolean
  readonly rightArrow: boolean
  readonly pageDown: boolean
  readonly pageUp: boolean
  readonly home: boolean
  readonly end: boolean
  readonly return: boolean
  readonly escape: boolean
  readonly ctrl: boolean
  readonly shift: boolean
  readonly tab: boolean
  readonly backspace: boolean
  readonly delete: boolean
  readonly meta: boolean
}

// [maint-note:bdd3ed12a654]
// 数据契约导航：KeyboardInputHandler；字段含义以消费端校验为准。
// 类型组合：(input: string, key: KeyboardKey) => void。
// [/maint-note:bdd3ed12a654]
export type KeyboardInputHandler = (input: string, key: KeyboardKey) => void

// [maint-note:50dab4eee6d5]
// 数据契约导航：DecodedKeyboardInput；字段含义以消费端校验为准。
// 成员 input：string。
// 成员 key：KeyboardKey。
// 成员 paste（可省略）：boolean。
// [/maint-note:50dab4eee6d5]
export interface DecodedKeyboardInput {
  readonly input: string
  readonly key: KeyboardKey
  /** Bracketed paste payload; controls never become keyboard actions. */
  readonly paste?: boolean
}

/** One synchronous owner transaction for all events decoded from one stdin chunk. */
// [maint-note:0da27d532125]
// 数据契约导航：KeyboardInputBatchHandler；字段含义以消费端校验为准。
// 类型组合：(events: readonly DecodedKeyboardInput[]) => void。
// [/maint-note:0da27d532125]
export type KeyboardInputBatchHandler = (events: readonly DecodedKeyboardInput[]) => void

// [maint-note:ea5b1c2ca2b3]
// 维护导航：emptyKey；下列为本函数源码中的接口与路径索引。
// 返回契约：KeyboardKey。
// [/maint-note:ea5b1c2ca2b3]
const emptyKey = (): KeyboardKey => ({
  upArrow: false, downArrow: false, leftArrow: false, rightArrow: false,
  pageDown: false, pageUp: false, home: false, end: false, return: false,
  escape: false, ctrl: false, shift: false, tab: false, backspace: false,
  delete: false, meta: false,
})

const specialKeys: Readonly<Record<string, keyof KeyboardKey>> = {
  '\r': 'return', '\x1b\r': 'return', '\t': 'tab', '\b': 'backspace',
  '\x7f': 'backspace', '\x1b\b': 'backspace', '\x1b\x7f': 'backspace',
  '\x1b': 'escape', '\x1b\x1b': 'escape',
  '\x1b[A': 'upArrow', '\x1bOA': 'upArrow', '\x1b[B': 'downArrow', '\x1bOB': 'downArrow',
  '\x1b[C': 'rightArrow', '\x1bOC': 'rightArrow', '\x1b[D': 'leftArrow', '\x1bOD': 'leftArrow',
  '\x1b[H': 'home', '\x1bOH': 'home', '\x1b[F': 'end', '\x1bOF': 'end',
  '\x1b[1~': 'home', '\x1b[4~': 'end', '\x1b[3~': 'delete', '\x1b[5~': 'pageUp',
  '\x1b[6~': 'pageDown', '\x1b[Z': 'tab',
}

const bracketedPasteMarkers = ['\x1b[200~', '\x1b[201~'] as const

// [maint-note:e86ad166aa66]
// 维护导航：isCsiFinal；下列为本函数源码中的接口与路径索引。
// 输入 char：string | undefined。
// 返回契约：boolean。
// 分支条件：!char。
// 返回路径：false。
// 返回路径：code >= 0x40 && code <= 0x7e。
// 直接调用：char.charCodeAt。
// [/maint-note:e86ad166aa66]
function isCsiFinal(char: string | undefined): boolean {
  if (!char) return false
  const code = char.charCodeAt(0)
  return code >= 0x40 && code <= 0x7e
}

// [maint-note:e2b8f1edb46f]
// 维护导航：isUnknownControlEnvelope；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：boolean。
// 返回路径：value.startsWith('\x1b[') || value.startsWith('\x1bO')。
// 直接调用：value.startsWith。
// [/maint-note:e2b8f1edb46f]
function isUnknownControlEnvelope(value: string): boolean {
  return value.startsWith('\x1b[') || value.startsWith('\x1bO')
}

// [maint-note:2d270257e44f]
// 维护导航：controlEnvelopeEnd；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 分支条件：isCsiFinal(value[index])。
// 遍历边界：let index = 2; index < value.length; index += 1。
// 返回路径：index + 1。
// 返回路径：-1。
// 写入目标：index。
// 直接调用：isCsiFinal。
// [/maint-note:2d270257e44f]
function controlEnvelopeEnd(value: string): number {
  for (let index = 2; index < value.length; index += 1) {
    if (isCsiFinal(value[index])) return index + 1
  }
  return -1
}

/** Decode terminal input without allowing control sequences into model text. */
// [maint-note:1c892d0c10b9]
// 维护导航：decodeKeyboardInput；下列为本函数源码中的接口与路径索引。
// 输入 raw：string | Uint8Array。
// 返回契约：{ input: string; key: KeyboardKey }。
// 分支条件：keyName !== undefined。
// 分支条件：value.length === 1。
// 分支条件：code >= 1 && code <= 26。
// 分支条件：value.includes('\x03')。
// 分支条件：value.startsWith('\x1b')。
// 返回路径：{ input: '', key: { ...key, [keyName]: true } }。
// 返回路径：{ input: String.fromCharCode(code + 96), key: { ...emptyKey(), ctrl: true } }。
// 返回路径：{ input: value, key: emptyKey() }。
// 返回路径：{ input: '', key: { ...emptyKey(), escape: true } }。
// 直接调用：new TextDecoder().decode。
// 直接调用：emptyKey。
// 直接调用：value.charCodeAt。
// 直接调用：String.fromCharCode。
// 直接调用：value.includes。
// 直接调用：value.startsWith。
// [/maint-note:1c892d0c10b9]
export function decodeKeyboardInput(raw: string | Uint8Array): { input: string; key: KeyboardKey } {
  const value = typeof raw === 'string' ? raw : new TextDecoder().decode(raw)
  const keyName = specialKeys[value]
  if (keyName !== undefined) {
    const key = emptyKey()
    return { input: '', key: { ...key, [keyName]: true } }
  }
  if (value.length === 1) {
    const code = value.charCodeAt(0)
    if (code >= 1 && code <= 26) return { input: String.fromCharCode(code + 96), key: { ...emptyKey(), ctrl: true } }
  }
  // Keep embedded Ctrl-C bytes intact so the existing chunk reducer preserves
  // their order relative to pasted text.
  if (value.includes('\x03')) return { input: value, key: emptyKey() }
  if (value.startsWith('\x1b')) return { input: '', key: { ...emptyKey(), escape: true } }
  return { input: value, key: emptyKey() }
}

/**
 * Decode a stream of terminal chunks without treating the first byte of a
 * split escape sequence as a standalone Escape key. PTYs are allowed to
 * split `ESC [ A` into separate data events; the REPL has one keyboard owner,
 * so it must perform this small amount of framing before routing the key.
 */
// [maint-note:bc79eebbf001]
// 维护导航：createKeyboardInputDecoder；下列为本函数源码中的接口与路径索引。
// 返回契约：{ push: (raw: string | Uint8Array) => readonly DecodedKeyboardInput[] flush: () => readonly DecodedKeyboardInput[] /** True while a control envelope still needs another stdin chunk. */ isWaitingForMoreInput: () => boolean /** Drop all decod。
// 返回路径：{ push(raw) { const value = typeof raw === 'string' ? raw : textDecoder.decode(raw, { stream: true }) return consume(value) }, flush() { // A timeout is only an Escape disambiguation boundary, not EOF. Keep an // open bracketed paste (and s。
// [/maint-note:bc79eebbf001]
export function createKeyboardInputDecoder(): {
  push: (raw: string | Uint8Array) => readonly DecodedKeyboardInput[]
  flush: () => readonly DecodedKeyboardInput[]
  /** True while a control envelope still needs another stdin chunk. */
  isWaitingForMoreInput: () => boolean
  /** Drop all decoder state without delivering a pending Escape or paste. */
  discard: () => void
} {
  let pending = ''
  let inBracketedPaste = false
  const textDecoder = new TextDecoder()
  const pasteNormalizer = new StreamingTerminalTextNormalizer()
  const pasteStart = bracketedPasteMarkers[0]
  const pasteEnd = bracketedPasteMarkers[1]

  // [maint-note:13728e4b165e]
  // 维护导航：emitPasteText；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 输入 decoded：DecodedKeyboardInput[]。
  // 返回契约：void。
  // 分支条件：normalized。
  // 遍历边界：let start = 0; start < value.length; start += 4096。
  // 写入目标：start。
  // 直接调用：pasteNormalizer.push。
  // 直接调用：value.slice。
  // 直接调用：decoded.push。
  // 直接调用：emptyKey。
  // [/maint-note:13728e4b165e]
  const emitPasteText = (value: string, decoded: DecodedKeyboardInput[]): void => {
    // The normalizer owns terminal grammar state and drops control payloads
    // incrementally. Chunking its input keeps both retained text and emitted
    // event size bounded even when one stdin callback contains a huge paste.
    for (let start = 0; start < value.length; start += 4096) {
      const normalized = pasteNormalizer.push(value.slice(start, start + 4096))
      if (normalized) decoded.push({ input: normalized, key: emptyKey(), paste: true })
    }
  }

  // [maint-note:bfa1c8f3e9ee]
  // 维护导航：processPastePending；下列为本函数源码中的接口与路径索引。
  // 输入 decoded：DecodedKeyboardInput[]。
  // 返回契约：boolean。
  // 分支条件：end >= 0。
  // 分支条件：normalizedTail。
  // 分支条件：pending.length > keep。
  // 返回路径：true。
  // 返回路径：false。
  // 写入目标：pending。
  // 写入目标：inBracketedPaste。
  // 直接调用：pending.indexOf。
  // 直接调用：emitPasteText。
  // 直接调用：pending.slice。
  // 直接调用：pasteNormalizer.finish。
  // 直接调用：decoded.push。
  // 直接调用：emptyKey。
  // 直接调用：pasteNormalizer.reset。
  // 直接调用：Math.min。
  // [/maint-note:bfa1c8f3e9ee]
  const processPastePending = (decoded: DecodedKeyboardInput[]): boolean => {
    const end = pending.indexOf(pasteEnd)
    if (end >= 0) {
      emitPasteText(pending.slice(0, end), decoded)
      const normalizedTail = pasteNormalizer.finish()
      if (normalizedTail) decoded.push({ input: normalizedTail, key: emptyKey(), paste: true })
      pending = pending.slice(end + pasteEnd.length)
      inBracketedPaste = false
      pasteNormalizer.reset()
      return true
    }

    // Keep only the suffix that could still be the beginning of the end
    // marker. The normalizer, rather than this decoder, retains any
    // incomplete CR, surrogate or terminal control state.
    const keep = Math.min(pending.length, pasteEnd.length - 1)
    if (pending.length > keep) {
      const pasteText = pending.slice(0, pending.length - keep)
      pending = pending.slice(pasteText.length)
      emitPasteText(pasteText, decoded)
    }
    return false
  }

  // [maint-note:966a8f281aa0]
  // 维护导航：consume；下列为本函数源码中的接口与路径索引。
  // 输入 value：string。
  // 返回契约：DecodedKeyboardInput[]。
  // 分支条件：inBracketedPaste。
  // 分支条件：remaining.length > 0。
  // 分支条件：remaining.length === 0。
  // 分支条件：!pending.startsWith('\x1b')。
  // 分支条件：specialIndex < 0。
  // 分支条件：specialIndex > 0。
  // 分支条件：pending === '\x1b'。
  // 分支条件：pending.startsWith(pasteStart)。
  // 分支条件：exact。
  // 分支条件：isPrefix。
  // 分支条件：isUnknownControlEnvelope(pending)。
  // 分支条件：end < 0。
  // 分支条件：pending.length === 0。
  // 遍历边界：remaining.length > 0 || pending.length > 0。
  // 遍历边界：pending && !inBracketedPaste。
  // 返回路径：decoded。
  // 写入目标：remaining。
  // 写入目标：pending。
  // 写入目标：inBracketedPaste。
  // 直接调用：remaining.slice。
  // 直接调用：processPastePending。
  // 直接调用：pending.startsWith。
  // 直接调用：findSpecialCharacter。
  // 直接调用：decoded.push。
  // 直接调用：decodeKeyboardInput。
  // 直接调用：pending.slice。
  // 直接调用：pasteNormalizer.reset。
  // 直接调用：Object.keys(specialKeys) .filter((sequence) => sequence.length > 1 && pending.startsWith(sequence)) .sort。
  // 直接调用：Object.keys(specialKeys) .filter。
  // 直接调用：Object.keys。
  // 直接调用：Object.keys(specialKeys).some。
  // 直接调用：isUnknownControlEnvelope。
  // 直接调用：controlEnvelopeEnd。
  // [/maint-note:966a8f281aa0]
  const consume = (value: string): DecodedKeyboardInput[] => {
    const decoded: DecodedKeyboardInput[] = []
    let remaining = value
    while (remaining.length > 0 || pending.length > 0) {
      if (inBracketedPaste) {
        if (remaining.length > 0) {
          const chunk = remaining.slice(0, 4096)
          remaining = remaining.slice(chunk.length)
          pending += chunk
        }
        processPastePending(decoded)
        if (inBracketedPaste) {
          if (remaining.length === 0) break
          continue
        }
        continue
      }

      if (remaining.length > 0) {
        const chunk = remaining.slice(0, 4096)
        remaining = remaining.slice(chunk.length)
        pending += chunk
      }

      while (pending && !inBracketedPaste) {
        if (!pending.startsWith('\x1b')) {
          const specialIndex = findSpecialCharacter(pending)
          if (specialIndex < 0) {
            decoded.push(decodeKeyboardInput(pending))
            pending = ''
            continue
          }
          if (specialIndex > 0) {
            decoded.push(decodeKeyboardInput(pending.slice(0, specialIndex)))
            pending = pending.slice(specialIndex)
            continue
          }
          decoded.push(decodeKeyboardInput(pending[0] ?? ''))
          pending = pending.slice(1)
          continue
        }

        if (pending === '\x1b') break

        if (pending.startsWith(pasteStart)) {
          // Paste framing is transport metadata. Enter paste mode so every
          // payload byte remains data until the closing marker arrives.
          pending = pending.slice(pasteStart.length)
          inBracketedPaste = true
          pasteNormalizer.reset()
          continue
        }

        const exactMatches = Object.keys(specialKeys)
          .filter((sequence) => sequence.length > 1 && pending.startsWith(sequence))
          .sort((left, right) => right.length - left.length)
        const exact = exactMatches[0]
        if (exact) {
          decoded.push(decodeKeyboardInput(exact))
          pending = pending.slice(exact.length)
          continue
        }
        const isPrefix = Object.keys(specialKeys).some(
          (sequence) => sequence.length > 1 && sequence.startsWith(pending),
        )
        if (isPrefix) break

        if (isUnknownControlEnvelope(pending)) {
          const end = controlEnvelopeEnd(pending)
          // Keep an incomplete CSI/SS3 envelope together until the next
          // stdin chunk. An unknown complete envelope is inert, including
          // modifier forms such as ESC[1;5A and vendor-private sequences.
          if (end < 0) break
          pending = pending.slice(end)
          continue
        }

        // An ESC followed by ordinary text is a standalone Escape followed
        // by that text, not an opaque control sequence.
        decoded.push(decodeKeyboardInput('\x1b'))
        pending = pending.slice(1)
      }

      if (inBracketedPaste) continue
      if (pending.length === 0) {
        if (remaining.length === 0) break
        continue
      }
      if (remaining.length === 0) break
    }
    return decoded
  }

  return {
    // [maint-note:378bc391a656]
    // 维护导航：push；下列为本函数源码中的接口与路径索引。
    // 输入 raw：类型由调用方与函数体约束。
    // 返回路径：consume(value)。
    // 直接调用：textDecoder.decode。
    // 直接调用：consume。
    // [/maint-note:378bc391a656]
    push(raw) {
      const value = typeof raw === 'string' ? raw : textDecoder.decode(raw, { stream: true })
      return consume(value)
    },
    // [maint-note:4822b25b3002]
    // 维护导航：flush；下列为本函数源码中的接口与路径索引。
    // 分支条件：isWaiting()。
    // 分支条件：inBracketedPaste。
    // 分支条件：pending。
    // 分支条件：!isUnknownControlEnvelope(pending)。
    // 返回路径：decoded。
    // 写入目标：pending。
    // 直接调用：consume。
    // 直接调用：isWaiting。
    // 直接调用：isUnknownControlEnvelope。
    // 直接调用：decoded.push。
    // 直接调用：decodeKeyboardInput。
    // [/maint-note:4822b25b3002]
    flush() {
      // A timeout is only an Escape disambiguation boundary, not EOF. Keep an
      // open bracketed paste (and split start/end envelopes) intact across
      // arbitrarily delayed stdin chunks. `discard()` is the explicit owner
      // revocation path for abandoning such a paste.
      // In particular, do not call TextDecoder.decode() without `stream:
      // true` here: an incomplete UTF-8 character may be waiting for the next
      // stdin chunk, and decoder EOF would turn it into replacement text.
      const decoded = consume('')
      if (isWaiting()) return decoded
      if (inBracketedPaste) {
        return decoded
      }
      if (pending) {
        if (!isUnknownControlEnvelope(pending)) decoded.push(decodeKeyboardInput(pending))
        pending = ''
      }
      return decoded
    },
    isWaitingForMoreInput: isWaiting,
    // [maint-note:45e977e8bdd5]
    // 维护导航：discard；下列为本函数源码中的接口与路径索引。
    // 写入目标：pending。
    // 写入目标：inBracketedPaste。
    // 直接调用：textDecoder.decode。
    // 直接调用：pasteNormalizer.reset。
    // [/maint-note:45e977e8bdd5]
    discard() {
      // Flush TextDecoder's incomplete UTF-8 state without exposing it.
      textDecoder.decode()
      pending = ''
      inBracketedPaste = false
      pasteNormalizer.reset()
    },
  }

  // [maint-note:d295817c7c57]
  // 维护导航：isWaiting；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：inBracketedPaste。
  // 分支条件：!pending.startsWith('\x1b') || pending === '\x1b'。
  // 分支条件：pasteStart.startsWith(pending)。
  // 分支条件：Object.keys(specialKeys).some((sequence) => sequence.length > 1 && sequence.startsWith(pending))。
  // 返回路径：true。
  // 返回路径：false。
  // 返回路径：pending.startsWith('\x1b[') || pending.startsWith('\x1bO')。
  // 直接调用：pending.startsWith。
  // 直接调用：pasteStart.startsWith。
  // 直接调用：Object.keys(specialKeys).some。
  // 直接调用：Object.keys。
  // [/maint-note:d295817c7c57]
  function isWaiting(): boolean {
    if (inBracketedPaste) return true
    if (!pending.startsWith('\x1b') || pending === '\x1b') return false
    if (pasteStart.startsWith(pending)) return true
    if (Object.keys(specialKeys).some((sequence) => sequence.length > 1 && sequence.startsWith(pending))) return true
    return pending.startsWith('\x1b[') || pending.startsWith('\x1bO')
  }
}

// [maint-note:ee30a129b4cb]
// 维护导航：findSpecialCharacter；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 分支条件：char === '\r' || char === '\t' || char === '\b' || char === '\x7f' || char === '\x03' || char === '\x1b'。
// 分支条件：code < 0x20 && code !== 0x0a。
// 遍历边界：let index = 0; index < value.length; index += 1。
// 返回路径：index。
// 返回路径：-1。
// 写入目标：index。
// 直接调用：char.charCodeAt。
// [/maint-note:ee30a129b4cb]
function findSpecialCharacter(value: string): number {
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index]
    if (char === '\r' || char === '\t' || char === '\b' || char === '\x7f' || char === '\x03' || char === '\x1b') {
      return index
    }
    // Every other C0 control must also frame its own event: a chunk like
    // Ctrl+O followed by text (\x0fZ) must decode as a ctrl key press plus a
    // text event, never as one literal string with an embedded control byte.
    // \n stays ordinary text (the Enter key arrives as \r).
    const code = char.charCodeAt(0)
    if (code < 0x20 && code !== 0x0a) return index
  }
  return -1
}
