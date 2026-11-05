import { StreamingTerminalTextNormalizer } from '../terminal/text.js'

/** The small, provider-independent key shape used by the single REPL owner. */
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

export type KeyboardInputHandler = (input: string, key: KeyboardKey) => void

export interface DecodedKeyboardInput {
  readonly input: string
  readonly key: KeyboardKey
  /** Bracketed paste payload; controls never become keyboard actions. */
  readonly paste?: boolean
}

/** One synchronous owner transaction for all events decoded from one stdin chunk. */
export type KeyboardInputBatchHandler = (events: readonly DecodedKeyboardInput[]) => void

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

function isCsiFinal(char: string | undefined): boolean {
  if (!char) return false
  const code = char.charCodeAt(0)
  return code >= 0x40 && code <= 0x7e
}

function isUnknownControlEnvelope(value: string): boolean {
  return value.startsWith('\x1b[') || value.startsWith('\x1bO')
}

function controlEnvelopeEnd(value: string): number {
  for (let index = 2; index < value.length; index += 1) {
    if (isCsiFinal(value[index])) return index + 1
  }
  return -1
}

/** Decode terminal input without allowing control sequences into model text. */
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

  const emitPasteText = (value: string, decoded: DecodedKeyboardInput[]): void => {
    // The normalizer owns terminal grammar state and drops control payloads
    // incrementally. Chunking its input keeps both retained text and emitted
    // event size bounded even when one stdin callback contains a huge paste.
    for (let start = 0; start < value.length; start += 4096) {
      const normalized = pasteNormalizer.push(value.slice(start, start + 4096))
      if (normalized) decoded.push({ input: normalized, key: emptyKey(), paste: true })
    }
  }

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
    push(raw) {
      const value = typeof raw === 'string' ? raw : textDecoder.decode(raw, { stream: true })
      return consume(value)
    },
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
    discard() {
      // Flush TextDecoder's incomplete UTF-8 state without exposing it.
      textDecoder.decode()
      pending = ''
      inBracketedPaste = false
      pasteNormalizer.reset()
    },
  }

  function isWaiting(): boolean {
    if (inBracketedPaste) return true
    if (!pending.startsWith('\x1b') || pending === '\x1b') return false
    if (pasteStart.startsWith(pending)) return true
    if (Object.keys(specialKeys).some((sequence) => sequence.length > 1 && sequence.startsWith(pending))) return true
    return pending.startsWith('\x1b[') || pending.startsWith('\x1bO')
  }
}

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
