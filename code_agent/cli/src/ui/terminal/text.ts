import stringWidth from 'string-width'
import stripAnsi from 'strip-ansi'
import wrapAnsi from 'wrap-ansi'

const ANSI_SEQUENCE = /\u001b(?:\[[0-?]*[ -/]*[@-~]|\][\s\S]*?(?:\u0007|\u001b\\)|[()][0-2A-Z])/gu
const segmenter = typeof Intl.Segmenter === 'function'
  ? new Intl.Segmenter(undefined, { granularity: 'grapheme' })
  : undefined

export interface WrapDisplayOptions {
  hard?: boolean
  trim?: boolean
  wordWrap?: boolean
}

export interface TruncateDisplayOptions {
  marker?: string
  position?: 'start' | 'end'
}

type TerminalEscapeState = 'normal' | 'escape' | 'escapeIntermediate' | 'csi' | 'string' | 'stringEscape'
type TerminalStringKind = 'osc' | 'dcs' | 'sos' | 'pm' | 'apc'

/**
 * The single source-normalization contract used by one-shot and streaming
 * Markdown rendering. Terminal sequences are discarded without retaining
 * their payload, so an unterminated CSI/OSC cannot create an unbounded cache.
 */
export class StreamingTerminalTextNormalizer {
  private state: TerminalEscapeState = 'normal'
  private stringKind: TerminalStringKind | undefined
  private pendingCR = false
  private pendingHighSurrogate = ''

  push(input: string): string {
    if (input.length === 0) return ''
    let output = ''
    if (this.pendingHighSurrogate) {
      if (isLowSurrogateCode(input.charCodeAt(0))) {
        input = this.pendingHighSurrogate + input
        this.pendingHighSurrogate = ''
      } else {
        output += this.process('\ufffd')
        this.pendingHighSurrogate = ''
      }
    }
    if (isHighSurrogateCode(input.charCodeAt(input.length - 1))) {
      this.pendingHighSurrogate = input.slice(-1)
      input = input.slice(0, -1)
    }
    return output + this.process(input)
  }

  private process(input: string): string {
    let output = ''
    let index = 0
    while (index < input.length) {
      const character = input[index] ?? ''
      if (this.state === 'normal') {
        if (this.pendingCR) {
          output += '\n'
          this.pendingCR = false
          if (character === '\n') {
            index += 1
            continue
          }
        }
        if (character === '\r') {
          this.pendingCR = true
          index += 1
          continue
        }
        if (character === '\u001b') {
          this.state = 'escape'
          index += 1
          continue
        }
        if (character === '\u009b') {
          this.state = 'csi'
          index += 1
          continue
        }
        const c1StringKind = getC1StringKind(character)
        if (c1StringKind) {
          this.beginString(c1StringKind)
          index += 1
          continue
        }
        if (character === '\u009c' || isTerminalControl(character)) {
          index += 1
          continue
        }
        const code = character.charCodeAt(0)
        if (isHighSurrogateCode(code)) {
          const nextCode = input.charCodeAt(index + 1)
          if (isLowSurrogateCode(nextCode)) {
            output += character + (input[index + 1] ?? '')
            index += 2
          } else {
            output += '\ufffd'
            index += 1
          }
          continue
        }
        if (isLowSurrogateCode(code)) {
          output += '\ufffd'
          index += 1
          continue
        }
        output += character
        index += 1
        continue
      }

      if (this.state === 'escape') {
        if (isSequenceCancel(character)) {
          this.endSequence()
          index += 1
          continue
        }
        if (character === '[') {
          this.state = 'csi'
          index += 1
          continue
        }
        if (character === ']') {
          this.beginString('osc')
          index += 1
          continue
        }
        const escapeStringKind = getEscapeStringKind(character)
        if (escapeStringKind) {
          this.beginString(escapeStringKind)
          index += 1
          continue
        }
        if (isEscapeIntermediate(character)) {
          this.state = 'escapeIntermediate'
          index += 1
          continue
        }
        if (isEscapeFinal(character)) {
          this.endSequence()
          index += 1
          continue
        }
        this.endSequence()
        // Malformed input is reprocessed as ordinary text, rather than being
        // swallowed together with the ESC introducer.
        continue
      }

      if (this.state === 'escapeIntermediate') {
        if (isSequenceCancel(character)) {
          this.endSequence()
          index += 1
          continue
        }
        if (isEscapeIntermediate(character)) {
          index += 1
          continue
        }
        if (isEscapeFinal(character)) {
          this.endSequence()
          index += 1
          continue
        }
        this.endSequence()
        continue
      }

      if (this.state === 'csi') {
        if (isSequenceCancel(character)) {
          this.endSequence()
          index += 1
          continue
        }
        if (isCsiFinalByte(character)) {
          this.endSequence()
          index += 1
          continue
        }
        if (isCsiParameterOrIntermediate(character)) {
          index += 1
          continue
        }
        // A malformed CSI ends before the invalid character. The invalid
        // character is ordinary input and must not be swallowed with it.
        this.endSequence()
        continue
      }

      if (this.state === 'string') {
        if (isSequenceCancel(character)) {
          this.endSequence()
          index += 1
          continue
        }
        if ((this.stringKind === 'osc' && character === '\u0007') || character === '\u009c') {
          this.endSequence()
          index += 1
          continue
        }
        if (character === '\u001b') {
          this.state = 'stringEscape'
          index += 1
          continue
        }
        index += 1
        continue
      }

      if (isSequenceCancel(character)) {
        this.endSequence()
        index += 1
        continue
      }
      if (character === '\\') {
        this.endSequence()
        index += 1
        continue
      }
      this.state = 'string'
      // The current character remains part of the string payload. BEL only
      // terminates OSC; DCS/SOS/PM/APC must wait for ST.
    }
    return output
  }

  finish(): string {
    let output = ''
    if (this.pendingCR) {
      this.pendingCR = false
      output += '\n'
    }
    if (this.pendingHighSurrogate) {
      output += this.process('\ufffd')
      this.pendingHighSurrogate = ''
    }
    // Any confirmed escape sequence, including an incomplete one, is
    // discarded at EOF. This is intentionally the same policy as push().
    this.endSequence()
    return output
  }

  reset(): void {
    this.endSequence()
    this.pendingCR = false
    this.pendingHighSurrogate = ''
  }

  private beginString(kind: TerminalStringKind): void {
    this.state = 'string'
    this.stringKind = kind
  }

  private endSequence(): void {
    this.state = 'normal'
    this.stringKind = undefined
  }
}

export function sanitizeTerminalText(text: string, options: { expandTabs?: boolean } = {}): string {
  const normalizer = new StreamingTerminalTextNormalizer()
  const safe = normalizer.push(text) + normalizer.finish()
  return options.expandTabs ? safe.replace(/\t/gu, '    ') : safe
}

function isTerminalControl(character: string): boolean {
  const code = character.charCodeAt(0)
  return (code >= 0 && code <= 0x08) || (code >= 0x0b && code <= 0x0c) || (code >= 0x0e && code <= 0x1f) || (code >= 0x7f && code <= 0x9f)
}

function getC1StringKind(character: string): TerminalStringKind | undefined {
  if (character === '\u0090') return 'dcs'
  if (character === '\u0098') return 'sos'
  if (character === '\u009e') return 'pm'
  if (character === '\u009f') return 'apc'
  if (character === '\u009d') return 'osc'
  return undefined
}

function getEscapeStringKind(character: string): TerminalStringKind | undefined {
  if (character === 'P') return 'dcs'
  if (character === 'X') return 'sos'
  if (character === '^') return 'pm'
  if (character === '_') return 'apc'
  return undefined
}

function isEscapeIntermediate(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x20 && code <= 0x2f
}

function isCsiFinalByte(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x40 && code <= 0x7e
}

function isCsiParameterOrIntermediate(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x20 && code <= 0x3f
}

function isEscapeFinal(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x30 && code <= 0x7e
}

function isSequenceCancel(character: string): boolean {
  return character === '\u0018' || character === '\u001a'
}

function isHighSurrogateCode(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff
}

function isLowSurrogateCode(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff
}

export function wrapDisplayText(text: string, width: number, options: WrapDisplayOptions = {}): string {
  const safeWidth = normalizeWidth(width)
  if (safeWidth === 0) return ''
  const wrapped = wrapAnsi(text.replace(/\r\n?/gu, '\n'), safeWidth, {
    hard: options.hard ?? true,
    trim: options.trim ?? false,
    wordWrap: options.wordWrap ?? true,
  })
  // wrap-ansi keeps an indivisible wide grapheme intact even when the
  // terminal has only one cell left. Apply the same grapheme-safe truncation
  // used by the public truncation helper so every returned line obeys width.
  return wrapped.split('\n').map((line) => truncateTrustedDisplayLine(line, safeWidth, { marker: '…' })).join('\n')
}

export function wrapDisplayLines(text: string, width: number, options: WrapDisplayOptions = {}): string[] {
  return sanitizeTerminalText(text).split('\n').flatMap((line) => wrapDisplayText(line, width, options).split('\n'))
}

export function truncateDisplayText(text: string, width: number, options: TruncateDisplayOptions = {}): string {
  const safeWidth = normalizeWidth(width, 0)
  return text.split(/\r\n?|\n/gu).map((line) => truncateTrustedDisplayLine(line, safeWidth, options)).join('\n')
}

export function fillDisplayWidth(fill: string, width: number): string {
  const safeWidth = normalizeWidth(width, 0)
  const safeFill = sanitizeTerminalText(fill, { expandTabs: true }).replace(/\n/gu, '')
  const unitWidth = visibleWidth(safeFill)
  if (safeWidth < 1 || unitWidth < 1) return ''
  const count = Math.floor(safeWidth / unitWidth)
  return safeFill.repeat(count)
}

export function sanitizeAndWrap(text: string, width: number, options: WrapDisplayOptions = {}): {
  lines: string[]
  hardWrapped: boolean
} {
  const safe = sanitizeTerminalText(text, { expandTabs: true })
  const lines = wrapDisplayLines(safe, width, options)
  return { lines, hardWrapped: lines.length > safe.split('\n').length }
}

/**
 * Trusted renderer text may contain SGR/ANSI sequences. It is never used for
 * model/tool input directly; untrusted values must go through
 * sanitizeAndWrap/sanitizeTerminalText first.
 */
function truncateTrustedDisplayLine(line: string, width: number, options: TruncateDisplayOptions): string {
  if (width <= 0) return ''
  if (visibleWidth(line) <= width) return closeTrustedSgr(line)
  const marker = fitGraphemes(options.marker ?? '…', width)
  const markerWidth = visibleWidth(marker)
  if (markerWidth >= width) return marker
  const budget = width - markerWidth
  const tokens = ansiGraphemes(line)
  const selected = options.position === 'start'
    ? takeFromEnd(tokens, budget)
    : takeFromStart(tokens, budget)
  const output = options.position === 'start' ? marker + selected : selected + marker
  return closeTrustedSgr(output)
}

function takeFromStart(tokens: readonly string[], width: number): string {
  let used = 0
  let output = ''
  for (const token of tokens) {
    const tokenWidth = visibleWidth(token)
    if (used + tokenWidth > width) break
    output += token
    used += tokenWidth
  }
  return output
}

function takeFromEnd(tokens: readonly string[], width: number): string {
  let used = 0
  let output = ''
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index] ?? ''
    const tokenWidth = visibleWidth(token)
    if (used + tokenWidth > width) break
    output = token + output
    used += tokenWidth
  }
  return output
}

function fitGraphemes(value: string, width: number): string {
  if (width <= 0) return ''
  if (visibleWidth(value) <= width) return value
  return takeFromStart(ansiGraphemes(value), width)
}

function ansiGraphemes(value: string): string[] {
  const result: string[] = []
  let cursor = 0
  for (const match of value.matchAll(ANSI_SEQUENCE)) {
    const start = match.index ?? 0
    if (start > cursor) result.push(...graphemes(value.slice(cursor, start)))
    result.push(match[0])
    cursor = start + match[0].length
  }
  if (cursor < value.length) result.push(...graphemes(value.slice(cursor)))
  return result
}

function graphemes(value: string): string[] {
  if (segmenter) return Array.from(segmenter.segment(value), (part) => part.segment)
  return Array.from(value)
}

function normalizeWidth(width: number, fallback = 1): number {
  if (typeof width === 'number' && Number.isFinite(width) && width >= 0) return Math.floor(width)
  return fallback
}

function visibleWidth(value: string): number {
  return stringWidth(stripAnsi(value))
}

function closeTrustedSgr(value: string): string {
  if (/\u001b\[[0-9;]*m/u.test(value) && !value.endsWith('\u001b[0m')) return `${value}\u001b[0m`
  return value
}
