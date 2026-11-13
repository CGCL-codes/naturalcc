import stringWidth from 'string-width'
import stripAnsi from 'strip-ansi'
import wrapAnsi from 'wrap-ansi'

const ANSI_SEQUENCE = /\u001b(?:\[[0-?]*[ -/]*[@-~]|\][\s\S]*?(?:\u0007|\u001b\\)|[()][0-2A-Z])/gu
const segmenter = typeof Intl.Segmenter === 'function'
  ? new Intl.Segmenter(undefined, { granularity: 'grapheme' })
  : undefined

// [maint-note:b9c27409565b]
// 数据契约导航：WrapDisplayOptions；字段含义以消费端校验为准。
// 成员 hard（可省略）：boolean。
// 成员 trim（可省略）：boolean。
// 成员 wordWrap（可省略）：boolean。
// [/maint-note:b9c27409565b]
export interface WrapDisplayOptions {
  hard?: boolean
  trim?: boolean
  wordWrap?: boolean
}

// [maint-note:f52791edeb72]
// 数据契约导航：TruncateDisplayOptions；字段含义以消费端校验为准。
// 成员 marker（可省略）：string。
// 成员 position（可省略）：'start' | 'end'。
// [/maint-note:f52791edeb72]
export interface TruncateDisplayOptions {
  marker?: string
  position?: 'start' | 'end'
}

// [maint-note:25b76f380f00]
// 数据契约导航：TerminalEscapeState；字段含义以消费端校验为准。
// 类型组合：'normal' | 'escape' | 'escapeIntermediate' | 'csi' | 'string' | 'stringEscape'。
// [/maint-note:25b76f380f00]
type TerminalEscapeState = 'normal' | 'escape' | 'escapeIntermediate' | 'csi' | 'string' | 'stringEscape'
// [maint-note:7a6ae129d46a]
// 数据契约导航：TerminalStringKind；字段含义以消费端校验为准。
// 类型组合：'osc' | 'dcs' | 'sos' | 'pm' | 'apc'。
// [/maint-note:7a6ae129d46a]
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

  // [maint-note:788b7ca69266]
  // 维护导航：push；下列为本函数源码中的接口与路径索引。
  // 输入 input：string。
  // 返回契约：string。
  // 分支条件：input.length === 0。
  // 分支条件：this.pendingHighSurrogate。
  // 分支条件：isLowSurrogateCode(input.charCodeAt(0))。
  // 分支条件：isHighSurrogateCode(input.charCodeAt(input.length - 1))。
  // 返回路径：''。
  // 返回路径：output + this.process(input)。
  // 写入目标：input。
  // 写入目标：this.pendingHighSurrogate。
  // 写入目标：output。
  // 直接调用：isLowSurrogateCode。
  // 直接调用：input.charCodeAt。
  // 直接调用：this.process。
  // 直接调用：isHighSurrogateCode。
  // 直接调用：input.slice。
  // [/maint-note:788b7ca69266]
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

  // [maint-note:45be4b6af17a]
  // 维护导航：process；下列为本函数源码中的接口与路径索引。
  // 输入 input：string。
  // 返回契约：string。
  // 分支条件：this.state === 'normal'。
  // 分支条件：this.pendingCR。
  // 分支条件：character === '\n'。
  // 分支条件：character === '\r'。
  // 分支条件：character === '\u001b'。
  // 分支条件：character === '\u009b'。
  // 分支条件：c1StringKind。
  // 分支条件：character === '\u009c' || isTerminalControl(character)。
  // 分支条件：isHighSurrogateCode(code)。
  // 分支条件：isLowSurrogateCode(nextCode)。
  // 分支条件：isLowSurrogateCode(code)。
  // 分支条件：this.state === 'escape'。
  // 分支条件：isSequenceCancel(character)。
  // 分支条件：character === '['。
  // 分支条件：character === ']'。
  // 分支条件：escapeStringKind。
  // 分支条件：isEscapeIntermediate(character)。
  // 分支条件：isEscapeFinal(character)。
  // 分支条件：this.state === 'escapeIntermediate'。
  // 分支条件：this.state === 'csi'。
  // 分支条件：isCsiFinalByte(character)。
  // 分支条件：isCsiParameterOrIntermediate(character)。
  // 分支条件：this.state === 'string'。
  // 分支条件：(this.stringKind === 'osc' && character === '\u0007') || character === '\u009c'。
  // 分支条件：character === '\\'。
  // 遍历边界：index < input.length。
  // 返回路径：output。
  // 写入目标：output。
  // 写入目标：this.pendingCR。
  // 写入目标：index。
  // 写入目标：this.state。
  // 直接调用：getC1StringKind。
  // 直接调用：this.beginString。
  // 直接调用：isTerminalControl。
  // 直接调用：character.charCodeAt。
  // 直接调用：isHighSurrogateCode。
  // 直接调用：input.charCodeAt。
  // 直接调用：isLowSurrogateCode。
  // 直接调用：isSequenceCancel。
  // 直接调用：this.endSequence。
  // 直接调用：getEscapeStringKind。
  // 直接调用：isEscapeIntermediate。
  // 直接调用：isEscapeFinal。
  // 直接调用：isCsiFinalByte。
  // 直接调用：isCsiParameterOrIntermediate。
  // [/maint-note:45be4b6af17a]
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

  // [maint-note:424780367443]
  // 维护导航：finish；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 分支条件：this.pendingCR。
  // 分支条件：this.pendingHighSurrogate。
  // 返回路径：output。
  // 写入目标：this.pendingCR。
  // 写入目标：output。
  // 写入目标：this.pendingHighSurrogate。
  // 直接调用：this.process。
  // 直接调用：this.endSequence。
  // [/maint-note:424780367443]
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

  // [maint-note:d1192f3844cf]
  // 维护导航：reset；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.pendingCR。
  // 写入目标：this.pendingHighSurrogate。
  // 直接调用：this.endSequence。
  // [/maint-note:d1192f3844cf]
  reset(): void {
    this.endSequence()
    this.pendingCR = false
    this.pendingHighSurrogate = ''
  }

  // [maint-note:7957dd989980]
  // 维护导航：beginString；下列为本函数源码中的接口与路径索引。
  // 输入 kind：TerminalStringKind。
  // 返回契约：void。
  // 写入目标：this.state。
  // 写入目标：this.stringKind。
  // [/maint-note:7957dd989980]
  private beginString(kind: TerminalStringKind): void {
    this.state = 'string'
    this.stringKind = kind
  }

  // [maint-note:ffff5eca75b2]
  // 维护导航：endSequence；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.state。
  // 写入目标：this.stringKind。
  // [/maint-note:ffff5eca75b2]
  private endSequence(): void {
    this.state = 'normal'
    this.stringKind = undefined
  }
}

// [maint-note:e3320a2f2a95]
// 维护导航：sanitizeTerminalText；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 options：{ expandTabs?: boolean }；缺省表达式 {}。
// 返回契约：string。
// 返回路径：options.expandTabs ? safe.replace(/\t/gu, ' ') : safe。
// 直接调用：normalizer.push。
// 直接调用：normalizer.finish。
// 直接调用：safe.replace。
// [/maint-note:e3320a2f2a95]
export function sanitizeTerminalText(text: string, options: { expandTabs?: boolean } = {}): string {
  const normalizer = new StreamingTerminalTextNormalizer()
  const safe = normalizer.push(text) + normalizer.finish()
  return options.expandTabs ? safe.replace(/\t/gu, '    ') : safe
}

// [maint-note:626a12ad7c9d]
// 维护导航：isTerminalControl；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：(code >= 0 && code <= 0x08) || (code >= 0x0b && code <= 0x0c) || (code >= 0x0e && code <= 0x1f) || (code >= 0x7f && code <= 0x9f)。
// 直接调用：character.charCodeAt。
// [/maint-note:626a12ad7c9d]
function isTerminalControl(character: string): boolean {
  const code = character.charCodeAt(0)
  return (code >= 0 && code <= 0x08) || (code >= 0x0b && code <= 0x0c) || (code >= 0x0e && code <= 0x1f) || (code >= 0x7f && code <= 0x9f)
}

// [maint-note:0e0ac7df0ed2]
// 维护导航：getC1StringKind；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：TerminalStringKind | undefined。
// 分支条件：character === '\u0090'。
// 分支条件：character === '\u0098'。
// 分支条件：character === '\u009e'。
// 分支条件：character === '\u009f'。
// 分支条件：character === '\u009d'。
// 返回路径：'dcs'。
// 返回路径：'sos'。
// 返回路径：'pm'。
// 返回路径：'apc'。
// 返回路径：'osc'。
// 返回路径：undefined。
// [/maint-note:0e0ac7df0ed2]
function getC1StringKind(character: string): TerminalStringKind | undefined {
  if (character === '\u0090') return 'dcs'
  if (character === '\u0098') return 'sos'
  if (character === '\u009e') return 'pm'
  if (character === '\u009f') return 'apc'
  if (character === '\u009d') return 'osc'
  return undefined
}

// [maint-note:e6e0264e0fce]
// 维护导航：getEscapeStringKind；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：TerminalStringKind | undefined。
// 分支条件：character === 'P'。
// 分支条件：character === 'X'。
// 分支条件：character === '^'。
// 分支条件：character === '_'。
// 返回路径：'dcs'。
// 返回路径：'sos'。
// 返回路径：'pm'。
// 返回路径：'apc'。
// 返回路径：undefined。
// [/maint-note:e6e0264e0fce]
function getEscapeStringKind(character: string): TerminalStringKind | undefined {
  if (character === 'P') return 'dcs'
  if (character === 'X') return 'sos'
  if (character === '^') return 'pm'
  if (character === '_') return 'apc'
  return undefined
}

// [maint-note:70908168a085]
// 维护导航：isEscapeIntermediate；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：code >= 0x20 && code <= 0x2f。
// 直接调用：character.charCodeAt。
// [/maint-note:70908168a085]
function isEscapeIntermediate(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x20 && code <= 0x2f
}

// [maint-note:53d68eeb17ce]
// 维护导航：isCsiFinalByte；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：code >= 0x40 && code <= 0x7e。
// 直接调用：character.charCodeAt。
// [/maint-note:53d68eeb17ce]
function isCsiFinalByte(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x40 && code <= 0x7e
}

// [maint-note:7dfcef258a98]
// 维护导航：isCsiParameterOrIntermediate；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：code >= 0x20 && code <= 0x3f。
// 直接调用：character.charCodeAt。
// [/maint-note:7dfcef258a98]
function isCsiParameterOrIntermediate(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x20 && code <= 0x3f
}

// [maint-note:01115ca00681]
// 维护导航：isEscapeFinal；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：code >= 0x30 && code <= 0x7e。
// 直接调用：character.charCodeAt。
// [/maint-note:01115ca00681]
function isEscapeFinal(character: string): boolean {
  const code = character.charCodeAt(0)
  return code >= 0x30 && code <= 0x7e
}

// [maint-note:e1a3b9b02fe5]
// 维护导航：isSequenceCancel；下列为本函数源码中的接口与路径索引。
// 输入 character：string。
// 返回契约：boolean。
// 返回路径：character === '\u0018' || character === '\u001a'。
// [/maint-note:e1a3b9b02fe5]
function isSequenceCancel(character: string): boolean {
  return character === '\u0018' || character === '\u001a'
}

// [maint-note:3022f7b01b4e]
// 维护导航：isHighSurrogateCode；下列为本函数源码中的接口与路径索引。
// 输入 code：number。
// 返回契约：boolean。
// 返回路径：code >= 0xd800 && code <= 0xdbff。
// [/maint-note:3022f7b01b4e]
function isHighSurrogateCode(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff
}

// [maint-note:b85978247896]
// 维护导航：isLowSurrogateCode；下列为本函数源码中的接口与路径索引。
// 输入 code：number。
// 返回契约：boolean。
// 返回路径：code >= 0xdc00 && code <= 0xdfff。
// [/maint-note:b85978247896]
function isLowSurrogateCode(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff
}

// [maint-note:3fe0c15f41ce]
// 维护导航：wrapDisplayText；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 width：number。
// 输入 options：WrapDisplayOptions；缺省表达式 {}。
// 返回契约：string。
// 分支条件：safeWidth === 0。
// 返回路径：''。
// 返回路径：wrapped.split('\n').map((line) => truncateTrustedDisplayLine(line, safeWidth, { marker: '…' })).join('\n')。
// 直接调用：normalizeWidth。
// 直接调用：wrapAnsi。
// 直接调用：text.replace。
// 直接调用：wrapped.split('\n').map((line) => truncateTrustedDisplayLine(line, safeWidth, { marker: '…' })).join。
// 直接调用：wrapped.split('\n').map。
// 直接调用：wrapped.split。
// [/maint-note:3fe0c15f41ce]
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

// [maint-note:51d1f2ba7a1e]
// 维护导航：wrapDisplayLines；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 width：number。
// 输入 options：WrapDisplayOptions；缺省表达式 {}。
// 返回契约：string[]。
// 返回路径：sanitizeTerminalText(text).split('\n').flatMap((line) => wrapDisplayText(line, width, options).split('\n'))。
// 直接调用：sanitizeTerminalText(text).split('\n').flatMap。
// 直接调用：sanitizeTerminalText(text).split。
// 直接调用：sanitizeTerminalText。
// [/maint-note:51d1f2ba7a1e]
export function wrapDisplayLines(text: string, width: number, options: WrapDisplayOptions = {}): string[] {
  return sanitizeTerminalText(text).split('\n').flatMap((line) => wrapDisplayText(line, width, options).split('\n'))
}

// [maint-note:e1cf5bad261c]
// 维护导航：truncateDisplayText；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 width：number。
// 输入 options：TruncateDisplayOptions；缺省表达式 {}。
// 返回契约：string。
// 返回路径：text.split(/\r\n?|\n/gu).map((line) => truncateTrustedDisplayLine(line, safeWidth, options)).join('\n')。
// 直接调用：normalizeWidth。
// 直接调用：text.split(/\r\n?|\n/gu).map((line) => truncateTrustedDisplayLine(line, safeWidth, options)).join。
// 直接调用：text.split(/\r\n?|\n/gu).map。
// 直接调用：text.split。
// [/maint-note:e1cf5bad261c]
export function truncateDisplayText(text: string, width: number, options: TruncateDisplayOptions = {}): string {
  const safeWidth = normalizeWidth(width, 0)
  return text.split(/\r\n?|\n/gu).map((line) => truncateTrustedDisplayLine(line, safeWidth, options)).join('\n')
}

// [maint-note:29825167fb4e]
// 维护导航：fillDisplayWidth；下列为本函数源码中的接口与路径索引。
// 输入 fill：string。
// 输入 width：number。
// 返回契约：string。
// 分支条件：safeWidth < 1 || unitWidth < 1。
// 返回路径：''。
// 返回路径：safeFill.repeat(count)。
// 直接调用：normalizeWidth。
// 直接调用：sanitizeTerminalText(fill, { expandTabs: true }).replace。
// 直接调用：sanitizeTerminalText。
// 直接调用：visibleWidth。
// 直接调用：Math.floor。
// 直接调用：safeFill.repeat。
// [/maint-note:29825167fb4e]
export function fillDisplayWidth(fill: string, width: number): string {
  const safeWidth = normalizeWidth(width, 0)
  const safeFill = sanitizeTerminalText(fill, { expandTabs: true }).replace(/\n/gu, '')
  const unitWidth = visibleWidth(safeFill)
  if (safeWidth < 1 || unitWidth < 1) return ''
  const count = Math.floor(safeWidth / unitWidth)
  return safeFill.repeat(count)
}

// [maint-note:8e4fa4602290]
// 维护导航：sanitizeAndWrap；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 输入 width：number。
// 输入 options：WrapDisplayOptions；缺省表达式 {}。
// 返回契约：{ lines: string[] hardWrapped: boolean }。
// 返回路径：{ lines, hardWrapped: lines.length > safe.split('\n').length }。
// 直接调用：sanitizeTerminalText。
// 直接调用：wrapDisplayLines。
// 直接调用：safe.split。
// [/maint-note:8e4fa4602290]
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
// [maint-note:883a562aaf44]
// 维护导航：truncateTrustedDisplayLine；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 输入 width：number。
// 输入 options：TruncateDisplayOptions。
// 返回契约：string。
// 分支条件：width <= 0。
// 分支条件：visibleWidth(line) <= width。
// 分支条件：markerWidth >= width。
// 返回路径：''。
// 返回路径：closeTrustedSgr(line)。
// 返回路径：marker。
// 返回路径：closeTrustedSgr(output)。
// 直接调用：visibleWidth。
// 直接调用：closeTrustedSgr。
// 直接调用：fitGraphemes。
// 直接调用：ansiGraphemes。
// 直接调用：takeFromEnd。
// 直接调用：takeFromStart。
// [/maint-note:883a562aaf44]
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

// [maint-note:9e64f8f24907]
// 维护导航：takeFromStart；下列为本函数源码中的接口与路径索引。
// 输入 tokens：readonly string[]。
// 输入 width：number。
// 返回契约：string。
// 分支条件：used + tokenWidth > width。
// 遍历边界：const token ← tokens。
// 返回路径：output。
// 写入目标：output。
// 写入目标：used。
// 直接调用：visibleWidth。
// [/maint-note:9e64f8f24907]
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

// [maint-note:5d0b719079c9]
// 维护导航：takeFromEnd；下列为本函数源码中的接口与路径索引。
// 输入 tokens：readonly string[]。
// 输入 width：number。
// 返回契约：string。
// 分支条件：used + tokenWidth > width。
// 遍历边界：let index = tokens.length - 1; index >= 0; index -= 1。
// 返回路径：output。
// 写入目标：index。
// 写入目标：output。
// 写入目标：used。
// 直接调用：visibleWidth。
// [/maint-note:5d0b719079c9]
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

// [maint-note:cb29717590b4]
// 维护导航：fitGraphemes；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 width：number。
// 返回契约：string。
// 分支条件：width <= 0。
// 分支条件：visibleWidth(value) <= width。
// 返回路径：''。
// 返回路径：value。
// 返回路径：takeFromStart(ansiGraphemes(value), width)。
// 直接调用：visibleWidth。
// 直接调用：takeFromStart。
// 直接调用：ansiGraphemes。
// [/maint-note:cb29717590b4]
function fitGraphemes(value: string, width: number): string {
  if (width <= 0) return ''
  if (visibleWidth(value) <= width) return value
  return takeFromStart(ansiGraphemes(value), width)
}

// [maint-note:24ebb383a10b]
// 维护导航：ansiGraphemes；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string[]。
// 分支条件：start > cursor。
// 分支条件：cursor < value.length。
// 遍历边界：const match ← value.matchAll(ANSI_SEQUENCE)。
// 返回路径：result。
// 写入目标：cursor。
// 直接调用：value.matchAll。
// 直接调用：result.push。
// 直接调用：graphemes。
// 直接调用：value.slice。
// [/maint-note:24ebb383a10b]
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

/** Split display text using terminal grapheme boundaries, not UTF-16/code points. */
// [maint-note:d719875a0eb7]
// 维护导航：splitDisplayGraphemes；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string[]。
// 分支条件：segmenter。
// 返回路径：Array.from(segmenter.segment(value), (part) => part.segment)。
// 返回路径：Array.from(value)。
// 直接调用：Array.from。
// 直接调用：segmenter.segment。
// [/maint-note:d719875a0eb7]
export function splitDisplayGraphemes(value: string): string[] {
  if (segmenter) return Array.from(segmenter.segment(value), (part) => part.segment)
  return Array.from(value)
}

// [maint-note:e48d37609d57]
// 维护导航：graphemes；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string[]。
// 返回路径：splitDisplayGraphemes(value)。
// 直接调用：splitDisplayGraphemes。
// [/maint-note:e48d37609d57]
function graphemes(value: string): string[] {
  return splitDisplayGraphemes(value)
}

// [maint-note:d10c0fb99084]
// 维护导航：normalizeWidth；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 1。
// 返回契约：number。
// 分支条件：typeof width === 'number' && Number.isFinite(width) && width >= 0。
// 返回路径：Math.floor(width)。
// 返回路径：fallback。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// [/maint-note:d10c0fb99084]
function normalizeWidth(width: number, fallback = 1): number {
  if (typeof width === 'number' && Number.isFinite(width) && width >= 0) return Math.floor(width)
  return fallback
}

// [maint-note:88916f928053]
// 维护导航：visibleWidth；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 返回路径：stringWidth(stripAnsi(value))。
// 直接调用：stringWidth。
// 直接调用：stripAnsi。
// [/maint-note:88916f928053]
function visibleWidth(value: string): number {
  return stringWidth(stripAnsi(value))
}

// [maint-note:bee1768924af]
// 维护导航：closeTrustedSgr；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 分支条件：/\u001b\[[0-9;]*m/u.test(value) && !value.endsWith('\u001b[0m')。
// 返回路径：`${value}\u001b[0m`。
// 返回路径：value。
// 直接调用：/\u001b\[[0-9;]*m/u.test。
// 直接调用：value.endsWith。
// [/maint-note:bee1768924af]
function closeTrustedSgr(value: string): string {
  if (/\u001b\[[0-9;]*m/u.test(value) && !value.endsWith('\u001b[0m')) return `${value}\u001b[0m`
  return value
}
