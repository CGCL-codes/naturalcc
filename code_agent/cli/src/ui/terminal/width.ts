import stringWidth from 'string-width'
import stripAnsi from 'strip-ansi'

export interface ContentWidthOptions {
  terminalWidth: number
  prefixWidth?: number | string
  paddingLeft?: number
  paddingRight?: number
  borderWidth?: number
  safetyMargin?: number
  minWidth?: number
  maxWidth?: number
}

export function normalizeTerminalWidth(columns: unknown, fallback = 80): number {
  const safeFallback = finitePositive(fallback) ? Math.floor(fallback) : 80
  return finitePositive(columns) ? Math.floor(columns as number) : safeFallback
}

export function calculateContentWidth(options: ContentWidthOptions): number {
  const terminalWidth = normalizeTerminalWidth(options.terminalWidth)
  const minWidth = finiteNonNegative(options.minWidth ?? 1)
  const maxWidth = options.maxWidth === undefined ? Number.POSITIVE_INFINITY : finiteNonNegative(options.maxWidth)
  const prefixWidth = typeof options.prefixWidth === 'string'
    ? displayWidth(options.prefixWidth)
    : finiteNonNegative(options.prefixWidth ?? 0)
  const reserved = prefixWidth + finiteNonNegative(options.paddingLeft ?? 0) +
    finiteNonNegative(options.paddingRight ?? 0) + finiteNonNegative(options.borderWidth ?? 0) +
    finiteNonNegative(options.safetyMargin ?? 0)
  const available = Math.max(0, terminalWidth - reserved)
  return Math.max(minWidth, Math.min(maxWidth, available))
}

export function displayWidth(text: string): number {
  return text.split(/\r\n?|\n/u).reduce((maximum, line) => Math.max(maximum, visibleWidth(line)), 0)
}

export function displayLineWidth(line: string): number {
  return visibleWidth(line)
}

function finitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function finiteNonNegative(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0
}

function visibleWidth(value: string): number {
  return stringWidth(stripAnsi(value).replace(/[\u0000-\u001f\u007f-\u009f]/gu, ''))
}
