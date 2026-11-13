import stringWidth from 'string-width'
import stripAnsi from 'strip-ansi'

// [maint-note:fe5983895047]
// 数据契约导航：ContentWidthOptions；字段含义以消费端校验为准。
// 成员 terminalWidth：number。
// 成员 prefixWidth（可省略）：number | string。
// 成员 paddingLeft（可省略）：number。
// 成员 paddingRight（可省略）：number。
// 成员 borderWidth（可省略）：number。
// 成员 safetyMargin（可省略）：number。
// 成员 minWidth（可省略）：number。
// 成员 maxWidth（可省略）：number。
// [/maint-note:fe5983895047]
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

// [maint-note:d340659d0371]
// 维护导航：normalizeTerminalWidth；下列为本函数源码中的接口与路径索引。
// 输入 columns：unknown。
// 输入 fallback：类型由调用方与函数体约束；缺省表达式 80。
// 返回契约：number。
// 返回路径：finitePositive(columns) ? Math.floor(columns as number) : safeFallback。
// 直接调用：finitePositive。
// 直接调用：Math.floor。
// [/maint-note:d340659d0371]
export function normalizeTerminalWidth(columns: unknown, fallback = 80): number {
  const safeFallback = finitePositive(fallback) ? Math.floor(fallback) : 80
  return finitePositive(columns) ? Math.floor(columns as number) : safeFallback
}

// [maint-note:70f8fbf5b079]
// 维护导航：calculateContentWidth；下列为本函数源码中的接口与路径索引。
// 输入 options：ContentWidthOptions。
// 返回契约：number。
// 返回路径：Math.max(minWidth, Math.min(maxWidth, available))。
// 直接调用：normalizeTerminalWidth。
// 直接调用：finiteNonNegative。
// 直接调用：displayWidth。
// 直接调用：Math.max。
// 直接调用：Math.min。
// [/maint-note:70f8fbf5b079]
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

// [maint-note:38e28b7d8303]
// 维护导航：displayWidth；下列为本函数源码中的接口与路径索引。
// 输入 text：string。
// 返回契约：number。
// 返回路径：text.split(/\r\n?|\n/u).reduce((maximum, line) => Math.max(maximum, visibleWidth(line)), 0)。
// 直接调用：text.split(/\r\n?|\n/u).reduce。
// 直接调用：text.split。
// [/maint-note:38e28b7d8303]
export function displayWidth(text: string): number {
  return text.split(/\r\n?|\n/u).reduce((maximum, line) => Math.max(maximum, visibleWidth(line)), 0)
}

// [maint-note:fae7061e15d1]
// 维护导航：displayLineWidth；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：number。
// 返回路径：visibleWidth(line)。
// 直接调用：visibleWidth。
// [/maint-note:fae7061e15d1]
export function displayLineWidth(line: string): number {
  return visibleWidth(line)
}

// [maint-note:a3dd0819bd49]
// 维护导航：finitePositive；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is number。
// 返回路径：typeof value === 'number' && Number.isFinite(value) && value > 0。
// 直接调用：Number.isFinite。
// [/maint-note:a3dd0819bd49]
function finitePositive(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

// [maint-note:f8ec995ffbaf]
// 维护导航：finiteNonNegative；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：number。
// 返回路径：typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0。
// 直接调用：Number.isFinite。
// 直接调用：Math.floor。
// [/maint-note:f8ec995ffbaf]
function finiteNonNegative(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0
}

// [maint-note:b59ebcff0071]
// 维护导航：visibleWidth；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：number。
// 返回路径：stringWidth(stripAnsi(value).replace(/[\u0000-\u001f\u007f-\u009f]/gu, ''))。
// 直接调用：stringWidth。
// 直接调用：stripAnsi(value).replace。
// 直接调用：stripAnsi。
// [/maint-note:b59ebcff0071]
function visibleWidth(value: string): number {
  return stringWidth(stripAnsi(value).replace(/[\u0000-\u001f\u007f-\u009f]/gu, ''))
}
