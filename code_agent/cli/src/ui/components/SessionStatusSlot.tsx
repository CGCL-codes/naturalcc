import { homedir } from 'node:os'
import { Text } from 'ink'

import {
  displayLineWidth,
  normalizeTerminalWidth,
  sanitizeTerminalText,
  splitDisplayGraphemes,
} from '../terminal/index.js'
import { UI_PALETTE, type UiPaletteColor } from '../theme.js'

// [maint-note:7c52272a0101]
// 数据契约导航：SessionStatusMetadata；字段含义以消费端校验为准。
// 成员 model：string。
// 成员 workspace：string。
// [/maint-note:7c52272a0101]
export interface SessionStatusMetadata {
  readonly model: string
  readonly workspace: string
}

export const DEFAULT_SESSION_STATUS: SessionStatusMetadata = Object.freeze({
  model: 'unknown',
  workspace: '.',
})

// Resolve the display-only home prefix once at module load. The component
// receives model/workspace as explicit metadata and does not reread process
// configuration on every render or timer tick.
const DEFAULT_HOME_DIRECTORY = homedir()
const EXIT_WARNING = 'press ctrl+c again to exit'
const COMPACT_EXIT_WARNING = 'ctrl+c again to exit'

// [maint-note:c20af93a94c4]
// 数据契约导航：SessionStatusSpanRole；字段含义以消费端校验为准。
// 类型组合：'label' | 'model' | 'workspace'。
// [/maint-note:c20af93a94c4]
export type SessionStatusSpanRole = 'label' | 'model' | 'workspace'

// [maint-note:4fe71e26bf66]
// 数据契约导航：SessionStatusSpan；字段含义以消费端校验为准。
// 成员 role：SessionStatusSpanRole。
// 成员 text：string。
// 成员 color：UiPaletteColor。
// [/maint-note:4fe71e26bf66]
export interface SessionStatusSpan {
  readonly role: SessionStatusSpanRole
  readonly text: string
  readonly color: UiPaletteColor
}

/** Remove terminal controls and line breaks from app-provided UI metadata. */
// [maint-note:cc8d986212ef]
// 维护导航：sanitizeSessionStatusValue；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：sanitizeTerminalText(value).replace(/[\r\n\t]+/gu, ' ').trim()。
// 直接调用：sanitizeTerminalText(value).replace(/[\r\n\t]+/gu, ' ').trim。
// 直接调用：sanitizeTerminalText(value).replace。
// 直接调用：sanitizeTerminalText。
// [/maint-note:cc8d986212ef]
export function sanitizeSessionStatusValue(value: string): string {
  return sanitizeTerminalText(value).replace(/[\r\n\t]+/gu, ' ').trim()
}

// [maint-note:00337bf0de13]
// 维护导航：displayWorkspacePath；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 home：类型由调用方与函数体约束；缺省表达式 DEFAULT_HOME_DIRECTORY。
// 返回契约：string。
// 分支条件：!safeHome。
// 分支条件：workspace === safeHome。
// 分支条件：workspace.startsWith(`${safeHome}/`) || workspace.startsWith(`${safeHome}\\`)。
// 返回路径：workspace。
// 返回路径：'~'。
// 返回路径：`~/${workspace.slice(safeHome.length + 1).replaceAll('\\', '/')}`。
// 直接调用：sanitizeSessionStatusValue。
// 直接调用：sanitizeSessionStatusValue(home).replace。
// 直接调用：workspace.startsWith。
// 直接调用：workspace.slice(safeHome.length + 1).replaceAll。
// 直接调用：workspace.slice。
// [/maint-note:00337bf0de13]
export function displayWorkspacePath(value: string, home = DEFAULT_HOME_DIRECTORY): string {
  const workspace = sanitizeSessionStatusValue(value) || '.'
  const safeHome = sanitizeSessionStatusValue(home).replace(/[\\/]+$/u, '')
  if (!safeHome) return workspace
  if (workspace === safeHome) return '~'
  if (workspace.startsWith(`${safeHome}/`) || workspace.startsWith(`${safeHome}\\`)) {
    return `~/${workspace.slice(safeHome.length + 1).replaceAll('\\', '/')}`
  }
  return workspace
}

// [maint-note:d236bcff0cd3]
// 维护导航：workspaceBasename；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：index >= 0 ? normalized.slice(index + 1) || '/' : normalized。
// 直接调用：value.replace。
// 直接调用：Math.max。
// 直接调用：normalized.lastIndexOf。
// 直接调用：normalized.slice。
// [/maint-note:d236bcff0cd3]
function workspaceBasename(value: string): string {
  const normalized = value.replace(/[\\/]+$/u, '')
  const index = Math.max(normalized.lastIndexOf('/'), normalized.lastIndexOf('\\'))
  return index >= 0 ? normalized.slice(index + 1) || '/' : normalized
}

/**
 * Pure, one-line status projection. Model is the stable priority when the
 * complete model/workspace pair cannot fit; workspace first degrades to its
 * basename before that final model-priority truncation.
 */
// [maint-note:041c9b6e5b55]
// 维护导航：projectSessionStatusSpans；下列为本函数源码中的接口与路径索引。
// 输入 metadata：SessionStatusMetadata。
// 输入 width：number。
// 返回契约：readonly SessionStatusSpan[]。
// 分支条件：displayLineWidth(joinStatusSpans(full)) <= safeWidth。
// 分支条件：displayLineWidth(joinStatusSpans(compact)) <= safeWidth。
// 返回路径：full。
// 返回路径：compact。
// 返回路径：truncateStatusSpans([ statusSpan('label', 'Model: '), statusSpan('model', model), ], safeWidth)。
// 直接调用：normalizeTerminalWidth。
// 直接调用：sanitizeSessionStatusValue。
// 直接调用：displayWorkspacePath。
// 直接调用：candidate。
// 直接调用：displayLineWidth。
// 直接调用：joinStatusSpans。
// 直接调用：workspaceBasename。
// 直接调用：truncateStatusSpans。
// 直接调用：statusSpan。
// [/maint-note:041c9b6e5b55]
export function projectSessionStatusSpans(metadata: SessionStatusMetadata, width: number): readonly SessionStatusSpan[] {
  const safeWidth = normalizeTerminalWidth(width, 1)
  const model = sanitizeSessionStatusValue(metadata.model) || 'unknown'
  const workspace = displayWorkspacePath(metadata.workspace)
  // [maint-note:80295813b8cc]
  // 维护导航：candidate；下列为本函数源码中的接口与路径索引。
  // 输入 workspaceValue：string。
  // 返回契约：SessionStatusSpan[]。
  // 直接调用：statusSpan。
  // [/maint-note:80295813b8cc]
  const candidate = (workspaceValue: string): SessionStatusSpan[] => [
    statusSpan('label', 'Model: '),
    statusSpan('model', model),
    statusSpan('label', ' · Workspace: '),
    statusSpan('workspace', workspaceValue),
  ]
  const full = candidate(workspace)
  if (displayLineWidth(joinStatusSpans(full)) <= safeWidth) return full

  const compact = candidate(workspaceBasename(workspace))
  if (displayLineWidth(joinStatusSpans(compact)) <= safeWidth) return compact

  return truncateStatusSpans([
    statusSpan('label', 'Model: '),
    statusSpan('model', model),
  ], safeWidth)
}

// [maint-note:eed9fe54131c]
// 维护导航：projectSessionStatus；下列为本函数源码中的接口与路径索引。
// 输入 metadata：SessionStatusMetadata。
// 输入 width：number。
// 返回契约：string。
// 返回路径：joinStatusSpans(projectSessionStatusSpans(metadata, width))。
// 直接调用：joinStatusSpans。
// 直接调用：projectSessionStatusSpans。
// [/maint-note:eed9fe54131c]
export function projectSessionStatus(metadata: SessionStatusMetadata, width: number): string {
  return joinStatusSpans(projectSessionStatusSpans(metadata, width))
}

// [maint-note:97ada83d0ede]
// 维护导航：projectSessionStatusWarning；下列为本函数源码中的接口与路径索引。
// 输入 width：number。
// 返回契约：string。
// 分支条件：displayLineWidth(EXIT_WARNING) <= safeWidth。
// 分支条件：displayLineWidth(COMPACT_EXIT_WARNING) <= safeWidth。
// 返回路径：EXIT_WARNING。
// 返回路径：COMPACT_EXIT_WARNING。
// 返回路径：joinStatusSpans(truncateStatusSpans([ { role: 'label', text: EXIT_WARNING, color: UI_PALETTE.danger }, ], safeWidth))。
// 直接调用：normalizeTerminalWidth。
// 直接调用：displayLineWidth。
// 直接调用：joinStatusSpans。
// 直接调用：truncateStatusSpans。
// [/maint-note:97ada83d0ede]
export function projectSessionStatusWarning(width: number): string {
  const safeWidth = normalizeTerminalWidth(width, 1)
  if (displayLineWidth(EXIT_WARNING) <= safeWidth) return EXIT_WARNING
  if (displayLineWidth(COMPACT_EXIT_WARNING) <= safeWidth) return COMPACT_EXIT_WARNING
  return joinStatusSpans(truncateStatusSpans([
    { role: 'label', text: EXIT_WARNING, color: UI_PALETTE.danger },
  ], safeWidth))
}

// [maint-note:94c6ba1e2b7d]
// 维护导航：SessionStatusSlot；下列为本函数源码中的接口与路径索引。
// 输入 { metadata = DEFAULT_SESSION_STATUS, exitArmed, width, renderMarker = '', }：{ metadata?: SessionStatusMetadata exitArmed: boolean width: number renderMarker?: string }。
// 分支条件：exitArmed。
// 返回路径：<Text color={UI_PALETTE.danger}>{renderMarker}{projectSessionStatusWarning(width)}</Text>。
// 返回路径：( <Text> {renderMarker} {spans.map((span, index) => <Text color={span.color} key={index}>{span.text}</Text>)} </Text> )。
// 直接调用：projectSessionStatusWarning。
// 直接调用：projectSessionStatusSpans。
// 直接调用：spans.map。
// [/maint-note:94c6ba1e2b7d]
export function SessionStatusSlot({
  metadata = DEFAULT_SESSION_STATUS,
  exitArmed,
  width,
  renderMarker = '',
}: {
  metadata?: SessionStatusMetadata
  exitArmed: boolean
  width: number
  renderMarker?: string
}) {
  if (exitArmed) {
    return <Text color={UI_PALETTE.danger}>{renderMarker}{projectSessionStatusWarning(width)}</Text>
  }
  const spans = projectSessionStatusSpans(metadata, width)
  return (
    <Text>
      {renderMarker}
      {spans.map((span, index) => <Text color={span.color} key={index}>{span.text}</Text>)}
    </Text>
  )
}

// [maint-note:012bcce215a8]
// 维护导航：statusSpan；下列为本函数源码中的接口与路径索引。
// 输入 role：SessionStatusSpanRole。
// 输入 text：string。
// 返回契约：SessionStatusSpan。
// 返回路径：{ role, text, color }。
// [/maint-note:012bcce215a8]
function statusSpan(role: SessionStatusSpanRole, text: string): SessionStatusSpan {
  const color = role === 'model'
    ? UI_PALETTE.model
    : role === 'workspace' ? UI_PALETTE.workspace : UI_PALETTE.neutral
  return { role, text, color }
}

// [maint-note:4fbb8c96dab8]
// 维护导航：joinStatusSpans；下列为本函数源码中的接口与路径索引。
// 输入 spans：readonly SessionStatusSpan[]。
// 返回契约：string。
// 返回路径：spans.map(span => span.text).join('')。
// 直接调用：spans.map(span => span.text).join。
// 直接调用：spans.map。
// [/maint-note:4fbb8c96dab8]
function joinStatusSpans(spans: readonly SessionStatusSpan[]): string {
  return spans.map(span => span.text).join('')
}

/** Grapheme-safe truncation that retains semantic ownership of every cell. */
// [maint-note:cb7564bc652a]
// 维护导航：truncateStatusSpans；下列为本函数源码中的接口与路径索引。
// 输入 spans：readonly SessionStatusSpan[]。
// 输入 width：number。
// 返回契约：readonly SessionStatusSpan[]。
// 分支条件：displayLineWidth(joinStatusSpans(spans)) <= safeWidth。
// 分支条件：safeWidth === 1。
// 分支条件：used + graphemeWidth > budget。
// 遍历边界：const span ← spans。
// 遍历边界：const grapheme ← splitDisplayGraphemes(span.text)。
// 返回路径：spans。
// 返回路径：[{ ...spans[0]!, text: '…' }]。
// 返回路径：result。
// 写入目标：markerOwner。
// 写入目标：used。
// 直接调用：normalizeTerminalWidth。
// 直接调用：displayLineWidth。
// 直接调用：joinStatusSpans。
// 直接调用：splitDisplayGraphemes。
// 直接调用：appendStatusSpan。
// [/maint-note:cb7564bc652a]
function truncateStatusSpans(spans: readonly SessionStatusSpan[], width: number): readonly SessionStatusSpan[] {
  const safeWidth = normalizeTerminalWidth(width, 1)
  if (displayLineWidth(joinStatusSpans(spans)) <= safeWidth) return spans
  if (safeWidth === 1) return [{ ...spans[0]!, text: '…' }]

  const result: SessionStatusSpan[] = []
  let used = 0
  let markerOwner = spans[0]!
  const budget = safeWidth - 1
  outer: for (const span of spans) {
    for (const grapheme of splitDisplayGraphemes(span.text)) {
      const graphemeWidth = displayLineWidth(grapheme)
      if (used + graphemeWidth > budget) {
        markerOwner = span
        break outer
      }
      appendStatusSpan(result, span, grapheme)
      used += graphemeWidth
      markerOwner = span
    }
  }
  appendStatusSpan(result, markerOwner, '…')
  return result
}

// [maint-note:896e7cd33014]
// 维护导航：appendStatusSpan；下列为本函数源码中的接口与路径索引。
// 输入 target：SessionStatusSpan[]。
// 输入 source：SessionStatusSpan。
// 输入 text：string。
// 返回契约：void。
// 分支条件：previous?.role === source.role && previous.color === source.color。
// 写入目标：target[target.length - 1]。
// 直接调用：target.at。
// 直接调用：target.push。
// [/maint-note:896e7cd33014]
function appendStatusSpan(target: SessionStatusSpan[], source: SessionStatusSpan, text: string): void {
  const previous = target.at(-1)
  if (previous?.role === source.role && previous.color === source.color) {
    target[target.length - 1] = { ...previous, text: previous.text + text }
  } else {
    target.push({ ...source, text })
  }
}
