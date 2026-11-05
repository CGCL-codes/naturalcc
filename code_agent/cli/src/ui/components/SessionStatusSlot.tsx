import { homedir } from 'node:os'
import { Text } from 'ink'

import {
  displayLineWidth,
  normalizeTerminalWidth,
  sanitizeTerminalText,
  splitDisplayGraphemes,
} from '../terminal/index.js'
import { UI_PALETTE, type UiPaletteColor } from '../theme.js'

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

export type SessionStatusSpanRole = 'label' | 'model' | 'workspace'

export interface SessionStatusSpan {
  readonly role: SessionStatusSpanRole
  readonly text: string
  readonly color: UiPaletteColor
}

/** Remove terminal controls and line breaks from app-provided UI metadata. */
export function sanitizeSessionStatusValue(value: string): string {
  return sanitizeTerminalText(value).replace(/[\r\n\t]+/gu, ' ').trim()
}

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
export function projectSessionStatusSpans(metadata: SessionStatusMetadata, width: number): readonly SessionStatusSpan[] {
  const safeWidth = normalizeTerminalWidth(width, 1)
  const model = sanitizeSessionStatusValue(metadata.model) || 'unknown'
  const workspace = displayWorkspacePath(metadata.workspace)
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

export function projectSessionStatus(metadata: SessionStatusMetadata, width: number): string {
  return joinStatusSpans(projectSessionStatusSpans(metadata, width))
}

export function projectSessionStatusWarning(width: number): string {
  const safeWidth = normalizeTerminalWidth(width, 1)
  if (displayLineWidth(EXIT_WARNING) <= safeWidth) return EXIT_WARNING
  if (displayLineWidth(COMPACT_EXIT_WARNING) <= safeWidth) return COMPACT_EXIT_WARNING
  return joinStatusSpans(truncateStatusSpans([
    { role: 'label', text: EXIT_WARNING, color: UI_PALETTE.danger },
  ], safeWidth))
}

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

function statusSpan(role: SessionStatusSpanRole, text: string): SessionStatusSpan {
  const color = role === 'model'
    ? UI_PALETTE.model
    : role === 'workspace' ? UI_PALETTE.workspace : UI_PALETTE.neutral
  return { role, text, color }
}

function joinStatusSpans(spans: readonly SessionStatusSpan[]): string {
  return spans.map(span => span.text).join('')
}

/** Grapheme-safe truncation that retains semantic ownership of every cell. */
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

function appendStatusSpan(target: SessionStatusSpan[], source: SessionStatusSpan, text: string): void {
  const previous = target.at(-1)
  if (previous?.role === source.role && previous.color === source.color) {
    target[target.length - 1] = { ...previous, text: previous.text + text }
  } else {
    target.push({ ...source, text })
  }
}
