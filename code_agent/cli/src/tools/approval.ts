import { createInterface } from 'node:readline/promises'
import { stdin, stderr } from 'node:process'
import { randomUUID } from 'node:crypto'

import { ToolInputError, type ToolSchema } from './types.js'

export type ApprovalMode = 'prompt' | 'deny' | 'allow'

/** The model-facing reason budget is deliberately independent of terminal width. */
export const MAX_APPROVAL_REASON_CHARS = 500

const dangerousApprovalControl = /[\u0000-\u001f\u007f-\u009f]/u

export interface ApprovalRequest {
  requestId: string
  callId: string
  toolName: string
  input: Record<string, unknown>
  reason: string
}

export interface ApprovalProvider {
  /** Identifies the single owner of a prompt presentation in one-shot mode. */
  readonly presentationOwner?: 'provider' | 'renderer'
  request(input: ApprovalRequest, signal?: AbortSignal): Promise<boolean>
}

export interface ApprovalController {
  decide(requestId: string, approved: boolean): void
}

export interface ToolPolicy {
  readonly mode: ApprovalMode
  readonly approvalProvider: ApprovalProvider
  requiresApproval(schema: ToolSchema): boolean
  beginTurn?(): void
  createRequestId?(callId: string, ordinal: number): string
}

export class FixedApprovalProvider implements ApprovalProvider {
  constructor(private readonly approved: boolean) {}

  async request(_input: ApprovalRequest, signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    return this.approved
  }
}

export class PromptApprovalProvider implements ApprovalProvider {
  readonly presentationOwner = 'provider' as const

  constructor(
    private readonly input: NodeJS.ReadableStream = stdin,
    private readonly output: NodeJS.WritableStream = stderr,
  ) {}

  async request(request: ApprovalRequest, signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    const lineReader = createInterface({ input: this.input, output: this.output })
    try {
      const prompt = formatApprovalPrompt(request)
      const answer = signal
        ? await lineReader.question(`${prompt}[y/N] `, { signal })
        : await lineReader.question(`${prompt}[y/N] `)
      throwIfAborted(signal)
      return /^(y|yes)$/i.test(answer.trim())
    } finally {
      lineReader.close()
    }
  }
}

/** Approval bridge for Ink: the UI owns the key handling, not readline. */
export class InteractiveApprovalProvider implements ApprovalProvider, ApprovalController {
  private pendingResolve: ((approved: boolean) => void) | undefined
  private pendingRequestId: string | undefined
  private removeAbortListener: (() => void) | undefined

  request(request: ApprovalRequest, signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    return new Promise<boolean>((resolve, reject) => {
      if (this.pendingResolve) {
        reject(new Error('Another approval request is already pending'))
        return
      }
      this.pendingResolve = resolve
      this.pendingRequestId = request.requestId
      if (signal) {
        const onAbort = () => {
          this.clearPending(request.requestId)
          reject(createAbortError())
        }
        signal.addEventListener('abort', onAbort, { once: true })
        this.removeAbortListener = () => signal.removeEventListener('abort', onAbort)
      }
    })
  }

  decide(requestId: string, approved: boolean): void {
    if (!this.pendingResolve || this.pendingRequestId !== requestId) return
    const resolve = this.pendingResolve
    this.clearPending(requestId)
    resolve(approved)
  }

  private clearPending(requestId?: string): void {
    if (requestId !== undefined && this.pendingRequestId !== requestId) return
    this.removeAbortListener?.()
    this.removeAbortListener = undefined
    this.pendingResolve = undefined
    this.pendingRequestId = undefined
  }
}

export function createApprovalProvider(
  mode: ApprovalMode,
  provider?: ApprovalProvider,
): ApprovalProvider {
  if (provider) return provider
  if (mode === 'allow') return new FixedApprovalProvider(true)
  if (mode === 'deny') return new FixedApprovalProvider(false)
  return new PromptApprovalProvider()
}

export function createToolPolicy(
  mode: ApprovalMode,
  provider?: ApprovalProvider,
): ToolPolicy {
  const policyNonce = randomUUID()
  let requestSequence = 0
  return {
    mode,
    approvalProvider: createApprovalProvider(mode, provider),
    requiresApproval: (schema) => schema.readOnly !== true && schema.requiresApproval === true,
    beginTurn: () => {},
    createRequestId: (callId) => `approval-${policyNonce}-${++requestSequence}-${callId}`,
  }
}

export function summarizeApprovalInput(input: Record<string, unknown>): string {
  const safeInput = sanitizeApprovalValue(input, 0, new Set())
  const json = JSON.stringify(safeInput)
  return truncateApprovalPreview(json, 240)
}

/** Validate the model-owned reason before it can reach approval or execution. */
export function validateApprovalReason(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new ToolInputError('Approval reason must be a non-empty string')
  }
  const length = [...value].length
  if (length > MAX_APPROVAL_REASON_CHARS) {
    throw new ToolInputError(
      `Approval reason is too long (maximum ${MAX_APPROVAL_REASON_CHARS} Unicode characters)`,
    )
  }
  if (dangerousApprovalControl.test(value)) {
    throw new ToolInputError('Approval reason must not contain terminal control characters')
  }
  return value
}

/** Require every model-visible mutating schema to declare the same reason contract. */
export function validateApprovalSchema(schema: ToolSchema): void {
  if (schema.readOnly === true || schema.requiresApproval !== true) return
  const parameters = schema.parameters
  const properties = isRecord(parameters.properties) ? parameters.properties : undefined
  const reason = properties && isRecord(properties.reason) ? properties.reason : undefined
  const required = Array.isArray(parameters.required) ? parameters.required : []
  if (
    reason?.type !== 'string' ||
    reason.minLength !== 1 ||
    reason.maxLength !== MAX_APPROVAL_REASON_CHARS ||
    typeof reason.description !== 'string' ||
    !reason.description.trim() ||
    !required.includes('reason')
  ) {
    throw new ToolInputError(
      `Tool ${schema.name} must require reason as a non-empty string with a maximum of ${MAX_APPROVAL_REASON_CHARS} Unicode characters`,
    )
  }
}

export function validateApprovalInput(schema: ToolSchema, input: Record<string, unknown>): string | undefined {
  if (schema.readOnly === true || schema.requiresApproval !== true) return undefined
  validateApprovalSchema(schema)
  return validateApprovalReason(input.reason)
}

export interface ApprovalPresentation {
  title: string
  instructionLabel: 'Command' | 'Patch' | 'Input'
  instruction: string
  reason: string
}

/** A shared, bounded presentation contract used by Ink and readline. */
export function formatApprovalPrompt(request: ApprovalRequest): string {
  const presentation = buildApprovalPresentation(request)
  return [
    presentation.title,
    '',
    `${presentation.instructionLabel}:`,
    presentation.instruction,
    '',
    'Reason:',
    presentation.reason,
    '',
  ].join('\n')
}

export function buildApprovalPresentation(request: ApprovalRequest): ApprovalPresentation {
  const metadata = approvalPresentationMetadata[request.toolName]
  const displayName = sanitizeApprovalDisplayName(metadata?.displayName ?? request.toolName)
  const instructionLabel = metadata?.instructionLabel ?? 'Input'
  const instruction = metadata
    ? metadata.instruction(request.input)
    : summarizeApprovalInputWithoutReason(request.input)
  return {
    title: `Approval required: ${displayName}`,
    instructionLabel,
    instruction: truncateApprovalPreview(redactSensitiveText(stripApprovalControls(instruction))),
    reason: truncateApprovalPreview(stripApprovalControls(request.reason)),
  }
}

interface ApprovalPresentationMetadata {
  displayName: string
  instructionLabel: ApprovalPresentation['instructionLabel']
  instruction: (input: Record<string, unknown>) => string
}

const approvalPresentationMetadata: Record<string, ApprovalPresentationMetadata> = {
  bash: {
    displayName: 'Bash',
    instructionLabel: 'Command',
    instruction: (input) => typeof input.command === 'string' ? input.command : summarizeApprovalInputWithoutReason(input),
  },
  apply_patch: {
    displayName: 'Apply patch',
    instructionLabel: 'Patch',
    instruction: (input) => typeof input.patch === 'string' ? input.patch : summarizeApprovalInputWithoutReason(input),
  },
}

export const MAX_APPROVAL_PREVIEW_CHARS = 1200
export const MAX_APPROVAL_PREVIEW_LINES = 24

export function truncateApprovalPreview(
  value: string,
  maxChars = MAX_APPROVAL_PREVIEW_CHARS,
  maxLines = MAX_APPROVAL_PREVIEW_LINES,
): string {
  const normalized = value.replace(/\r\n?/g, '\n')
  const sourceChars = [...normalized]
  const sourceLines = normalized.split('\n')
  if (sourceChars.length <= maxChars && sourceLines.length <= maxLines) return normalized
  if (sourceLines.length === 1) return truncateSingleLine(sourceChars, maxChars)

  const lineBudget = Math.max(1, Math.floor(maxLines))
  const retainedLineBudget = Math.min(lineBudget - 1, sourceLines.length)
  const headLineCount = Math.floor(retainedLineBudget / 2)
  const tailLineCount = retainedLineBudget - headLineCount
  const headSource = sourceLines.slice(0, headLineCount).join('\n')
  const tailSource = tailLineCount > 0 ? sourceLines.slice(-tailLineCount).join('\n') : ''

  let head = ''
  let tail = ''
  let omittedLines = Math.max(0, sourceLines.length - retainedLineBudget)
  let marker = approvalOmittedMarker(sourceChars.length, omittedLines)
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const omittedChars = sourceChars.length - [...head, ...tail].length
    marker = approvalOmittedMarker(omittedChars, omittedLines)
    const separatorBudget = headSource && tailSource ? 2 : 1
    const contentBudget = Math.max(0, maxChars - [...marker].length - separatorBudget)
    const headBudget = Math.floor(contentBudget / 2)
    const tailBudget = contentBudget - headBudget
    head = trimPreviewBoundary(takePreviewHead(headSource, headBudget))
    tail = trimPreviewBoundary(takePreviewTail(tailSource, tailBudget))

    // Character truncation can remove entire lines from the initially selected
    // head/tail ranges. Count only source lines still represented by content in
    // the final projection; the marker itself is not a source line.
    const representedLines = (head ? head.split('\n').length : 0) + (tail ? tail.split('\n').length : 0)
    const nextOmittedLines = Math.max(0, sourceLines.length - representedLines)
    const nextOmittedChars = sourceChars.length - [...head, ...tail].length
    if (nextOmittedLines === omittedLines && nextOmittedChars === omittedChars) break
    omittedLines = nextOmittedLines
  }

  marker = approvalOmittedMarker(sourceChars.length - [...head, ...tail].length, omittedLines)
  return [head, marker, tail].filter(Boolean).join('\n')
}

function truncateSingleLine(sourceChars: readonly string[], maxChars: number): string {
  let head = ''
  let tail = ''
  let marker = approvalOmittedMarker(sourceChars.length, 0)
  for (let attempt = 0; attempt < 20; attempt += 1) {
    marker = approvalOmittedMarker(sourceChars.length - [...head, ...tail].length, 0)
    const contentBudget = Math.max(0, maxChars - [...marker].length)
    const headBudget = Math.floor(contentBudget / 2)
    const tailBudget = contentBudget - headBudget
    head = sourceChars.slice(0, headBudget).join('')
    tail = tailBudget > 0 ? sourceChars.slice(-tailBudget).join('') : ''
  }
  marker = approvalOmittedMarker(sourceChars.length - [...head, ...tail].length, 0)
  return `${head}${marker}${tail}`
}

function approvalOmittedMarker(omittedChars: number, omittedLines: number): string {
  return omittedLines > 0
    ? `… [${omittedChars} chars omitted; ${omittedLines} lines omitted]`
    : `… [${omittedChars} chars omitted]`
}

function takePreviewHead(value: string, count: number): string {
  return [...value].slice(0, Math.max(0, count)).join('')
}

function takePreviewTail(value: string, count: number): string {
  return count > 0 ? [...value].slice(-count).join('') : ''
}

function trimPreviewBoundary(value: string): string {
  return value.replace(/^\n+|\n+$/g, '')
}

export function stripApprovalControls(value: string): string {
  return value
    .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '')
    .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '')
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/gu, '')
    .replace(/\r/g, '')
}

function redactSensitiveText(value: string): string {
  return value
    .replace(/((?:--?)(?:api[-_]?key|token|password|secret|authorization)\s+)([^\s]+)/giu, '$1[redacted]')
    .replace(/((?:api[-_]?key|token|password|secret|authorization)\s*=\s*)([^\s]+)/giu, '$1[redacted]')
}

function sanitizeApprovalDisplayName(value: string): string {
  const safe = stripApprovalControls(value).trim()
  return truncateApprovalPreview(safe || 'tool', 80)
}

function summarizeApprovalInputWithoutReason(input: Record<string, unknown>): string {
  const copy = { ...input }
  delete copy.reason
  return summarizeApprovalInput(copy)
}

const MAX_APPROVAL_DEPTH = 3
const MAX_APPROVAL_ITEMS = 8

function sanitizeApprovalValue(value: unknown, depth: number, seen: Set<object>): unknown {
  if (typeof value === 'string') {
    const chars = [...value]
    return chars.length > 120 ? `${chars.slice(0, 117).join('')}...` : value
  }
  if (value === null || typeof value === 'number' || typeof value === 'boolean') return value
  if (depth >= MAX_APPROVAL_DEPTH || typeof value !== 'object') return '[omitted]'
  if (seen.has(value)) return '[circular]'
  seen.add(value)
  if (Array.isArray(value)) {
    const result = value.slice(0, MAX_APPROVAL_ITEMS).map((item) => sanitizeApprovalValue(item, depth + 1, seen))
    if (value.length > MAX_APPROVAL_ITEMS) result.push(`[+${value.length - MAX_APPROVAL_ITEMS} items]`)
    seen.delete(value)
    return result
  }
  const result: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value).slice(0, MAX_APPROVAL_ITEMS)) {
    result[key] = /(api[-_]?key|token|secret|password|authorization|credential)/i.test(key)
      ? '[redacted]'
      : sanitizeApprovalValue(item, depth + 1, seen)
  }
  if (Object.keys(value).length > MAX_APPROVAL_ITEMS) result._truncated = true
  seen.delete(value)
  return result
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const error = new Error('The approval request was interrupted')
    error.name = 'AbortError'
    throw error
  }
}

function createAbortError(): Error {
  const error = new Error('The approval request was interrupted')
  error.name = 'AbortError'
  return error
}
