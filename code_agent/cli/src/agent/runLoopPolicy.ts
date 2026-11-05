import { createHash } from 'node:crypto'

import type { ToolCall, ToolSchema } from '../tools/types.js'
import type { BatchToolResult } from './toolBatchExecutor.js'

export interface RunLoopPolicy {
  softTurnThreshold: number
  maxNoProgressRounds: number
  maxRecoveryPrompts: number
}

export const DEFAULT_RUN_LOOP_POLICY: RunLoopPolicy = Object.freeze({
  softTurnThreshold: 12,
  maxNoProgressRounds: 3,
  maxRecoveryPrompts: 2,
})

export interface ToolRepeatPolicy {
  beforeAttempt(input: {
    call: ToolCall
    schema?: ToolSchema
    validationCode?: string
  }): { code: 'repeated_tool_call'; message: string } | undefined
  observe(input: {
    call: ToolCall
    schema?: ToolSchema
    result: BatchToolResult
    resultCode?: string
  }): void
}

export interface RunRoundSummary {
  progress: boolean
  noProgressRounds: number
}

export interface EligibleProgressReceipt {
  readonly serial: number
}

export function createRunLoopPolicy(
  maxTurns: number,
  overrides: Partial<RunLoopPolicy> = {},
): RunLoopPolicy {
  const softTurnThreshold = bounded(overrides.softTurnThreshold, DEFAULT_RUN_LOOP_POLICY.softTurnThreshold, 0)
  const maxNoProgressRounds = bounded(overrides.maxNoProgressRounds, DEFAULT_RUN_LOOP_POLICY.maxNoProgressRounds, 1)
  const maxRecoveryPrompts = bounded(overrides.maxRecoveryPrompts, DEFAULT_RUN_LOOP_POLICY.maxRecoveryPrompts, 0)
  return {
    softTurnThreshold: Math.min(softTurnThreshold, Math.max(1, maxTurns - 1)),
    maxNoProgressRounds,
    maxRecoveryPrompts,
  }
}

/**
 * Run-local loop guard. It deliberately owns no thread/session state and is
 * bounded so a pathological model cannot turn fingerprints into a memory leak.
 */
export class RunLoopGuard implements ToolRepeatPolicy {
  private readonly observations = new Map<string, string>()
  private readonly errorCodes = new Set<string>()
  private readonly completedAttempts = new Set<string>()
  private readonly deterministicBases = new Set<string>()
  private readonly deterministicAttempts = new Set<string>()
  private readonly mutationSuccessRevisions = new Map<string, number>()
  /** B-only evidence identity; opaque revision changes never invalidate it. */
  private readonly eligibleReadObservations = new Map<string, string>()
  private readonly recentBatches: string[][] = []
  private currentBatch: string[] = []
  private readonly recentRounds: boolean[] = []
  private workspaceRevision = 0
  private roundProgress = false
  private noProgressRounds = 0
  private recoveryPrompts = 0
  private eligibleSerial = 0

  constructor(private readonly policy: RunLoopPolicy) {}

  beginRound(): void {
    this.roundProgress = false
    this.completedAttempts.clear()
    this.currentBatch = []
  }

  finishRound(): RunRoundSummary {
    if (this.roundProgress) this.noProgressRounds = 0
    else this.noProgressRounds += 1
    this.recentRounds.push(this.roundProgress)
    if (this.recentRounds.length > 8) this.recentRounds.shift()
    this.recentBatches.push(this.currentBatch.slice(0, 128))
    if (this.recentBatches.length > 8) this.recentBatches.shift()
    return { progress: this.roundProgress, noProgressRounds: this.noProgressRounds }
  }

  recoverySummary(): { noProgressRounds: number; recoveryPrompts: number; workspaceRevision: number; errorCodes: string[]; batchCount: number } {
    return {
      noProgressRounds: this.noProgressRounds,
      recoveryPrompts: this.recoveryPrompts,
      workspaceRevision: this.workspaceRevision,
      errorCodes: [...this.errorCodes].slice(-8),
      batchCount: this.recentBatches.length,
    }
  }

  shouldRequestRecovery(): boolean {
    return this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts < this.policy.maxRecoveryPrompts
  }

  consumeRecoveryPrompt(): boolean {
    if (!this.shouldRequestRecovery()) return false
    this.recoveryPrompts += 1
    this.noProgressRounds = 0
    return true
  }

  shouldPause(): boolean {
    return this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts >= this.policy.maxRecoveryPrompts
  }

  /** Run budget receipt; deliberately independent of repeat-failure roundProgress. */
  eligibleProgressReceipt(): EligibleProgressReceipt {
    return Object.freeze({ serial: this.eligibleSerial })
  }

  beforeAttempt(input: {
    call: ToolCall
    schema?: ToolSchema
    validationCode?: string
  }): { code: 'repeated_tool_call'; message: string } | undefined {
    const base = this.baseAttemptKey(input.call)
    if (this.completedAttempts.has(base)) return repeatedMessage()

    const key = this.attemptKey(input.call, input.schema)
    if (input.schema?.repeatKind === 'workspaceMutation' && this.mutationSuccessRevisions.get(base) === this.workspaceRevision) {
      return repeatedMessage()
    }
    // Validation/unknown/denied outcomes are deterministic regardless of a
    // later workspace revision or a changed approval reason.
    if (this.deterministicBases.has(base)) return repeatedMessage()
    if (
      input.schema?.repeatKind === 'workspaceMutation' ||
      this.deterministicAttempts.has(key)
    ) {
      if (this.observations.has(key)) return repeatedMessage()
    }
    return undefined
  }

  observe(input: {
    call: ToolCall
    schema?: ToolSchema
    result: BatchToolResult
    resultCode?: string
  }): void {
    const base = this.baseAttemptKey(input.call)
    this.completedAttempts.add(base)
    this.currentBatch.push(base)
    const key = this.attemptKey(input.call, input.schema)
    // ToolResult.content is already the bounded canonical payload exposed to
    // the model. Hash all of it: truncating the prefix would make two
    // materially different results look identical to the no-progress guard.
    const resultFingerprint = fingerprintResult(
      input.result.isError,
      input.resultCode,
      input.result.content,
    )
    const previous = this.observations.get(key)
    this.observations.set(key, resultFingerprint)
    if (this.observations.size > 128) this.observations.delete(this.observations.keys().next().value as string)

    this.noteError(input.resultCode ?? (input.result.isError ? 'tool_execution_error' : ''))
    if (input.resultCode === 'repeated_tool_call') return

    if (input.schema?.repeatKind === 'read' && !input.result.isError) {
      const eligiblePrevious = this.eligibleReadObservations.get(base)
      if (eligiblePrevious !== resultFingerprint) this.eligibleSerial += 1
      this.eligibleReadObservations.set(base, resultFingerprint)
      if (this.eligibleReadObservations.size > 128) {
        this.eligibleReadObservations.delete(this.eligibleReadObservations.keys().next().value as string)
      }
    }

    if (input.schema?.repeatKind === 'workspaceMutation' && !input.result.isError) {
      this.workspaceRevision += 1
      this.mutationSuccessRevisions.set(base, this.workspaceRevision)
      if (this.mutationSuccessRevisions.size > 128) this.mutationSuccessRevisions.delete(this.mutationSuccessRevisions.keys().next().value as string)
      this.roundProgress = true
      this.eligibleSerial += 1
      return
    }
    if (
      input.resultCode === 'tool_not_found' ||
      input.resultCode === 'tool_validation_error' ||
      input.resultCode === 'tool_denied'
    ) {
      this.deterministicBases.add(base)
      if (this.deterministicBases.size > 128) this.deterministicBases.delete(this.deterministicBases.values().next().value as string)
    }
    if (input.schema?.repeatKind === 'workspaceMutation' && input.result.isError) {
      this.deterministicAttempts.add(key)
      if (this.deterministicAttempts.size > 128) this.deterministicAttempts.delete(this.deterministicAttempts.values().next().value as string)
    }
    if (!previous) {
      if (input.schema?.repeatKind === 'read' || !input.result.isError) this.roundProgress = true
      if (isOpaque(input.schema) && isExecutedResult(input.resultCode)) this.workspaceRevision += 1
      return
    }
    if (previous !== resultFingerprint) {
      this.roundProgress = true
    }
    else this.noteError(input.resultCode ?? (input.result.isError ? 'tool_execution_error' : ''))

    if (isOpaque(input.schema) && isExecutedResult(input.resultCode)) {
      this.workspaceRevision += 1
    }
  }

  private baseAttemptKey(call: ToolCall): string {
    return fingerprint({ name: call.name, arguments: canonicalArguments(call.arguments) })
  }

  private attemptKey(call: ToolCall, schema?: ToolSchema): string {
    return `${isOpaque(schema) ? 'opaque' : this.workspaceRevision}:${this.baseAttemptKey(call)}`
  }

  private noteError(code: string): void {
    if (!code) return
    this.errorCodes.add(code)
    if (this.errorCodes.size > 8) this.errorCodes.delete(this.errorCodes.values().next().value as string)
  }
}

function repeatedMessage(): { code: 'repeated_tool_call'; message: string } {
  return {
    code: 'repeated_tool_call',
    message: 'The same tool call was already attempted in this run; choose a different action or provide a final response.',
  }
}

function bounded(value: number | undefined, fallback: number, minimum: number): number {
  return value !== undefined && Number.isSafeInteger(value) && value >= minimum && value <= 256 ? value : fallback
}

function canonicalArguments(value: unknown, depth = 0): unknown {
  if (Array.isArray(value)) return value.map((item) => canonicalArguments(item, depth + 1))
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([key]) => depth !== 0 || key !== 'reason')
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalArguments(item, depth + 1)]))
  }
  return value
}

function isOpaque(schema?: ToolSchema): boolean {
  return schema?.repeatKind === 'opaque' || (schema !== undefined && schema.repeatKind === undefined && schema.readOnly !== true)
}

function isExecutedResult(resultCode?: string): boolean {
  return resultCode === undefined || resultCode === 'interrupted' || resultCode === 'tool_execution_error'
}

function fingerprint(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex')
}

function fingerprintResult(isError: boolean, resultCode: string | undefined, content: string): string {
  return createHash('sha256')
    .update(JSON.stringify({ isError, resultCode, contentLength: content.length }))
    .update('\u0000')
    .update(content)
    .digest('hex')
}
