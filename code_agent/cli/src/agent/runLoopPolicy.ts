import { createHash } from 'node:crypto'

import type { ToolCall, ToolSchema } from '../tools/types.js'
import type { BatchToolResult } from './toolBatchExecutor.js'

// [maint-note:688a1bf763c1]
// 数据契约导航：RunLoopPolicy；字段含义以消费端校验为准。
// 成员 softTurnThreshold：number。
// 成员 maxNoProgressRounds：number。
// 成员 maxRecoveryPrompts：number。
// [/maint-note:688a1bf763c1]
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

// [maint-note:ab13609d9eca]
// 数据契约导航：ToolRepeatPolicy；字段含义以消费端校验为准。
// 成员 beforeAttempt：{ code: 'repeated_tool_call'; message: string } | undefined。
// 成员 observe：void。
// [/maint-note:ab13609d9eca]
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

// [maint-note:13e6e9e86b58]
// 数据契约导航：RunRoundSummary；字段含义以消费端校验为准。
// 成员 progress：boolean。
// 成员 noProgressRounds：number。
// [/maint-note:13e6e9e86b58]
export interface RunRoundSummary {
  progress: boolean
  noProgressRounds: number
}

// [maint-note:1049257cdb6e]
// 数据契约导航：EligibleProgressReceipt；字段含义以消费端校验为准。
// 成员 serial：number。
// [/maint-note:1049257cdb6e]
export interface EligibleProgressReceipt {
  readonly serial: number
}

// [maint-note:27330ff06e52]
// 维护导航：createRunLoopPolicy；下列为本函数源码中的接口与路径索引。
// 输入 maxTurns：number。
// 输入 overrides：Partial<RunLoopPolicy>；缺省表达式 {}。
// 返回契约：RunLoopPolicy。
// 返回路径：{ softTurnThreshold: Math.min(softTurnThreshold, Math.max(1, maxTurns - 1)), maxNoProgressRounds, maxRecoveryPrompts, }。
// 直接调用：bounded。
// 直接调用：Math.min。
// 直接调用：Math.max。
// [/maint-note:27330ff06e52]
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

  // [maint-note:c456ad81bc09]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 policy：RunLoopPolicy。
  // [/maint-note:c456ad81bc09]
  constructor(private readonly policy: RunLoopPolicy) {}

  // [maint-note:6318a8752e9c]
  // 维护导航：beginRound；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.roundProgress。
  // 写入目标：this.currentBatch。
  // 直接调用：this.completedAttempts.clear。
  // [/maint-note:6318a8752e9c]
  beginRound(): void {
    this.roundProgress = false
    this.completedAttempts.clear()
    this.currentBatch = []
  }

  // [maint-note:4525b4c15c1a]
  // 维护导航：finishRound；下列为本函数源码中的接口与路径索引。
  // 返回契约：RunRoundSummary。
  // 分支条件：this.roundProgress。
  // 分支条件：this.recentRounds.length > 8。
  // 分支条件：this.recentBatches.length > 8。
  // 返回路径：{ progress: this.roundProgress, noProgressRounds: this.noProgressRounds }。
  // 写入目标：this.noProgressRounds。
  // 直接调用：this.recentRounds.push。
  // 直接调用：this.recentRounds.shift。
  // 直接调用：this.recentBatches.push。
  // 直接调用：this.currentBatch.slice。
  // 直接调用：this.recentBatches.shift。
  // [/maint-note:4525b4c15c1a]
  finishRound(): RunRoundSummary {
    if (this.roundProgress) this.noProgressRounds = 0
    else this.noProgressRounds += 1
    this.recentRounds.push(this.roundProgress)
    if (this.recentRounds.length > 8) this.recentRounds.shift()
    this.recentBatches.push(this.currentBatch.slice(0, 128))
    if (this.recentBatches.length > 8) this.recentBatches.shift()
    return { progress: this.roundProgress, noProgressRounds: this.noProgressRounds }
  }

  // [maint-note:1461811a8839]
  // 维护导航：recoverySummary；下列为本函数源码中的接口与路径索引。
  // 返回契约：{ noProgressRounds: number; recoveryPrompts: number; workspaceRevision: number; errorCodes: string[]; batchCount: number }。
  // 返回路径：{ noProgressRounds: this.noProgressRounds, recoveryPrompts: this.recoveryPrompts, workspaceRevision: this.workspaceRevision, errorCodes: [...this.errorCodes].slice(-8), batchCount: this.recentBatches.length, }。
  // 直接调用：[...this.errorCodes].slice。
  // [/maint-note:1461811a8839]
  recoverySummary(): { noProgressRounds: number; recoveryPrompts: number; workspaceRevision: number; errorCodes: string[]; batchCount: number } {
    return {
      noProgressRounds: this.noProgressRounds,
      recoveryPrompts: this.recoveryPrompts,
      workspaceRevision: this.workspaceRevision,
      errorCodes: [...this.errorCodes].slice(-8),
      batchCount: this.recentBatches.length,
    }
  }

  // [maint-note:93eab91276f2]
  // 维护导航：shouldRequestRecovery；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts < this.policy.maxRecoveryPrompts。
  // [/maint-note:93eab91276f2]
  shouldRequestRecovery(): boolean {
    return this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts < this.policy.maxRecoveryPrompts
  }

  // [maint-note:b76c64011187]
  // 维护导航：consumeRecoveryPrompt；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：!this.shouldRequestRecovery()。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.recoveryPrompts。
  // 写入目标：this.noProgressRounds。
  // 直接调用：this.shouldRequestRecovery。
  // [/maint-note:b76c64011187]
  consumeRecoveryPrompt(): boolean {
    if (!this.shouldRequestRecovery()) return false
    this.recoveryPrompts += 1
    this.noProgressRounds = 0
    return true
  }

  // [maint-note:176a7068580e]
  // 维护导航：shouldPause；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts >= this.policy.maxRecoveryPrompts。
  // [/maint-note:176a7068580e]
  shouldPause(): boolean {
    return this.noProgressRounds >= this.policy.maxNoProgressRounds && this.recoveryPrompts >= this.policy.maxRecoveryPrompts
  }

  /** Run budget receipt; deliberately independent of repeat-failure roundProgress. */
  // [maint-note:b1954f9e6275]
  // 维护导航：eligibleProgressReceipt；下列为本函数源码中的接口与路径索引。
  // 返回契约：EligibleProgressReceipt。
  // 返回路径：Object.freeze({ serial: this.eligibleSerial })。
  // 直接调用：Object.freeze。
  // [/maint-note:b1954f9e6275]
  eligibleProgressReceipt(): EligibleProgressReceipt {
    return Object.freeze({ serial: this.eligibleSerial })
  }

  // [maint-note:ea983267e067]
  // 维护导航：beforeAttempt；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ call: ToolCall schema?: ToolSchema validationCode?: string }。
  // 返回契约：{ code: 'repeated_tool_call'; message: string } | undefined。
  // 分支条件：this.completedAttempts.has(base)。
  // 分支条件：input.schema?.repeatKind === 'workspaceMutation' && this.mutationSuccessRevisions.get(base) === this.workspaceRevision。
  // 分支条件：this.deterministicBases.has(base)。
  // 分支条件：input.schema?.repeatKind === 'workspaceMutation' || this.deterministicAttempts.has(key)。
  // 分支条件：this.observations.has(key)。
  // 返回路径：repeatedMessage()。
  // 返回路径：undefined。
  // 直接调用：this.baseAttemptKey。
  // 直接调用：this.completedAttempts.has。
  // 直接调用：repeatedMessage。
  // 直接调用：this.attemptKey。
  // 直接调用：this.mutationSuccessRevisions.get。
  // 直接调用：this.deterministicBases.has。
  // 直接调用：this.deterministicAttempts.has。
  // 直接调用：this.observations.has。
  // [/maint-note:ea983267e067]
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

  // [maint-note:28ad838b1d23]
  // 维护导航：observe；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ call: ToolCall schema?: ToolSchema result: BatchToolResult resultCode?: string }。
  // 返回契约：void。
  // 分支条件：this.observations.size > 128。
  // 分支条件：input.resultCode === 'repeated_tool_call'。
  // 分支条件：input.schema?.repeatKind === 'read' && !input.result.isError。
  // 分支条件：eligiblePrevious !== resultFingerprint。
  // 分支条件：this.eligibleReadObservations.size > 128。
  // 分支条件：input.schema?.repeatKind === 'workspaceMutation' && !input.result.isError。
  // 分支条件：this.mutationSuccessRevisions.size > 128。
  // 分支条件：input.resultCode === 'tool_not_found' || input.resultCode === 'tool_validation_error' || input.resultCode === 'tool_denied'。
  // 分支条件：this.deterministicBases.size > 128。
  // 分支条件：input.schema?.repeatKind === 'workspaceMutation' && input.result.isError。
  // 分支条件：this.deterministicAttempts.size > 128。
  // 分支条件：!previous。
  // 分支条件：input.schema?.repeatKind === 'read' || !input.result.isError。
  // 分支条件：isOpaque(input.schema) && isExecutedResult(input.resultCode)。
  // 分支条件：previous !== resultFingerprint。
  // 返回路径：无返回值。
  // 写入目标：this.eligibleSerial。
  // 写入目标：this.workspaceRevision。
  // 写入目标：this.roundProgress。
  // 直接调用：this.baseAttemptKey。
  // 直接调用：this.completedAttempts.add。
  // 直接调用：this.currentBatch.push。
  // 直接调用：this.attemptKey。
  // 直接调用：fingerprintResult。
  // 直接调用：this.observations.get。
  // 直接调用：this.observations.set。
  // 直接调用：this.observations.delete。
  // 直接调用：this.observations.keys().next。
  // 直接调用：this.observations.keys。
  // 直接调用：this.noteError。
  // 直接调用：this.eligibleReadObservations.get。
  // 直接调用：this.eligibleReadObservations.set。
  // 直接调用：this.eligibleReadObservations.delete。
  // 直接调用：this.eligibleReadObservations.keys().next。
  // 直接调用：this.eligibleReadObservations.keys。
  // 直接调用：this.mutationSuccessRevisions.set。
  // 直接调用：this.mutationSuccessRevisions.delete。
  // 直接调用：this.mutationSuccessRevisions.keys().next。
  // 直接调用：this.mutationSuccessRevisions.keys。
  // 直接调用：this.deterministicBases.add。
  // 直接调用：this.deterministicBases.delete。
  // 直接调用：this.deterministicBases.values().next。
  // 直接调用：this.deterministicBases.values。
  // 直接调用：this.deterministicAttempts.add。
  // 直接调用：this.deterministicAttempts.delete。
  // 直接调用：this.deterministicAttempts.values().next。
  // 直接调用：this.deterministicAttempts.values。
  // 直接调用：isOpaque。
  // 直接调用：isExecutedResult。
  // [/maint-note:28ad838b1d23]
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

  // [maint-note:3a5889b98a32]
  // 维护导航：baseAttemptKey；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 返回契约：string。
  // 返回路径：fingerprint({ name: call.name, arguments: canonicalArguments(call.arguments) })。
  // 直接调用：fingerprint。
  // 直接调用：canonicalArguments。
  // [/maint-note:3a5889b98a32]
  private baseAttemptKey(call: ToolCall): string {
    return fingerprint({ name: call.name, arguments: canonicalArguments(call.arguments) })
  }

  // [maint-note:394da275daa5]
  // 维护导航：attemptKey；下列为本函数源码中的接口与路径索引。
  // 输入 call：ToolCall。
  // 输入 schema：ToolSchema（可省略）。
  // 返回契约：string。
  // 返回路径：`${isOpaque(schema) ? 'opaque' : this.workspaceRevision}:${this.baseAttemptKey(call)}`。
  // 直接调用：isOpaque。
  // 直接调用：this.baseAttemptKey。
  // [/maint-note:394da275daa5]
  private attemptKey(call: ToolCall, schema?: ToolSchema): string {
    return `${isOpaque(schema) ? 'opaque' : this.workspaceRevision}:${this.baseAttemptKey(call)}`
  }

  // [maint-note:71fb14c5d9b3]
  // 维护导航：noteError；下列为本函数源码中的接口与路径索引。
  // 输入 code：string。
  // 返回契约：void。
  // 分支条件：!code。
  // 分支条件：this.errorCodes.size > 8。
  // 返回路径：无返回值。
  // 直接调用：this.errorCodes.add。
  // 直接调用：this.errorCodes.delete。
  // 直接调用：this.errorCodes.values().next。
  // 直接调用：this.errorCodes.values。
  // [/maint-note:71fb14c5d9b3]
  private noteError(code: string): void {
    if (!code) return
    this.errorCodes.add(code)
    if (this.errorCodes.size > 8) this.errorCodes.delete(this.errorCodes.values().next().value as string)
  }
}

// [maint-note:91047a167899]
// 维护导航：repeatedMessage；下列为本函数源码中的接口与路径索引。
// 返回契约：{ code: 'repeated_tool_call'; message: string }。
// 返回路径：{ code: 'repeated_tool_call', message: 'The same tool call was already attempted in this run; choose a different action or provide a final response.', }。
// [/maint-note:91047a167899]
function repeatedMessage(): { code: 'repeated_tool_call'; message: string } {
  return {
    code: 'repeated_tool_call',
    message: 'The same tool call was already attempted in this run; choose a different action or provide a final response.',
  }
}

// [maint-note:fe9b9d7ee767]
// 维护导航：bounded；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 fallback：number。
// 输入 minimum：number。
// 返回契约：number。
// 返回路径：value !== undefined && Number.isSafeInteger(value) && value >= minimum && value <= 256 ? value : fallback。
// 直接调用：Number.isSafeInteger。
// [/maint-note:fe9b9d7ee767]
function bounded(value: number | undefined, fallback: number, minimum: number): number {
  return value !== undefined && Number.isSafeInteger(value) && value >= minimum && value <= 256 ? value : fallback
}

// [maint-note:f01095265252]
// 维护导航：canonicalArguments；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 depth：类型由调用方与函数体约束；缺省表达式 0。
// 返回契约：unknown。
// 分支条件：Array.isArray(value)。
// 分支条件：value && typeof value === 'object'。
// 返回路径：value.map((item) => canonicalArguments(item, depth + 1))。
// 返回路径：Object.fromEntries(Object.entries(value as Record<string, unknown>) .filter(([key]) => depth !== 0 || key !== 'reason') .sort(([left], [right]) => left.localeCompare(right)) .map(([key, item]) => [key, canonicalArguments(item, depth + 1)]))。
// 返回路径：value。
// 直接调用：Array.isArray。
// 直接调用：value.map。
// 直接调用：Object.fromEntries。
// 直接调用：Object.entries(value as Record<string, unknown>) .filter(([key]) => depth !== 0 || key !== 'reason') .sort(([left], [right]) => left.localeCompare(right)) .map。
// 直接调用：Object.entries(value as Record<string, unknown>) .filter(([key]) => depth !== 0 || key !== 'reason') .sort。
// 直接调用：Object.entries(value as Record<string, unknown>) .filter。
// 直接调用：Object.entries。
// [/maint-note:f01095265252]
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

// [maint-note:0e8a1d1bd03f]
// 维护导航：isOpaque；下列为本函数源码中的接口与路径索引。
// 输入 schema：ToolSchema（可省略）。
// 返回契约：boolean。
// 返回路径：schema?.repeatKind === 'opaque' || (schema !== undefined && schema.repeatKind === undefined && schema.readOnly !== true)。
// [/maint-note:0e8a1d1bd03f]
function isOpaque(schema?: ToolSchema): boolean {
  return schema?.repeatKind === 'opaque' || (schema !== undefined && schema.repeatKind === undefined && schema.readOnly !== true)
}

// [maint-note:0ea2d8be18fa]
// 维护导航：isExecutedResult；下列为本函数源码中的接口与路径索引。
// 输入 resultCode：string（可省略）。
// 返回契约：boolean。
// 返回路径：resultCode === undefined || resultCode === 'interrupted' || resultCode === 'tool_execution_error'。
// [/maint-note:0ea2d8be18fa]
function isExecutedResult(resultCode?: string): boolean {
  return resultCode === undefined || resultCode === 'interrupted' || resultCode === 'tool_execution_error'
}

// [maint-note:649a22621bb2]
// 维护导航：fingerprint；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：string。
// 返回路径：createHash('sha256').update(JSON.stringify(value)).digest('hex')。
// 直接调用：createHash('sha256').update(JSON.stringify(value)).digest。
// 直接调用：createHash('sha256').update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:649a22621bb2]
function fingerprint(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex')
}

// [maint-note:9dbf2ce6800e]
// 维护导航：fingerprintResult；下列为本函数源码中的接口与路径索引。
// 输入 isError：boolean。
// 输入 resultCode：string | undefined。
// 输入 content：string。
// 返回契约：string。
// 返回路径：createHash('sha256') .update(JSON.stringify({ isError, resultCode, contentLength: content.length })) .update('\u0000') .update(content) .digest('hex')。
// 直接调用：createHash('sha256') .update(JSON.stringify({ isError, resultCode, contentLength: content.length })) .update('\u0000') .update(content) .digest。
// 直接调用：createHash('sha256') .update(JSON.stringify({ isError, resultCode, contentLength: content.length })) .update('\u0000') .update。
// 直接调用：createHash('sha256') .update(JSON.stringify({ isError, resultCode, contentLength: content.length })) .update。
// 直接调用：createHash('sha256') .update。
// 直接调用：createHash。
// 直接调用：JSON.stringify。
// [/maint-note:9dbf2ce6800e]
function fingerprintResult(isError: boolean, resultCode: string | undefined, content: string): string {
  return createHash('sha256')
    .update(JSON.stringify({ isError, resultCode, contentLength: content.length }))
    .update('\u0000')
    .update(content)
    .digest('hex')
}
