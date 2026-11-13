import { createInterface } from 'node:readline/promises'
import { stdin, stderr } from 'node:process'
import { randomUUID } from 'node:crypto'

import { ToolInputError, type ToolSchema } from './types.js'
import { inputError } from './diagnostics.js'

// [maint-note:66eae2d5c4dc]
// 数据契约导航：ApprovalMode；字段含义以消费端校验为准。
// 类型组合：'prompt' | 'deny' | 'allow'。
// [/maint-note:66eae2d5c4dc]
export type ApprovalMode = 'prompt' | 'deny' | 'allow'

/** The model-facing reason budget is deliberately independent of terminal width. */
export const MAX_APPROVAL_REASON_CHARS = 500

const dangerousApprovalControl = /[\u0000-\u001f\u007f-\u009f]/u

// [maint-note:9e8cc985429f]
// 数据契约导航：ApprovalRequest；字段含义以消费端校验为准。
// 成员 requestId：string。
// 成员 callId：string。
// 成员 toolName：string。
// 成员 input：Record<string, unknown>。
// 成员 reason：string。
// [/maint-note:9e8cc985429f]
export interface ApprovalRequest {
  requestId: string
  callId: string
  toolName: string
  input: Record<string, unknown>
  reason: string
}

// [maint-note:1119dfdfb9df]
// 数据契约导航：ApprovalDenial；字段含义以消费端校验为准。
// 类型组合：| { readonly origin: 'policy'; readonly category: 'format_mismatch' | 'unsafe_command' | 'permission_denied'; readonly message?: string } | { readonly origin: 'user'; readonly category: 'declined'; readonly message?: string } | { readonly o。
// [/maint-note:1119dfdfb9df]
export type ApprovalDenial =
  | { readonly origin: 'policy'; readonly category: 'format_mismatch' | 'unsafe_command' | 'permission_denied'; readonly message?: string }
  | { readonly origin: 'user'; readonly category: 'declined'; readonly message?: string }
  | { readonly origin: 'unspecified'; readonly category: 'unspecified'; readonly message?: string }

// [maint-note:2b4badca7864]
// 数据契约导航：ApprovalResponse；字段含义以消费端校验为准。
// 类型组合：boolean | { readonly approved: true } | { readonly approved: false; readonly denial: ApprovalDenial }。
// [/maint-note:2b4badca7864]
export type ApprovalResponse = boolean
  | { readonly approved: true }
  | { readonly approved: false; readonly denial: ApprovalDenial }

// [maint-note:f2ebb611a123]
// 数据契约导航：NormalizedApproval；字段含义以消费端校验为准。
// 类型组合：{ readonly approved: true } | { readonly approved: false; readonly denial: { readonly origin: ApprovalDenial['origin'] | 'legacy' readonly category: ApprovalDenial['category'] readonly message: string } }。
// [/maint-note:f2ebb611a123]
export type NormalizedApproval = { readonly approved: true }
  | { readonly approved: false; readonly denial: {
      readonly origin: ApprovalDenial['origin'] | 'legacy'
      readonly category: ApprovalDenial['category']
      readonly message: string
    } }

/** Validate the trusted provider boundary before touching the attempt ledger. */
// [maint-note:0f564aaf9c9e]
// 维护导航：normalizeApprovalResponse；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：NormalizedApproval。
// 分支条件：value === true。
// 分支条件：value === false。
// 分支条件：decision.approved === true && Object.keys(decision).length === 1。
// 分支条件：decision.approved !== false || Object.keys(decision).length !== 2。
// 分支条件：typeof denial.origin !== 'string' || typeof denial.category !== 'string' || !Object.prototype.hasOwnProperty.call(allowed, denial.origin) || !allowed[denial.origin]!.includes(denial.category)。
// 分支条件：typeof message !== 'string' || !message.trim() || [...message].length > 400 || Buffer.byteLength(message, 'utf8') > 1600 || /[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u.test(message)。
// 返回路径：Object.freeze({ approved: true })。
// 返回路径：Object.freeze({ approved: false, denial: Object.freeze({ origin: 'legacy', category: 'unspecified', message: 'User denied this tool call.', }) })。
// 返回路径：Object.freeze({ approved: false, denial: Object.freeze({ origin: denial.origin as ApprovalDenial['origin'], category: denial.category as ApprovalDenial['category'], message, }) })。
// 显式抛出：new Error()。
// 显式抛出：new Error('Invalid approval provider decision')。
// 直接调用：Object.freeze。
// 直接调用：approvalData。
// 直接调用：Object.keys。
// 直接调用：Object.prototype.hasOwnProperty.call。
// 直接调用：allowed[denial.origin]!.includes。
// 直接调用：message.trim。
// 直接调用：Buffer.byteLength。
// 直接调用：/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u.test。
// [/maint-note:0f564aaf9c9e]
export function normalizeApprovalResponse(value: unknown): NormalizedApproval {
  try {
    if (value === true) return Object.freeze({ approved: true })
    if (value === false) return Object.freeze({ approved: false, denial: Object.freeze({
      origin: 'legacy', category: 'unspecified', message: 'User denied this tool call.',
    }) })
    const decision = approvalData(value, ['approved', 'denial'])
    if (decision.approved === true && Object.keys(decision).length === 1) {
      return Object.freeze({ approved: true })
    }
    if (decision.approved !== false || Object.keys(decision).length !== 2) throw new Error()
    const denial = approvalData(decision.denial, ['origin', 'category', 'message'])
    const allowed: Record<string, readonly string[]> = {
      policy: ['format_mismatch', 'unsafe_command', 'permission_denied'],
      user: ['declined'], unspecified: ['unspecified'],
    }
    if (typeof denial.origin !== 'string' || typeof denial.category !== 'string'
      || !Object.prototype.hasOwnProperty.call(allowed, denial.origin)
      || !allowed[denial.origin]!.includes(denial.category)) throw new Error()
    const message = Object.prototype.hasOwnProperty.call(denial, 'message')
      ? denial.message
      : denial.origin === 'user' ? 'User denied this tool call.' : 'Tool call rejected by the approval provider.'
    if (typeof message !== 'string' || !message.trim() || [...message].length > 400
      || Buffer.byteLength(message, 'utf8') > 1600
      || /[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u.test(message)) throw new Error()
    return Object.freeze({ approved: false, denial: Object.freeze({
      origin: denial.origin as ApprovalDenial['origin'],
      category: denial.category as ApprovalDenial['category'], message,
    }) })
  } catch {
    // Never expose an invalid return value, getter or proxy exception in history.
    throw new Error('Invalid approval provider decision')
  }
}

// [maint-note:30fb7883e9c1]
// 维护导航：approvalData；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 keys：readonly string[]。
// 返回契约：Record<string, unknown>。
// 分支条件：!value || typeof value !== 'object' || Array.isArray(value)。
// 分支条件：prototype !== Object.prototype && prototype !== null。
// 分支条件：Reflect.ownKeys(value).some((key) => typeof key !== 'string' || !keys.includes(key))。
// 分支条件：!('value' in descriptor)。
// 遍历边界：const [key, descriptor] ← Object.entries(descriptors)。
// 返回路径：data。
// 显式抛出：new Error()。
// 写入目标：data[key]。
// 直接调用：Array.isArray。
// 直接调用：Object.getPrototypeOf。
// 直接调用：Object.getOwnPropertyDescriptors。
// 直接调用：Reflect.ownKeys(value).some。
// 直接调用：Reflect.ownKeys。
// 直接调用：Object.create。
// 直接调用：Object.entries。
// [/maint-note:30fb7883e9c1]
function approvalData(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error()
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) throw new Error()
  const descriptors = Object.getOwnPropertyDescriptors(value)
  if (Reflect.ownKeys(value).some((key) => typeof key !== 'string' || !keys.includes(key))) throw new Error()
  const data: Record<string, unknown> = Object.create(null)
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!('value' in descriptor)) throw new Error()
    data[key] = descriptor.value
  }
  return data
}

// [maint-note:c1a4a7a28f6d]
// 数据契约导航：ApprovalProvider；字段含义以消费端校验为准。
// 成员 presentationOwner（可省略）：'provider' | 'renderer'。
// 成员 request：Promise<ApprovalResponse>。
// [/maint-note:c1a4a7a28f6d]
export interface ApprovalProvider {
  /** Identifies the single owner of a prompt presentation in one-shot mode. */
  readonly presentationOwner?: 'provider' | 'renderer'
  request(input: ApprovalRequest, signal?: AbortSignal): Promise<ApprovalResponse>
}

// [maint-note:a95b417ee029]
// 数据契约导航：ApprovalController；字段含义以消费端校验为准。
// 成员 decide：void。
// [/maint-note:a95b417ee029]
export interface ApprovalController {
  decide(requestId: string, approved: boolean): void
}

// [maint-note:258e0da84904]
// 数据契约导航：ToolPolicy；字段含义以消费端校验为准。
// 成员 mode：ApprovalMode。
// 成员 approvalProvider：ApprovalProvider。
// 成员 requiresApproval：boolean。
// 成员 beginTurn（可省略）：void。
// 成员 createRequestId（可省略）：string。
// [/maint-note:258e0da84904]
export interface ToolPolicy {
  readonly mode: ApprovalMode
  readonly approvalProvider: ApprovalProvider
  requiresApproval(schema: ToolSchema): boolean
  beginTurn?(): void
  createRequestId?(callId: string, ordinal: number, runOwner?: string): string
}

export class FixedApprovalProvider implements ApprovalProvider {
  // [maint-note:5b80f368a639]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 approved：boolean。
  // [/maint-note:5b80f368a639]
  constructor(private readonly approved: boolean) {}

  // [maint-note:037d77fe743c]
  // 维护导航：request；下列为本函数源码中的接口与路径索引。
  // 输入 _input：ApprovalRequest。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<boolean>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 返回路径：this.approved。
  // 直接调用：throwIfAborted。
  // [/maint-note:037d77fe743c]
  async request(_input: ApprovalRequest, signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    return this.approved
  }
}

export class PromptApprovalProvider implements ApprovalProvider {
  readonly presentationOwner = 'provider' as const

  // [maint-note:00e0c45fce4f]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 input：NodeJS.ReadableStream；缺省表达式 stdin。
  // 输入 output：NodeJS.WritableStream；缺省表达式 stderr。
  // [/maint-note:00e0c45fce4f]
  constructor(
    private readonly input: NodeJS.ReadableStream = stdin,
    private readonly output: NodeJS.WritableStream = stderr,
  ) {}

  // [maint-note:65d2b64b4a98]
  // 维护导航：request；下列为本函数源码中的接口与路径索引。
  // 输入 request：ApprovalRequest。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<boolean>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 返回路径：/^(y|yes)$/i.test(answer.trim())。
  // 直接调用：throwIfAborted。
  // 直接调用：createInterface。
  // 直接调用：formatApprovalPrompt。
  // 直接调用：lineReader.question。
  // 直接调用：/^(y|yes)$/i.test。
  // 直接调用：answer.trim。
  // 直接调用：lineReader.close。
  // [/maint-note:65d2b64b4a98]
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

  // [maint-note:d2be2e50a35d]
  // 维护导航：request；下列为本函数源码中的接口与路径索引。
  // 输入 request：ApprovalRequest。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<boolean>。
  // 返回路径：new Promise<boolean>((resolve, reject) => { if (this.pendingResolve) { reject(new Error('Another approval request is already pending')) return } this.pendingResolve = resolve this.pendingRequestId = request.requestId if (signal) { const onA。
  // 直接调用：throwIfAborted。
  // [/maint-note:d2be2e50a35d]
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
        // [maint-note:393d2b4f00f6]
        // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
        // 直接调用：this.clearPending。
        // 直接调用：reject。
        // 直接调用：createAbortError。
        // [/maint-note:393d2b4f00f6]
        const onAbort = () => {
          this.clearPending(request.requestId)
          reject(createAbortError())
        }
        signal.addEventListener('abort', onAbort, { once: true })
        this.removeAbortListener = () => signal.removeEventListener('abort', onAbort)
      }
    })
  }

  // [maint-note:d4dd1fad150c]
  // 维护导航：decide；下列为本函数源码中的接口与路径索引。
  // 输入 requestId：string。
  // 输入 approved：boolean。
  // 返回契约：void。
  // 分支条件：!this.pendingResolve || this.pendingRequestId !== requestId。
  // 返回路径：无返回值。
  // 直接调用：this.clearPending。
  // 直接调用：resolve。
  // [/maint-note:d4dd1fad150c]
  decide(requestId: string, approved: boolean): void {
    if (!this.pendingResolve || this.pendingRequestId !== requestId) return
    const resolve = this.pendingResolve
    this.clearPending(requestId)
    resolve(approved)
  }

  // [maint-note:1165445f3fad]
  // 维护导航：clearPending；下列为本函数源码中的接口与路径索引。
  // 输入 requestId：string（可省略）。
  // 返回契约：void。
  // 分支条件：requestId !== undefined && this.pendingRequestId !== requestId。
  // 返回路径：无返回值。
  // 写入目标：this.removeAbortListener。
  // 写入目标：this.pendingResolve。
  // 写入目标：this.pendingRequestId。
  // 直接调用：this.removeAbortListener。
  // [/maint-note:1165445f3fad]
  private clearPending(requestId?: string): void {
    if (requestId !== undefined && this.pendingRequestId !== requestId) return
    this.removeAbortListener?.()
    this.removeAbortListener = undefined
    this.pendingResolve = undefined
    this.pendingRequestId = undefined
  }
}

// [maint-note:99976d84831d]
// 维护导航：createApprovalProvider；下列为本函数源码中的接口与路径索引。
// 输入 mode：ApprovalMode。
// 输入 provider：ApprovalProvider（可省略）。
// 返回契约：ApprovalProvider。
// 分支条件：provider。
// 分支条件：mode === 'allow'。
// 分支条件：mode === 'deny'。
// 返回路径：provider。
// 返回路径：new FixedApprovalProvider(true)。
// 返回路径：new FixedApprovalProvider(false)。
// 返回路径：new PromptApprovalProvider()。
// [/maint-note:99976d84831d]
export function createApprovalProvider(
  mode: ApprovalMode,
  provider?: ApprovalProvider,
): ApprovalProvider {
  if (provider) return provider
  if (mode === 'allow') return new FixedApprovalProvider(true)
  if (mode === 'deny') return new FixedApprovalProvider(false)
  return new PromptApprovalProvider()
}

// [maint-note:65224d19be2e]
// 维护导航：createToolPolicy；下列为本函数源码中的接口与路径索引。
// 输入 mode：ApprovalMode。
// 输入 provider：ApprovalProvider（可省略）。
// 返回契约：ToolPolicy。
// 返回路径：{ mode, approvalProvider: createApprovalProvider(mode, provider), requiresApproval: (schema) => schema.readOnly !== true && schema.requiresApproval === true, beginTurn: () => {}, createRequestId: (callId, ordinal, runOwner) => `approval-${r。
// 直接调用：randomUUID。
// 直接调用：createApprovalProvider。
// [/maint-note:65224d19be2e]
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
    createRequestId: (callId, ordinal, runOwner) =>
      `approval-${runOwner ?? policyNonce}-${runOwner === undefined ? ++requestSequence : ordinal + 1}-${callId}`,
  }
}

// [maint-note:ac71d9a46a60]
// 维护导航：summarizeApprovalInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：string。
// 返回路径：truncateApprovalPreview(json, 240)。
// 直接调用：sanitizeApprovalValue。
// 直接调用：JSON.stringify。
// 直接调用：truncateApprovalPreview。
// [/maint-note:ac71d9a46a60]
export function summarizeApprovalInput(input: Record<string, unknown>): string {
  const safeInput = sanitizeApprovalValue(input, 0, new Set())
  const json = JSON.stringify(safeInput)
  return truncateApprovalPreview(json, 240)
}

/** Validate the model-owned reason before it can reach approval or execution. */
// [maint-note:f88dc7e4085a]
// 维护导航：validateApprovalReason；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：string。
// 分支条件：typeof value !== 'string' || !value.trim()。
// 分支条件：length > MAX_APPROVAL_REASON_CHARS。
// 分支条件：dangerousApprovalControl.test(value)。
// 返回路径：value。
// 显式抛出：inputError('Approval reason must be a non-empty string', value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', 'reason', 'non-empty string')。
// 显式抛出：inputError( `Approval reason is too long (maximum ${MAX_APPROVAL_REASON_CHARS} Unicode characters)`, 'out_of_range', 'reason', `at most ${MAX_APPROVAL_REASON_CHARS} Unicode characters`, )。
// 显式抛出：inputError('Approval reason must not contain terminal control characters', 'invalid_value', 'reason', 'text without terminal controls')。
// 直接调用：value.trim。
// 直接调用：inputError。
// 直接调用：dangerousApprovalControl.test。
// [/maint-note:f88dc7e4085a]
export function validateApprovalReason(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw inputError('Approval reason must be a non-empty string', value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', 'reason', 'non-empty string')
  }
  const length = [...value].length
  if (length > MAX_APPROVAL_REASON_CHARS) {
    throw inputError(
      `Approval reason is too long (maximum ${MAX_APPROVAL_REASON_CHARS} Unicode characters)`,
      'out_of_range', 'reason', `at most ${MAX_APPROVAL_REASON_CHARS} Unicode characters`,
    )
  }
  if (dangerousApprovalControl.test(value)) {
    throw inputError('Approval reason must not contain terminal control characters', 'invalid_value', 'reason', 'text without terminal controls')
  }
  return value
}

/** Require every model-visible mutating schema to declare the same reason contract. */
// [maint-note:df0743d5ad52]
// 维护导航：validateApprovalSchema；下列为本函数源码中的接口与路径索引。
// 输入 schema：ToolSchema。
// 返回契约：void。
// 分支条件：schema.readOnly === true || schema.requiresApproval !== true。
// 分支条件：reason?.type !== 'string' || reason.minLength !== 1 || reason.maxLength !== MAX_APPROVAL_REASON_CHARS || typeof reason.description !== 'string' || !reason.description.trim() || !required.includes('reason')。
// 返回路径：无返回值。
// 显式抛出：new ToolInputError( `Tool ${schema.name} must require reason as a non-empty string with a maximum of ${MAX_APPROVAL_REASON_CHARS} Unicode characters`, )。
// 直接调用：isRecord。
// 直接调用：Array.isArray。
// 直接调用：reason.description.trim。
// 直接调用：required.includes。
// [/maint-note:df0743d5ad52]
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

// [maint-note:c14864b50e41]
// 维护导航：validateApprovalInput；下列为本函数源码中的接口与路径索引。
// 输入 schema：ToolSchema。
// 输入 input：Record<string, unknown>。
// 返回契约：string | undefined。
// 分支条件：schema.readOnly === true || schema.requiresApproval !== true。
// 返回路径：undefined。
// 返回路径：validateApprovalReason(input.reason)。
// 直接调用：validateApprovalSchema。
// 直接调用：validateApprovalReason。
// [/maint-note:c14864b50e41]
export function validateApprovalInput(schema: ToolSchema, input: Record<string, unknown>): string | undefined {
  if (schema.readOnly === true || schema.requiresApproval !== true) return undefined
  validateApprovalSchema(schema)
  return validateApprovalReason(input.reason)
}

// [maint-note:87824b6f7a26]
// 数据契约导航：ApprovalPresentation；字段含义以消费端校验为准。
// 成员 title：string。
// 成员 instructionLabel：'Command' | 'Patch' | 'Input'。
// 成员 instruction：string。
// 成员 reason：string。
// [/maint-note:87824b6f7a26]
export interface ApprovalPresentation {
  title: string
  instructionLabel: 'Command' | 'Patch' | 'Input'
  instruction: string
  reason: string
}

/** A shared, bounded presentation contract used by Ink and readline. */
// [maint-note:5601ccf08f23]
// 维护导航：formatApprovalPrompt；下列为本函数源码中的接口与路径索引。
// 输入 request：ApprovalRequest。
// 返回契约：string。
// 返回路径：[ presentation.title, '', `${presentation.instructionLabel}:`, presentation.instruction, '', 'Reason:', presentation.reason, '', ].join('\n')。
// 直接调用：buildApprovalPresentation。
// 直接调用：[ presentation.title, '', `${presentation.instructionLabel}:`, presentation.instruction, '', 'Reason:', presentation.reason, '', ].join。
// [/maint-note:5601ccf08f23]
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

// [maint-note:10d5577e87e6]
// 维护导航：buildApprovalPresentation；下列为本函数源码中的接口与路径索引。
// 输入 request：ApprovalRequest。
// 返回契约：ApprovalPresentation。
// 返回路径：{ title: `Approval required: ${displayName}`, instructionLabel, instruction: truncateApprovalPreview(redactSensitiveText(stripApprovalControls(instruction))), reason: truncateApprovalPreview(stripApprovalControls(request.reason)), }。
// 直接调用：sanitizeApprovalDisplayName。
// 直接调用：metadata.instruction。
// 直接调用：summarizeApprovalInputWithoutReason。
// 直接调用：truncateApprovalPreview。
// 直接调用：redactSensitiveText。
// 直接调用：stripApprovalControls。
// [/maint-note:10d5577e87e6]
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

// [maint-note:0d2d6cf865bd]
// 数据契约导航：ApprovalPresentationMetadata；字段含义以消费端校验为准。
// 成员 displayName：string。
// 成员 instructionLabel：ApprovalPresentation['instructionLabel']。
// 成员 instruction：(input: Record<string, unknown>) => string。
// [/maint-note:0d2d6cf865bd]
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

// [maint-note:7a14ccc0f573]
// 维护导航：truncateApprovalPreview；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 maxChars：类型由调用方与函数体约束；缺省表达式 MAX_APPROVAL_PREVIEW_CHARS。
// 输入 maxLines：类型由调用方与函数体约束；缺省表达式 MAX_APPROVAL_PREVIEW_LINES。
// 返回契约：string。
// 分支条件：sourceChars.length <= maxChars && sourceLines.length <= maxLines。
// 分支条件：sourceLines.length === 1。
// 分支条件：nextOmittedLines === omittedLines && nextOmittedChars === omittedChars。
// 遍历边界：let attempt = 0; attempt < 20; attempt += 1。
// 返回路径：normalized。
// 返回路径：truncateSingleLine(sourceChars, maxChars)。
// 返回路径：[head, marker, tail].filter(Boolean).join('\n')。
// 写入目标：attempt。
// 写入目标：marker。
// 写入目标：head。
// 写入目标：tail。
// 写入目标：omittedLines。
// 直接调用：value.replace。
// 直接调用：normalized.split。
// 直接调用：truncateSingleLine。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：Math.min。
// 直接调用：sourceLines.slice(0, headLineCount).join。
// 直接调用：sourceLines.slice。
// 直接调用：sourceLines.slice(-tailLineCount).join。
// 直接调用：approvalOmittedMarker。
// 直接调用：trimPreviewBoundary。
// 直接调用：takePreviewHead。
// 直接调用：takePreviewTail。
// 直接调用：head.split。
// 直接调用：tail.split。
// 直接调用：[head, marker, tail].filter(Boolean).join。
// 直接调用：[head, marker, tail].filter。
// [/maint-note:7a14ccc0f573]
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

// [maint-note:598a14227bad]
// 维护导航：truncateSingleLine；下列为本函数源码中的接口与路径索引。
// 输入 sourceChars：readonly string[]。
// 输入 maxChars：number。
// 返回契约：string。
// 遍历边界：let attempt = 0; attempt < 20; attempt += 1。
// 返回路径：`${head}${marker}${tail}`。
// 写入目标：attempt。
// 写入目标：marker。
// 写入目标：head。
// 写入目标：tail。
// 直接调用：approvalOmittedMarker。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// 直接调用：sourceChars.slice(0, headBudget).join。
// 直接调用：sourceChars.slice。
// 直接调用：sourceChars.slice(-tailBudget).join。
// [/maint-note:598a14227bad]
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

// [maint-note:63e53fa4192f]
// 维护导航：approvalOmittedMarker；下列为本函数源码中的接口与路径索引。
// 输入 omittedChars：number。
// 输入 omittedLines：number。
// 返回契约：string。
// 返回路径：omittedLines > 0 ? `… [${omittedChars} chars omitted; ${omittedLines} lines omitted]` : `… [${omittedChars} chars omitted]`。
// [/maint-note:63e53fa4192f]
function approvalOmittedMarker(omittedChars: number, omittedLines: number): string {
  return omittedLines > 0
    ? `… [${omittedChars} chars omitted; ${omittedLines} lines omitted]`
    : `… [${omittedChars} chars omitted]`
}

// [maint-note:312c20a71329]
// 维护导航：takePreviewHead；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 count：number。
// 返回契约：string。
// 返回路径：[...value].slice(0, Math.max(0, count)).join('')。
// 直接调用：[...value].slice(0, Math.max(0, count)).join。
// 直接调用：[...value].slice。
// 直接调用：Math.max。
// [/maint-note:312c20a71329]
function takePreviewHead(value: string, count: number): string {
  return [...value].slice(0, Math.max(0, count)).join('')
}

// [maint-note:f553fa7bf93e]
// 维护导航：takePreviewTail；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 count：number。
// 返回契约：string。
// 返回路径：count > 0 ? [...value].slice(-count).join('') : ''。
// 直接调用：[...value].slice(-count).join。
// 直接调用：[...value].slice。
// [/maint-note:f553fa7bf93e]
function takePreviewTail(value: string, count: number): string {
  return count > 0 ? [...value].slice(-count).join('') : ''
}

// [maint-note:224269242a76]
// 维护导航：trimPreviewBoundary；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：value.replace(/^\n+|\n+$/g, '')。
// 直接调用：value.replace。
// [/maint-note:224269242a76]
function trimPreviewBoundary(value: string): string {
  return value.replace(/^\n+|\n+$/g, '')
}

// [maint-note:88b19671e98f]
// 维护导航：stripApprovalControls；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/gu, '') .replace(/\r/g, '')。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/gu, '') .replace。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '') .replace。
// 直接调用：value .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '') .replace。
// 直接调用：value .replace。
// [/maint-note:88b19671e98f]
export function stripApprovalControls(value: string): string {
  return value
    .replace(/\u001b\][\s\S]*?(?:\u0007|\u001b\\)/gu, '')
    .replace(/\u001b(?:\[[0-?]*[ -/]*[@-~]|[()][0-2A-Z])/gu, '')
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/gu, '')
    .replace(/\r/g, '')
}

// [maint-note:3403f003ab8a]
// 维护导航：redactSensitiveText；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：value .replace(/((?:--?)(?:api[-_]?key|token|password|secret|authorization)\s+)([^\s]+)/giu, '$1[redacted]') .replace(/((?:api[-_]?key|token|password|secret|authorization)\s*=\s*)([^\s]+)/giu, '$1[redacted]')。
// 直接调用：value .replace(/((?:--?)(?:api[-_]?key|token|password|secret|authorization)\s+)([^\s]+)/giu, '$1[redacted]') .replace。
// 直接调用：value .replace。
// [/maint-note:3403f003ab8a]
function redactSensitiveText(value: string): string {
  return value
    .replace(/((?:--?)(?:api[-_]?key|token|password|secret|authorization)\s+)([^\s]+)/giu, '$1[redacted]')
    .replace(/((?:api[-_]?key|token|password|secret|authorization)\s*=\s*)([^\s]+)/giu, '$1[redacted]')
}

// [maint-note:9a81e85ca692]
// 维护导航：sanitizeApprovalDisplayName；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 返回契约：string。
// 返回路径：truncateApprovalPreview(safe || 'tool', 80)。
// 直接调用：stripApprovalControls(value).trim。
// 直接调用：stripApprovalControls。
// 直接调用：truncateApprovalPreview。
// [/maint-note:9a81e85ca692]
function sanitizeApprovalDisplayName(value: string): string {
  const safe = stripApprovalControls(value).trim()
  return truncateApprovalPreview(safe || 'tool', 80)
}

// [maint-note:a642894e99f4]
// 维护导航：summarizeApprovalInputWithoutReason；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：string。
// 返回路径：summarizeApprovalInput(copy)。
// 直接调用：summarizeApprovalInput。
// [/maint-note:a642894e99f4]
function summarizeApprovalInputWithoutReason(input: Record<string, unknown>): string {
  const copy = { ...input }
  delete copy.reason
  return summarizeApprovalInput(copy)
}

const MAX_APPROVAL_DEPTH = 3
const MAX_APPROVAL_ITEMS = 8

// [maint-note:d4c1479935c5]
// 维护导航：sanitizeApprovalValue；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 depth：number。
// 输入 seen：Set<object>。
// 返回契约：unknown。
// 分支条件：typeof value === 'string'。
// 分支条件：value === null || typeof value === 'number' || typeof value === 'boolean'。
// 分支条件：depth >= MAX_APPROVAL_DEPTH || typeof value !== 'object'。
// 分支条件：seen.has(value)。
// 分支条件：Array.isArray(value)。
// 分支条件：value.length > MAX_APPROVAL_ITEMS。
// 分支条件：Object.keys(value).length > MAX_APPROVAL_ITEMS。
// 遍历边界：const [key, item] ← Object.entries(value).slice(0, MAX_APPROVAL_ITEMS)。
// 返回路径：chars.length > 120 ? `${chars.slice(0, 117).join('')}...` : value。
// 返回路径：value。
// 返回路径：'[omitted]'。
// 返回路径：'[circular]'。
// 返回路径：result。
// 写入目标：result[key]。
// 写入目标：result._truncated。
// 直接调用：chars.slice(0, 117).join。
// 直接调用：chars.slice。
// 直接调用：seen.has。
// 直接调用：seen.add。
// 直接调用：Array.isArray。
// 直接调用：value.slice(0, MAX_APPROVAL_ITEMS).map。
// 直接调用：value.slice。
// 直接调用：result.push。
// 直接调用：seen.delete。
// 直接调用：Object.entries(value).slice。
// 直接调用：Object.entries。
// 直接调用：/(api[-_]?key|token|secret|password|authorization|credential)/i.test。
// 直接调用：sanitizeApprovalValue。
// 直接调用：Object.keys。
// [/maint-note:d4c1479935c5]
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

// [maint-note:3ad11d552855]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, unknown>。
// 返回路径：Boolean(value) && typeof value === 'object' && !Array.isArray(value)。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:3ad11d552855]
function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

// [maint-note:adc58c74ea7b]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：error。
// 写入目标：error.name。
// [/maint-note:adc58c74ea7b]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    const error = new Error('The approval request was interrupted')
    error.name = 'AbortError'
    throw error
  }
}

// [maint-note:2a9f6ccb3759]
// 维护导航：createAbortError；下列为本函数源码中的接口与路径索引。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:2a9f6ccb3759]
function createAbortError(): Error {
  const error = new Error('The approval request was interrupted')
  error.name = 'AbortError'
  return error
}
