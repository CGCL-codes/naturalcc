import type { RunBudgetConfig } from './types/types.js'
import type {
  ModelBudgetCapabilities,
  ModelRequestMeasurement,
  ModelUsage,
} from '../model/types/types.js'

// [maint-note:bdf33c9223e0]
// 数据契约导航：ModelDispatchKind；字段含义以消费端校验为准。
// 类型组合：'agent' | 'closing' | 'compaction' | 'repair'。
// [/maint-note:bdf33c9223e0]
export type ModelDispatchKind = 'agent' | 'closing' | 'compaction' | 'repair'

// [maint-note:99091c222158]
// 数据契约导航：PreparedBudgetDispatch；字段含义以消费端校验为准。
// 成员 kind：ModelDispatchKind。
// 成员 closing：boolean。
// 成员 validate：RunBudgetRejection | undefined。
// 成员 publish：BudgetDispatchReceipt | undefined。
// [/maint-note:99091c222158]
export interface PreparedBudgetDispatch {
  readonly kind: ModelDispatchKind
  readonly closing: boolean
  validate(): RunBudgetRejection | undefined
  publish(): BudgetDispatchReceipt | undefined
}

// [maint-note:3230cd4dba74]
// 数据契约导航：RunBudgetRejection；字段含义以消费端校验为准。
// 成员 code：'request_budget_exhausted'。
// 成员 reason：| 'absolute_requests' | 'segment_requests' | 'elapsed' | 'context_window' | 'total_tokens' | 'capability' | 'stale'。
// 成员 message：string。
// [/maint-note:3230cd4dba74]
export interface RunBudgetRejection {
  readonly code: 'request_budget_exhausted'
  readonly reason:
    | 'absolute_requests'
    | 'segment_requests'
    | 'elapsed'
    | 'context_window'
    | 'total_tokens'
    | 'capability'
    | 'stale'
  readonly message: string
}

// [maint-note:bdff128550eb]
// 数据契约导航：BudgetDispatchReceipt；字段含义以消费端校验为准。
// 成员 id：number。
// 成员 kind：ModelDispatchKind。
// 成员 reservedTokens（可省略）：number。
// 成员 reservedInputTokens（可省略）：number。
// 成员 outputTokenLimit（可省略）：number。
// 成员 safetyMarginTokens（可省略）：number。
// [/maint-note:bdff128550eb]
export interface BudgetDispatchReceipt {
  readonly id: number
  readonly kind: ModelDispatchKind
  readonly reservedTokens?: number
  readonly reservedInputTokens?: number
  readonly outputTokenLimit?: number
  readonly safetyMarginTokens?: number
}

// [maint-note:223c82811a4e]
// 数据契约导航：RunBudgetSnapshot；字段含义以消费端校验为准。
// 成员 totalModelDispatches：number。
// 成员 segmentModelDispatches：number。
// 成员 segmentIndex：number。
// 成员 segmentRenewalCount：number。
// 成员 dispatchesByKind：Readonly<Record<ModelDispatchKind, number>>。
// 成员 observedToolCalls：number。
// 成员 admittedToolAttempts：number。
// 成员 elapsedMs：number。
// 成员 reservedTokens（可省略）：number。
// 成员 knownInputTokens（可省略）：number。
// 成员 knownOutputTokens（可省略）：number。
// 成员 knownActualTokens（可省略）：number。
// 成员 unknownTokenDebt（可省略）：number。
// 成员 usageComplete：boolean。
// 成员 overspent：boolean。
// [/maint-note:223c82811a4e]
export interface RunBudgetSnapshot {
  readonly totalModelDispatches: number
  readonly segmentModelDispatches: number
  readonly segmentIndex: number
  readonly segmentRenewalCount: number
  readonly dispatchesByKind: Readonly<Record<ModelDispatchKind, number>>
  readonly observedToolCalls: number
  readonly admittedToolAttempts: number
  readonly elapsedMs: number
  readonly reservedTokens?: number
  readonly knownInputTokens?: number
  readonly knownOutputTokens?: number
  readonly knownActualTokens?: number
  readonly unknownTokenDebt?: number
  readonly usageComplete: boolean
  readonly overspent: boolean
}

const DEFAULT_SEGMENT_MODEL_REQUESTS = 32

/**
 * The single run-local admission owner. Preparation is read-only; publish is a
 * short synchronous commit and settlement is first-wins per dispatch receipt.
 */
export class RunBudgetLedger {
  private readonly startedAt: number
  private readonly segmentLimit: number
  private totalDispatches = 0
  private segmentDispatches = 0
  private segmentIndex = 0
  private segmentRenewals = 0
  private serial = 0
  private version = 0
  private observedTools = 0
  private admittedTools = 0
  private reservedTokenTotal = 0n
  private knownInputTokenTotal: bigint | undefined
  private knownOutputTokenTotal: bigint | undefined
  private knownActualTokenTotal: bigint | undefined
  /** Per-receipt conservative spend used for admission; never exposed as provider total. */
  private settledTokenFloorTotal = 0n
  private unknownTokenTotal = 0n
  private usageIsComplete = true
  private overspent = false
  private closedAt: number | undefined
  private readonly outstanding = new Map<number, BudgetDispatchReceipt>()
  private readonly byKind: Record<ModelDispatchKind, number> = {
    agent: 0,
    closing: 0,
    compaction: 0,
    repair: 0,
  }

  // [maint-note:cbb77c7ad4ca]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 absoluteMaxModelRequests：number。
  // 输入 config：RunBudgetConfig；缺省表达式 {}。
  // 输入 now：() => number；缺省表达式 Date.now。
  // 写入目标：this.segmentLimit。
  // 写入目标：this.startedAt。
  // 直接调用：Math.min。
  // 直接调用：now。
  // [/maint-note:cbb77c7ad4ca]
  constructor(
    private readonly absoluteMaxModelRequests: number,
    private readonly config: RunBudgetConfig = {},
    private readonly now: () => number = Date.now,
  ) {
    this.segmentLimit = Math.min(
      absoluteMaxModelRequests,
      config.segmentMaxModelRequests ?? DEFAULT_SEGMENT_MODEL_REQUESTS,
    )
    this.startedAt = now()
  }

  // [maint-note:a5de35cf6f17]
  // 维护导航：prepareDispatch；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ readonly kind: ModelDispatchKind readonly closing: boolean readonly measurement?: ModelRequestMeasurement readonly capabilities?: ModelBudgetCapabilities readonly firstAgentCompatibility?: boolean }。
  // 返回契约：PreparedBudgetDispatch。
  // 返回路径：Object.freeze({ kind: input.kind, closing: input.closing, validate, publish: (): BudgetDispatchReceipt | undefined => { // validate() is the only admission read. In particular, publish does // not sample the clock again after other request 。
  // 直接调用：reservationTokens。
  // 直接调用：Object.freeze。
  // [/maint-note:a5de35cf6f17]
  prepareDispatch(input: {
    readonly kind: ModelDispatchKind
    readonly closing: boolean
    readonly measurement?: ModelRequestMeasurement
    readonly capabilities?: ModelBudgetCapabilities
    readonly firstAgentCompatibility?: boolean
  }): PreparedBudgetDispatch {
    const version = this.version
    let published = false
    let commitReady = false
    let validationAttempted = false
    const reservedTokens = reservationTokens(input.measurement, this.config)
    // [maint-note:61c36c479c3e]
    // 维护导航：validate；下列为本函数源码中的接口与路径索引。
    // 返回契约：RunBudgetRejection | undefined。
    // 分支条件：published || version !== this.version。
    // 分支条件：!denied。
    // 返回路径：rejection('stale', 'The prepared budget reservation is stale')。
    // 返回路径：denied。
    // 写入目标：validationAttempted。
    // 写入目标：commitReady。
    // 直接调用：rejection。
    // 直接调用：this.checkDispatch。
    // [/maint-note:61c36c479c3e]
    const validate = (): RunBudgetRejection | undefined => {
      validationAttempted = true
      commitReady = false
      if (published || version !== this.version) return rejection('stale', 'The prepared budget reservation is stale')
      const denied = this.checkDispatch(input, reservedTokens)
      if (!denied) commitReady = true
      return denied
    }
    return Object.freeze({
      kind: input.kind,
      closing: input.closing,
      validate,
      publish: (): BudgetDispatchReceipt | undefined => {
        // validate() is the only admission read. In particular, publish does
        // not sample the clock again after other request receipts have been
        // synchronously published; this is the commit permit validated by
        // the Runner's no-await transaction.
        // Direct ledger callers keep the legacy convenience path. The Runner
        // always validates explicitly, so its publish never re-samples time.
        if (!validationAttempted && validate()) return undefined
        if (!commitReady || published || version !== this.version) return undefined
        commitReady = false
        published = true
        const receipt = Object.freeze({
          id: ++this.serial,
          kind: input.kind,
          ...(reservedTokens === undefined ? {} : { reservedTokens }),
          ...(input.measurement?.inputTokens === undefined
            ? {}
            : { reservedInputTokens: input.measurement.inputTokens }),
          ...(input.measurement?.outputLimit?.tokens === undefined
            ? {}
            : { outputTokenLimit: input.measurement.outputLimit.tokens }),
          ...(this.config.safetyMarginTokens === undefined
            ? {}
            : { safetyMarginTokens: this.config.safetyMarginTokens }),
        })
        this.totalDispatches += 1
        this.segmentDispatches += 1
        this.byKind[input.kind] += 1
        if (reservedTokens !== undefined) {
          this.reservedTokenTotal = addTokenCounts(this.reservedTokenTotal, reservedTokens)
          this.latchAccountingOverflow()
        }
        this.outstanding.set(receipt.id, receipt)
        this.version += 1
        return receipt
      },
    })
  }

  // [maint-note:c4cf748e4826]
  // 维护导航：settleDispatch；下列为本函数源码中的接口与路径索引。
  // 输入 receipt：BudgetDispatchReceipt。
  // 输入 usage：ModelUsage（可省略）。
  // 输入 options：{ readonly usageComplete?: boolean }；缺省表达式 {}。
  // 返回契约：boolean。
  // 分支条件：this.closedAt !== undefined || this.outstanding.get(receipt.id) !== receipt。
  // 分支条件：!complete。
  // 分支条件：inputTokens !== undefined。
  // 分支条件：outputTokens !== undefined。
  // 分支条件：receipt.outputTokenLimit !== undefined && outputTokens > receipt.outputTokenLimit。
  // 分支条件：receipt.reservedTokens !== undefined。
  // 分支条件：totalTokens === undefined。
  // 分支条件：receipt.reservedTokens !== undefined && totalTokens > receipt.reservedTokens。
  // 分支条件：this.config.maxTotalTokens !== undefined && addTokenCounts(this.settledTokenFloorTotal, this.reservedTokenTotal) > BigInt(this.config.maxTotalTokens)。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.usageIsComplete。
  // 写入目标：this.knownInputTokenTotal。
  // 写入目标：this.knownOutputTokenTotal。
  // 写入目标：this.overspent。
  // 写入目标：this.reservedTokenTotal。
  // 写入目标：this.unknownTokenTotal。
  // 写入目标：this.knownActualTokenTotal。
  // 写入目标：this.settledTokenFloorTotal。
  // 直接调用：this.outstanding.get。
  // 直接调用：this.outstanding.delete。
  // 直接调用：knownUsageValue。
  // 直接调用：addTokenCounts。
  // 直接调用：maxTokenCount。
  // 直接调用：BigInt。
  // 直接调用：usageLowerBound。
  // 直接调用：this.latchAccountingOverflow。
  // [/maint-note:c4cf748e4826]
  settleDispatch(
    receipt: BudgetDispatchReceipt,
    usage?: ModelUsage,
    options: { readonly usageComplete?: boolean } = {},
  ): boolean {
    if (this.closedAt !== undefined || this.outstanding.get(receipt.id) !== receipt) return false
    this.outstanding.delete(receipt.id)
    const inputTokens = knownUsageValue(usage?.inputTokens)
    const outputTokens = knownUsageValue(usage?.outputTokens)
    const totalTokens = knownUsageValue(usage?.totalTokens)
    const complete = inputTokens !== undefined &&
      outputTokens !== undefined &&
      totalTokens !== undefined &&
      options.usageComplete !== false
    if (!complete) this.usageIsComplete = false
    if (inputTokens !== undefined) {
      this.knownInputTokenTotal = addTokenCounts(this.knownInputTokenTotal, inputTokens)
    }
    if (outputTokens !== undefined) {
      this.knownOutputTokenTotal = addTokenCounts(this.knownOutputTokenTotal, outputTokens)
      if (receipt.outputTokenLimit !== undefined && outputTokens > receipt.outputTokenLimit) {
        this.overspent = true
      }
    }
    if (receipt.reservedTokens !== undefined) {
      this.reservedTokenTotal = maxTokenCount(
        0n,
        this.reservedTokenTotal - BigInt(receipt.reservedTokens),
      )
    }
    if (totalTokens === undefined) {
      if (receipt.reservedTokens !== undefined) {
        this.unknownTokenTotal = addTokenCounts(this.unknownTokenTotal, receipt.reservedTokens)
      }
    } else {
      this.knownActualTokenTotal = addTokenCounts(this.knownActualTokenTotal, totalTokens)
      if (receipt.reservedTokens !== undefined && totalTokens > receipt.reservedTokens) this.overspent = true
    }
    const knownLowerBound = usageLowerBound(inputTokens, outputTokens, totalTokens)
    const conservativeSpend = totalTokens === undefined
      ? maxTokenCount(BigInt(receipt.reservedTokens ?? 0), knownLowerBound)
      : knownLowerBound
    this.settledTokenFloorTotal = addTokenCounts(this.settledTokenFloorTotal, conservativeSpend)
    this.latchAccountingOverflow()
    if (
      this.config.maxTotalTokens !== undefined &&
      addTokenCounts(this.settledTokenFloorTotal, this.reservedTokenTotal) >
        BigInt(this.config.maxTotalTokens)
    ) {
      this.overspent = true
    }
    return true
  }

  // [maint-note:53f3cd564ba6]
  // 维护导航：settleUnknown；下列为本函数源码中的接口与路径索引。
  // 输入 receipt：BudgetDispatchReceipt。
  // 返回契约：boolean。
  // 返回路径：this.settleDispatch(receipt)。
  // 直接调用：this.settleDispatch。
  // [/maint-note:53f3cd564ba6]
  settleUnknown(receipt: BudgetDispatchReceipt): boolean {
    return this.settleDispatch(receipt)
  }

  /** Freeze terminal accounting; unresolved committed reservations become sticky unknown debt. */
  // [maint-note:2b202f7cb368]
  // 维护导航：close；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：this.closedAt !== undefined。
  // 分支条件：this.outstanding.size > 0。
  // 分支条件：receipt.reservedTokens === undefined。
  // 遍历边界：const receipt ← this.outstanding.values()。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.closedAt。
  // 写入目标：this.usageIsComplete。
  // 写入目标：this.reservedTokenTotal。
  // 写入目标：this.unknownTokenTotal。
  // 写入目标：this.settledTokenFloorTotal。
  // 写入目标：this.version。
  // 直接调用：this.now。
  // 直接调用：this.outstanding.values。
  // 直接调用：maxTokenCount。
  // 直接调用：BigInt。
  // 直接调用：addTokenCounts。
  // 直接调用：this.outstanding.clear。
  // 直接调用：this.latchAccountingOverflow。
  // [/maint-note:2b202f7cb368]
  close(): boolean {
    if (this.closedAt !== undefined) return false
    this.closedAt = this.now()
    if (this.outstanding.size > 0) this.usageIsComplete = false
    for (const receipt of this.outstanding.values()) {
      if (receipt.reservedTokens === undefined) continue
      this.reservedTokenTotal = maxTokenCount(
        0n,
        this.reservedTokenTotal - BigInt(receipt.reservedTokens),
      )
      this.unknownTokenTotal = addTokenCounts(this.unknownTokenTotal, receipt.reservedTokens)
      this.settledTokenFloorTotal = addTokenCounts(
        this.settledTokenFloorTotal,
        receipt.reservedTokens,
      )
    }
    this.outstanding.clear()
    this.latchAccountingOverflow()
    this.version += 1
    return true
  }

  // [maint-note:c6f117650dec]
  // 维护导航：canRenewSegment；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：this.closedAt !== undefined。
  // 分支条件：this.segmentDispatches < this.segmentLimit - 1。
  // 分支条件：this.totalDispatches >= this.absoluteMaxModelRequests - 1。
  // 分支条件：this.elapsedExceeded() || this.overspent || this.hasAccountingOverflow()。
  // 返回路径：false。
  // 返回路径：true。
  // 直接调用：this.elapsedExceeded。
  // 直接调用：this.hasAccountingOverflow。
  // [/maint-note:c6f117650dec]
  canRenewSegment(): boolean {
    if (this.closedAt !== undefined) return false
    if (this.segmentDispatches < this.segmentLimit - 1) return false
    if (this.totalDispatches >= this.absoluteMaxModelRequests - 1) return false
    if (this.elapsedExceeded() || this.overspent || this.hasAccountingOverflow()) return false
    return true
  }

  // [maint-note:92a34c15ddd0]
  // 维护导航：previewAfterSegmentRenewal；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ readonly kind: ModelDispatchKind readonly closing: false readonly measurement?: ModelRequestMeasurement readonly capabilities?: ModelBudgetCapabilities }。
  // 返回契约：RunBudgetRejection | undefined。
  // 分支条件：!this.canRenewSegment()。
  // 分支条件：denied。
  // 返回路径：rejection('segment_requests', 'The current model request segment cannot be renewed')。
  // 返回路径：denied。
  // 返回路径：validateRenewalMeasurement(input.measurement, input.capabilities)。
  // 直接调用：this.canRenewSegment。
  // 直接调用：rejection。
  // 直接调用：this.checkDispatch。
  // 直接调用：reservationTokens。
  // 直接调用：validateRenewalMeasurement。
  // [/maint-note:92a34c15ddd0]
  previewAfterSegmentRenewal(input: {
    readonly kind: ModelDispatchKind
    readonly closing: false
    readonly measurement?: ModelRequestMeasurement
    readonly capabilities?: ModelBudgetCapabilities
  }): RunBudgetRejection | undefined {
    if (!this.canRenewSegment()) {
      return rejection('segment_requests', 'The current model request segment cannot be renewed')
    }
    const denied = this.checkDispatch(input, reservationTokens(input.measurement, this.config), 0)
    if (denied) return denied
    return validateRenewalMeasurement(input.measurement, input.capabilities)
  }

  // [maint-note:3c444073bdb6]
  // 维护导航：renewSegment；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 分支条件：!this.canRenewSegment()。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.segmentDispatches。
  // 写入目标：this.segmentIndex。
  // 写入目标：this.segmentRenewals。
  // 写入目标：this.version。
  // 直接调用：this.canRenewSegment。
  // [/maint-note:3c444073bdb6]
  renewSegment(): boolean {
    if (!this.canRenewSegment()) return false
    this.segmentDispatches = 0
    this.segmentIndex += 1
    this.segmentRenewals += 1
    this.version += 1
    return true
  }

  // [maint-note:e1b1861dbc95]
  // 维护导航：observeToolCalls；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 返回契约：void。
  // 分支条件：this.closedAt !== undefined。
  // 返回路径：无返回值。
  // 写入目标：this.observedTools。
  // [/maint-note:e1b1861dbc95]
  observeToolCalls(count: number): void {
    if (this.closedAt !== undefined) return
    this.observedTools += count
  }

  // [maint-note:3c653532e07e]
  // 维护导航：admitToolBatch；下列为本函数源码中的接口与路径索引。
  // 输入 count：number。
  // 返回契约：boolean。
  // 分支条件：this.closedAt !== undefined。
  // 分支条件：maximum !== undefined && this.admittedTools + count > maximum。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.admittedTools。
  // [/maint-note:3c653532e07e]
  admitToolBatch(count: number): boolean {
    if (this.closedAt !== undefined) return false
    const maximum = this.config.maxToolAttempts
    if (maximum !== undefined && this.admittedTools + count > maximum) return false
    this.admittedTools += count
    return true
  }

  // [maint-note:82b9304f4928]
  // 维护导航：remainingAbsoluteRequests；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：Math.max(0, this.absoluteMaxModelRequests - this.totalDispatches)。
  // 直接调用：Math.max。
  // [/maint-note:82b9304f4928]
  get remainingAbsoluteRequests(): number {
    return Math.max(0, this.absoluteMaxModelRequests - this.totalDispatches)
  }

  // [maint-note:47d9b31f0050]
  // 维护导航：remainingSegmentRequests；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：Math.max(0, this.segmentLimit - this.segmentDispatches)。
  // 直接调用：Math.max。
  // [/maint-note:47d9b31f0050]
  get remainingSegmentRequests(): number {
    return Math.max(0, this.segmentLimit - this.segmentDispatches)
  }

  /** Read-only pressure probe used only to decide whether A maintenance is useful. */
  // [maint-note:6ffd7cc0f917]
  // 维护导航：exceedsContextWindow；下列为本函数源码中的接口与路径索引。
  // 输入 measurement：ModelRequestMeasurement | undefined。
  // 输入 capabilities：ModelBudgetCapabilities（可省略）。
  // 返回契约：boolean。
  // 分支条件：validateMeasurement(measurement, this.config, capabilities) || !measurement。
  // 分支条件：input === undefined || context === undefined。
  // 返回路径：false。
  // 返回路径：input + (measurement.outputLimit?.tokens ?? 0) + (this.config.safetyMarginTokens ?? 0) > context。
  // 直接调用：validateMeasurement。
  // [/maint-note:6ffd7cc0f917]
  exceedsContextWindow(
    measurement: ModelRequestMeasurement | undefined,
    capabilities?: ModelBudgetCapabilities,
  ): boolean {
    if (validateMeasurement(measurement, this.config, capabilities) || !measurement) return false
    const input = measurement.inputTokens
    const context = measurement.contextWindowTokens
    if (input === undefined || context === undefined) return false
    return input + (measurement.outputLimit?.tokens ?? 0) + (this.config.safetyMarginTokens ?? 0) > context
  }

  // [maint-note:83fb07c9b925]
  // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
  // 返回契约：RunBudgetSnapshot。
  // 返回路径：Object.freeze({ totalModelDispatches: this.totalDispatches, segmentModelDispatches: this.segmentDispatches, segmentIndex: this.segmentIndex, segmentRenewalCount: this.segmentRenewals, dispatchesByKind: Object.freeze({ ...this.byKind }), obs。
  // 直接调用：visibleTokenCount。
  // 直接调用：this.hasAccountingOverflow。
  // 直接调用：Object.freeze。
  // 直接调用：Math.max。
  // 直接调用：this.now。
  // [/maint-note:83fb07c9b925]
  snapshot(): RunBudgetSnapshot {
    const reservedTokens = visibleTokenCount(this.reservedTokenTotal)
    const knownInputTokens = visibleTokenCount(this.knownInputTokenTotal)
    const knownOutputTokens = visibleTokenCount(this.knownOutputTokenTotal)
    const knownActualTokens = visibleTokenCount(this.knownActualTokenTotal)
    const unknownTokenDebt = visibleTokenCount(this.unknownTokenTotal)
    const accountingOverflow = this.hasAccountingOverflow()
    return Object.freeze({
      totalModelDispatches: this.totalDispatches,
      segmentModelDispatches: this.segmentDispatches,
      segmentIndex: this.segmentIndex,
      segmentRenewalCount: this.segmentRenewals,
      dispatchesByKind: Object.freeze({ ...this.byKind }),
      observedToolCalls: this.observedTools,
      admittedToolAttempts: this.admittedTools,
      elapsedMs: Math.max(0, (this.closedAt ?? this.now()) - this.startedAt),
      ...(reservedTokens !== undefined && reservedTokens > 0 ? { reservedTokens } : {}),
      ...(knownInputTokens === undefined ? {} : { knownInputTokens }),
      ...(knownOutputTokens === undefined ? {} : { knownOutputTokens }),
      ...(knownActualTokens === undefined ? {} : { knownActualTokens }),
      ...(unknownTokenDebt !== undefined && unknownTokenDebt > 0 ? { unknownTokenDebt } : {}),
      usageComplete: this.usageIsComplete && !accountingOverflow,
      overspent: this.overspent || accountingOverflow,
    })
  }

  // [maint-note:b11833280477]
  // 维护导航：checkDispatch；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ readonly kind: ModelDispatchKind readonly closing: boolean readonly measurement?: ModelRequestMeasurement readonly capabilities?: ModelBudgetCapabilities readonly firstAgentCompatibility?: boolean }。
  // 输入 reservedTokens：number | undefined。
  // 输入 segmentDispatches：类型由调用方与函数体约束；缺省表达式 this.segmentDispatches。
  // 返回契约：RunBudgetRejection | undefined。
  // 分支条件：this.closedAt !== undefined。
  // 分支条件：this.elapsedExceeded()。
  // 分支条件：this.overspent || this.hasAccountingOverflow()。
  // 分支条件：measurementError。
  // 分支条件：this.remainingAbsoluteRequests < absoluteNeeded。
  // 分支条件：Math.max(0, this.segmentLimit - segmentDispatches) < segmentNeeded。
  // 分支条件：this.config.maxOutputTokensPerRequest !== undefined && ( measurement?.outputLimit?.enforcement !== 'provider' || measurement.outputLimit.covers !== 'completion_and_reasoning' || measurement.outputLimit.tokens !== this.config.maxOutputTokens。
  // 分支条件：this.config.maxTotalTokens !== undefined && reservedTokens === undefined。
  // 分支条件：measurement?.contextWindowTokens !== undefined && measurement.inputTokens !== undefined。
  // 分支条件：measurement.inputTokens + output + margin > measurement.contextWindowTokens。
  // 分支条件：this.config.maxTotalTokens !== undefined && reservedTokens !== undefined && addTokenCounts( this.settledTokenFloorTotal, this.reservedTokenTotal, reservedTokens, closingTokenReserve(input, this.config), ) > BigInt(this.config.maxTotalTokens。
  // 返回路径：rejection('stale', 'The run budget ledger is closed')。
  // 返回路径：rejection('elapsed', 'The run elapsed-time admission budget is exhausted')。
  // 返回路径：rejection('total_tokens', 'The run token budget was exceeded by a settled request')。
  // 返回路径：measurementError。
  // 返回路径：rejection('absolute_requests', 'The absolute model request budget cannot admit this dispatch')。
  // 返回路径：rejection('segment_requests', 'The current model request segment cannot admit this dispatch')。
  // 返回路径：rejection('capability', 'The prepared request does not enforce the configured completion-and-reasoning output cap')。
  // 返回路径：rejection('capability', 'A hard token budget requires exact input measurement and a provider output cap')。
  // 返回路径：rejection('context_window', 'The prepared request exceeds the trusted model context window')。
  // 返回路径：rejection('total_tokens', 'The prepared request exceeds the run token budget')。
  // 返回路径：undefined。
  // 直接调用：rejection。
  // 直接调用：this.elapsedExceeded。
  // 直接调用：this.hasAccountingOverflow。
  // 直接调用：validateMeasurement。
  // 直接调用：Math.max。
  // 直接调用：addTokenCounts。
  // 直接调用：closingTokenReserve。
  // 直接调用：BigInt。
  // [/maint-note:b11833280477]
  private checkDispatch(
    input: {
      readonly kind: ModelDispatchKind
      readonly closing: boolean
      readonly measurement?: ModelRequestMeasurement
      readonly capabilities?: ModelBudgetCapabilities
      readonly firstAgentCompatibility?: boolean
    },
    reservedTokens: number | undefined,
    segmentDispatches = this.segmentDispatches,
  ): RunBudgetRejection | undefined {
    if (this.closedAt !== undefined) return rejection('stale', 'The run budget ledger is closed')
    if (this.elapsedExceeded()) return rejection('elapsed', 'The run elapsed-time admission budget is exhausted')
    if (this.overspent || this.hasAccountingOverflow()) {
      return rejection('total_tokens', 'The run token budget was exceeded by a settled request')
    }

    const measurementError = validateMeasurement(input.measurement, this.config, input.capabilities)
    if (measurementError) return measurementError

    const absoluteNeeded = input.closing
      ? 1
      : input.kind === 'compaction' || input.kind === 'repair'
        ? 3
        : input.firstAgentCompatibility ? 1 : 2
    if (this.remainingAbsoluteRequests < absoluteNeeded) {
      return rejection('absolute_requests', 'The absolute model request budget cannot admit this dispatch')
    }
    const segmentNeeded = input.closing ? 1 : input.firstAgentCompatibility ? 1 : 2
    if (Math.max(0, this.segmentLimit - segmentDispatches) < segmentNeeded) {
      return rejection('segment_requests', 'The current model request segment cannot admit this dispatch')
    }

    const measurement = input.measurement
    if (
      this.config.maxOutputTokensPerRequest !== undefined &&
      (
        measurement?.outputLimit?.enforcement !== 'provider' ||
        measurement.outputLimit.covers !== 'completion_and_reasoning' ||
        measurement.outputLimit.tokens !== this.config.maxOutputTokensPerRequest
      )
    ) {
      return rejection('capability', 'The prepared request does not enforce the configured completion-and-reasoning output cap')
    }
    if (this.config.maxTotalTokens !== undefined && reservedTokens === undefined) {
      return rejection('capability', 'A hard token budget requires exact input measurement and a provider output cap')
    }
    if (measurement?.contextWindowTokens !== undefined && measurement.inputTokens !== undefined) {
      const output = measurement.outputLimit?.tokens ?? 0
      const margin = this.config.safetyMarginTokens ?? 0
      if (measurement.inputTokens + output + margin > measurement.contextWindowTokens) {
        return rejection('context_window', 'The prepared request exceeds the trusted model context window')
      }
    }
    if (
      this.config.maxTotalTokens !== undefined &&
      reservedTokens !== undefined &&
      addTokenCounts(
        this.settledTokenFloorTotal,
        this.reservedTokenTotal,
        reservedTokens,
        closingTokenReserve(input, this.config),
      ) > BigInt(this.config.maxTotalTokens)
    ) {
      return rejection('total_tokens', 'The prepared request exceeds the run token budget')
    }
    return undefined
  }

  // [maint-note:74042f2a43c4]
  // 维护导航：elapsedExceeded；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.config.maxElapsedMs !== undefined && this.now() - this.startedAt >= this.config.maxElapsedMs。
  // 直接调用：this.now。
  // [/maint-note:74042f2a43c4]
  private elapsedExceeded(): boolean {
    return this.config.maxElapsedMs !== undefined && this.now() - this.startedAt >= this.config.maxElapsedMs
  }

  // [maint-note:24ec5c576bb3]
  // 维护导航：hasAccountingOverflow；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：[ this.reservedTokenTotal, this.knownInputTokenTotal, this.knownOutputTokenTotal, this.knownActualTokenTotal, this.settledTokenFloorTotal, this.unknownTokenTotal, ].some((value) => value !== undefined && value > MAX_SAFE_TOKEN_COUNT)。
  // 直接调用：[ this.reservedTokenTotal, this.knownInputTokenTotal, this.knownOutputTokenTotal, this.knownActualTokenTotal, this.settledTokenFloorTotal, this.unknownTokenTotal, ].some。
  // [/maint-note:24ec5c576bb3]
  private hasAccountingOverflow(): boolean {
    return [
      this.reservedTokenTotal,
      this.knownInputTokenTotal,
      this.knownOutputTokenTotal,
      this.knownActualTokenTotal,
      this.settledTokenFloorTotal,
      this.unknownTokenTotal,
    ].some((value) => value !== undefined && value > MAX_SAFE_TOKEN_COUNT)
  }

  // [maint-note:18593125d60e]
  // 维护导航：latchAccountingOverflow；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：!this.hasAccountingOverflow()。
  // 返回路径：无返回值。
  // 写入目标：this.usageIsComplete。
  // 写入目标：this.overspent。
  // 直接调用：this.hasAccountingOverflow。
  // [/maint-note:18593125d60e]
  private latchAccountingOverflow(): void {
    if (!this.hasAccountingOverflow()) return
    this.usageIsComplete = false
    this.overspent = true
  }
}

// [maint-note:fb6f733407aa]
// 维护导航：reservationTokens；下列为本函数源码中的接口与路径索引。
// 输入 measurement：ModelRequestMeasurement | undefined。
// 输入 config：RunBudgetConfig。
// 返回契约：number | undefined。
// 分支条件：measurement?.inputTokens === undefined。
// 分支条件：output === undefined。
// 返回路径：undefined。
// 返回路径：measurement.inputTokens + output + (config.safetyMarginTokens ?? 0)。
// [/maint-note:fb6f733407aa]
function reservationTokens(
  measurement: ModelRequestMeasurement | undefined,
  config: RunBudgetConfig,
): number | undefined {
  if (measurement?.inputTokens === undefined) return undefined
  const output = measurement.outputLimit?.tokens
  if (output === undefined) return undefined
  return measurement.inputTokens + output + (config.safetyMarginTokens ?? 0)
}

// [maint-note:188f19089562]
// 维护导航：closingTokenReserve；下列为本函数源码中的接口与路径索引。
// 输入 input：{ readonly closing: boolean; readonly firstAgentCompatibility?: boolean }。
// 输入 config：RunBudgetConfig。
// 返回契约：number。
// 分支条件：input.closing || input.firstAgentCompatibility || config.maxTotalTokens === undefined。
// 返回路径：0。
// 返回路径：config.maxOutputTokensPerRequest ?? 0。
// [/maint-note:188f19089562]
function closingTokenReserve(
  input: { readonly closing: boolean; readonly firstAgentCompatibility?: boolean },
  config: RunBudgetConfig,
): number {
  if (input.closing || input.firstAgentCompatibility || config.maxTotalTokens === undefined) return 0
  return config.maxOutputTokensPerRequest ?? 0
}

// [maint-note:934032ade342]
// 维护导航：validateMeasurement；下列为本函数源码中的接口与路径索引。
// 输入 measurement：ModelRequestMeasurement | undefined。
// 输入 config：RunBudgetConfig。
// 输入 capabilities：ModelBudgetCapabilities（可省略）。
// 返回契约：RunBudgetRejection | undefined。
// 分支条件：!validNonNegativeInteger(config.safetyMarginTokens, true)。
// 分支条件：!measurement。
// 分支条件：typeof measurement.providerShapeId !== 'string' || !measurement.providerShapeId || !validNonNegativeInteger(measurement.serializedBodyBytes)。
// 分支条件：measurement.trust !== 'exact_tokenizer' && measurement.trust !== 'provider_estimate' && measurement.trust !== 'char_fallback' && measurement.trust !== 'unknown'。
// 分支条件：!validNonNegativeInteger(measurement.inputTokens, true)。
// 分支条件：!validNonNegativeInteger(measurement.contextWindowTokens, true, false)。
// 分支条件：measurement.outputLimit。
// 分支条件：measurement.outputLimit.enforcement !== 'provider' || (measurement.outputLimit.field !== 'max_completion_tokens' && measurement.outputLimit.field !== 'max_tokens') || (measurement.outputLimit.covers !== 'visible_output' && measurement.outpu。
// 分支条件：capabilities。
// 分支条件：measurement.trust !== capabilities.inputTokenEstimation。
// 分支条件：(capabilities.contextWindowTokens !== undefined || measurement.contextWindowTokens !== undefined) && measurement.contextWindowTokens !== capabilities.contextWindowTokens。
// 分支条件：measurement.outputLimit && !capabilities.outputLimit。
// 分支条件：measurement.outputLimit && capabilities.outputLimit && ( measurement.outputLimit.field !== capabilities.outputLimit.field || measurement.outputLimit.covers !== capabilities.outputLimit.covers )。
// 分支条件：config.maxTotalTokens !== undefined && measurement.trust !== 'exact_tokenizer'。
// 分支条件：inputTokens !== undefined && outputTokens !== undefined && !validNonNegativeInteger(inputTokens + outputTokens + margin)。
// 返回路径：rejection('capability', 'The request safety margin must be a non-negative safe integer')。
// 返回路径：undefined。
// 返回路径：rejection('capability', 'The prepared request measurement is invalid')。
// 返回路径：rejection('capability', 'The prepared input measurement trust is invalid')。
// 返回路径：rejection('capability', 'The prepared input token measurement is invalid')。
// 返回路径：rejection('capability', 'The prepared context window measurement is invalid')。
// 返回路径：rejection('capability', 'The prepared output token limit is invalid')。
// 返回路径：rejection('capability', 'The prepared input measurement trust does not match the model capability')。
// 返回路径：rejection('capability', 'The prepared context window does not match the model capability')。
// 返回路径：rejection('capability', 'The prepared output limit is not declared by the model capability')。
// 返回路径：rejection('capability', 'The prepared output limit does not match the model capability')。
// 返回路径：rejection('capability', 'A hard token budget requires an exact prepared input measurement')。
// 返回路径：rejection('capability', 'The prepared token reservation is not a safe integer')。
// 直接调用：validNonNegativeInteger。
// 直接调用：rejection。
// [/maint-note:934032ade342]
function validateMeasurement(
  measurement: ModelRequestMeasurement | undefined,
  config: RunBudgetConfig,
  capabilities?: ModelBudgetCapabilities,
): RunBudgetRejection | undefined {
  if (!validNonNegativeInteger(config.safetyMarginTokens, true)) {
    return rejection('capability', 'The request safety margin must be a non-negative safe integer')
  }
  if (!measurement) return undefined
  if (
    typeof measurement.providerShapeId !== 'string' ||
    !measurement.providerShapeId ||
    !validNonNegativeInteger(measurement.serializedBodyBytes)
  ) {
    return rejection('capability', 'The prepared request measurement is invalid')
  }
  if (
    measurement.trust !== 'exact_tokenizer' &&
    measurement.trust !== 'provider_estimate' &&
    measurement.trust !== 'char_fallback' &&
    measurement.trust !== 'unknown'
  ) {
    return rejection('capability', 'The prepared input measurement trust is invalid')
  }
  if (!validNonNegativeInteger(measurement.inputTokens, true)) {
    return rejection('capability', 'The prepared input token measurement is invalid')
  }
  if (!validNonNegativeInteger(measurement.contextWindowTokens, true, false)) {
    return rejection('capability', 'The prepared context window measurement is invalid')
  }
  if (measurement.outputLimit) {
    if (
      measurement.outputLimit.enforcement !== 'provider' ||
      (measurement.outputLimit.field !== 'max_completion_tokens' && measurement.outputLimit.field !== 'max_tokens') ||
      (measurement.outputLimit.covers !== 'visible_output' && measurement.outputLimit.covers !== 'completion_and_reasoning') ||
      !validNonNegativeInteger(measurement.outputLimit.tokens, false, false)
    ) {
      return rejection('capability', 'The prepared output token limit is invalid')
    }
  }
  if (capabilities) {
    if (measurement.trust !== capabilities.inputTokenEstimation) {
      return rejection('capability', 'The prepared input measurement trust does not match the model capability')
    }
    if (
      (capabilities.contextWindowTokens !== undefined || measurement.contextWindowTokens !== undefined) &&
      measurement.contextWindowTokens !== capabilities.contextWindowTokens
    ) {
      return rejection('capability', 'The prepared context window does not match the model capability')
    }
    if (measurement.outputLimit && !capabilities.outputLimit) {
      return rejection('capability', 'The prepared output limit is not declared by the model capability')
    }
    if (
      measurement.outputLimit &&
      capabilities.outputLimit &&
      (
        measurement.outputLimit.field !== capabilities.outputLimit.field ||
        measurement.outputLimit.covers !== capabilities.outputLimit.covers
      )
    ) {
      return rejection('capability', 'The prepared output limit does not match the model capability')
    }
  }
  if (config.maxTotalTokens !== undefined && measurement.trust !== 'exact_tokenizer') {
    return rejection('capability', 'A hard token budget requires an exact prepared input measurement')
  }
  const inputTokens = measurement.inputTokens
  const outputTokens = measurement.outputLimit?.tokens
  const margin = config.safetyMarginTokens ?? 0
  if (
    inputTokens !== undefined &&
    outputTokens !== undefined &&
    !validNonNegativeInteger(inputTokens + outputTokens + margin)
  ) {
    return rejection('capability', 'The prepared token reservation is not a safe integer')
  }
  return undefined
}

// [maint-note:a48678dbb68d]
// 维护导航：validateRenewalMeasurement；下列为本函数源码中的接口与路径索引。
// 输入 measurement：ModelRequestMeasurement | undefined。
// 输入 capabilities：ModelBudgetCapabilities | undefined。
// 返回契约：RunBudgetRejection | undefined。
// 分支条件：!measurement || !capabilities || (measurement.trust !== 'exact_tokenizer' && measurement.trust !== 'provider_estimate') || measurement.trust !== capabilities.inputTokenEstimation || measurement.inputTokens === undefined || measurement.conte。
// 返回路径：rejection( 'capability', 'Segment renewal requires a complete trusted measurement of this prepared request', )。
// 返回路径：undefined。
// 直接调用：rejection。
// [/maint-note:a48678dbb68d]
function validateRenewalMeasurement(
  measurement: ModelRequestMeasurement | undefined,
  capabilities: ModelBudgetCapabilities | undefined,
): RunBudgetRejection | undefined {
  if (
    !measurement ||
    !capabilities ||
    (measurement.trust !== 'exact_tokenizer' && measurement.trust !== 'provider_estimate') ||
    measurement.trust !== capabilities.inputTokenEstimation ||
    measurement.inputTokens === undefined ||
    measurement.contextWindowTokens === undefined ||
    measurement.contextWindowTokens !== capabilities.contextWindowTokens ||
    measurement.outputLimit?.enforcement !== 'provider' ||
    measurement.outputLimit.covers !== 'completion_and_reasoning' ||
    measurement.outputLimit.field !== capabilities.outputLimit?.field ||
    measurement.outputLimit.covers !== capabilities.outputLimit.covers
  ) {
    return rejection(
      'capability',
      'Segment renewal requires a complete trusted measurement of this prepared request',
    )
  }
  return undefined
}

// [maint-note:ced233b80bab]
// 维护导航：usageLowerBound；下列为本函数源码中的接口与路径索引。
// 输入 inputTokens：number | undefined。
// 输入 outputTokens：number | undefined。
// 输入 totalTokens：number | undefined。
// 返回契约：bigint。
// 返回路径：maxTokenCount(knownAxes, BigInt(totalTokens ?? 0))。
// 直接调用：addTokenCounts。
// 直接调用：maxTokenCount。
// 直接调用：BigInt。
// [/maint-note:ced233b80bab]
function usageLowerBound(
  inputTokens: number | undefined,
  outputTokens: number | undefined,
  totalTokens: number | undefined,
): bigint {
  const knownAxes = addTokenCounts(inputTokens ?? 0, outputTokens ?? 0)
  return maxTokenCount(knownAxes, BigInt(totalTokens ?? 0))
}

// [maint-note:bc3343933f7f]
// 维护导航：addTokenCounts；下列为本函数源码中的接口与路径索引。
// 输入 values：Array<number | bigint | undefined>。
// 返回契约：bigint。
// 分支条件：value === undefined。
// 遍历边界：const value ← values。
// 返回路径：total。
// 写入目标：total。
// 直接调用：BigInt。
// [/maint-note:bc3343933f7f]
function addTokenCounts(...values: Array<number | bigint | undefined>): bigint {
  let total = 0n
  for (const value of values) {
    if (value === undefined) continue
    total += typeof value === 'bigint' ? value : BigInt(value)
  }
  return total
}

const MAX_SAFE_TOKEN_COUNT = BigInt(Number.MAX_SAFE_INTEGER)

// [maint-note:e9e0ebf3f3e5]
// 维护导航：maxTokenCount；下列为本函数源码中的接口与路径索引。
// 输入 left：bigint。
// 输入 right：bigint。
// 返回契约：bigint。
// 返回路径：left > right ? left : right。
// [/maint-note:e9e0ebf3f3e5]
function maxTokenCount(left: bigint, right: bigint): bigint {
  return left > right ? left : right
}

// [maint-note:bcc24f50f869]
// 维护导航：visibleTokenCount；下列为本函数源码中的接口与路径索引。
// 输入 value：bigint | undefined。
// 返回契约：number | undefined。
// 分支条件：value === undefined || value > MAX_SAFE_TOKEN_COUNT。
// 返回路径：undefined。
// 返回路径：Number(value)。
// 直接调用：Number。
// [/maint-note:bcc24f50f869]
function visibleTokenCount(value: bigint | undefined): number | undefined {
  if (value === undefined || value > MAX_SAFE_TOKEN_COUNT) return undefined
  return Number(value)
}

// [maint-note:df477d503264]
// 维护导航：validNonNegativeInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 optional：类型由调用方与函数体约束；缺省表达式 false。
// 输入 allowZero：类型由调用方与函数体约束；缺省表达式 true。
// 返回契约：boolean。
// 分支条件：value === undefined。
// 返回路径：optional。
// 返回路径：Number.isSafeInteger(value) && value >= (allowZero ? 0 : 1)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:df477d503264]
function validNonNegativeInteger(
  value: number | undefined,
  optional = false,
  allowZero = true,
): boolean {
  if (value === undefined) return optional
  return Number.isSafeInteger(value) && value >= (allowZero ? 0 : 1)
}

// [maint-note:4138d2f2d9da]
// 维护导航：knownUsageValue；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 返回契约：number | undefined。
// 返回路径：validNonNegativeInteger(value, true) ? value : undefined。
// 直接调用：validNonNegativeInteger。
// [/maint-note:4138d2f2d9da]
function knownUsageValue(value: number | undefined): number | undefined {
  return validNonNegativeInteger(value, true) ? value : undefined
}

// [maint-note:73f95ae468b2]
// 维护导航：rejection；下列为本函数源码中的接口与路径索引。
// 输入 reason：RunBudgetRejection['reason']。
// 输入 message：string。
// 返回契约：RunBudgetRejection。
// 返回路径：Object.freeze({ code: 'request_budget_exhausted', reason, message })。
// 直接调用：Object.freeze。
// [/maint-note:73f95ae468b2]
function rejection(
  reason: RunBudgetRejection['reason'],
  message: string,
): RunBudgetRejection {
  return Object.freeze({ code: 'request_budget_exhausted', reason, message })
}
