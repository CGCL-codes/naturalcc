import type { RunBudgetConfig } from './types/types.js'
import type {
  ModelBudgetCapabilities,
  ModelRequestMeasurement,
  ModelUsage,
} from '../model/types/types.js'

export type ModelDispatchKind = 'agent' | 'closing' | 'compaction' | 'repair'

export interface PreparedBudgetDispatch {
  readonly kind: ModelDispatchKind
  readonly closing: boolean
  validate(): RunBudgetRejection | undefined
  publish(): BudgetDispatchReceipt | undefined
}

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

export interface BudgetDispatchReceipt {
  readonly id: number
  readonly kind: ModelDispatchKind
  readonly reservedTokens?: number
  readonly reservedInputTokens?: number
  readonly outputTokenLimit?: number
  readonly safetyMarginTokens?: number
}

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

  settleUnknown(receipt: BudgetDispatchReceipt): boolean {
    return this.settleDispatch(receipt)
  }

  /** Freeze terminal accounting; unresolved committed reservations become sticky unknown debt. */
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

  canRenewSegment(): boolean {
    if (this.closedAt !== undefined) return false
    if (this.segmentDispatches < this.segmentLimit - 1) return false
    if (this.totalDispatches >= this.absoluteMaxModelRequests - 1) return false
    if (this.elapsedExceeded() || this.overspent || this.hasAccountingOverflow()) return false
    return true
  }

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

  renewSegment(): boolean {
    if (!this.canRenewSegment()) return false
    this.segmentDispatches = 0
    this.segmentIndex += 1
    this.segmentRenewals += 1
    this.version += 1
    return true
  }

  observeToolCalls(count: number): void {
    if (this.closedAt !== undefined) return
    this.observedTools += count
  }

  admitToolBatch(count: number): boolean {
    if (this.closedAt !== undefined) return false
    const maximum = this.config.maxToolAttempts
    if (maximum !== undefined && this.admittedTools + count > maximum) return false
    this.admittedTools += count
    return true
  }

  get remainingAbsoluteRequests(): number {
    return Math.max(0, this.absoluteMaxModelRequests - this.totalDispatches)
  }

  get remainingSegmentRequests(): number {
    return Math.max(0, this.segmentLimit - this.segmentDispatches)
  }

  /** Read-only pressure probe used only to decide whether A maintenance is useful. */
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

  private elapsedExceeded(): boolean {
    return this.config.maxElapsedMs !== undefined && this.now() - this.startedAt >= this.config.maxElapsedMs
  }

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

  private latchAccountingOverflow(): void {
    if (!this.hasAccountingOverflow()) return
    this.usageIsComplete = false
    this.overspent = true
  }
}

function reservationTokens(
  measurement: ModelRequestMeasurement | undefined,
  config: RunBudgetConfig,
): number | undefined {
  if (measurement?.inputTokens === undefined) return undefined
  const output = measurement.outputLimit?.tokens
  if (output === undefined) return undefined
  return measurement.inputTokens + output + (config.safetyMarginTokens ?? 0)
}

function closingTokenReserve(
  input: { readonly closing: boolean; readonly firstAgentCompatibility?: boolean },
  config: RunBudgetConfig,
): number {
  if (input.closing || input.firstAgentCompatibility || config.maxTotalTokens === undefined) return 0
  return config.maxOutputTokensPerRequest ?? 0
}

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

function usageLowerBound(
  inputTokens: number | undefined,
  outputTokens: number | undefined,
  totalTokens: number | undefined,
): bigint {
  const knownAxes = addTokenCounts(inputTokens ?? 0, outputTokens ?? 0)
  return maxTokenCount(knownAxes, BigInt(totalTokens ?? 0))
}

function addTokenCounts(...values: Array<number | bigint | undefined>): bigint {
  let total = 0n
  for (const value of values) {
    if (value === undefined) continue
    total += typeof value === 'bigint' ? value : BigInt(value)
  }
  return total
}

const MAX_SAFE_TOKEN_COUNT = BigInt(Number.MAX_SAFE_INTEGER)

function maxTokenCount(left: bigint, right: bigint): bigint {
  return left > right ? left : right
}

function visibleTokenCount(value: bigint | undefined): number | undefined {
  if (value === undefined || value > MAX_SAFE_TOKEN_COUNT) return undefined
  return Number(value)
}

function validNonNegativeInteger(
  value: number | undefined,
  optional = false,
  allowZero = true,
): boolean {
  if (value === undefined) return optional
  return Number.isSafeInteger(value) && value >= (allowZero ? 0 : 1)
}

function knownUsageValue(value: number | undefined): number | undefined {
  return validNonNegativeInteger(value, true) ? value : undefined
}

function rejection(
  reason: RunBudgetRejection['reason'],
  message: string,
): RunBudgetRejection {
  return Object.freeze({ code: 'request_budget_exhausted', reason, message })
}
