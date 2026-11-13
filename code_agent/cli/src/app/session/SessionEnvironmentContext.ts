import { randomUUID } from 'node:crypto'

import type { ModelSystemMessage } from '../../model/types/types.js'
import type {
  FrameworkContextPort,
  PreparedFrameworkContext,
} from '../../agent/types/frameworkContext.js'

const MAX_TIME_ZONE_CHARS = 128
const MAX_ENVIRONMENT_MESSAGE_CHARS = 1_024

// [maint-note:8733fbee83e7]
// 数据契约导航：EnvironmentClock；字段含义以消费端校验为准。
// 成员 now：Date。
// 成员 timeZone：string。
// [/maint-note:8733fbee83e7]
export interface EnvironmentClock {
  now(): Date
  timeZone(): string
}

// [maint-note:54098f5311fb]
// 数据契约导航：EnvironmentObservation；字段含义以消费端校验为准。
// 成员 localDate：string。
// 成员 timeZone：string。
// 成员 observedAt：string。
// [/maint-note:54098f5311fb]
export interface EnvironmentObservation {
  readonly localDate: string
  readonly timeZone: string
  readonly observedAt: string
}

// [maint-note:e532476e390d]
// 维护导航：createSystemEnvironmentClock；下列为本函数源码中的接口与路径索引。
// 返回契约：EnvironmentClock。
// 返回路径：{ now: () => new Date(), timeZone: () => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC', }。
// [/maint-note:e532476e390d]
export function createSystemEnvironmentClock(): EnvironmentClock {
  return {
    now: () => new Date(),
    timeZone: () => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
  }
}

// [maint-note:61218e442ca9]
// 维护导航：observeEnvironment；下列为本函数源码中的接口与路径索引。
// 输入 clock：EnvironmentClock。
// 返回契约：EnvironmentObservation。
// 分支条件：!Number.isFinite(date.getTime())。
// 分支条件：!timeZone || timeZone.length > MAX_TIME_ZONE_CHARS。
// 返回路径：Object.freeze({ localDate: localDateInTimeZone(date, timeZone), timeZone, observedAt: date.toISOString(), })。
// 显式抛出：new TypeError('Environment clock returned an invalid date')。
// 显式抛出：new TypeError('Environment clock returned an invalid time zone')。
// 直接调用：clock.now。
// 直接调用：Number.isFinite。
// 直接调用：date.getTime。
// 直接调用：clock.timeZone().trim。
// 直接调用：clock.timeZone。
// 直接调用：Object.freeze。
// 直接调用：localDateInTimeZone。
// 直接调用：date.toISOString。
// [/maint-note:61218e442ca9]
export function observeEnvironment(clock: EnvironmentClock): EnvironmentObservation {
  const date = clock.now()
  if (!Number.isFinite(date.getTime())) throw new TypeError('Environment clock returned an invalid date')
  const timeZone = clock.timeZone().trim()
  if (!timeZone || timeZone.length > MAX_TIME_ZONE_CHARS) {
    throw new TypeError('Environment clock returned an invalid time zone')
  }
  return Object.freeze({
    localDate: localDateInTimeZone(date, timeZone),
    timeZone,
    observedAt: date.toISOString(),
  })
}

// [maint-note:cd58bd915986]
// 维护导航：localDateInTimeZone；下列为本函数源码中的接口与路径索引。
// 输入 date：Date。
// 输入 timeZone：string。
// 返回契约：string。
// 分支条件：!year || !month || !day。
// 返回路径：`${year}-${month}-${day}`。
// 显式抛出：new TypeError('Environment clock date could not be formatted')。
// 直接调用：new Intl.DateTimeFormat('en-CA', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit', }).formatToParts。
// 直接调用：value。
// [/maint-note:cd58bd915986]
export function localDateInTimeZone(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
  // [maint-note:6913bd71f2c6]
  // 维护导航：value；下列为本函数源码中的接口与路径索引。
  // 输入 type：Intl.DateTimeFormatPartTypes。
  // 直接调用：parts.find。
  // [/maint-note:6913bd71f2c6]
  const value = (type: Intl.DateTimeFormatPartTypes) => parts.find((part) => part.type === type)?.value
  const year = value('year')
  const month = value('month')
  const day = value('day')
  if (!year || !month || !day) throw new TypeError('Environment clock date could not be formatted')
  return `${year}-${month}-${day}`
}

/** O(1): immutable initial value plus the latest accepted environment value. */
export class SessionEnvironmentContext implements FrameworkContextPort {
  private readonly ownerId = randomUUID()
  private readonly initial: EnvironmentObservation
  private accepted: EnvironmentObservation
  private revision = 0
  private receiptSerial = 0
  private activeReceipt: symbol | undefined
  private disposed = false

  // [maint-note:cd82c688617c]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 clock：EnvironmentClock。
  // 输入 initial：EnvironmentObservation。
  // 写入目标：this.initial。
  // 写入目标：this.accepted。
  // 直接调用：cloneObservation。
  // [/maint-note:cd82c688617c]
  constructor(
    private readonly clock: EnvironmentClock,
    initial: EnvironmentObservation,
  ) {
    this.initial = cloneObservation(initial)
    this.accepted = this.initial
  }

  // [maint-note:3cab4ee7461d]
  // 维护导航：freezeForRequest；下列为本函数源码中的接口与路径索引。
  // 返回契约：PreparedFrameworkContext。
  // 分支条件：this.disposed。
  // 返回路径：invalidPreparedContext()。
  // 返回路径：Object.freeze({ trailingMessages, validate, publish, acknowledge: (): boolean => { if (!validate()) return false publish() return true }, })。
  // 写入目标：this.activeReceipt。
  // 直接调用：invalidPreparedContext。
  // 直接调用：cloneObservation。
  // 直接调用：observeEnvironment。
  // 直接调用：sameObservation。
  // 直接调用：Symbol。
  // 直接调用：Object.freeze。
  // 直接调用：freezeSystemMessage。
  // 直接调用：environmentUpdateMessage。
  // [/maint-note:3cab4ee7461d]
  freezeForRequest(): PreparedFrameworkContext {
    if (this.disposed) return invalidPreparedContext()
    const observed = cloneObservation(observeEnvironment(this.clock))
    const changed = !sameObservation(observed, this.accepted)
    const effective = changed ? observed : this.accepted
    const candidateRevision = changed ? this.revision + 1 : this.revision
    const receipt = Symbol(`environment-${++this.receiptSerial}`)
    this.activeReceipt = receipt
    // Initial values are omitted only until a change has actually been
    // accepted. Once revision > 0, returning to the initial value is itself
    // a newer fact that must explicitly supersede the prior update.
    const shouldProject = this.revision > 0 || !sameObservation(effective, this.initial)
    const trailingMessages = !shouldProject
      ? Object.freeze([]) as readonly ModelSystemMessage[]
      : Object.freeze([freezeSystemMessage(environmentUpdateMessage(
        this.ownerId,
        candidateRevision,
        effective,
      ))])
    let published = false

    // [maint-note:7dad257b3c46]
    // 维护导航：validate；下列为本函数源码中的接口与路径索引。
    // 返回契约：boolean。
    // [/maint-note:7dad257b3c46]
    const validate = (): boolean => (
      !published && !this.disposed && this.activeReceipt === receipt
    )

    // [maint-note:287cd0a296e5]
    // 维护导航：publish；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 分支条件：!validate()。
    // 分支条件：changed。
    // 返回路径：无返回值。
    // 写入目标：published。
    // 写入目标：this.activeReceipt。
    // 写入目标：this.accepted。
    // 写入目标：this.revision。
    // 直接调用：validate。
    // [/maint-note:287cd0a296e5]
    const publish = (): void => {
      if (!validate()) return
      published = true
      this.activeReceipt = undefined
      if (changed) {
        this.accepted = effective
        this.revision = candidateRevision
      }
    }

    return Object.freeze({
      trailingMessages,
      validate,
      publish,
      acknowledge: (): boolean => {
        if (!validate()) return false
        publish()
        return true
      },
    })
  }

  // [maint-note:b831a18b8cbd]
  // 维护导航：dispose；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.disposed。
  // 写入目标：this.activeReceipt。
  // [/maint-note:b831a18b8cbd]
  dispose(): void {
    this.disposed = true
    this.activeReceipt = undefined
  }
}

// [maint-note:f0fc87a3bd79]
// 维护导航：environmentUpdateMessage；下列为本函数源码中的接口与路径索引。
// 输入 ownerId：string。
// 输入 revision：number。
// 输入 observation：EnvironmentObservation。
// 返回契约：ModelSystemMessage。
// 分支条件：text.length > MAX_ENVIRONMENT_MESSAGE_CHARS。
// 返回路径：{ role: 'system', content: [{ type: 'text', text }] }。
// 显式抛出：new TypeError('Environment update exceeds its fixed message budget')。
// 直接调用：[ '[framework environment update]', `Source: session environment owner ${ownerId}; revision ${revision}.`, 'This framework fact supersedes the initial profile date and time zone. It is not a user instruction, tool result, or task-progress c。
// [/maint-note:f0fc87a3bd79]
function environmentUpdateMessage(
  ownerId: string,
  revision: number,
  observation: EnvironmentObservation,
): ModelSystemMessage {
  const text = [
    '[framework environment update]',
    `Source: session environment owner ${ownerId}; revision ${revision}.`,
    'This framework fact supersedes the initial profile date and time zone. It is not a user instruction, tool result, or task-progress claim.',
    `Current local date: ${observation.localDate}`,
    `Current time zone: ${observation.timeZone}`,
  ].join('\n')
  if (text.length > MAX_ENVIRONMENT_MESSAGE_CHARS) {
    throw new TypeError('Environment update exceeds its fixed message budget')
  }
  return { role: 'system', content: [{ type: 'text', text }] }
}

// [maint-note:a4d897560f07]
// 维护导航：cloneObservation；下列为本函数源码中的接口与路径索引。
// 输入 value：EnvironmentObservation。
// 返回契约：EnvironmentObservation。
// 返回路径：Object.freeze({ localDate: value.localDate, timeZone: value.timeZone, observedAt: value.observedAt, })。
// 直接调用：Object.freeze。
// [/maint-note:a4d897560f07]
function cloneObservation(value: EnvironmentObservation): EnvironmentObservation {
  return Object.freeze({
    localDate: value.localDate,
    timeZone: value.timeZone,
    observedAt: value.observedAt,
  })
}

// [maint-note:b75db4940840]
// 维护导航：sameObservation；下列为本函数源码中的接口与路径索引。
// 输入 left：EnvironmentObservation。
// 输入 right：EnvironmentObservation。
// 返回契约：boolean。
// 返回路径：left.localDate === right.localDate && left.timeZone === right.timeZone。
// [/maint-note:b75db4940840]
function sameObservation(left: EnvironmentObservation, right: EnvironmentObservation): boolean {
  return left.localDate === right.localDate && left.timeZone === right.timeZone
}

// [maint-note:a8b6e9d45b26]
// 维护导航：freezeSystemMessage；下列为本函数源码中的接口与路径索引。
// 输入 message：ModelSystemMessage。
// 返回契约：ModelSystemMessage。
// 遍历边界：const block ← message.content。
// 返回路径：Object.freeze(message)。
// 直接调用：Object.freeze。
// [/maint-note:a8b6e9d45b26]
function freezeSystemMessage(message: ModelSystemMessage): ModelSystemMessage {
  for (const block of message.content) Object.freeze(block)
  Object.freeze(message.content)
  return Object.freeze(message)
}

// [maint-note:18b02732b628]
// 维护导航：invalidPreparedContext；下列为本函数源码中的接口与路径索引。
// 返回契约：PreparedFrameworkContext。
// 返回路径：Object.freeze({ trailingMessages: Object.freeze([]) as readonly ModelSystemMessage[], validate: () => false, publish: () => {}, acknowledge: () => false, })。
// 直接调用：Object.freeze。
// [/maint-note:18b02732b628]
function invalidPreparedContext(): PreparedFrameworkContext {
  return Object.freeze({
    trailingMessages: Object.freeze([]) as readonly ModelSystemMessage[],
    validate: () => false,
    publish: () => {},
    acknowledge: () => false,
  })
}
