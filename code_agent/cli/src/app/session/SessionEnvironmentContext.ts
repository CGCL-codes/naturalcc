import { randomUUID } from 'node:crypto'

import type { ModelSystemMessage } from '../../model/types/types.js'
import type {
  FrameworkContextPort,
  PreparedFrameworkContext,
} from '../../agent/types/frameworkContext.js'

const MAX_TIME_ZONE_CHARS = 128
const MAX_ENVIRONMENT_MESSAGE_CHARS = 1_024

export interface EnvironmentClock {
  now(): Date
  timeZone(): string
}

export interface EnvironmentObservation {
  readonly localDate: string
  readonly timeZone: string
  readonly observedAt: string
}

export function createSystemEnvironmentClock(): EnvironmentClock {
  return {
    now: () => new Date(),
    timeZone: () => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
  }
}

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

export function localDateInTimeZone(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
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

  constructor(
    private readonly clock: EnvironmentClock,
    initial: EnvironmentObservation,
  ) {
    this.initial = cloneObservation(initial)
    this.accepted = this.initial
  }

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

    const validate = (): boolean => (
      !published && !this.disposed && this.activeReceipt === receipt
    )

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

  dispose(): void {
    this.disposed = true
    this.activeReceipt = undefined
  }
}

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

function cloneObservation(value: EnvironmentObservation): EnvironmentObservation {
  return Object.freeze({
    localDate: value.localDate,
    timeZone: value.timeZone,
    observedAt: value.observedAt,
  })
}

function sameObservation(left: EnvironmentObservation, right: EnvironmentObservation): boolean {
  return left.localDate === right.localDate && left.timeZone === right.timeZone
}

function freezeSystemMessage(message: ModelSystemMessage): ModelSystemMessage {
  for (const block of message.content) Object.freeze(block)
  Object.freeze(message.content)
  return Object.freeze(message)
}

function invalidPreparedContext(): PreparedFrameworkContext {
  return Object.freeze({
    trailingMessages: Object.freeze([]) as readonly ModelSystemMessage[],
    validate: () => false,
    publish: () => {},
    acknowledge: () => false,
  })
}
