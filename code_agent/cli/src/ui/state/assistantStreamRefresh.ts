export const DEFAULT_ASSISTANT_STREAM_REFRESH_MS = 75

export interface AssistantStreamOwner {
  readonly runId: number
  readonly generation: number
}

export interface AssistantStreamRefreshStats {
  readonly scheduledCallbacks: number
  readonly flushes: number
  readonly deliveredChars: number
  readonly pendingChars: number
  readonly hasPendingCallback: boolean
}

export interface AssistantStreamRefreshClock<Handle = ReturnType<typeof setTimeout>> {
  readonly schedule: (callback: () => void, delayMs: number) => Handle
  readonly cancel: (handle: Handle) => void
}

const defaultClock: AssistantStreamRefreshClock = {
  schedule: (callback, delayMs) => setTimeout(callback, delayMs),
  cancel: handle => clearTimeout(handle),
}

/**
 * UI-only delta refresh owner. It never delays AgentEvent consumption or
 * mutates canonical model/thread state: it only batches the deltas delivered
 * to the React message projection. One owner has at most one pending callback.
 * Every callback carries an epoch so even an uncancellable late callback is
 * unable to revive a cleared, replaced, terminal, or unmounted draft.
 */
export class AssistantStreamRefreshController<Handle = ReturnType<typeof setTimeout>> {
  private owner: AssistantStreamOwner | null = null
  private chunks: string[] = []
  private pending: Handle | null = null
  private epoch = 0
  private leadingAvailable = true
  private scheduledCallbacks = 0
  private flushes = 0
  private deliveredChars = 0

  constructor(
    private readonly deliver: (owner: AssistantStreamOwner, content: string) => void,
    private readonly clock: AssistantStreamRefreshClock<Handle> = defaultClock as unknown as AssistantStreamRefreshClock<Handle>,
    private readonly delayMs = DEFAULT_ASSISTANT_STREAM_REFRESH_MS,
  ) {
    if (!Number.isFinite(delayMs) || delayMs < 0) throw new Error('Assistant stream refresh delay must be non-negative')
  }

  begin(owner: AssistantStreamOwner): void {
    this.invalidate()
    this.owner = { ...owner }
    this.leadingAvailable = true
  }

  push(owner: AssistantStreamOwner, delta: string): boolean {
    if (!delta || !sameOwner(this.owner, owner)) return false
    this.chunks.push(delta)
    if (this.pending === null) {
      if (this.leadingAvailable) {
        this.leadingAvailable = false
        this.deliverPending(owner)
      }
      const epoch = this.epoch
      this.scheduledCallbacks += 1
      this.pending = this.clock.schedule(() => this.onScheduled(epoch), this.delayMs)
    }
    return true
  }

  flush(owner: AssistantStreamOwner): boolean {
    if (!sameOwner(this.owner, owner)) return false
    this.cancelPending()
    if (this.chunks.length === 0) return false
    return this.deliverPending(owner)
  }

  discard(owner: AssistantStreamOwner): void {
    if (!sameOwner(this.owner, owner)) return
    this.cancelPending()
    this.chunks = []
    this.epoch += 1
    this.leadingAvailable = true
  }

  invalidate(): void {
    this.cancelPending()
    this.owner = null
    this.chunks = []
    this.epoch += 1
  }

  dispose(): void {
    this.invalidate()
  }

  stats(): AssistantStreamRefreshStats {
    return {
      scheduledCallbacks: this.scheduledCallbacks,
      flushes: this.flushes,
      deliveredChars: this.deliveredChars,
      pendingChars: this.chunks.reduce((total, chunk) => total + chunk.length, 0),
      hasPendingCallback: this.pending !== null,
    }
  }

  private onScheduled(epoch: number): void {
    if (epoch !== this.epoch || this.pending === null || this.owner === null) return
    this.pending = null
    this.deliverPending(this.owner)
    this.leadingAvailable = true
  }

  private deliverPending(owner: AssistantStreamOwner): boolean {
    if (this.chunks.length === 0 || !sameOwner(this.owner, owner)) return false
    const content = this.chunks.join('')
    this.chunks = []
    this.flushes += 1
    this.deliveredChars += content.length
    this.deliver({ ...owner }, content)
    return true
  }

  private cancelPending(): void {
    if (this.pending === null) return
    this.clock.cancel(this.pending)
    this.pending = null
  }
}

function sameOwner(current: AssistantStreamOwner | null, candidate: AssistantStreamOwner): boolean {
  return current?.runId === candidate.runId && current.generation === candidate.generation
}
