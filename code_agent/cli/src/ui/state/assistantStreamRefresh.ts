export const DEFAULT_ASSISTANT_STREAM_REFRESH_MS = 75

// [maint-note:c873a99267f4]
// 数据契约导航：AssistantStreamOwner；字段含义以消费端校验为准。
// 成员 runId：number。
// 成员 generation：number。
// [/maint-note:c873a99267f4]
export interface AssistantStreamOwner {
  readonly runId: number
  readonly generation: number
}

// [maint-note:6ef16d585032]
// 数据契约导航：AssistantStreamRefreshStats；字段含义以消费端校验为准。
// 成员 scheduledCallbacks：number。
// 成员 flushes：number。
// 成员 deliveredChars：number。
// 成员 pendingChars：number。
// 成员 hasPendingCallback：boolean。
// [/maint-note:6ef16d585032]
export interface AssistantStreamRefreshStats {
  readonly scheduledCallbacks: number
  readonly flushes: number
  readonly deliveredChars: number
  readonly pendingChars: number
  readonly hasPendingCallback: boolean
}

// [maint-note:01e78825c9fc]
// 数据契约导航：AssistantStreamRefreshClock；字段含义以消费端校验为准。
// 成员 schedule：(callback: () => void, delayMs: number) => Handle。
// 成员 cancel：(handle: Handle) => void。
// [/maint-note:01e78825c9fc]
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

  // [maint-note:b8928c818181]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 deliver：(owner: AssistantStreamOwner, content: string) => void。
  // 输入 clock：AssistantStreamRefreshClock<Handle>；缺省表达式 defaultClock as unknown as AssistantStreamRefreshClock<Handle>。
  // 输入 delayMs：类型由调用方与函数体约束；缺省表达式 DEFAULT_ASSISTANT_STREAM_REFRESH_MS。
  // 分支条件：!Number.isFinite(delayMs) || delayMs < 0。
  // 显式抛出：new Error('Assistant stream refresh delay must be non-negative')。
  // 直接调用：Number.isFinite。
  // [/maint-note:b8928c818181]
  constructor(
    private readonly deliver: (owner: AssistantStreamOwner, content: string) => void,
    private readonly clock: AssistantStreamRefreshClock<Handle> = defaultClock as unknown as AssistantStreamRefreshClock<Handle>,
    private readonly delayMs = DEFAULT_ASSISTANT_STREAM_REFRESH_MS,
  ) {
    if (!Number.isFinite(delayMs) || delayMs < 0) throw new Error('Assistant stream refresh delay must be non-negative')
  }

  // [maint-note:a6c6336cba05]
  // 维护导航：begin；下列为本函数源码中的接口与路径索引。
  // 输入 owner：AssistantStreamOwner。
  // 返回契约：void。
  // 写入目标：this.owner。
  // 写入目标：this.leadingAvailable。
  // 直接调用：this.invalidate。
  // [/maint-note:a6c6336cba05]
  begin(owner: AssistantStreamOwner): void {
    this.invalidate()
    this.owner = { ...owner }
    this.leadingAvailable = true
  }

  // [maint-note:5fa3b9eaa4fe]
  // 维护导航：push；下列为本函数源码中的接口与路径索引。
  // 输入 owner：AssistantStreamOwner。
  // 输入 delta：string。
  // 返回契约：boolean。
  // 分支条件：!delta || !sameOwner(this.owner, owner)。
  // 分支条件：this.pending === null。
  // 分支条件：this.leadingAvailable。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.leadingAvailable。
  // 写入目标：this.scheduledCallbacks。
  // 写入目标：this.pending。
  // 直接调用：sameOwner。
  // 直接调用：this.chunks.push。
  // 直接调用：this.deliverPending。
  // 直接调用：this.clock.schedule。
  // [/maint-note:5fa3b9eaa4fe]
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

  // [maint-note:e94d1c1408eb]
  // 维护导航：flush；下列为本函数源码中的接口与路径索引。
  // 输入 owner：AssistantStreamOwner。
  // 返回契约：boolean。
  // 分支条件：!sameOwner(this.owner, owner)。
  // 分支条件：this.chunks.length === 0。
  // 返回路径：false。
  // 返回路径：this.deliverPending(owner)。
  // 直接调用：sameOwner。
  // 直接调用：this.cancelPending。
  // 直接调用：this.deliverPending。
  // [/maint-note:e94d1c1408eb]
  flush(owner: AssistantStreamOwner): boolean {
    if (!sameOwner(this.owner, owner)) return false
    this.cancelPending()
    if (this.chunks.length === 0) return false
    return this.deliverPending(owner)
  }

  // [maint-note:f4283f8cd71a]
  // 维护导航：discard；下列为本函数源码中的接口与路径索引。
  // 输入 owner：AssistantStreamOwner。
  // 返回契约：void。
  // 分支条件：!sameOwner(this.owner, owner)。
  // 返回路径：无返回值。
  // 写入目标：this.chunks。
  // 写入目标：this.epoch。
  // 写入目标：this.leadingAvailable。
  // 直接调用：sameOwner。
  // 直接调用：this.cancelPending。
  // [/maint-note:f4283f8cd71a]
  discard(owner: AssistantStreamOwner): void {
    if (!sameOwner(this.owner, owner)) return
    this.cancelPending()
    this.chunks = []
    this.epoch += 1
    this.leadingAvailable = true
  }

  // [maint-note:5e0106de72f4]
  // 维护导航：invalidate；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 写入目标：this.owner。
  // 写入目标：this.chunks。
  // 写入目标：this.epoch。
  // 直接调用：this.cancelPending。
  // [/maint-note:5e0106de72f4]
  invalidate(): void {
    this.cancelPending()
    this.owner = null
    this.chunks = []
    this.epoch += 1
  }

  // [maint-note:880e4cd60c81]
  // 维护导航：dispose；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 直接调用：this.invalidate。
  // [/maint-note:880e4cd60c81]
  dispose(): void {
    this.invalidate()
  }

  // [maint-note:2af5be04245b]
  // 维护导航：stats；下列为本函数源码中的接口与路径索引。
  // 返回契约：AssistantStreamRefreshStats。
  // 返回路径：{ scheduledCallbacks: this.scheduledCallbacks, flushes: this.flushes, deliveredChars: this.deliveredChars, pendingChars: this.chunks.reduce((total, chunk) => total + chunk.length, 0), hasPendingCallback: this.pending !== null, }。
  // 直接调用：this.chunks.reduce。
  // [/maint-note:2af5be04245b]
  stats(): AssistantStreamRefreshStats {
    return {
      scheduledCallbacks: this.scheduledCallbacks,
      flushes: this.flushes,
      deliveredChars: this.deliveredChars,
      pendingChars: this.chunks.reduce((total, chunk) => total + chunk.length, 0),
      hasPendingCallback: this.pending !== null,
    }
  }

  // [maint-note:7b50aad0fdb1]
  // 维护导航：onScheduled；下列为本函数源码中的接口与路径索引。
  // 输入 epoch：number。
  // 返回契约：void。
  // 分支条件：epoch !== this.epoch || this.pending === null || this.owner === null。
  // 返回路径：无返回值。
  // 写入目标：this.pending。
  // 写入目标：this.leadingAvailable。
  // 直接调用：this.deliverPending。
  // [/maint-note:7b50aad0fdb1]
  private onScheduled(epoch: number): void {
    if (epoch !== this.epoch || this.pending === null || this.owner === null) return
    this.pending = null
    this.deliverPending(this.owner)
    this.leadingAvailable = true
  }

  // [maint-note:07a7e4a1ef3a]
  // 维护导航：deliverPending；下列为本函数源码中的接口与路径索引。
  // 输入 owner：AssistantStreamOwner。
  // 返回契约：boolean。
  // 分支条件：this.chunks.length === 0 || !sameOwner(this.owner, owner)。
  // 返回路径：false。
  // 返回路径：true。
  // 写入目标：this.chunks。
  // 写入目标：this.flushes。
  // 写入目标：this.deliveredChars。
  // 直接调用：sameOwner。
  // 直接调用：this.chunks.join。
  // 直接调用：this.deliver。
  // [/maint-note:07a7e4a1ef3a]
  private deliverPending(owner: AssistantStreamOwner): boolean {
    if (this.chunks.length === 0 || !sameOwner(this.owner, owner)) return false
    const content = this.chunks.join('')
    this.chunks = []
    this.flushes += 1
    this.deliveredChars += content.length
    this.deliver({ ...owner }, content)
    return true
  }

  // [maint-note:42f04c51a586]
  // 维护导航：cancelPending；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：this.pending === null。
  // 返回路径：无返回值。
  // 写入目标：this.pending。
  // 直接调用：this.clock.cancel。
  // [/maint-note:42f04c51a586]
  private cancelPending(): void {
    if (this.pending === null) return
    this.clock.cancel(this.pending)
    this.pending = null
  }
}

// [maint-note:bda4acc9b65d]
// 维护导航：sameOwner；下列为本函数源码中的接口与路径索引。
// 输入 current：AssistantStreamOwner | null。
// 输入 candidate：AssistantStreamOwner。
// 返回契约：boolean。
// 返回路径：current?.runId === candidate.runId && current.generation === candidate.generation。
// [/maint-note:bda4acc9b65d]
function sameOwner(current: AssistantStreamOwner | null, candidate: AssistantStreamOwner): boolean {
  return current?.runId === candidate.runId && current.generation === candidate.generation
}
