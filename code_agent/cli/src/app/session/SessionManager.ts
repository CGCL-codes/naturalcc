import { AgentError, createAbortError } from '../../agent/errors.js'
import type { CodingAgentSession } from './CodingAgentSession.js'

// [maint-note:0c6fe79a588d]
// 数据契约导航：SessionFactoryContext；字段含义以消费端校验为准。
// 成员 generation：number。
// 成员 signal：AbortSignal。
// [/maint-note:0c6fe79a588d]
export interface SessionFactoryContext {
  readonly generation: number
  readonly signal: AbortSignal
}

// [maint-note:066012043c80]
// 数据契约导航：CodingAgentSessionFactory；字段含义以消费端校验为准。
// 类型组合：( context: SessionFactoryContext, ) => CodingAgentSession | Promise<CodingAgentSession>。
// [/maint-note:066012043c80]
export type CodingAgentSessionFactory = (
  context: SessionFactoryContext,
) => CodingAgentSession | Promise<CodingAgentSession>

export class SessionManager {
  private activeSession: CodingAgentSession | undefined
  private disposed = false
  private lifecycleVersion = 0
  private nextFactoryGeneration = 1
  private operationQueue: Promise<void> = Promise.resolve()
  private activeCandidate: AbortController | undefined

  // [maint-note:00010c1e79d1]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 createSession：CodingAgentSessionFactory。
  // [/maint-note:00010c1e79d1]
  constructor(private readonly createSession: CodingAgentSessionFactory) {}

  // [maint-note:849fa25f7f65]
  // 维护导航：create；下列为本函数源码中的接口与路径索引。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<CodingAgentSession>。
  // 返回路径：this.enqueue(() => this.replaceLocked(signal))。
  // 直接调用：this.enqueue。
  // [/maint-note:849fa25f7f65]
  create(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

  // [maint-note:08db3b4fd07a]
  // 维护导航：current；下列为本函数源码中的接口与路径索引。
  // 返回契约：CodingAgentSession | undefined。
  // 返回路径：this.activeSession。
  // [/maint-note:08db3b4fd07a]
  current(): CodingAgentSession | undefined {
    return this.activeSession
  }

  // [maint-note:c7b98ff4abb6]
  // 维护导航：replace；下列为本函数源码中的接口与路径索引。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<CodingAgentSession>。
  // 返回路径：this.enqueue(() => this.replaceLocked(signal))。
  // 直接调用：this.enqueue。
  // [/maint-note:c7b98ff4abb6]
  replace(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

  // [maint-note:3c35632938d5]
  // 维护导航：clear；下列为本函数源码中的接口与路径索引。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<CodingAgentSession>。
  // 返回路径：this.enqueue(() => this.replaceLocked(signal))。
  // 直接调用：this.enqueue。
  // [/maint-note:3c35632938d5]
  clear(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

  // [maint-note:40544e38296f]
  // 维护导航：dispose；下列为本函数源码中的接口与路径索引。
  // 返回契约：Promise<void>。
  // 分支条件：!this.disposed。
  // 返回路径：this.enqueue(async () => { await session?.dispose() })。
  // 写入目标：this.disposed。
  // 写入目标：this.lifecycleVersion。
  // 写入目标：this.activeSession。
  // 直接调用：this.activeCandidate?.abort。
  // 直接调用：this.enqueue。
  // [/maint-note:40544e38296f]
  dispose(): Promise<void> {
    if (!this.disposed) {
      this.disposed = true
      this.lifecycleVersion += 1
      this.activeCandidate?.abort()
    }
    const session = this.activeSession
    this.activeSession = undefined

    return this.enqueue(async () => {
      await session?.dispose()
    })
  }

  // [maint-note:9a5004da8fa3]
  // 维护导航：enqueue；下列为本函数源码中的接口与路径索引。
  // 输入 operation：() => Promise<T>。
  // 返回契约：Promise<T>。
  // 返回路径：run。
  // 写入目标：this.operationQueue。
  // 直接调用：this.operationQueue.then。
  // 直接调用：run.then。
  // [/maint-note:9a5004da8fa3]
  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.operationQueue.then(operation, operation)
    this.operationQueue = run.then(
      () => undefined,
      () => undefined,
    )
    return run
  }

  // [maint-note:5446be64f794]
  // 维护导航：replaceLocked；下列为本函数源码中的接口与路径索引。
  // 输入 signal：AbortSignal（可省略）。
  // 返回契约：Promise<CodingAgentSession>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：this.disposed || version !== this.lifecycleVersion。
  // 分支条件：this.activeCandidate === controller。
  // 分支条件：signal?.aborted。
  // 返回路径：next。
  // 显式抛出：new AgentError('config_error', 'The session manager is no longer active')。
  // 显式抛出：error。
  // 显式抛出：createAbortError()。
  // 写入目标：this.activeCandidate。
  // 写入目标：next。
  // 写入目标：this.activeSession。
  // 直接调用：this.assertActive。
  // 直接调用：throwIfAborted。
  // 直接调用：signal?.addEventListener。
  // 直接调用：Promise.resolve().then。
  // 直接调用：Promise.resolve。
  // 直接调用：raceWithAbort。
  // 直接调用：controller.abort。
  // 直接调用：candidatePromise.then( (late) => late.dispose(), () => undefined, ).catch。
  // 直接调用：candidatePromise.then。
  // 直接调用：signal?.removeEventListener。
  // 直接调用：next.dispose。
  // 直接调用：createAbortError。
  // 直接调用：previous?.dispose。
  // [/maint-note:5446be64f794]
  private async replaceLocked(signal?: AbortSignal): Promise<CodingAgentSession> {
    this.assertActive()
    throwIfAborted(signal)
    const version = this.lifecycleVersion
    const generation = this.nextFactoryGeneration++
    const controller = new AbortController()
    // [maint-note:cfffead7ce40]
    // 维护导航：abortCandidate；下列为本函数源码中的接口与路径索引。
    // 直接调用：controller.abort。
    // [/maint-note:cfffead7ce40]
    const abortCandidate = () => controller.abort()
    signal?.addEventListener('abort', abortCandidate, { once: true })
    this.activeCandidate = controller
    const candidatePromise = Promise.resolve().then(() => this.createSession({
      generation,
      signal: controller.signal,
    }))
    let next: CodingAgentSession
    try {
      next = await raceWithAbort(candidatePromise, controller.signal)
    } catch (error) {
      // The candidate owns this signal even when its factory fails for a
      // non-cancellation reason. End sibling provider/listener resources
      // before dropping the only controller reference, while preserving the
      // factory's original error for the caller.
      controller.abort()
      // A non-cooperative factory must not hold the queue. Its eventual
      // success is disposed and its eventual rejection is observed.
      void candidatePromise.then(
        (late) => late.dispose(),
        () => undefined,
      ).catch(() => undefined)
      if (this.disposed || version !== this.lifecycleVersion) {
        throw new AgentError('config_error', 'The session manager is no longer active')
      }
      throw error
    } finally {
      signal?.removeEventListener('abort', abortCandidate)
      if (this.activeCandidate === controller) this.activeCandidate = undefined
    }

    if (this.disposed || version !== this.lifecycleVersion) {
      await next.dispose()
      throw new AgentError('config_error', 'The session manager is no longer active')
    }

    // Final cancellation boundary. Once teardown starts the old Session may
    // already have cancelled a run and is not resurrected.
    if (signal?.aborted) {
      await next.dispose()
      throw createAbortError()
    }

    const previous = this.activeSession
    try {
      await previous?.dispose()
    } catch (error) {
      await next.dispose()
      throw error
    }

    if (this.disposed || version !== this.lifecycleVersion) {
      await next.dispose()
      throw new AgentError('config_error', 'The session manager is no longer active')
    }

    this.activeSession = next
    return next
  }

  // [maint-note:e6e93158e8c2]
  // 维护导航：assertActive；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：this.disposed。
  // 显式抛出：new AgentError('config_error', 'The session manager has been disposed')。
  // [/maint-note:e6e93158e8c2]
  private assertActive(): void {
    if (this.disposed) {
      throw new AgentError('config_error', 'The session manager has been disposed')
    }
  }
}

// [maint-note:64b35f40e477]
// 维护导航：throwIfAborted；下列为本函数源码中的接口与路径索引。
// 输入 signal：AbortSignal（可省略）。
// 返回契约：void。
// 分支条件：signal?.aborted。
// 显式抛出：createAbortError()。
// 直接调用：createAbortError。
// [/maint-note:64b35f40e477]
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

// [maint-note:cf76cab8fb1e]
// 维护导航：raceWithAbort；下列为本函数源码中的接口与路径索引。
// 输入 promise：Promise<T>。
// 输入 signal：AbortSignal。
// 返回契约：Promise<T>。
// 分支条件：signal.aborted。
// 返回路径：Promise.reject(createAbortError())。
// 返回路径：new Promise<T>((resolve, reject) => { let settled = false const cleanup = () => signal.removeEventListener('abort', onAbort) const finish = (callback: () => void) => { if (settled) return settled = true cleanup() callback() } const onAbort 。
// 直接调用：Promise.reject。
// 直接调用：createAbortError。
// [/maint-note:cf76cab8fb1e]
function raceWithAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(createAbortError())
  return new Promise<T>((resolve, reject) => {
    let settled = false
    // [maint-note:9269e518f658]
    // 维护导航：cleanup；下列为本函数源码中的接口与路径索引。
    // 直接调用：signal.removeEventListener。
    // [/maint-note:9269e518f658]
    const cleanup = () => signal.removeEventListener('abort', onAbort)
    // [maint-note:0cce4822456b]
    // 维护导航：finish；下列为本函数源码中的接口与路径索引。
    // 输入 callback：() => void。
    // 分支条件：settled。
    // 返回路径：无返回值。
    // 写入目标：settled。
    // 直接调用：cleanup。
    // 直接调用：callback。
    // [/maint-note:0cce4822456b]
    const finish = (callback: () => void) => {
      if (settled) return
      settled = true
      cleanup()
      callback()
    }
    // [maint-note:700ed2c94bd2]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 直接调用：finish。
    // [/maint-note:700ed2c94bd2]
    const onAbort = () => finish(() => reject(createAbortError()))
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(error)),
    )
    if (signal.aborted) onAbort()
  })
}
