import { AgentError, createAbortError } from '../../agent/errors.js'
import type { CodingAgentSession } from './CodingAgentSession.js'

export interface SessionFactoryContext {
  readonly generation: number
  readonly signal: AbortSignal
}

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

  constructor(private readonly createSession: CodingAgentSessionFactory) {}

  create(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

  current(): CodingAgentSession | undefined {
    return this.activeSession
  }

  replace(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

  clear(signal?: AbortSignal): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked(signal))
  }

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

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.operationQueue.then(operation, operation)
    this.operationQueue = run.then(
      () => undefined,
      () => undefined,
    )
    return run
  }

  private async replaceLocked(signal?: AbortSignal): Promise<CodingAgentSession> {
    this.assertActive()
    throwIfAborted(signal)
    const version = this.lifecycleVersion
    const generation = this.nextFactoryGeneration++
    const controller = new AbortController()
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

  private assertActive(): void {
    if (this.disposed) {
      throw new AgentError('config_error', 'The session manager has been disposed')
    }
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw createAbortError()
}

function raceWithAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(createAbortError())
  return new Promise<T>((resolve, reject) => {
    let settled = false
    const cleanup = () => signal.removeEventListener('abort', onAbort)
    const finish = (callback: () => void) => {
      if (settled) return
      settled = true
      cleanup()
      callback()
    }
    const onAbort = () => finish(() => reject(createAbortError()))
    signal.addEventListener('abort', onAbort, { once: true })
    promise.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(error)),
    )
    if (signal.aborted) onAbort()
  })
}
