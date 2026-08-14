import { AgentError } from '../../agent/errors.js'
import type { CodingAgentSession } from './CodingAgentSession.js'

export type CodingAgentSessionFactory = () => CodingAgentSession | Promise<CodingAgentSession>

export class SessionManager {
  private activeSession: CodingAgentSession | undefined
  private disposed = false
  private lifecycleVersion = 0
  private operationQueue: Promise<void> = Promise.resolve()

  constructor(private readonly createSession: CodingAgentSessionFactory) {}

  create(): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked())
  }

  current(): CodingAgentSession | undefined {
    return this.activeSession
  }

  replace(): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked())
  }

  clear(): Promise<CodingAgentSession> {
    return this.enqueue(() => this.replaceLocked())
  }

  dispose(): Promise<void> {
    if (!this.disposed) {
      this.disposed = true
      this.lifecycleVersion += 1
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

  private async replaceLocked(): Promise<CodingAgentSession> {
    this.assertActive()
    const version = this.lifecycleVersion
    const next = await this.createSession()

    if (this.disposed || version !== this.lifecycleVersion) {
      await next.dispose()
      throw new AgentError('config_error', 'The session manager is no longer active')
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
