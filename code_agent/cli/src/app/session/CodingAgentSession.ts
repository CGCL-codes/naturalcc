import type { AgentRunner } from '../../agent/AgentRunner.js'
import type { AgentEvent } from '../../agent/types/event.js'
import type { AgentRuntime } from '../../agent/types/types.js'
import type { AgentThread } from '../../agent/thread/AgentThread.js'
import type { CodingAgentProfile } from '../profile.js'

export interface CodingAgentSessionOptions {
  profile: CodingAgentProfile
  runtime: AgentRuntime
  thread: AgentThread
  runner: AgentRunner
}

export class CodingAgentSession {
  readonly profile: CodingAgentProfile
  readonly runtime: AgentRuntime
  readonly thread: AgentThread
  readonly runner: AgentRunner
  private disposed = false
  private generation = 0
  private readonly activeTurns = new Set<ActiveTurn>()
  private disposePromise: Promise<void> | undefined

  constructor(options: CodingAgentSessionOptions) {
    this.profile = options.profile
    this.runtime = options.runtime
    this.thread = options.thread
    this.runner = options.runner
  }

  get id(): string {
    return this.thread.id
  }

  get isDisposed(): boolean {
    return this.disposed
  }

  snapshot() {
    return this.thread.snapshot()
  }

  async *runTurn(input: {
    prompt: string
    signal?: AbortSignal
  }): AsyncGenerator<AgentEvent> {
    if (this.disposed) {
      yield {
        type: 'error',
        code: 'config_error',
        message: 'The coding agent session has been disposed',
        recoverable: false,
        workspaceMayHaveChanged: false,
      }
      return
    }

    const generation = this.generation
    const controller = new AbortController()
    const activeTurn = createActiveTurn(controller)
    this.activeTurns.add(activeTurn)

    const abortFromCaller = () => controller.abort()
    if (input.signal) {
      if (input.signal.aborted) controller.abort()
      else input.signal.addEventListener('abort', abortFromCaller, { once: true })
    }
    let callerListenerCleaned = false
    activeTurn.cleanup = () => {
      if (callerListenerCleaned) return
      callerListenerCleaned = true
      input.signal?.removeEventListener('abort', abortFromCaller)
    }

    const runnerTurn = this.runner.runTurn({
      thread: this.thread,
      prompt: input.prompt,
      runtime: this.runtime,
      signal: controller.signal,
    })

    const closeRunner = (): Promise<void> => {
      if (!activeTurn.closePromise) {
        activeTurn.closePromise = Promise.resolve()
          .then(() => runnerTurn.return(undefined))
          .then(
            () => undefined,
            () => undefined,
          )
          .then(() => {
            activeTurn.resolveRunnerClosed()
            activeTurn.resolveClosed()
          })
      }
      return activeTurn.closePromise
    }

    const drainRunner = (): Promise<void> => {
      if (!activeTurn.drainPromise) {
        activeTurn.drainPromise = (async () => {
          try {
            while (true) {
              const result = await runnerTurn.next()
              if (result.done) break
              if (
                isTerminalEvent(result.value) &&
                !activeTurn.drainTerminal
              ) {
                activeTurn.drainTerminal = result.value
              }
            }
          } catch {
            // AgentRunner normally converts failures to AgentEvents. The
            // fallback below still makes a detached stream terminate safely
            // if a future runner violates that boundary.
          } finally {
            if (!activeTurn.drainTerminal && !activeTurn.terminalDelivered) {
              activeTurn.drainTerminal = fallbackTerminal(
                activeTurn.lastYielded,
                this.runtime,
              )
            }
            activeTurn.cleanup?.()
            this.activeTurns.delete(activeTurn)
            activeTurn.resolveRunnerClosed()
            activeTurn.resolveClosed()
          }
        })()
      }
      return activeTurn.drainPromise
    }

    activeTurn.close = () => activeTurn.drainPromise
      ? activeTurn.drainPromise
      : closeRunner()
    activeTurn.drain = drainRunner

    try {
      while (true) {
        if (activeTurn.detached) {
          await activeTurn.drain?.()
          if (
            activeTurn.drainTerminal &&
            !activeTurn.terminalDelivered
          ) {
            activeTurn.terminalDelivered = true
            activeTurn.pausedForConsumer = true
            try {
              yield activeTurn.drainTerminal
            } finally {
              activeTurn.pausedForConsumer = false
            }
          }
          return
        }

        activeTurn.awaitingRunner = true
        const result = await runnerTurn.next()
        activeTurn.awaitingRunner = false
        if (result.done) {
          activeTurn.resolveRunnerClosed()
          if (
            !activeTurn.terminalDelivered &&
            (this.disposed || generation !== this.generation)
          ) {
            activeTurn.terminalDelivered = true
            activeTurn.pausedForConsumer = true
            try {
              yield fallbackTerminal(activeTurn.lastYielded, this.runtime)
            } finally {
              activeTurn.pausedForConsumer = false
            }
          }
          break
        }

        const event = result.value
        if (this.disposed || generation !== this.generation) {
          activeTurn.lastYielded = event

          if (event.type === 'assistant_message') {
            activeTurn.detached = true
            await drainRunner()
            activeTurn.pausedForConsumer = true
            try {
              yield event
            } finally {
              activeTurn.pausedForConsumer = false
            }
            continue
          }

          if (isTerminalEvent(event)) {
            activeTurn.terminalDelivered = true
            activeTurn.detached = true
            await drainRunner()
            activeTurn.pausedForConsumer = true
            try {
              yield event
            } finally {
              activeTurn.pausedForConsumer = false
            }
            return
          }

          continue
        }

        activeTurn.lastYielded = event
        if (isTerminalEvent(event)) activeTurn.terminalDelivered = true
        activeTurn.pausedForConsumer = true
        try {
          yield event
        } finally {
          activeTurn.pausedForConsumer = false
        }
      }
    } finally {
      await activeTurn.close?.()
      activeTurn.cleanup?.()
      this.activeTurns.delete(activeTurn)
      activeTurn.resolveClosed()
    }
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise

    this.disposed = true
    this.generation += 1
    const pendingTurns: Promise<void>[] = []

    for (const activeTurn of this.activeTurns) {
      activeTurn.controller.abort()
      activeTurn.cleanup?.()

      if (activeTurn.pausedForConsumer) {
        activeTurn.detached = true
        pendingTurns.push(activeTurn.drain?.() ?? activeTurn.close?.() ?? activeTurn.closed)
      } else {
        pendingTurns.push(
          activeTurn.awaitingRunner
            ? activeTurn.runnerClosed
            : activeTurn.close?.() ?? activeTurn.runnerClosed,
        )
      }
    }

    this.disposePromise = Promise.all(pendingTurns).then(() => undefined)
    return this.disposePromise
  }
}

interface ActiveTurn {
  controller: AbortController
  awaitingRunner: boolean
  pausedForConsumer: boolean
  detached: boolean
  lastYielded?: AgentEvent
  drainTerminal?: AgentEvent
  terminalDelivered: boolean
  runnerClosed: Promise<void>
  resolveRunnerClosed: () => void
  closed: Promise<void>
  resolveClosed: () => void
  close?: () => Promise<void>
  closePromise?: Promise<void>
  drain?: () => Promise<void>
  drainPromise?: Promise<void>
  cleanup?: () => void
}

function createActiveTurn(controller: AbortController): ActiveTurn {
  let resolveRunnerClosed!: () => void
  let resolveClosed!: () => void
  const runnerClosed = new Promise<void>((resolve) => {
    resolveRunnerClosed = resolve
  })
  const closed = new Promise<void>((resolve) => {
    resolveClosed = resolve
  })

  return {
    controller,
    awaitingRunner: false,
    pausedForConsumer: false,
    detached: false,
    terminalDelivered: false,
    runnerClosed,
    resolveRunnerClosed,
    closed,
    resolveClosed,
  }
}

function fallbackTerminal(
  event: AgentEvent | undefined,
  runtime: AgentRuntime,
): AgentEvent {
  if (event?.type === 'assistant_message') {
    return { type: 'done', content: event.content }
  }

  const needsRecheck = event?.type === 'tool_start' || event?.type === 'tool_result'
    ? hasSideEffectTool(event.name, runtime)
    : false
  return {
    type: 'interrupted',
    message: 'Agent turn interrupted.',
    workspaceMayHaveChanged: needsRecheck,
  }
}

function hasSideEffectTool(
  name: string,
  runtime: AgentRuntime,
): boolean {
  try {
    return runtime.tools.schemas().some((schema) => (
      schema.name === name && schema.readOnly !== true
    ))
  } catch {
    return false
  }
}

function isTerminalEvent(event: AgentEvent): boolean {
  return event.type === 'done' || event.type === 'error' || event.type === 'interrupted'
}
