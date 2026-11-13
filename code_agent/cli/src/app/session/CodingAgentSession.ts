import type { AgentRunner } from '../../agent/AgentRunner.js'
import type { AgentEvent } from '../../agent/types/event.js'
import { isTerminalEvent } from '../../agent/terminal.js'
import type { AgentRuntime } from '../../agent/types/types.js'
import type { AgentThread } from '../../agent/thread/AgentThread.js'
import type { CodingAgentProfile } from '../profile.js'
import type { RunTaskSnapshot } from '../../task/contracts.js'
import type { FrameworkContextPort } from '../../agent/types/frameworkContext.js'

import { TaskLedger } from './TaskLedger.js'

// [maint-note:8522ec00449c]
// 数据契约导航：CodingAgentSessionOptions；字段含义以消费端校验为准。
// 成员 profile：CodingAgentProfile。
// 成员 runtime：AgentRuntime。
// 成员 thread：AgentThread。
// 成员 runner：AgentRunner。
// 成员 taskLedger（可省略）：TaskLedger。
// 成员 frameworkContext（可省略）：FrameworkContextPort。
// [/maint-note:8522ec00449c]
export interface CodingAgentSessionOptions {
  profile: CodingAgentProfile
  runtime: AgentRuntime
  thread: AgentThread
  runner: AgentRunner
  taskLedger?: TaskLedger
  frameworkContext?: FrameworkContextPort
}

/** Public factory for a session-consistent ledger. */
// [maint-note:27c2c114cb29]
// 维护导航：createTaskLedgerForThread；下列为本函数源码中的接口与路径索引。
// 输入 thread：AgentThread。
// 输入 generation：string（可省略）。
// 返回契约：TaskLedger。
// 返回路径：new TaskLedger({ sessionId: thread.id, ...(generation === undefined ? {} : { generation }), resolveSource: ({ ref }) => thread.resolveSource(ref), })。
// [/maint-note:27c2c114cb29]
export function createTaskLedgerForThread(thread: AgentThread, generation?: string): TaskLedger {
  return new TaskLedger({
    sessionId: thread.id,
    ...(generation === undefined ? {} : { generation }),
    resolveSource: ({ ref }) => thread.resolveSource(ref),
  })
}

export class CodingAgentSession {
  readonly profile: CodingAgentProfile
  readonly runtime: AgentRuntime
  readonly thread: AgentThread
  readonly runner: AgentRunner
  private readonly taskLedger: TaskLedger
  private readonly frameworkContext?: FrameworkContextPort
  private disposed = false
  private generation = 0
  private readonly activeTurns = new Set<ActiveTurn>()
  private disposePromise: Promise<void> | undefined

  // [maint-note:496bc96769b1]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：CodingAgentSessionOptions。
  // 写入目标：this.profile。
  // 写入目标：this.runtime。
  // 写入目标：this.thread。
  // 写入目标：this.runner。
  // 写入目标：this.taskLedger。
  // 写入目标：this.frameworkContext。
  // 直接调用：createTaskLedgerForThread。
  // [/maint-note:496bc96769b1]
  constructor(options: CodingAgentSessionOptions) {
    this.profile = options.profile
    this.runtime = options.runtime
    this.thread = options.thread
    this.runner = options.runner
    // the ledger keeps only bounded previews/hashes; the raw run
    // source stays owned by this session's committed thread and every
    // range read re-validates the LIVE thread, so rollback/clear/dispose
    // surface as a real unavailability instead of a silent fallback.
    this.taskLedger = options.taskLedger ?? createTaskLedgerForThread(options.thread)
    this.frameworkContext = options.frameworkContext
  }

  // [maint-note:8d0ed8ba2021]
  // 维护导航：id；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 返回路径：this.thread.id。
  // [/maint-note:8d0ed8ba2021]
  get id(): string {
    return this.thread.id
  }

  // [maint-note:a005f3dce6df]
  // 维护导航：isDisposed；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.disposed。
  // [/maint-note:a005f3dce6df]
  get isDisposed(): boolean {
    return this.disposed
  }

  // [maint-note:e0da6fba92e3]
  // 维护导航：snapshot；下列为本函数源码中的接口与路径索引。
  // 返回路径：this.thread.snapshot()。
  // 直接调用：this.thread.snapshot。
  // [/maint-note:e0da6fba92e3]
  snapshot() {
    return this.thread.snapshot()
  }

  // [maint-note:182f53e24b28]
  // 维护导航：taskSnapshots；下列为本函数源码中的接口与路径索引。
  // 返回契约：readonly RunTaskSnapshot[]。
  // 返回路径：this.taskLedger.snapshots()。
  // 直接调用：this.taskLedger.snapshots。
  // [/maint-note:182f53e24b28]
  taskSnapshots(): readonly RunTaskSnapshot[] {
    return this.taskLedger.snapshots()
  }

  // [maint-note:aacd28342ce3]
  // 维护导航：runTurn；下列为本函数源码中的接口与路径索引。
  // 输入 input：{ prompt: string signal?: AbortSignal }。
  // 返回契约：AsyncGenerator<AgentEvent>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：this.disposed。
  // 分支条件：input.signal。
  // 分支条件：input.signal.aborted。
  // 分支条件：activeTurn.detached。
  // 分支条件：activeTurn.drainTerminal && !activeTurn.terminalDelivered。
  // 分支条件：result.done。
  // 分支条件：!activeTurn.terminalDelivered && (this.disposed || generation !== this.generation)。
  // 分支条件：this.disposed || generation !== this.generation。
  // 分支条件：event.type === 'assistant_message'。
  // 分支条件：isTerminalEvent(event)。
  // 遍历边界：true。
  // 返回路径：无返回值。
  // 写入目标：activeTurn.cleanup。
  // 写入目标：activeTurn.close。
  // 写入目标：activeTurn.drain。
  // 写入目标：activeTurn.terminalDelivered。
  // 写入目标：activeTurn.pausedForConsumer。
  // 写入目标：activeTurn.awaitingRunner。
  // 写入目标：activeTurn.lastYielded。
  // 写入目标：activeTurn.detached。
  // 直接调用：createActiveTurn。
  // 直接调用：this.activeTurns.add。
  // 直接调用：controller.abort。
  // 直接调用：input.signal.addEventListener。
  // 直接调用：this.taskLedger.reserveRun。
  // 直接调用：this.runner.runTurn。
  // 直接调用：activeTurn.drain。
  // 直接调用：runnerTurn.next。
  // 直接调用：activeTurn.resolveRunnerClosed。
  // 直接调用：fallbackTerminal。
  // 直接调用：drainRunner。
  // 直接调用：isTerminalEvent。
  // 直接调用：activeTurn.close。
  // 直接调用：activeTurn.cleanup。
  // 直接调用：this.activeTurns.delete。
  // 直接调用：activeTurn.resolveClosed。
  // [/maint-note:aacd28342ce3]
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

    // [maint-note:d1cf4c732876]
    // 维护导航：abortFromCaller；下列为本函数源码中的接口与路径索引。
    // 直接调用：controller.abort。
    // [/maint-note:d1cf4c732876]
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

    const taskRun = this.taskLedger.reserveRun(input.prompt)
    const runnerTurn = this.runner.runTurn({
      thread: this.thread,
      prompt: input.prompt,
      runtime: this.runtime,
      signal: controller.signal,
      taskRun,
      ...(this.frameworkContext ? { frameworkContext: this.frameworkContext } : {}),
    })

    // [maint-note:5140c470125f]
    // 维护导航：closeRunner；下列为本函数源码中的接口与路径索引。
    // 返回契约：Promise<void>。
    // 分支条件：!activeTurn.closePromise。
    // 返回路径：activeTurn.closePromise。
    // 写入目标：activeTurn.closePromise。
    // 直接调用：Promise.resolve() .then(() => runnerTurn.return(undefined)) .then( () => undefined, () => undefined, ) .then。
    // 直接调用：Promise.resolve() .then(() => runnerTurn.return(undefined)) .then。
    // 直接调用：Promise.resolve() .then。
    // 直接调用：Promise.resolve。
    // [/maint-note:5140c470125f]
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

    // [maint-note:63dd50d8f491]
    // 维护导航：drainRunner；下列为本函数源码中的接口与路径索引。
    // 返回契约：Promise<void>。
    // 分支条件：!activeTurn.drainPromise。
    // 返回路径：activeTurn.drainPromise。
    // 写入目标：activeTurn.drainPromise。
    // 直接调用：(async () => { try { while (true) { const result = await runnerTurn.next() if (result.done) break if ( isTerminalEvent(result.value) && !activeTurn.drainTerminal ) { activeTurn.drainTerminal = result.value } } } catch { // AgentRunner norma。
    // [/maint-note:63dd50d8f491]
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

  // [maint-note:f0930c6dcf0c]
  // 维护导航：dispose；下列为本函数源码中的接口与路径索引。
  // 返回契约：Promise<void>。
  // 分支条件：this.disposePromise。
  // 分支条件：activeTurn.pausedForConsumer。
  // 遍历边界：const activeTurn ← this.activeTurns。
  // 返回路径：this.disposePromise。
  // 写入目标：this.disposed。
  // 写入目标：this.generation。
  // 写入目标：activeTurn.detached。
  // 写入目标：this.disposePromise。
  // 直接调用：this.frameworkContext?.dispose。
  // 直接调用：activeTurn.controller.abort。
  // 直接调用：activeTurn.cleanup。
  // 直接调用：pendingTurns.push。
  // 直接调用：activeTurn.drain。
  // 直接调用：activeTurn.close。
  // 直接调用：Promise.all(pendingTurns).then。
  // 直接调用：Promise.all。
  // [/maint-note:f0930c6dcf0c]
  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise

    this.disposed = true
    this.generation += 1
    this.frameworkContext?.dispose()
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

    this.disposePromise = Promise.all(pendingTurns).then(() => {
      this.taskLedger.dispose()
    })
    return this.disposePromise
  }
}

// [maint-note:dd11babfd31e]
// 数据契约导航：ActiveTurn；字段含义以消费端校验为准。
// 成员 controller：AbortController。
// 成员 awaitingRunner：boolean。
// 成员 pausedForConsumer：boolean。
// 成员 detached：boolean。
// 成员 lastYielded（可省略）：AgentEvent。
// 成员 drainTerminal（可省略）：AgentEvent。
// 成员 terminalDelivered：boolean。
// 成员 runnerClosed：Promise<void>。
// 成员 resolveRunnerClosed：() => void。
// 成员 closed：Promise<void>。
// 成员 resolveClosed：() => void。
// 成员 close（可省略）：() => Promise<void>。
// 成员 closePromise（可省略）：Promise<void>。
// 成员 drain（可省略）：() => Promise<void>。
// 成员 drainPromise（可省略）：Promise<void>。
// 成员 cleanup（可省略）：() => void。
// [/maint-note:dd11babfd31e]
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

// [maint-note:84c08d3fa655]
// 维护导航：createActiveTurn；下列为本函数源码中的接口与路径索引。
// 输入 controller：AbortController。
// 返回契约：ActiveTurn。
// 返回路径：{ controller, awaitingRunner: false, pausedForConsumer: false, detached: false, terminalDelivered: false, runnerClosed, resolveRunnerClosed, closed, resolveClosed, }。
// [/maint-note:84c08d3fa655]
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

// [maint-note:7b263ef0bb63]
// 维护导航：fallbackTerminal；下列为本函数源码中的接口与路径索引。
// 输入 event：AgentEvent | undefined。
// 输入 runtime：AgentRuntime。
// 返回契约：AgentEvent。
// 分支条件：event?.type === 'assistant_message'。
// 返回路径：{ type: 'done', content: event.content }。
// 返回路径：{ type: 'interrupted', message: 'Agent turn interrupted.', workspaceMayHaveChanged: needsRecheck, }。
// 直接调用：hasSideEffectTool。
// [/maint-note:7b263ef0bb63]
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

// [maint-note:dc0de64fc7b4]
// 维护导航：hasSideEffectTool；下列为本函数源码中的接口与路径索引。
// 输入 name：string。
// 输入 runtime：AgentRuntime。
// 返回契约：boolean。
// 返回路径：runtime.tools.schemas().some((schema) => ( schema.name === name && schema.readOnly !== true ))。
// 返回路径：false。
// 直接调用：runtime.tools.schemas().some。
// 直接调用：runtime.tools.schemas。
// [/maint-note:dc0de64fc7b4]
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
