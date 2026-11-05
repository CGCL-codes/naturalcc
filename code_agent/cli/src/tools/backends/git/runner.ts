import { spawn } from 'node:child_process'

import { GitChangesError } from './types.js'

const DEFAULT_WORK_TIMEOUT_MS = 5_000
const MAX_WORK_TIMEOUT_MS = 5_000
const MAX_COMMANDS = 17
const MAX_STDOUT_BYTES = 32 * 1024 * 1024
const MAX_STDERR_BYTES = 16 * 1024
const TERM_GRACE_MS = 250
const KILL_GRACE_MS = 250

export interface GitSpawnOptions {
  cwd: string
  shell: false
  detached: boolean
  stdio: ['pipe', 'pipe', 'pipe']
  env: NodeJS.ProcessEnv
}

export type GitSpawn = (
  executable: string,
  args: readonly string[],
  options: GitSpawnOptions,
) => ReturnType<typeof spawn>

export interface GitCommandResult {
  stdout: Buffer
  stderr: Buffer
  exitCode: number
}

export interface GitCommandSessionOptions {
  executable?: string
  signal?: AbortSignal
  timeoutMs?: number
  path?: string
  spawnProcess?: GitSpawn
  now?: () => number
  onCommand?: (request: { args: readonly string[]; cwd: string }) => void
  onMetadataRead?: (path: string) => void
  onRetainedBudget?: (event: GitRetainedBudgetEvent) => void
  limits?: Partial<GitCommandLimits>
}

export interface GitRetainedBudgetEvent {
  kind: 'reserve' | 'release' | 'reject'
  label: string
  bytes: number
  currentBytes: number
  highWaterBytes: number
  limitBytes: number
}

export interface GitCommandLimits {
  maxCommands: number
  maxStdoutBytes: number
  maxStderrBytes: number
  maxRetainedBytes: number
}

export class GitCommandSession {
  readonly requests: Array<{ args: readonly string[]; cwd: string }> = []
  private readonly executable: string
  private readonly signal?: AbortSignal
  private readonly deadline: number
  private readonly spawnProcess: GitSpawn
  private readonly now: () => number
  private readonly env: NodeJS.ProcessEnv
  private readonly onCommand?: GitCommandSessionOptions['onCommand']
  private readonly onMetadataRead?: GitCommandSessionOptions['onMetadataRead']
  private readonly onRetainedBudget?: GitCommandSessionOptions['onRetainedBudget']
  private readonly limits: GitCommandLimits
  private commandCount = 0
  private stdoutBytes = 0
  private stderrBytes = 0
  private retainedBytes = 0
  private retainedHighWaterBytes = 0

  constructor(options: GitCommandSessionOptions = {}) {
    this.executable = options.executable ?? 'git'
    this.signal = options.signal
    this.now = options.now ?? Date.now
    const timeout = Math.min(
      MAX_WORK_TIMEOUT_MS,
      Math.max(1, Math.floor(options.timeoutMs ?? DEFAULT_WORK_TIMEOUT_MS)),
    )
    this.deadline = this.now() + timeout
    this.spawnProcess = options.spawnProcess ?? ((executable, args, spawnOptions) => (
      spawn(executable, [...args], spawnOptions)
    ))
    this.onCommand = options.onCommand
    this.onMetadataRead = options.onMetadataRead
    this.onRetainedBudget = options.onRetainedBudget
    this.limits = {
      maxCommands: boundedLimit(options.limits?.maxCommands, MAX_COMMANDS),
      maxStdoutBytes: boundedLimit(options.limits?.maxStdoutBytes, MAX_STDOUT_BYTES),
      maxStderrBytes: boundedLimit(options.limits?.maxStderrBytes, MAX_STDERR_BYTES),
      maxRetainedBytes: boundedLimit(options.limits?.maxRetainedBytes, MAX_STDOUT_BYTES),
    }
    this.env = {
      ...(options.path ? { PATH: options.path } : process.env.PATH ? { PATH: process.env.PATH } : {}),
      LC_ALL: 'C',
      LANG: 'C',
      GIT_TERMINAL_PROMPT: '0',
      GIT_OPTIONAL_LOCKS: '0',
      GIT_CONFIG_NOSYSTEM: '1',
      GIT_CONFIG_GLOBAL: process.platform === 'win32' ? 'NUL' : '/dev/null',
      GIT_ATTR_NOSYSTEM: '1',
      GIT_NO_LAZY_FETCH: '1',
      GIT_PAGER: 'cat',
      PAGER: 'cat',
    }
  }

  consumeRetainedBytes(bytes: number, label = 'retained'): void {
    this.throwIfStopped()
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new GitChangesError(
        'git_changes_observation_limit',
        'Invalid retained-byte accounting',
        'invalid_retained_byte_count',
      )
    }
    const next = this.retainedBytes + bytes
    if (!Number.isSafeInteger(next) || next > this.limits.maxRetainedBytes) {
      this.reportRetainedBudget('reject', label, bytes)
      throw new GitChangesError(
        'git_changes_observation_limit',
        'Git observation retained-byte budget exceeded',
        'retained_bytes_exceeded',
      )
    }
    this.retainedBytes = next
    this.retainedHighWaterBytes = Math.max(this.retainedHighWaterBytes, next)
    this.reportRetainedBudget('reserve', label, bytes)
  }

  async run(args: readonly string[], cwd: string, stdin?: Buffer): Promise<GitCommandResult> {
    this.throwIfStopped()
    this.commandCount += 1
    if (this.commandCount > this.limits.maxCommands) {
      throw new GitChangesError(
        'git_changes_observation_limit',
        'Git command budget exceeded',
        'command_count_exceeded',
      )
    }
    this.requests.push({ args: [...args], cwd })
    this.onCommand?.({ args: [...args], cwd })
    // The diagnostic hook is deliberately inside the dispatch boundary. A
    // cancellation raised there must prevent the process from being created.
    this.throwIfStopped()

    const remaining = this.deadline - this.now()
    if (remaining <= 0) throw timeoutError()

    return new Promise<GitCommandResult>((resolve, reject) => {
      let child: ReturnType<typeof spawn>
      try {
        child = this.spawnProcess(this.executable, args, {
          cwd,
          shell: false,
          detached: process.platform !== 'win32',
          stdio: ['pipe', 'pipe', 'pipe'],
          env: this.env,
        })
      } catch (error) {
        reject(new GitChangesError(
          'git_changes_unsupported_git',
          `Unable to start Git: ${message(error)}`,
          'git_unavailable',
        ))
        return
      }

      const stdout: Buffer[] = []
      const stderr: Buffer[] = []
      let localStdout = 0
      let localStderr = 0
      let settled = false
      let firstError: unknown
      let cleanupStarted = false
      let termTimer: ReturnType<typeof setTimeout> | undefined
      let settleTimer: ReturnType<typeof setTimeout> | undefined

      const timeout = setTimeout(() => stop(timeoutError()), remaining)
      const onAbort = () => stop(abortError())
      this.signal?.addEventListener('abort', onAbort, { once: true })
      if (this.signal?.aborted) onAbort()

      child.stdout?.on('data', (value: Buffer | string) => {
        if (settled || firstError) return
        const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value)
        const nextStdout = this.stdoutBytes + chunk.length
        const nextRetained = this.retainedBytes + chunk.length
        if (
          nextStdout > this.limits.maxStdoutBytes ||
          nextRetained > this.limits.maxRetainedBytes
        ) {
          stop(limitError('stdout_bytes_exceeded'))
          return
        }
        localStdout += chunk.length
        this.stdoutBytes = nextStdout
        this.retainedBytes = nextRetained
        this.retainedHighWaterBytes = Math.max(this.retainedHighWaterBytes, nextRetained)
        this.reportRetainedBudget('reserve', 'git_stdout', chunk.length)
        stdout.push(chunk)
      })
      child.stderr?.on('data', (value: Buffer | string) => {
        if (settled || firstError) return
        const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value)
        localStderr += chunk.length
        this.stderrBytes += chunk.length
        if (this.stderrBytes > this.limits.maxStderrBytes) {
          stop(limitError('stderr_bytes_exceeded'))
          return
        }
        stderr.push(Buffer.from(chunk))
      })
      child.on('error', (error) => stop(new GitChangesError(
        'git_changes_unsupported_git',
        `Unable to run Git: ${message(error)}`,
        'git_unavailable',
      )))
      child.on('close', (code) => {
        if (!firstError || !cleanupStarted || !ownedProcessGroupAlive(child)) finish(code ?? 1)
      })

      if (stdin && child.stdin) {
        child.stdin.write(stdin, (error) => {
          if (error) {
            stop(new GitChangesError('git_changes_git_failed', 'Unable to send Git command input', 'stdin_write_failed'))
            return
          }
          child.stdin?.end()
        })
      } else child.stdin?.end()

      function finish(exitCode: number): void {
        if (settled) return
        settled = true
        clearTimeout(timeout)
        if (termTimer) clearTimeout(termTimer)
        if (settleTimer) clearTimeout(settleTimer)
        thisSession.signal?.removeEventListener('abort', onAbort)
        if (firstError) {
          reject(firstError)
          return
        }
        try {
          // A child close is not the end of the whole observation budget.
          // Re-check a fake/real clock before publishing its result.
          thisSession.throwIfStopped()
          let combinedStdout: Buffer
          if (stdout.length === 0) combinedStdout = Buffer.alloc(0)
          else if (stdout.length === 1) combinedStdout = stdout[0]!
          else {
            thisSession.consumeRetainedBytes(localStdout, 'stdout_concat')
            combinedStdout = Buffer.concat(stdout, localStdout)
            stdout.length = 0
            thisSession.releaseRetainedBytes(localStdout, 'stdout_chunks')
          }
          resolve({ stdout: combinedStdout, stderr: Buffer.concat(stderr, localStderr), exitCode })
        } catch (error) {
          reject(error)
        }
      }

      const thisSession = this
      function stop(error: unknown): void {
        if (settled || firstError) return
        firstError = error
        if (cleanupStarted) return
        cleanupStarted = true
        terminate(child, 'SIGTERM')
        termTimer = setTimeout(() => {
          if (!ownedProcessGroupAlive(child)) {
            finish(1)
            return
          }
          terminate(child, 'SIGKILL')
        }, TERM_GRACE_MS)
        settleTimer = setTimeout(() => {
          finish(1)
        }, TERM_GRACE_MS + KILL_GRACE_MS)
      }
    })
  }

  releaseRetainedBytes(bytes: number, label = 'retained'): void {
    const released = Math.min(this.retainedBytes, Math.max(0, bytes))
    this.retainedBytes -= released
    if (released > 0) this.reportRetainedBudget('release', label, released)
  }

  remainingRetainedBytes(): number {
    return this.limits.maxRetainedBytes - this.retainedBytes
  }

  beforeMetadataRead(path: string): void {
    this.throwIfStopped()
    this.onMetadataRead?.(path)
    this.throwIfStopped()
  }

  throwIfStopped(): void {
    if (this.signal?.aborted) throw abortError()
    if (this.now() >= this.deadline) throw timeoutError()
  }

  private reportRetainedBudget(kind: GitRetainedBudgetEvent['kind'], label: string, bytes: number): void {
    this.onRetainedBudget?.({
      kind,
      label,
      bytes,
      currentBytes: this.retainedBytes,
      highWaterBytes: this.retainedHighWaterBytes,
      limitBytes: this.limits.maxRetainedBytes,
    })
  }
}

function boundedLimit(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback
  if (!Number.isSafeInteger(value) || value <= 0 || value > fallback) {
    throw new Error(`Git command limit must be a positive safe integer no greater than ${fallback}`)
  }
  return value
}

function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals): void {
  try {
    if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* bounded cleanup is best-effort */ }
  }
}

function ownedProcessGroupAlive(child: ReturnType<typeof spawn>): boolean {
  if (process.platform === 'win32' || !child.pid) return false
  try {
    process.kill(-child.pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== 'ESRCH'
  }
}

function abortError(): Error {
  const error = new Error('Git observation interrupted')
  error.name = 'AbortError'
  return error
}

function timeoutError(): GitChangesError {
  return new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)
}

function limitError(reason: string): GitChangesError {
  return new GitChangesError('git_changes_observation_limit', 'Git observation byte budget exceeded', reason)
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
