import { spawn } from 'node:child_process'

import { GitChangesError } from './types.js'

const DEFAULT_WORK_TIMEOUT_MS = 5_000
const MAX_WORK_TIMEOUT_MS = 5_000
const MAX_COMMANDS = 17
const MAX_STDOUT_BYTES = 32 * 1024 * 1024
const MAX_STDERR_BYTES = 16 * 1024
const TERM_GRACE_MS = 250
const KILL_GRACE_MS = 250

// [maint-note:c7231f47721e]
// 数据契约导航：GitSpawnOptions；字段含义以消费端校验为准。
// 成员 cwd：string。
// 成员 shell：false。
// 成员 detached：boolean。
// 成员 stdio：['pipe', 'pipe', 'pipe']。
// 成员 env：NodeJS.ProcessEnv。
// [/maint-note:c7231f47721e]
export interface GitSpawnOptions {
  cwd: string
  shell: false
  detached: boolean
  stdio: ['pipe', 'pipe', 'pipe']
  env: NodeJS.ProcessEnv
}

// [maint-note:b6ef23ce35db]
// 数据契约导航：GitSpawn；字段含义以消费端校验为准。
// 类型组合：( executable: string, args: readonly string[], options: GitSpawnOptions, ) => ReturnType<typeof spawn>。
// [/maint-note:b6ef23ce35db]
export type GitSpawn = (
  executable: string,
  args: readonly string[],
  options: GitSpawnOptions,
) => ReturnType<typeof spawn>

// [maint-note:c34677e436d4]
// 数据契约导航：GitCommandResult；字段含义以消费端校验为准。
// 成员 stdout：Buffer。
// 成员 stderr：Buffer。
// 成员 exitCode：number。
// [/maint-note:c34677e436d4]
export interface GitCommandResult {
  stdout: Buffer
  stderr: Buffer
  exitCode: number
}

// [maint-note:74b4700f1a6d]
// 数据契约导航：GitCommandSessionOptions；字段含义以消费端校验为准。
// 成员 executable（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// 成员 timeoutMs（可省略）：number。
// 成员 path（可省略）：string。
// 成员 spawnProcess（可省略）：GitSpawn。
// 成员 now（可省略）：() => number。
// 成员 onCommand（可省略）：(request: { args: readonly string[]; cwd: string }) => void。
// 成员 onMetadataRead（可省略）：(path: string) => void。
// 成员 onRetainedBudget（可省略）：(event: GitRetainedBudgetEvent) => void。
// 成员 limits（可省略）：Partial<GitCommandLimits>。
// [/maint-note:74b4700f1a6d]
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

// [maint-note:a43d84afaf5d]
// 数据契约导航：GitRetainedBudgetEvent；字段含义以消费端校验为准。
// 成员 kind：'reserve' | 'release' | 'reject'。
// 成员 label：string。
// 成员 bytes：number。
// 成员 currentBytes：number。
// 成员 highWaterBytes：number。
// 成员 limitBytes：number。
// [/maint-note:a43d84afaf5d]
export interface GitRetainedBudgetEvent {
  kind: 'reserve' | 'release' | 'reject'
  label: string
  bytes: number
  currentBytes: number
  highWaterBytes: number
  limitBytes: number
}

// [maint-note:f6ce75024035]
// 数据契约导航：GitCommandLimits；字段含义以消费端校验为准。
// 成员 maxCommands：number。
// 成员 maxStdoutBytes：number。
// 成员 maxStderrBytes：number。
// 成员 maxRetainedBytes：number。
// [/maint-note:f6ce75024035]
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

  // [maint-note:a662716a636d]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 options：GitCommandSessionOptions；缺省表达式 {}。
  // 写入目标：this.executable。
  // 写入目标：this.signal。
  // 写入目标：this.now。
  // 写入目标：this.deadline。
  // 写入目标：this.spawnProcess。
  // 写入目标：this.onCommand。
  // 写入目标：this.onMetadataRead。
  // 写入目标：this.onRetainedBudget。
  // 写入目标：this.limits。
  // 写入目标：this.env。
  // 直接调用：Math.min。
  // 直接调用：Math.max。
  // 直接调用：Math.floor。
  // 直接调用：this.now。
  // 直接调用：boundedLimit。
  // [/maint-note:a662716a636d]
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

  // [maint-note:50da49f631c5]
  // 维护导航：consumeRetainedBytes；下列为本函数源码中的接口与路径索引。
  // 输入 bytes：number。
  // 输入 label：类型由调用方与函数体约束；缺省表达式 'retained'。
  // 返回契约：void。
  // 分支条件：!Number.isSafeInteger(bytes) || bytes < 0。
  // 分支条件：!Number.isSafeInteger(next) || next > this.limits.maxRetainedBytes。
  // 显式抛出：new GitChangesError( 'git_changes_observation_limit', 'Invalid retained-byte accounting', 'invalid_retained_byte_count', )。
  // 显式抛出：new GitChangesError( 'git_changes_observation_limit', 'Git observation retained-byte budget exceeded', 'retained_bytes_exceeded', )。
  // 写入目标：this.retainedBytes。
  // 写入目标：this.retainedHighWaterBytes。
  // 直接调用：this.throwIfStopped。
  // 直接调用：Number.isSafeInteger。
  // 直接调用：this.reportRetainedBudget。
  // 直接调用：Math.max。
  // [/maint-note:50da49f631c5]
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

  // [maint-note:3952ddb345f4]
  // 维护导航：run；下列为本函数源码中的接口与路径索引。
  // 输入 args：readonly string[]。
  // 输入 cwd：string。
  // 输入 stdin：Buffer（可省略）。
  // 返回契约：Promise<GitCommandResult>。
  // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
  // 分支条件：this.commandCount > this.limits.maxCommands。
  // 分支条件：remaining <= 0。
  // 返回路径：new Promise<GitCommandResult>((resolve, reject) => { let child: ReturnType<typeof spawn> try { child = this.spawnProcess(this.executable, args, { cwd, shell: false, detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe'], en。
  // 显式抛出：new GitChangesError( 'git_changes_observation_limit', 'Git command budget exceeded', 'command_count_exceeded', )。
  // 显式抛出：timeoutError()。
  // 写入目标：this.commandCount。
  // 直接调用：this.throwIfStopped。
  // 直接调用：this.requests.push。
  // 直接调用：this.onCommand。
  // 直接调用：this.now。
  // 直接调用：timeoutError。
  // [/maint-note:3952ddb345f4]
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
      // [maint-note:3dc46bb8ee54]
      // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
      // 直接调用：stop。
      // 直接调用：abortError。
      // [/maint-note:3dc46bb8ee54]
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

      // [maint-note:a01b6b115bfb]
      // 维护导航：finish；下列为本函数源码中的接口与路径索引。
      // 输入 exitCode：number。
      // 返回契约：void。
      // 分支条件：settled。
      // 分支条件：termTimer。
      // 分支条件：settleTimer。
      // 分支条件：firstError。
      // 分支条件：stdout.length === 0。
      // 分支条件：stdout.length === 1。
      // 返回路径：无返回值。
      // 写入目标：settled。
      // 写入目标：combinedStdout。
      // 写入目标：stdout.length。
      // 直接调用：clearTimeout。
      // 直接调用：thisSession.signal?.removeEventListener。
      // 直接调用：reject。
      // 直接调用：thisSession.throwIfStopped。
      // 直接调用：Buffer.alloc。
      // 直接调用：thisSession.consumeRetainedBytes。
      // 直接调用：Buffer.concat。
      // 直接调用：thisSession.releaseRetainedBytes。
      // 直接调用：resolve。
      // [/maint-note:a01b6b115bfb]
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
      // [maint-note:e86207ba7ebc]
      // 维护导航：stop；下列为本函数源码中的接口与路径索引。
      // 输入 error：unknown。
      // 返回契约：void。
      // 分支条件：settled || firstError。
      // 分支条件：cleanupStarted。
      // 返回路径：无返回值。
      // 写入目标：firstError。
      // 写入目标：cleanupStarted。
      // 写入目标：termTimer。
      // 写入目标：settleTimer。
      // 直接调用：terminate。
      // 直接调用：setTimeout。
      // [/maint-note:e86207ba7ebc]
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

  // [maint-note:9d57c05069d7]
  // 维护导航：releaseRetainedBytes；下列为本函数源码中的接口与路径索引。
  // 输入 bytes：number。
  // 输入 label：类型由调用方与函数体约束；缺省表达式 'retained'。
  // 返回契约：void。
  // 分支条件：released > 0。
  // 写入目标：this.retainedBytes。
  // 直接调用：Math.min。
  // 直接调用：Math.max。
  // 直接调用：this.reportRetainedBudget。
  // [/maint-note:9d57c05069d7]
  releaseRetainedBytes(bytes: number, label = 'retained'): void {
    const released = Math.min(this.retainedBytes, Math.max(0, bytes))
    this.retainedBytes -= released
    if (released > 0) this.reportRetainedBudget('release', label, released)
  }

  // [maint-note:17f9a8f36d00]
  // 维护导航：remainingRetainedBytes；下列为本函数源码中的接口与路径索引。
  // 返回契约：number。
  // 返回路径：this.limits.maxRetainedBytes - this.retainedBytes。
  // [/maint-note:17f9a8f36d00]
  remainingRetainedBytes(): number {
    return this.limits.maxRetainedBytes - this.retainedBytes
  }

  // [maint-note:32fc9896c137]
  // 维护导航：beforeMetadataRead；下列为本函数源码中的接口与路径索引。
  // 输入 path：string。
  // 返回契约：void。
  // 直接调用：this.throwIfStopped。
  // 直接调用：this.onMetadataRead。
  // [/maint-note:32fc9896c137]
  beforeMetadataRead(path: string): void {
    this.throwIfStopped()
    this.onMetadataRead?.(path)
    this.throwIfStopped()
  }

  // [maint-note:07d6b83a6de8]
  // 维护导航：throwIfStopped；下列为本函数源码中的接口与路径索引。
  // 返回契约：void。
  // 分支条件：this.signal?.aborted。
  // 分支条件：this.now() >= this.deadline。
  // 显式抛出：abortError()。
  // 显式抛出：timeoutError()。
  // 直接调用：abortError。
  // 直接调用：this.now。
  // 直接调用：timeoutError。
  // [/maint-note:07d6b83a6de8]
  throwIfStopped(): void {
    if (this.signal?.aborted) throw abortError()
    if (this.now() >= this.deadline) throw timeoutError()
  }

  // [maint-note:37c61dd04fd2]
  // 维护导航：reportRetainedBudget；下列为本函数源码中的接口与路径索引。
  // 输入 kind：GitRetainedBudgetEvent['kind']。
  // 输入 label：string。
  // 输入 bytes：number。
  // 返回契约：void。
  // 直接调用：this.onRetainedBudget。
  // [/maint-note:37c61dd04fd2]
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

// [maint-note:70c355a5b468]
// 维护导航：boundedLimit；下列为本函数源码中的接口与路径索引。
// 输入 value：number | undefined。
// 输入 fallback：number。
// 返回契约：number。
// 分支条件：value === undefined。
// 分支条件：!Number.isSafeInteger(value) || value <= 0 || value > fallback。
// 返回路径：fallback。
// 返回路径：value。
// 显式抛出：new Error(`Git command limit must be a positive safe integer no greater than ${fallback}`)。
// 直接调用：Number.isSafeInteger。
// [/maint-note:70c355a5b468]
function boundedLimit(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback
  if (!Number.isSafeInteger(value) || value <= 0 || value > fallback) {
    throw new Error(`Git command limit must be a positive safe integer no greater than ${fallback}`)
  }
  return value
}

// [maint-note:51f7292f9c75]
// 维护导航：terminate；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 输入 signal：NodeJS.Signals。
// 返回契约：void。
// 分支条件：process.platform !== 'win32' && child.pid。
// 直接调用：process.kill。
// 直接调用：child.kill。
// [/maint-note:51f7292f9c75]
function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals): void {
  try {
    if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* bounded cleanup is best-effort */ }
  }
}

// [maint-note:80e84860db85]
// 维护导航：ownedProcessGroupAlive；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 返回契约：boolean。
// 分支条件：process.platform === 'win32' || !child.pid。
// 返回路径：false。
// 返回路径：true。
// 返回路径：(error as NodeJS.ErrnoException).code !== 'ESRCH'。
// 直接调用：process.kill。
// [/maint-note:80e84860db85]
function ownedProcessGroupAlive(child: ReturnType<typeof spawn>): boolean {
  if (process.platform === 'win32' || !child.pid) return false
  try {
    process.kill(-child.pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== 'ESRCH'
  }
}

// [maint-note:9d5059ff248d]
// 维护导航：abortError；下列为本函数源码中的接口与路径索引。
// 返回契约：Error。
// 返回路径：error。
// 写入目标：error.name。
// [/maint-note:9d5059ff248d]
function abortError(): Error {
  const error = new Error('Git observation interrupted')
  error.name = 'AbortError'
  return error
}

// [maint-note:ef61ce83cd36]
// 维护导航：timeoutError；下列为本函数源码中的接口与路径索引。
// 返回契约：GitChangesError。
// 返回路径：new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)。
// [/maint-note:ef61ce83cd36]
function timeoutError(): GitChangesError {
  return new GitChangesError('git_changes_timeout', 'Git observation timed out', 'deadline_exceeded', true)
}

// [maint-note:bef7ab6beb39]
// 维护导航：limitError；下列为本函数源码中的接口与路径索引。
// 输入 reason：string。
// 返回契约：GitChangesError。
// 返回路径：new GitChangesError('git_changes_observation_limit', 'Git observation byte budget exceeded', reason)。
// [/maint-note:bef7ab6beb39]
function limitError(reason: string): GitChangesError {
  return new GitChangesError('git_changes_observation_limit', 'Git observation byte budget exceeded', reason)
}

// [maint-note:8207fe57aa82]
// 维护导航：message；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：string。
// 返回路径：error instanceof Error ? error.message : String(error)。
// 直接调用：String。
// [/maint-note:8207fe57aa82]
function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
