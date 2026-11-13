import { spawn } from 'node:child_process'

import { RipgrepError, type RipgrepRequest, type RipgrepResult, type RipgrepRunner } from './types.js'

const DEFAULT_TIMEOUT_MS = 120_000
const KILL_GRACE_MS = 100
/**
 * Independent hard cap for a single in-flight record while `stdoutRecordRetention`
 * is `"discard"`. Kept far below `maxOutputBytes` so a discard-mode consumer (e.g.
 * `list_files` count) never buffers more than O(one path) regardless of match count.
 */
const DISCARD_PENDING_RECORD_BYTES = 65_536

// [maint-note:ef9a612e541d]
// 数据契约导航：RipgrepRunnerOptions；字段含义以消费端校验为准。
// 成员 spawnProcess（可省略）：RipgrepSpawn。
// [/maint-note:ef9a612e541d]
export interface RipgrepRunnerOptions {
  /** Test-only process injection; production callers should use the default spawn. */
  spawnProcess?: RipgrepSpawn
}

// [maint-note:427c01e459ea]
// 数据契约导航：RipgrepSpawn；字段含义以消费端校验为准。
// 类型组合：( executable: string, args: string[], options: { cwd: string shell: false detached: boolean stdio: ['ignore', 'pipe', 'pipe'] }, ) => ReturnType<typeof spawn>。
// [/maint-note:427c01e459ea]
export type RipgrepSpawn = (
  executable: string,
  args: string[],
  options: {
    cwd: string
    shell: false
    detached: boolean
    stdio: ['ignore', 'pipe', 'pipe']
  },
) => ReturnType<typeof spawn>

// [maint-note:60f7d89a7a4b]
// 维护导航：createRipgrepRunner；下列为本函数源码中的接口与路径索引。
// 输入 executable：类型由调用方与函数体约束；缺省表达式 'rg'。
// 输入 healthTimeoutMs：类型由调用方与函数体约束；缺省表达式 5_000。
// 输入 options：RipgrepRunnerOptions；缺省表达式 {}。
// 返回契约：RipgrepRunner。
// 返回路径：{ run: (request) => runRipgrep(executable, request, spawnProcess), async healthCheck(signal?: AbortSignal) { if (signal?.aborted) { throw new RipgrepError('ripgrep health check was interrupted', 'interrupted') } const result = await runRipg。
// [/maint-note:60f7d89a7a4b]
export function createRipgrepRunner(
  executable = 'rg',
  healthTimeoutMs = 5_000,
  options: RipgrepRunnerOptions = {},
): RipgrepRunner {
  const spawnProcess = options.spawnProcess ?? ((command, args, spawnOptions) => (
    spawn(command, args, spawnOptions)
  ))
  return {
    run: (request) => runRipgrep(executable, request, spawnProcess),
    // [maint-note:b4dd70cd8a39]
    // 维护导航：healthCheck；下列为本函数源码中的接口与路径索引。
    // 输入 signal：AbortSignal（可省略）。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 分支条件：signal?.aborted。
    // 分支条件：result.timedOut。
    // 分支条件：result.interrupted。
    // 分支条件：result.exitCode !== 0。
    // 分支条件：!/^ripgrep\s+\d+\.\d+\.\d+(?:\s|$)/i.test(result.stdout.trim())。
    // 显式抛出：new RipgrepError('ripgrep health check was interrupted', 'interrupted')。
    // 显式抛出：new RipgrepError('ripgrep health check timed out', 'timeout')。
    // 显式抛出：new RipgrepError( `ripgrep health check failed: ${result.stderr || result.stdout || `exit code ${result.exitCode}`}`, 'failed', )。
    // 显式抛出：new RipgrepError( 'ripgrep health check returned an invalid version signature', 'failed', )。
    // 直接调用：runRipgrep。
    // 直接调用：process.cwd。
    // 直接调用：/^ripgrep\s+\d+\.\d+\.\d+(?:\s|$)/i.test。
    // 直接调用：result.stdout.trim。
    // [/maint-note:b4dd70cd8a39]
    async healthCheck(signal?: AbortSignal) {
      if (signal?.aborted) {
        throw new RipgrepError('ripgrep health check was interrupted', 'interrupted')
      }
      const result = await runRipgrep(executable, {
        args: ['--no-config', '--version'],
        cwd: process.cwd(),
        signal,
        timeoutMs: healthTimeoutMs,
        maxOutputBytes: 4096,
      }, spawnProcess)
      if (result.timedOut) {
        throw new RipgrepError('ripgrep health check timed out', 'timeout')
      }
      if (result.interrupted) {
        throw new RipgrepError('ripgrep health check was interrupted', 'interrupted')
      }
      if (result.exitCode !== 0) {
        throw new RipgrepError(
          `ripgrep health check failed: ${result.stderr || result.stdout || `exit code ${result.exitCode}`}`,
          'failed',
        )
      }
      if (!/^ripgrep\s+\d+\.\d+\.\d+(?:\s|$)/i.test(result.stdout.trim())) {
        throw new RipgrepError(
          'ripgrep health check returned an invalid version signature',
          'failed',
        )
      }
    },
  }
}

// [maint-note:133f2b8585cb]
// 数据契约导航：TerminationReason；字段含义以消费端校验为准。
// 类型组合：'output' | 'consumer' | 'timeout' | 'abort' | 'callback'。
// [/maint-note:133f2b8585cb]
type TerminationReason = 'output' | 'consumer' | 'timeout' | 'abort' | 'callback'

// [maint-note:3f70cc21c84f]
// 维护导航：runRipgrep；下列为本函数源码中的接口与路径索引。
// 输入 executable：string。
// 输入 request：RipgrepRequest。
// 输入 spawnProcess：RipgrepSpawn。
// 返回契约：Promise<RipgrepResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：request.signal?.aborted。
// 返回路径：new Promise<RipgrepResult>((resolve, reject) => { let child: ReturnType<typeof spawn> try { child = spawnProcess(executable, [...request.args], { cwd: request.cwd, shell: false, detached: process.platform !== 'win32', stdio: ['ignore', 'pip。
// 显式抛出：new RipgrepError('ripgrep request was interrupted', 'interrupted')。
// 直接调用：Math.max。
// 直接调用：Math.floor。
// [/maint-note:3f70cc21c84f]
async function runRipgrep(
  executable: string,
  request: RipgrepRequest,
  spawnProcess: RipgrepSpawn,
): Promise<RipgrepResult> {
  const maxOutputBytes = Math.max(1, Math.floor(request.maxOutputBytes))
  if (request.signal?.aborted) {
    throw new RipgrepError('ripgrep request was interrupted', 'interrupted')
  }
  const delimiter = request.stdoutRecordDelimiter
  const recordConsumer = delimiter === undefined ? undefined : request.onStdoutRecord
  const discard = request.stdoutRecordRetention === 'discard'

  return new Promise<RipgrepResult>((resolve, reject) => {
    let child: ReturnType<typeof spawn>
    try {
      child = spawnProcess(executable, [...request.args], {
        cwd: request.cwd,
        shell: false,
        detached: process.platform !== 'win32',
        stdio: ['ignore', 'pipe', 'pipe'],
      })
    } catch (error) {
      reject(new RipgrepError(`Unable to start ripgrep: ${errorMessage(error)}`, 'unavailable'))
      return
    }

    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let stdoutPending = Buffer.alloc(0)
    let retainedBytes = 0
    let truncated = false
    let stoppedEarly = false
    let settled = false
    let terminationReason: TerminationReason | undefined
    let termination: Promise<void> | undefined
    let callbackError: unknown

    const timeout = setTimeout(
      () => requestTermination('timeout'),
      request.timeoutMs && request.timeoutMs > 0 ? request.timeoutMs : DEFAULT_TIMEOUT_MS,
    )

    // [maint-note:dc5548929230]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 直接调用：requestTermination。
    // [/maint-note:dc5548929230]
    const onAbort = () => requestTermination('abort')
    request.signal?.addEventListener('abort', onAbort, { once: true })
    if (request.signal?.aborted) onAbort()

    child.stdout?.on('data', (chunk: Buffer) => {
      if (settled || terminationReason) return
      if (recordConsumer && delimiter !== undefined) consumeRecords(chunk, delimiter)
      else appendRaw(stdout, chunk)
    })
    child.stderr?.on('data', (chunk: Buffer) => {
      if (settled) return
      appendRaw(stderr, chunk)
    })
    child.on('error', (error) => {
      if (settled || terminationReason) return
      settled = true
      cleanup()
      reject(new RipgrepError(`Unable to run ripgrep: ${errorMessage(error)}`, 'unavailable'))
    })
    child.on('close', (exitCode) => {
      // [maint-note:80d3c3d679cb]
      // 维护导航：finish；下列为本函数源码中的接口与路径索引。
      // 分支条件：settled。
      // 分支条件：recordConsumer && stdoutPending.length > 0 && !terminationReason。
      // 分支条件：callbackError。
      // 返回路径：无返回值。
      // 写入目标：settled。
      // 写入目标：truncated。
      // 直接调用：cleanup。
      // 直接调用：reject。
      // 直接调用：resolve。
      // 直接调用：Buffer.concat(stdout).toString。
      // 直接调用：Buffer.concat。
      // 直接调用：Buffer.concat(stderr).toString。
      // [/maint-note:80d3c3d679cb]
      const finish = () => {
        if (settled) return
        settled = true
        cleanup()
        if (recordConsumer && stdoutPending.length > 0 && !terminationReason) {
          truncated = true
        }
        if (callbackError) {
          reject(callbackError)
          return
        }
        resolve({
          stdout: Buffer.concat(stdout).toString('utf8'),
          stderr: Buffer.concat(stderr).toString('utf8'),
          exitCode: exitCode ?? 1,
          truncated,
          timedOut: terminationReason === 'timeout',
          interrupted: terminationReason === 'abort',
          outputBytes: retainedBytes,
          stoppedEarly,
          terminationReason,
        })
      }
      if (termination) void termination.then(finish, finish)
      else finish()
    })

    // [maint-note:ff47abf86834]
    // 维护导航：consumeRecords；下列为本函数源码中的接口与路径索引。
    // 输入 chunk：Buffer。
    // 输入 recordDelimiter：number。
    // 返回契约：void。
    // 分支条件：end === -1。
    // 分支条件：discard。
    // 分支条件：stdoutPending.length + chunkPart.length + delimiterByte.length > DISCARD_PENDING_RECORD_BYTES。
    // 分支条件：retainedBytes + stdoutPending.length + chunkPart.length + delimiterByte.length > maxOutputBytes。
    // 分支条件：!discard。
    // 分支条件：!recordConsumer。
    // 分支条件：!recordConsumer(Buffer.from(record))。
    // 遍历边界：offset < chunk.length && !terminationReason。
    // 返回路径：无返回值。
    // 写入目标：stdoutPending。
    // 写入目标：truncated。
    // 写入目标：retainedBytes。
    // 写入目标：offset。
    // 写入目标：stoppedEarly。
    // 写入目标：callbackError。
    // 直接调用：Buffer.from。
    // 直接调用：chunk.indexOf。
    // 直接调用：appendIncompleteRecord。
    // 直接调用：chunk.subarray。
    // 直接调用：Buffer.concat。
    // 直接调用：Buffer.alloc。
    // 直接调用：requestTermination。
    // 直接调用：stdout.push。
    // 直接调用：recordConsumer。
    // [/maint-note:ff47abf86834]
    function consumeRecords(chunk: Buffer, recordDelimiter: number): void {
      const delimiterByte = Buffer.from([recordDelimiter])
      let offset = 0
      while (offset < chunk.length && !terminationReason) {
        const end = chunk.indexOf(delimiterByte, offset)
        if (end === -1) {
          appendIncompleteRecord(chunk.subarray(offset))
          return
        }

        const chunkPart = chunk.subarray(offset, end)
        const record = stdoutPending.length === 0
          ? chunkPart
          : Buffer.concat([stdoutPending, chunkPart])
        const completeLength = record.length + delimiterByte.length

        if (discard) {
          if (stdoutPending.length + chunkPart.length + delimiterByte.length > DISCARD_PENDING_RECORD_BYTES) {
            stdoutPending = Buffer.alloc(0)
            truncated = true
            requestTermination('output')
            return
          }
        } else if (retainedBytes + stdoutPending.length + chunkPart.length + delimiterByte.length > maxOutputBytes) {
          stdoutPending = Buffer.alloc(0)
          truncated = true
          requestTermination('output')
          return
        }

        if (!discard) {
          stdout.push(Buffer.concat([record, delimiterByte]))
          retainedBytes += completeLength
        }
        stdoutPending = Buffer.alloc(0)
        offset = end + delimiterByte.length

        if (!recordConsumer) continue
        try {
          if (!recordConsumer(Buffer.from(record))) {
            stoppedEarly = true
            requestTermination('consumer')
            return
          }
        } catch (error) {
          callbackError = error
          requestTermination('callback')
          return
        }
      }
    }

    // [maint-note:922613d52bd7]
    // 维护导航：appendIncompleteRecord；下列为本函数源码中的接口与路径索引。
    // 输入 part：Buffer。
    // 返回契约：void。
    // 分支条件：discard。
    // 分支条件：available <= 0。
    // 分支条件：part.length > available。
    // 返回路径：无返回值。
    // 写入目标：truncated。
    // 写入目标：stdoutPending。
    // 直接调用：requestTermination。
    // 直接调用：Buffer.concat。
    // 直接调用：part.subarray。
    // [/maint-note:922613d52bd7]
    function appendIncompleteRecord(part: Buffer): void {
      if (discard) {
        const available = DISCARD_PENDING_RECORD_BYTES - stdoutPending.length
        if (available <= 0) {
          truncated = true
          requestTermination('output')
          return
        }
        if (part.length > available) {
          stdoutPending = Buffer.concat([stdoutPending, part.subarray(0, available)])
          truncated = true
          requestTermination('output')
          return
        }
        stdoutPending = Buffer.concat([stdoutPending, part])
        return
      }
      const available = maxOutputBytes - retainedBytes - stdoutPending.length
      if (available <= 0) {
        truncated = true
        requestTermination('output')
        return
      }
      if (part.length > available) {
        stdoutPending = Buffer.concat([stdoutPending, part.subarray(0, available)])
        truncated = true
        requestTermination('output')
        return
      }
      stdoutPending = Buffer.concat([stdoutPending, part])
    }

    // [maint-note:8635caf49ed2]
    // 维护导航：appendRaw；下列为本函数源码中的接口与路径索引。
    // 输入 target：Buffer[]。
    // 输入 chunk：Buffer。
    // 返回契约：void。
    // 分支条件：bufferedBytes >= maxOutputBytes。
    // 分支条件：accepted.length < chunk.length。
    // 返回路径：无返回值。
    // 写入目标：truncated。
    // 写入目标：retainedBytes。
    // 直接调用：requestTermination。
    // 直接调用：chunk.subarray。
    // 直接调用：target.push。
    // 直接调用：Buffer.from。
    // [/maint-note:8635caf49ed2]
    function appendRaw(target: Buffer[], chunk: Buffer): void {
      const bufferedBytes = retainedBytes + stdoutPending.length
      if (bufferedBytes >= maxOutputBytes) {
        truncated = true
        requestTermination('output')
        return
      }
      const remaining = maxOutputBytes - bufferedBytes
      const accepted = chunk.subarray(0, remaining)
      target.push(Buffer.from(accepted))
      retainedBytes += accepted.length
      if (accepted.length < chunk.length) {
        truncated = true
        requestTermination('output')
      }
    }

    // [maint-note:342259cda9fe]
    // 维护导航：requestTermination；下列为本函数源码中的接口与路径索引。
    // 输入 reason：TerminationReason。
    // 返回契约：void。
    // 分支条件：terminationReason。
    // 返回路径：无返回值。
    // 写入目标：terminationReason。
    // 写入目标：termination。
    // 直接调用：terminate。
    // 直接调用：waitForTermination。
    // [/maint-note:342259cda9fe]
    function requestTermination(reason: TerminationReason): void {
      if (terminationReason) return
      terminationReason = reason
      terminate(child)
      termination = waitForTermination(child)
    }

    // [maint-note:d0c4a4a5d012]
    // 维护导航：cleanup；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 直接调用：clearTimeout。
    // 直接调用：request.signal?.removeEventListener。
    // [/maint-note:d0c4a4a5d012]
    function cleanup(): void {
      clearTimeout(timeout)
      request.signal?.removeEventListener('abort', onAbort)
    }
  })
}

// [maint-note:1da337970753]
// 维护导航：waitForTermination；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 返回契约：Promise<void>。
// 分支条件：process.platform === 'win32' || !child.pid。
// 返回路径：Promise.resolve()。
// 返回路径：new Promise<void>((resolve) => { let finished = false let killTimer: ReturnType<typeof setTimeout> | undefined let pollTimer: ReturnType<typeof setTimeout> | undefined const finish = () => { if (finished) return finished = true if (killTime。
// 直接调用：Promise.resolve。
// [/maint-note:1da337970753]
function waitForTermination(child: ReturnType<typeof spawn>): Promise<void> {
  if (process.platform === 'win32' || !child.pid) return Promise.resolve()

  return new Promise<void>((resolve) => {
    let finished = false
    let killTimer: ReturnType<typeof setTimeout> | undefined
    let pollTimer: ReturnType<typeof setTimeout> | undefined

    // [maint-note:6f5fbd2546fc]
    // 维护导航：finish；下列为本函数源码中的接口与路径索引。
    // 分支条件：finished。
    // 分支条件：killTimer。
    // 分支条件：pollTimer。
    // 返回路径：无返回值。
    // 写入目标：finished。
    // 直接调用：clearTimeout。
    // 直接调用：resolve。
    // [/maint-note:6f5fbd2546fc]
    const finish = () => {
      if (finished) return
      finished = true
      if (killTimer) clearTimeout(killTimer)
      if (pollTimer) clearTimeout(pollTimer)
      resolve()
    }

    // [maint-note:103680bbf49e]
    // 维护导航：poll；下列为本函数源码中的接口与路径索引。
    // 分支条件：!isProcessGroupAlive(child.pid)。
    // 返回路径：无返回值。
    // 写入目标：pollTimer。
    // 直接调用：isProcessGroupAlive。
    // 直接调用：finish。
    // 直接调用：setTimeout。
    // [/maint-note:103680bbf49e]
    const poll = () => {
      if (!isProcessGroupAlive(child.pid)) {
        finish()
        return
      }
      pollTimer = setTimeout(poll, 5)
    }

    killTimer = setTimeout(() => {
      terminate(child, 'SIGKILL')
      poll()
    }, KILL_GRACE_MS)
    poll()
  })
}

// [maint-note:874944d19947]
// 维护导航：isProcessGroupAlive；下列为本函数源码中的接口与路径索引。
// 输入 pid：number | undefined。
// 返回契约：boolean。
// 分支条件：!pid || process.platform === 'win32'。
// 返回路径：false。
// 返回路径：true。
// 直接调用：process.kill。
// [/maint-note:874944d19947]
function isProcessGroupAlive(pid: number | undefined): boolean {
  if (!pid || process.platform === 'win32') return false
  try {
    process.kill(-pid, 0)
    return true
  } catch {
    return false
  }
}

// [maint-note:b218fa0f129a]
// 维护导航：terminate；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 输入 signal：NodeJS.Signals；缺省表达式 'SIGTERM'。
// 返回契约：void。
// 分支条件：child.pid && process.platform !== 'win32'。
// 直接调用：process.kill。
// 直接调用：child.kill。
// [/maint-note:b218fa0f129a]
function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals = 'SIGTERM'): void {
  try {
    if (child.pid && process.platform !== 'win32') process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* already exited */ }
  }
}

// [maint-note:9f2a21848647]
// 维护导航：errorMessage；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：string。
// 返回路径：error instanceof Error ? error.message : String(error)。
// 直接调用：String。
// [/maint-note:9f2a21848647]
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
