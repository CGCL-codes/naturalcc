import { spawn } from 'node:child_process'

import { RipgrepError, type RipgrepRequest, type RipgrepResult, type RipgrepRunner } from './types.js'

const DEFAULT_TIMEOUT_MS = 120_000
const KILL_GRACE_MS = 100

export interface RipgrepRunnerOptions {
  /** Test-only process injection; production callers should use the default spawn. */
  spawnProcess?: RipgrepSpawn
}

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

type TerminationReason = 'output' | 'consumer' | 'timeout' | 'abort' | 'callback'

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
        if (retainedBytes + stdoutPending.length + chunkPart.length + delimiterByte.length > maxOutputBytes) {
          stdoutPending = Buffer.alloc(0)
          truncated = true
          requestTermination('output')
          return
        }

        stdout.push(Buffer.concat([record, delimiterByte]))
        retainedBytes += completeLength
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

    function appendIncompleteRecord(part: Buffer): void {
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

    function requestTermination(reason: TerminationReason): void {
      if (terminationReason) return
      terminationReason = reason
      terminate(child)
      termination = waitForTermination(child)
    }

    function cleanup(): void {
      clearTimeout(timeout)
      request.signal?.removeEventListener('abort', onAbort)
    }
  })
}

function waitForTermination(child: ReturnType<typeof spawn>): Promise<void> {
  if (process.platform === 'win32' || !child.pid) return Promise.resolve()

  return new Promise<void>((resolve) => {
    let finished = false
    let killTimer: ReturnType<typeof setTimeout> | undefined
    let pollTimer: ReturnType<typeof setTimeout> | undefined

    const finish = () => {
      if (finished) return
      finished = true
      if (killTimer) clearTimeout(killTimer)
      if (pollTimer) clearTimeout(pollTimer)
      resolve()
    }

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

function isProcessGroupAlive(pid: number | undefined): boolean {
  if (!pid || process.platform === 'win32') return false
  try {
    process.kill(-pid, 0)
    return true
  } catch {
    return false
  }
}

function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals = 'SIGTERM'): void {
  try {
    if (child.pid && process.platform !== 'win32') process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* already exited */ }
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
