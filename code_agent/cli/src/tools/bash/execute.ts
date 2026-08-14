import { spawn } from 'node:child_process'

import { ToolInputError, type ToolExecutionContext, type ToolExecutionResult } from '../types.js'
import { assertOutputBudget, boundedJsonResult, throwIfAborted } from '../toolUtils.js'
import { validateApprovalReason } from '../approval.js'

const DEFAULT_MAX_OUTPUT_CHARS = 8000

export interface BashResult {
  exitCode: number
  stderr: string
  stdout: string
  timedOut: boolean
  interrupted: boolean
  truncated: boolean
}

interface ExecuteBashOptions {
  cwd?: string
  signal?: AbortSignal
  timeoutMs?: number
  maxOutputChars?: number
}

export function parseBashInput(input: Record<string, unknown>): { command: string; reason: string } {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new ToolInputError('Tool arguments must be a JSON object')
  }
  for (const key of Object.keys(input)) {
    if (key !== 'command' && key !== 'reason') throw new ToolInputError(`Unknown parameter: ${key}`)
  }
  const command = input.command
  if (typeof command !== 'string' || !command.trim()) {
    throw new ToolInputError('bash command is required')
  }
  if (command.includes('\0')) throw new ToolInputError('bash command must not contain NUL')
  if (command.length > 100_000) throw new ToolInputError('bash command is too long')
  return { command, reason: validateApprovalReason(input.reason) }
}

export async function executeBashTool(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
): Promise<ToolExecutionResult> {
  const { command } = parseBashInput(input)

  return {
    content: await executeBash(command, {
      cwd: context.projectDir,
      signal: context.signal,
      timeoutMs: context.timeoutMs,
      maxOutputChars: context.maxOutputChars,
    }),
  }
}

export async function executeBash(
  command: string,
  options: ExecuteBashOptions = {},
): Promise<string> {
  throwIfAborted(options.signal)
  const maxOutputChars = assertOutputBudget(options.maxOutputChars)
  const child = spawn('sh', ['-c', command], {
    cwd: options.cwd ?? process.cwd(),
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  const {
    exitCode,
    stdout,
    stderr,
    timedOut,
    interrupted,
    stdoutTruncated,
    stderrTruncated,
  } = await collectChildProcess(
    child,
    options.timeoutMs,
    options.signal,
    maxOutputChars,
  )
  const resolvedStderr = timedOut
    ? appendLine(stderr, `Command timed out after ${options.timeoutMs}ms.`)
    : stderr

  const streamBudget = Math.max(0, maxOutputChars - 96)
  const payload: BashResult = {
    stdout: truncate(stdout, Math.floor(streamBudget / 2)),
    stderr: truncate(resolvedStderr, Math.floor(streamBudget / 2)),
    exitCode,
    timedOut,
    interrupted,
    truncated: stdoutTruncated || stderrTruncated,
  }

  return boundedJsonResult(payload, maxOutputChars, {
    stdout: '',
    stderr: '',
    exitCode,
    timedOut,
    interrupted,
    truncated: true,
  })
}

function collectChildProcess(
  child: ReturnType<typeof spawn>,
  timeoutMs?: number,
  signal?: AbortSignal,
  maxOutputChars = DEFAULT_MAX_OUTPUT_CHARS,
): Promise<{
  exitCode: number
  stdout: string
  stderr: string
  timedOut: boolean
  interrupted: boolean
  stdoutTruncated: boolean
  stderrTruncated: boolean
}> {
  return new Promise((resolve, reject) => {
    const streamBudget = Math.max(0, maxOutputChars - 96)
    const stdout = new BoundedCollector(Math.floor(streamBudget / 2))
    const stderr = new BoundedCollector(Math.floor(streamBudget / 2))
    let timedOut = false
    let interrupted = false
    let settled = false
    let terminationPromise: Promise<void> | undefined
    const timeout =
      timeoutMs && timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true
            requestTermination()
          }, timeoutMs)
        : undefined

    const onAbort = () => {
      interrupted = true
      requestTermination()
    }
    signal?.addEventListener('abort', onAbort, { once: true })
    if (signal?.aborted) onAbort()

    child.stdout?.on('data', (chunk: Buffer) => stdout.append(chunk))
    child.stderr?.on('data', (chunk: Buffer) => stderr.append(chunk))
    child.on('error', (error) => {
      if (settled) return
      if (interrupted || timedOut) return
      finishError(error)
    })
    child.on('close', (code) => {
      const finish = () => {
        if (settled) return
        settled = true
        cleanup()
        resolve({
          exitCode: interrupted ? 130 : timedOut ? 124 : code ?? 1,
          stdout: stdout.value(),
          stderr: stderr.value(),
          timedOut,
          interrupted,
          stdoutTruncated: stdout.isTruncated(),
          stderrTruncated: stderr.isTruncated(),
        })
      }
      if (terminationPromise) void terminationPromise.then(finish, finish)
      else finish()
    })

    function requestTermination(): void {
      if (terminationPromise) return
      terminate(child)
      terminationPromise = process.platform === 'win32'
        ? Promise.resolve()
        : new Promise<void>((resolveGrace) => {
            setTimeout(() => {
              terminate(child, 'SIGKILL')
              void waitForProcessGroupExit(child.pid).then(resolveGrace)
            }, 100)
          })
    }

    function finishError(error: Error): void {
      settled = true
      cleanup()
      reject(error)
    }

    function cleanup(): void {
      if (timeout) clearTimeout(timeout)
      signal?.removeEventListener('abort', onAbort)
    }
  })
}

function truncate(value: string, max: number): string {
  if (max <= 0) return ''
  if (value.length <= max) return value
  const marker = '\n...(truncated)'
  if (max <= marker.length) return value.slice(0, max)
  return `${value.slice(0, max - marker.length)}${marker}`
}

function appendLine(value: string, line: string): string {
  return value ? `${value}\n${line}` : line
}

class BoundedCollector {
  private full = ''
  private head = ''
  private tail = ''
  private seen = 0
  private truncated = false

  constructor(private readonly maxChars: number) {}

  append(chunk: Buffer): void {
    const value = chunk.toString('utf8')
    this.seen += value.length
    if (this.maxChars <= 0) {
      this.truncated = true
      return
    }
    if (!this.truncated && this.full.length + value.length <= this.maxChars) {
      this.full += value
      return
    }
    if (!this.truncated) {
      this.truncated = true
      const headSize = Math.ceil(this.maxChars / 2)
      const tailSize = Math.floor(this.maxChars / 2)
      this.head = this.full.slice(0, headSize)
      this.tail = tailSize > 0
        ? `${this.full.slice(-tailSize)}${value}`.slice(-tailSize)
        : ''
      this.full = ''
      return
    }
    const tailSize = Math.floor(this.maxChars / 2)
    this.tail = tailSize > 0 ? `${this.tail}${value}`.slice(-tailSize) : ''
  }

  value(): string {
    if (!this.truncated) return this.full
    if (this.maxChars <= 0) return ''
    const omitted = Math.max(0, this.seen - this.head.length - this.tail.length)
    const marker = `\n…(${omitted} chars omitted)…\n`
    if (this.maxChars <= marker.length) return marker.slice(0, this.maxChars)
    const budget = Math.max(0, this.maxChars - marker.length)
    return `${this.head.slice(0, Math.ceil(budget / 2))}${marker}${this.tail.slice(-Math.floor(budget / 2))}`
  }

  isTruncated(): boolean {
    return this.truncated
  }
}

async function waitForProcessGroupExit(pid: number | undefined): Promise<void> {
  if (!pid || process.platform === 'win32') return
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      process.kill(-pid, 0)
    } catch {
      return
    }
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
}

function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals = 'SIGTERM'): void {
  try {
    if (child.pid) process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* already exited */ }
  }
}
