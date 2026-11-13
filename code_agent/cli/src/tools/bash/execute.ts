import { spawn } from 'node:child_process'
import { inputError } from '../diagnostics.js'

import type { ToolExecutionContext, ToolExecutionResult } from '../types.js'
import { assertOutputBudget, boundedJsonResult, throwIfAborted } from '../toolUtils.js'
import { validateApprovalReason } from '../approval.js'

const DEFAULT_MAX_OUTPUT_CHARS = 8000

// [maint-note:fac1907c2a87]
// 数据契约导航：BashResult；字段含义以消费端校验为准。
// 成员 exitCode：number。
// 成员 stderr：string。
// 成员 stdout：string。
// 成员 timedOut：boolean。
// 成员 interrupted：boolean。
// 成员 truncated：boolean。
// [/maint-note:fac1907c2a87]
export interface BashResult {
  exitCode: number
  stderr: string
  stdout: string
  timedOut: boolean
  interrupted: boolean
  truncated: boolean
}

// [maint-note:5108a4ed5894]
// 数据契约导航：ExecuteBashOptions；字段含义以消费端校验为准。
// 成员 cwd（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// 成员 timeoutMs（可省略）：number。
// 成员 maxOutputChars（可省略）：number。
// 成员 env（可省略）：NodeJS.ProcessEnv。
// [/maint-note:5108a4ed5894]
interface ExecuteBashOptions {
  cwd?: string
  signal?: AbortSignal
  timeoutMs?: number
  maxOutputChars?: number
  env?: NodeJS.ProcessEnv
}

// [maint-note:29318bf01879]
// 维护导航：parseBashInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：{ command: string; reason: string }。
// 分支条件：!input || typeof input !== 'object' || Array.isArray(input)。
// 分支条件：key !== 'command' && key !== 'reason'。
// 分支条件：typeof command !== 'string' || !command.trim()。
// 分支条件：command.includes('\0')。
// 分支条件：command.length > 100_000。
// 遍历边界：const key ← Object.keys(input)。
// 返回路径：{ command, reason: validateApprovalReason(input.reason) }。
// 显式抛出：inputError('Tool arguments must be a JSON object', 'invalid_argument_shape')。
// 显式抛出：inputError(`Unknown parameter: ${key}`, 'unknown_field')。
// 显式抛出：inputError('bash command is required', command === undefined ? 'missing_required_field' : typeof command !== 'string' ? 'invalid_type' : 'invalid_value', 'command', 'non-empty string')。
// 显式抛出：inputError('bash command must not contain NUL', 'invalid_value', 'command', 'string without NUL')。
// 显式抛出：inputError('bash command is too long', 'out_of_range', 'command', 'at most 100000 characters')。
// 直接调用：Array.isArray。
// 直接调用：inputError。
// 直接调用：Object.keys。
// 直接调用：command.trim。
// 直接调用：command.includes。
// 直接调用：validateApprovalReason。
// [/maint-note:29318bf01879]
export function parseBashInput(input: Record<string, unknown>): { command: string; reason: string } {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw inputError('Tool arguments must be a JSON object', 'invalid_argument_shape')
  }
  for (const key of Object.keys(input)) {
    if (key !== 'command' && key !== 'reason') throw inputError(`Unknown parameter: ${key}`, 'unknown_field')
  }
  const command = input.command
  if (typeof command !== 'string' || !command.trim()) {
    throw inputError('bash command is required', command === undefined ? 'missing_required_field' : typeof command !== 'string' ? 'invalid_type' : 'invalid_value', 'command', 'non-empty string')
  }
  if (command.includes('\0')) throw inputError('bash command must not contain NUL', 'invalid_value', 'command', 'string without NUL')
  if (command.length > 100_000) throw inputError('bash command is too long', 'out_of_range', 'command', 'at most 100000 characters')
  return { command, reason: validateApprovalReason(input.reason) }
}

// [maint-note:741c9fb2fa82]
// 维护导航：executeBashTool；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 返回契约：Promise<ToolExecutionResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：{ content: await executeBash(command, { cwd: context.projectDir, signal: context.signal, timeoutMs: context.timeoutMs, maxOutputChars: context.maxOutputChars, env: context.env, }), }。
// 直接调用：parseBashInput。
// 直接调用：executeBash。
// [/maint-note:741c9fb2fa82]
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
      env: context.env,
    }),
  }
}

// [maint-note:0b82bbb4a607]
// 维护导航：executeBash；下列为本函数源码中的接口与路径索引。
// 输入 command：string。
// 输入 options：ExecuteBashOptions；缺省表达式 {}。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：boundedJsonResult(payload, maxOutputChars, { stdout: '', stderr: '', exitCode, timedOut, interrupted, truncated: true, })。
// 直接调用：throwIfAborted。
// 直接调用：assertOutputBudget。
// 直接调用：spawn。
// 直接调用：process.cwd。
// 直接调用：collectChildProcess。
// 直接调用：appendLine。
// 直接调用：Math.max。
// 直接调用：truncate。
// 直接调用：Math.floor。
// 直接调用：boundedJsonResult。
// [/maint-note:0b82bbb4a607]
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
    ...(options.env ? { env: options.env } : {}),
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

// [maint-note:260b0da01061]
// 维护导航：collectChildProcess；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 输入 timeoutMs：number（可省略）。
// 输入 signal：AbortSignal（可省略）。
// 输入 maxOutputChars：类型由调用方与函数体约束；缺省表达式 DEFAULT_MAX_OUTPUT_CHARS。
// 返回契约：Promise<{ exitCode: number stdout: string stderr: string timedOut: boolean interrupted: boolean stdoutTruncated: boolean stderrTruncated: boolean }>。
// 返回路径：new Promise((resolve, reject) => { const streamBudget = Math.max(0, maxOutputChars - 96) const stdout = new BoundedCollector(Math.floor(streamBudget / 2)) const stderr = new BoundedCollector(Math.floor(streamBudget / 2)) let timedOut = fals。
// [/maint-note:260b0da01061]
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

    // [maint-note:50fa1035cf58]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 写入目标：interrupted。
    // 直接调用：requestTermination。
    // [/maint-note:50fa1035cf58]
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
      // [maint-note:5bc17d122d97]
      // 维护导航：finish；下列为本函数源码中的接口与路径索引。
      // 分支条件：settled。
      // 返回路径：无返回值。
      // 写入目标：settled。
      // 直接调用：cleanup。
      // 直接调用：resolve。
      // 直接调用：stdout.value。
      // 直接调用：stderr.value。
      // 直接调用：stdout.isTruncated。
      // 直接调用：stderr.isTruncated。
      // [/maint-note:5bc17d122d97]
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

    // [maint-note:b338637d9bc0]
    // 维护导航：requestTermination；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 分支条件：terminationPromise。
    // 返回路径：无返回值。
    // 写入目标：terminationPromise。
    // 直接调用：terminate。
    // 直接调用：Promise.resolve。
    // [/maint-note:b338637d9bc0]
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

    // [maint-note:af2df95e2388]
    // 维护导航：finishError；下列为本函数源码中的接口与路径索引。
    // 输入 error：Error。
    // 返回契约：void。
    // 写入目标：settled。
    // 直接调用：cleanup。
    // 直接调用：reject。
    // [/maint-note:af2df95e2388]
    function finishError(error: Error): void {
      settled = true
      cleanup()
      reject(error)
    }

    // [maint-note:7a16850de291]
    // 维护导航：cleanup；下列为本函数源码中的接口与路径索引。
    // 返回契约：void。
    // 分支条件：timeout。
    // 直接调用：clearTimeout。
    // 直接调用：signal?.removeEventListener。
    // [/maint-note:7a16850de291]
    function cleanup(): void {
      if (timeout) clearTimeout(timeout)
      signal?.removeEventListener('abort', onAbort)
    }
  })
}

// [maint-note:97a417cbb267]
// 维护导航：truncate；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 max：number。
// 返回契约：string。
// 分支条件：max <= 0。
// 分支条件：value.length <= max。
// 分支条件：max <= marker.length。
// 返回路径：''。
// 返回路径：value。
// 返回路径：value.slice(0, max)。
// 返回路径：`${value.slice(0, max - marker.length)}${marker}`。
// 直接调用：value.slice。
// [/maint-note:97a417cbb267]
function truncate(value: string, max: number): string {
  if (max <= 0) return ''
  if (value.length <= max) return value
  const marker = '\n...(truncated)'
  if (max <= marker.length) return value.slice(0, max)
  return `${value.slice(0, max - marker.length)}${marker}`
}

// [maint-note:aff529b18ad4]
// 维护导航：appendLine；下列为本函数源码中的接口与路径索引。
// 输入 value：string。
// 输入 line：string。
// 返回契约：string。
// 返回路径：value ? `${value}\n${line}` : line。
// [/maint-note:aff529b18ad4]
function appendLine(value: string, line: string): string {
  return value ? `${value}\n${line}` : line
}

class BoundedCollector {
  private full = ''
  private head = ''
  private tail = ''
  private seen = 0
  private truncated = false

  // [maint-note:d13a8812b727]
  // 维护导航：constructor；下列为本函数源码中的接口与路径索引。
  // 输入 maxChars：number。
  // [/maint-note:d13a8812b727]
  constructor(private readonly maxChars: number) {}

  // [maint-note:db19eaa04979]
  // 维护导航：append；下列为本函数源码中的接口与路径索引。
  // 输入 chunk：Buffer。
  // 返回契约：void。
  // 分支条件：this.maxChars <= 0。
  // 分支条件：!this.truncated && this.full.length + value.length <= this.maxChars。
  // 分支条件：!this.truncated。
  // 返回路径：无返回值。
  // 写入目标：this.seen。
  // 写入目标：this.truncated。
  // 写入目标：this.full。
  // 写入目标：this.head。
  // 写入目标：this.tail。
  // 直接调用：chunk.toString。
  // 直接调用：Math.ceil。
  // 直接调用：Math.floor。
  // 直接调用：this.full.slice。
  // 直接调用：`${this.full.slice(-tailSize)}${value}`.slice。
  // 直接调用：`${this.tail}${value}`.slice。
  // [/maint-note:db19eaa04979]
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

  // [maint-note:4b1457e2a0de]
  // 维护导航：value；下列为本函数源码中的接口与路径索引。
  // 返回契约：string。
  // 分支条件：!this.truncated。
  // 分支条件：this.maxChars <= 0。
  // 分支条件：this.maxChars <= marker.length。
  // 返回路径：this.full。
  // 返回路径：''。
  // 返回路径：marker.slice(0, this.maxChars)。
  // 返回路径：`${this.head.slice(0, Math.ceil(budget / 2))}${marker}${this.tail.slice(-Math.floor(budget / 2))}`。
  // 直接调用：Math.max。
  // 直接调用：marker.slice。
  // 直接调用：this.head.slice。
  // 直接调用：Math.ceil。
  // 直接调用：this.tail.slice。
  // 直接调用：Math.floor。
  // [/maint-note:4b1457e2a0de]
  value(): string {
    if (!this.truncated) return this.full
    if (this.maxChars <= 0) return ''
    const omitted = Math.max(0, this.seen - this.head.length - this.tail.length)
    const marker = `\n…(${omitted} chars omitted)…\n`
    if (this.maxChars <= marker.length) return marker.slice(0, this.maxChars)
    const budget = Math.max(0, this.maxChars - marker.length)
    return `${this.head.slice(0, Math.ceil(budget / 2))}${marker}${this.tail.slice(-Math.floor(budget / 2))}`
  }

  // [maint-note:2bc5a2bc6d21]
  // 维护导航：isTruncated；下列为本函数源码中的接口与路径索引。
  // 返回契约：boolean。
  // 返回路径：this.truncated。
  // [/maint-note:2bc5a2bc6d21]
  isTruncated(): boolean {
    return this.truncated
  }
}

// [maint-note:b0914356203d]
// 维护导航：waitForProcessGroupExit；下列为本函数源码中的接口与路径索引。
// 输入 pid：number | undefined。
// 返回契约：Promise<void>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!pid || process.platform === 'win32'。
// 遍历边界：let attempt = 0; attempt < 100; attempt++。
// 返回路径：无返回值。
// 直接调用：process.kill。
// [/maint-note:b0914356203d]
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

// [maint-note:2d8aed5fb7f3]
// 维护导航：terminate；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 输入 signal：NodeJS.Signals；缺省表达式 'SIGTERM'。
// 返回契约：void。
// 分支条件：child.pid。
// 直接调用：process.kill。
// 直接调用：child.kill。
// [/maint-note:2d8aed5fb7f3]
function terminate(child: ReturnType<typeof spawn>, signal: NodeJS.Signals = 'SIGTERM'): void {
  try {
    if (child.pid) process.kill(-child.pid, signal)
    else child.kill(signal)
  } catch {
    try { child.kill(signal) } catch { /* already exited */ }
  }
}
