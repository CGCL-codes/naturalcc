import { spawn } from 'node:child_process'
import { basename, resolve } from 'node:path'
import { TextDecoder } from 'node:util'

import { createAbortError } from '../errors.js'
import type {
  ContextBuildInput,
  ContextProvider,
  DynamicContext,
  GitSessionSnapshot,
  GitStatusResult,
  GitStatusSampler,
} from '../types/content.js'

const DEFAULT_GIT_TIMEOUT_MS = 1_000
const DEFAULT_GIT_MAX_OUTPUT_BYTES = 16_000
const GIT_KILL_GRACE_MS = 100
const NON_GIT_DIAGNOSTIC = /^fatal: not a git repository \(or any of the parent directories\): \.git\r?\n?$/

// [maint-note:bbbf9b91224a]
// 数据契约导航：GitStatusOptions；字段含义以消费端校验为准。
// 成员 timeoutMs（可省略）：number。
// 成员 maxOutputBytes（可省略）：number。
// 成员 spawnProcess（可省略）：typeof spawn。
// [/maint-note:bbbf9b91224a]
export interface GitStatusOptions {
  timeoutMs?: number
  maxOutputBytes?: number
  spawnProcess?: typeof spawn
}

// [maint-note:ae8b318768ac]
// 维护导航：createDynamicContextProvider；下列为本函数源码中的接口与路径索引。
// 输入 sampleGit：GitStatusSampler；缺省表达式 sampleGitStatus。
// 返回契约：ContextProvider。
// 返回路径：{ name: 'dynamic-context', async build(input: ContextBuildInput) { return getDynamicContext(input, sampleGit) }, }。
// [/maint-note:ae8b318768ac]
export function createDynamicContextProvider(
  sampleGit: GitStatusSampler = sampleGitStatus,
): ContextProvider {
  return {
    name: 'dynamic-context',
    // [maint-note:4e0d70659755]
    // 维护导航：build；下列为本函数源码中的接口与路径索引。
    // 输入 input：ContextBuildInput。
    // 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
    // 返回路径：getDynamicContext(input, sampleGit)。
    // 直接调用：getDynamicContext。
    // [/maint-note:4e0d70659755]
    async build(input: ContextBuildInput) {
      return getDynamicContext(input, sampleGit)
    },
  }
}

export const dynamicContextProvider: ContextProvider = createDynamicContextProvider()

// [maint-note:3607bd154424]
// 数据契约导航：DynamicContextInput；字段含义以消费端校验为准。
// 成员 projectDir：string。
// 成员 now：Date。
// 成员 timeZone（可省略）：string。
// 成员 signal（可省略）：AbortSignal。
// [/maint-note:3607bd154424]
export interface DynamicContextInput {
  projectDir: string
  now: Date
  timeZone?: string
  signal?: AbortSignal
}

// [maint-note:c1de22c373fc]
// 维护导航：getDynamicContext；下列为本函数源码中的接口与路径索引。
// 输入 input：DynamicContextInput。
// 输入 sampleGit：GitStatusSampler；缺省表达式 sampleGitStatus。
// 返回契约：Promise<DynamicContext>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：{ gitStatus: git.result.kind === 'dirty' ? git.result.status : '', projectName: environment.projectName, currentDate: `Today's date is ${localDate}.`, environment, }。
// 直接调用：resolve。
// 直接调用：Intl.DateTimeFormat().resolvedOptions。
// 直接调用：Intl.DateTimeFormat。
// 直接调用：formatDateInTimeZone。
// 直接调用：sampleGit。
// 直接调用：input.now.toISOString。
// 直接调用：cloneGitSnapshot。
// 直接调用：Object.freeze。
// 直接调用：getProjectName。
// [/maint-note:c1de22c373fc]
export async function getDynamicContext(
  input: DynamicContextInput,
  sampleGit: GitStatusSampler = sampleGitStatus,
): Promise<DynamicContext> {
  const workspaceIdentity = resolve(input.projectDir)
  const timeZone = input.timeZone ?? (Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  const localDate = formatDateInTimeZone(input.now, timeZone)
  const sampledGit = await sampleGit(workspaceIdentity, input.now.toISOString(), input.signal)
  const git = cloneGitSnapshot(
    sampledGit,
    workspaceIdentity,
    input.now.toISOString(),
  )
  const environment = Object.freeze({
    workspaceIdentity,
    projectName: getProjectName(workspaceIdentity),
    localDate,
    timeZone,
    git,
  })

  return {
    gitStatus: git.result.kind === 'dirty' ? git.result.status : '',
    projectName: environment.projectName,
    currentDate: `Today's date is ${localDate}.`,
    environment,
  }
}

// [maint-note:d8bbe151039d]
// 维护导航：getProjectName；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 返回契约：string。
// 返回路径：basename(resolved) || resolved。
// 直接调用：resolve。
// 直接调用：basename。
// [/maint-note:d8bbe151039d]
function getProjectName(projectDir: string): string {
  const resolved = resolve(projectDir)
  return basename(resolved) || resolved
}

// [maint-note:465e2b513e87]
// 维护导航：sampleGitStatus；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 sampledAt：string。
// 输入 signal：AbortSignal（可省略）。
// 输入 options：GitStatusOptions；缺省表达式 {}。
// 返回契约：Promise<GitSessionSnapshot>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：Object.freeze({ workspaceIdentity, sampledAt, result: Object.freeze(result), })。
// 直接调用：resolve。
// 直接调用：collectGitStatus。
// 直接调用：Object.freeze。
// [/maint-note:465e2b513e87]
export async function sampleGitStatus(
  projectDir: string,
  sampledAt: string,
  signal?: AbortSignal,
  options: GitStatusOptions = {},
): Promise<GitSessionSnapshot> {
  const workspaceIdentity = resolve(projectDir)
  const result = await collectGitStatus(workspaceIdentity, signal, options)
  return Object.freeze({
    workspaceIdentity,
    sampledAt,
    result: Object.freeze(result),
  })
}

/** Compatibility helper: unavailable, non-Git and clean all map to empty. */
// [maint-note:119177a9c06f]
// 维护导航：getGitStatus；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 signal：AbortSignal（可省略）。
// 输入 options：GitStatusOptions；缺省表达式 {}。
// 返回契约：Promise<string>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：snapshot.result.kind === 'dirty' ? snapshot.result.status : ''。
// 直接调用：sampleGitStatus。
// 直接调用：new Date().toISOString。
// [/maint-note:119177a9c06f]
export async function getGitStatus(
  projectDir: string,
  signal?: AbortSignal,
  options: GitStatusOptions = {},
): Promise<string> {
  const snapshot = await sampleGitStatus(projectDir, new Date().toISOString(), signal, options)
  return snapshot.result.kind === 'dirty' ? snapshot.result.status : ''
}

// [maint-note:6832853f1d8f]
// 维护导航：collectGitStatus；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 signal：AbortSignal | undefined。
// 输入 options：GitStatusOptions。
// 返回契约：Promise<GitStatusResult>。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：signal?.aborted。
// 分支条件：termination === 'external'。
// 分支条件：termination === 'output'。
// 分支条件：termination === 'timeout'。
// 分支条件：stdout === undefined || stderr === undefined。
// 分支条件：collected.exitCode === 0。
// 分支条件：NON_GIT_DIAGNOSTIC.test(stderr)。
// 分支条件：isMissingCommand(error)。
// 分支条件：forceKillTimer && termination !== 'external'。
// 返回路径：{ kind: 'unavailable', reason: 'output_limit' }。
// 返回路径：{ kind: 'unavailable', reason: 'timeout' }。
// 返回路径：{ kind: 'unavailable', reason: 'invalid_utf8' }。
// 返回路径：status ? { kind: 'dirty', status } : { kind: 'clean' }。
// 返回路径：{ kind: 'non_git' }。
// 返回路径：{ kind: 'unavailable', reason: 'command_failed' }。
// 返回路径：{ kind: 'unavailable', reason: 'command_missing' }。
// 返回路径：termination === 'output' ? { kind: 'unavailable', reason: 'output_limit' } : termination === 'timeout' ? { kind: 'unavailable', reason: 'timeout' } : { kind: 'unavailable', reason: 'command_failed' }。
// 显式抛出：createAbortError()。
// 写入目标：termination。
// 直接调用：createAbortError。
// 直接调用：setTimeout。
// 直接调用：signal?.addEventListener。
// 直接调用：spawnProcess。
// 直接调用：collectChildProcess。
// 直接调用：decodeUtf8。
// 直接调用：stdout.replace。
// 直接调用：NON_GIT_DIAGNOSTIC.test。
// 直接调用：isMissingCommand。
// 直接调用：clearTimeout。
// 直接调用：signal?.removeEventListener。
// [/maint-note:6832853f1d8f]
async function collectGitStatus(
  projectDir: string,
  signal: AbortSignal | undefined,
  options: GitStatusOptions,
): Promise<GitStatusResult> {
  if (signal?.aborted) throw createAbortError()
  const timeoutMs = options.timeoutMs ?? DEFAULT_GIT_TIMEOUT_MS
  const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_GIT_MAX_OUTPUT_BYTES
  const spawnProcess = options.spawnProcess ?? spawn
  const controller = new AbortController()
  let termination: 'external' | 'timeout' | 'output' | undefined
  // [maint-note:6b8b574f8faa]
  // 维护导航：onExternalAbort；下列为本函数源码中的接口与路径索引。
  // 分支条件：termination。
  // 返回路径：无返回值。
  // 写入目标：termination。
  // 直接调用：controller.abort。
  // [/maint-note:6b8b574f8faa]
  const onExternalAbort = () => {
    if (termination) return
    termination = 'external'
    controller.abort()
  }
  const timer = setTimeout(() => {
    if (!termination) termination = 'timeout'
    controller.abort()
  }, timeoutMs)
  signal?.addEventListener('abort', onExternalAbort, { once: true })
  let forceKillTimer: ReturnType<typeof setTimeout> | undefined

  try {
    if (signal?.aborted) {
      termination = 'external'
      throw createAbortError()
    }
    const child = spawnProcess('git', ['--no-optional-locks', 'status', '--short'], {
      cwd: projectDir,
      signal: controller.signal,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, LC_ALL: 'C', LANG: 'C' },
    })
    // [maint-note:920b2e794297]
    // 维护导航：stopChild；下列为本函数源码中的接口与路径索引。
    // 分支条件：!forceKillTimer。
    // 写入目标：forceKillTimer。
    // 直接调用：child.kill。
    // 直接调用：setTimeout。
    // [/maint-note:920b2e794297]
    const stopChild = () => {
      try { child.kill('SIGTERM') } catch { /* already closed */ }
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => {
          try { child.kill('SIGKILL') } catch { /* already closed */ }
        }, GIT_KILL_GRACE_MS)
      }
    }
    const collected = await collectChildProcess(
      child,
      maxOutputBytes,
      stopChild,
      controller.signal,
      () => {
        if (!termination) termination = 'output'
      },
      () => termination === 'external',
    )
    if (termination === 'external') throw createAbortError()
    if (termination === 'output') return { kind: 'unavailable', reason: 'output_limit' }
    if (termination === 'timeout') return { kind: 'unavailable', reason: 'timeout' }

    const stdout = decodeUtf8(collected.stdout)
    const stderr = decodeUtf8(collected.stderr)
    if (stdout === undefined || stderr === undefined) {
      return { kind: 'unavailable', reason: 'invalid_utf8' }
    }
    if (collected.exitCode === 0) {
      // Preserve the two-column `git status --short` prefix. Only Git's
      // trailing record terminator is presentation noise.
      const status = stdout.replace(/[\r\n]+$/u, '')
      return status ? { kind: 'dirty', status } : { kind: 'clean' }
    }
    if (NON_GIT_DIAGNOSTIC.test(stderr)) return { kind: 'non_git' }
    return { kind: 'unavailable', reason: 'command_failed' }
  } catch (error) {
    if (termination === 'external') throw createAbortError()
    if (isMissingCommand(error)) return { kind: 'unavailable', reason: 'command_missing' }
    return termination === 'output'
      ? { kind: 'unavailable', reason: 'output_limit' }
      : termination === 'timeout'
        ? { kind: 'unavailable', reason: 'timeout' }
        : { kind: 'unavailable', reason: 'command_failed' }
  } finally {
    clearTimeout(timer)
    if (forceKillTimer && termination !== 'external') clearTimeout(forceKillTimer)
    signal?.removeEventListener('abort', onExternalAbort)
  }
}

// [maint-note:c6170eb43a11]
// 维护导航：formatLocalDate；下列为本函数源码中的接口与路径索引。
// 输入 date：Date。
// 输入 timeZone：string（可省略）。
// 返回契约：string。
// 返回路径：`Today's date is ${formatDateInTimeZone(date, zone)}.`。
// 直接调用：Intl.DateTimeFormat().resolvedOptions。
// 直接调用：Intl.DateTimeFormat。
// 直接调用：formatDateInTimeZone。
// [/maint-note:c6170eb43a11]
export function formatLocalDate(date: Date, timeZone?: string): string {
  const zone = timeZone ?? (Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  return `Today's date is ${formatDateInTimeZone(date, zone)}.`
}

// [maint-note:c3c10daba286]
// 维护导航：formatDateInTimeZone；下列为本函数源码中的接口与路径索引。
// 输入 date：Date。
// 输入 timeZone：string。
// 返回契约：string。
// 返回路径：`${value('year')}-${value('month')}-${value('day')}`。
// 直接调用：new Intl.DateTimeFormat('en-CA', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit', }).formatToParts。
// 直接调用：value。
// [/maint-note:c3c10daba286]
function formatDateInTimeZone(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
  // [maint-note:91c33d8dff67]
  // 维护导航：value；下列为本函数源码中的接口与路径索引。
  // 输入 type：Intl.DateTimeFormatPartTypes。
  // 直接调用：parts.find。
  // [/maint-note:91c33d8dff67]
  const value = (type: Intl.DateTimeFormatPartTypes) => parts.find((part) => part.type === type)?.value
  return `${value('year')}-${value('month')}-${value('day')}`
}

// [maint-note:80ac784fab9e]
// 维护导航：collectChildProcess；下列为本函数源码中的接口与路径索引。
// 输入 child：ReturnType<typeof spawn>。
// 输入 maxOutputBytes：number。
// 输入 stop：() => void。
// 输入 signal：AbortSignal。
// 输入 onOutputLimit：() => void。
// 输入 isImmediateAbort：() => boolean。
// 返回契约：Promise<{ stdout: Buffer; stderr: Buffer; exitCode: number }>。
// 返回路径：new Promise((resolve, reject) => { const stdout: Buffer[] = [] const stderr: Buffer[] = [] let settled = false let stopTimer: ReturnType<typeof setTimeout> | undefined let stdoutBytes = 0 let stderrBytes = 0 let cleanup = () => { signal.rem。
// [/maint-note:80ac784fab9e]
function collectChildProcess(
  child: ReturnType<typeof spawn>,
  maxOutputBytes: number,
  stop: () => void,
  signal: AbortSignal,
  onOutputLimit: () => void,
  isImmediateAbort: () => boolean,
): Promise<{ stdout: Buffer; stderr: Buffer; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let settled = false
    let stopTimer: ReturnType<typeof setTimeout> | undefined
    let stdoutBytes = 0
    let stderrBytes = 0

    // [maint-note:b62ec5567b62]
    // 维护导航：cleanup；下列为本函数源码中的接口与路径索引。
    // 分支条件：stopTimer。
    // 直接调用：signal.removeEventListener。
    // 直接调用：clearTimeout。
    // [/maint-note:b62ec5567b62]
    let cleanup = () => {
      signal.removeEventListener('abort', onAbort)
      if (stopTimer) clearTimeout(stopTimer)
    }
    // [maint-note:3e0559419cac]
    // 维护导航：finish；下列为本函数源码中的接口与路径索引。
    // 输入 exitCode：number。
    // 分支条件：settled。
    // 返回路径：无返回值。
    // 写入目标：settled。
    // 直接调用：cleanup。
    // 直接调用：resolve。
    // 直接调用：Buffer.concat。
    // [/maint-note:3e0559419cac]
    const finish = (exitCode: number) => {
      if (settled) return
      settled = true
      cleanup()
      resolve({ exitCode, stdout: Buffer.concat(stdout), stderr: Buffer.concat(stderr) })
    }
    // [maint-note:9a2653842af1]
    // 维护导航：fail；下列为本函数源码中的接口与路径索引。
    // 输入 error：unknown。
    // 分支条件：settled。
    // 返回路径：无返回值。
    // 写入目标：settled。
    // 直接调用：cleanup。
    // 直接调用：reject。
    // [/maint-note:9a2653842af1]
    const fail = (error: unknown) => {
      if (settled) return
      settled = true
      cleanup()
      reject(error)
    }
    // [maint-note:0695e6c22493]
    // 维护导航：requestStop；下列为本函数源码中的接口与路径索引。
    // 分支条件：settled。
    // 分支条件：!stopTimer。
    // 返回路径：无返回值。
    // 写入目标：stopTimer。
    // 直接调用：stop。
    // 直接调用：setTimeout。
    // [/maint-note:0695e6c22493]
    const requestStop = () => {
      if (settled) return
      stop()
      if (!stopTimer) stopTimer = setTimeout(() => finish(143), GIT_KILL_GRACE_MS)
    }
    // [maint-note:bc1506c329c1]
    // 维护导航：onAbort；下列为本函数源码中的接口与路径索引。
    // 分支条件：isImmediateAbort()。
    // 直接调用：isImmediateAbort。
    // 直接调用：requestStop。
    // 直接调用：finish。
    // [/maint-note:bc1506c329c1]
    const onAbort = () => {
      if (isImmediateAbort()) {
        requestStop()
        finish(143)
      } else {
        requestStop()
      }
    }
    // [maint-note:bf811756bd9e]
    // 维护导航：onStdoutData；下列为本函数源码中的接口与路径索引。
    // 输入 chunk：Buffer。
    // 分支条件：settled。
    // 分支条件：accepted.length > 0。
    // 分支条件：accepted.length < chunk.length。
    // 返回路径：无返回值。
    // 写入目标：stdoutBytes。
    // 直接调用：Math.max。
    // 直接调用：chunk.subarray。
    // 直接调用：stdout.push。
    // 直接调用：onOutputLimit。
    // 直接调用：requestStop。
    // [/maint-note:bf811756bd9e]
    const onStdoutData = (chunk: Buffer) => {
      if (settled) return
      const remaining = Math.max(0, maxOutputBytes - stdoutBytes)
      const accepted = chunk.subarray(0, remaining)
      if (accepted.length > 0) stdout.push(accepted)
      stdoutBytes += accepted.length
      if (accepted.length < chunk.length) {
        onOutputLimit()
        requestStop()
      }
    }
    // [maint-note:e8876cb87cf3]
    // 维护导航：onStderrData；下列为本函数源码中的接口与路径索引。
    // 输入 chunk：Buffer。
    // 分支条件：settled。
    // 分支条件：accepted.length > 0。
    // 分支条件：accepted.length < chunk.length。
    // 返回路径：无返回值。
    // 写入目标：stderrBytes。
    // 直接调用：Math.max。
    // 直接调用：chunk.subarray。
    // 直接调用：stderr.push。
    // 直接调用：onOutputLimit。
    // 直接调用：requestStop。
    // [/maint-note:e8876cb87cf3]
    const onStderrData = (chunk: Buffer) => {
      if (settled) return
      const remaining = Math.max(0, maxOutputBytes - stderrBytes)
      const accepted = chunk.subarray(0, remaining)
      if (accepted.length > 0) stderr.push(accepted)
      stderrBytes += accepted.length
      if (accepted.length < chunk.length) {
        onOutputLimit()
        requestStop()
      }
    }
    // [maint-note:a0186bca24b5]
    // 维护导航：onError；下列为本函数源码中的接口与路径索引。
    // 输入 error：unknown。
    // 分支条件：signal.aborted。
    // 直接调用：requestStop。
    // 直接调用：fail。
    // [/maint-note:a0186bca24b5]
    const onError = (error: unknown) => {
      if (signal.aborted) requestStop()
      else fail(error)
    }
    // [maint-note:3f0a6718ecda]
    // 维护导航：onClose；下列为本函数源码中的接口与路径索引。
    // 输入 code：number | null。
    // 直接调用：finish。
    // [/maint-note:3f0a6718ecda]
    const onClose = (code: number | null) => finish(code ?? 1)
    signal.addEventListener('abort', onAbort, { once: true })
    child.stdout?.on('data', onStdoutData)
    child.stderr?.on('data', onStderrData)
    child.on('error', onError)
    child.on('close', onClose)

    const previousCleanup = cleanup
    cleanup = () => {
      previousCleanup()
      child.stdout?.removeListener('data', onStdoutData)
      child.stderr?.removeListener('data', onStderrData)
      child.removeListener('error', onError)
      child.removeListener('close', onClose)
    }
  })
}

// [maint-note:fb074af1f384]
// 维护导航：decodeUtf8；下列为本函数源码中的接口与路径索引。
// 输入 value：Buffer。
// 返回契约：string | undefined。
// 返回路径：new TextDecoder('utf-8', { fatal: true }).decode(value)。
// 返回路径：undefined。
// 直接调用：new TextDecoder('utf-8', { fatal: true }).decode。
// [/maint-note:fb074af1f384]
function decodeUtf8(value: Buffer): string | undefined {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(value)
  } catch {
    return undefined
  }
}

// [maint-note:88fb8846c371]
// 维护导航：isMissingCommand；下列为本函数源码中的接口与路径索引。
// 输入 error：unknown。
// 返回契约：boolean。
// 返回路径：Boolean(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')。
// 直接调用：Boolean。
// [/maint-note:88fb8846c371]
function isMissingCommand(error: unknown): boolean {
  return Boolean(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')
}

// [maint-note:2f53a313bdf5]
// 维护导航：cloneGitSnapshot；下列为本函数源码中的接口与路径索引。
// 输入 snapshot：GitSessionSnapshot。
// 输入 workspaceIdentity：类型由调用方与函数体约束；缺省表达式 snapshot.workspaceIdentity。
// 输入 sampledAt：类型由调用方与函数体约束；缺省表达式 snapshot.sampledAt。
// 返回契约：GitSessionSnapshot。
// 返回路径：Object.freeze({ workspaceIdentity, sampledAt, result: Object.freeze({ ...snapshot.result }), })。
// 直接调用：Object.freeze。
// [/maint-note:2f53a313bdf5]
function cloneGitSnapshot(
  snapshot: GitSessionSnapshot,
  workspaceIdentity = snapshot.workspaceIdentity,
  sampledAt = snapshot.sampledAt,
): GitSessionSnapshot {
  return Object.freeze({
    workspaceIdentity,
    sampledAt,
    result: Object.freeze({ ...snapshot.result }),
  })
}
