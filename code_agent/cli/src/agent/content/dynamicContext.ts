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

export interface GitStatusOptions {
  timeoutMs?: number
  maxOutputBytes?: number
  spawnProcess?: typeof spawn
}

export function createDynamicContextProvider(
  sampleGit: GitStatusSampler = sampleGitStatus,
): ContextProvider {
  return {
    name: 'dynamic-context',
    async build(input: ContextBuildInput) {
      return getDynamicContext(input, sampleGit)
    },
  }
}

export const dynamicContextProvider: ContextProvider = createDynamicContextProvider()

export interface DynamicContextInput {
  projectDir: string
  now: Date
  timeZone?: string
  signal?: AbortSignal
}

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

function getProjectName(projectDir: string): string {
  const resolved = resolve(projectDir)
  return basename(resolved) || resolved
}

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
export async function getGitStatus(
  projectDir: string,
  signal?: AbortSignal,
  options: GitStatusOptions = {},
): Promise<string> {
  const snapshot = await sampleGitStatus(projectDir, new Date().toISOString(), signal, options)
  return snapshot.result.kind === 'dirty' ? snapshot.result.status : ''
}

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

export function formatLocalDate(date: Date, timeZone?: string): string {
  const zone = timeZone ?? (Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  return `Today's date is ${formatDateInTimeZone(date, zone)}.`
}

function formatDateInTimeZone(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
  const value = (type: Intl.DateTimeFormatPartTypes) => parts.find((part) => part.type === type)?.value
  return `${value('year')}-${value('month')}-${value('day')}`
}

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

    let cleanup = () => {
      signal.removeEventListener('abort', onAbort)
      if (stopTimer) clearTimeout(stopTimer)
    }
    const finish = (exitCode: number) => {
      if (settled) return
      settled = true
      cleanup()
      resolve({ exitCode, stdout: Buffer.concat(stdout), stderr: Buffer.concat(stderr) })
    }
    const fail = (error: unknown) => {
      if (settled) return
      settled = true
      cleanup()
      reject(error)
    }
    const requestStop = () => {
      if (settled) return
      stop()
      if (!stopTimer) stopTimer = setTimeout(() => finish(143), GIT_KILL_GRACE_MS)
    }
    const onAbort = () => {
      if (isImmediateAbort()) {
        requestStop()
        finish(143)
      } else {
        requestStop()
      }
    }
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
    const onError = (error: unknown) => {
      if (signal.aborted) requestStop()
      else fail(error)
    }
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

function decodeUtf8(value: Buffer): string | undefined {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(value)
  } catch {
    return undefined
  }
}

function isMissingCommand(error: unknown): boolean {
  return Boolean(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')
}

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
