import { spawn } from 'node:child_process'
import { basename, resolve } from 'node:path'
import { TextDecoder } from 'node:util'

import { createAbortError } from '../errors.js'
import type { ContextBuildInput, ContextProvider, DynamicContext } from '../types/content.js'

const DEFAULT_GIT_TIMEOUT_MS = 1_000
const DEFAULT_GIT_MAX_OUTPUT_BYTES = 16_000
const GIT_KILL_GRACE_MS = 100

export interface GitStatusOptions {
  timeoutMs?: number
  maxOutputBytes?: number
  spawnProcess?: typeof spawn
}

export const dynamicContextProvider: ContextProvider = {
  name: 'dynamic-context',
  async build(input: ContextBuildInput) {
    return getDynamicContext(input)
  },
}

export interface DynamicContextInput {
  projectDir: string
  now: Date
  signal?: AbortSignal
}

export async function getDynamicContext(
  input: DynamicContextInput,
): Promise<DynamicContext> {
  const gitStatus = await getGitStatus(input.projectDir, input.signal)

  return {
    gitStatus,
    projectName: getProjectName(input.projectDir),
    currentDate: formatLocalDate(input.now),
  }
}

function getProjectName(projectDir: string): string {
  const resolved = resolve(projectDir)
  return basename(resolved) || resolved
}

export async function getGitStatus(
  projectDir: string,
  signal?: AbortSignal,
  options: GitStatusOptions = {},
): Promise<string> {
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
    const child = spawnProcess('git', ['status', '--short'], {
      cwd: projectDir,
      signal: controller.signal,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const stopChild = () => {
      try { child.kill('SIGTERM') } catch { /* already closed */ }
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => {
          try { child.kill('SIGKILL') } catch { /* already closed */ }
        }, GIT_KILL_GRACE_MS)
      }
    }
    const result = await collectChildProcess(
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
    if (termination === 'output') return ''
    if (termination === 'timeout') return ''
    return result.exitCode === 0 ? result.stdout.trim() : ''
  } catch {
    if (termination === 'external') throw createAbortError()
    return ''
  } finally {
    clearTimeout(timer)
    // An externally cancelled request settles immediately, but keeps its
    // scheduled hard kill alive so a child that ignores SIGTERM is cleaned up.
    if (forceKillTimer && termination !== 'external') clearTimeout(forceKillTimer)
    signal?.removeEventListener('abort', onExternalAbort)
  }
}

export function formatLocalDate(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `Today's date is ${year}-${month}-${day}.`
}

function collectChildProcess(
  child: ReturnType<typeof spawn>,
  maxOutputBytes: number,
  stop: () => void,
  signal: AbortSignal,
  onOutputLimit: () => void,
  isImmediateAbort: () => boolean,
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let settled = false
    let stopTimer: ReturnType<typeof setTimeout> | undefined

    let outputBytes = 0
    let cleanup = () => {
      signal.removeEventListener('abort', onAbort)
      if (stopTimer) clearTimeout(stopTimer)
    }
    const finish = (result: { stdout: string; stderr: string; exitCode: number }) => {
      if (settled) return
      settled = true
      cleanup()
      resolve(result)
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
      if (!stopTimer) {
        stopTimer = setTimeout(() => finish({
          exitCode: 143,
          stdout: decodeUtf8(stdout),
          stderr: decodeUtf8(stderr),
        }), GIT_KILL_GRACE_MS)
      }
    }
    const onAbort = () => {
      if (isImmediateAbort()) {
        requestStop()
        finish({
          exitCode: 143,
          stdout: decodeUtf8(stdout),
          stderr: decodeUtf8(stderr),
        })
        return
      }
      requestStop()
    }
    const onStdoutData = (chunk: Buffer) => {
      if (settled) return
      if (outputBytes >= maxOutputBytes) {
        if (chunk.length > 0) {
          onOutputLimit()
          requestStop()
        }
        return
      }
      const remaining = maxOutputBytes - outputBytes
      const accepted = chunk.subarray(0, remaining)
      stdout.push(accepted)
      outputBytes += accepted.length
      if (accepted.length < chunk.length) {
        onOutputLimit()
        requestStop()
      }
    }
    const onStderrData = (chunk: Buffer) => {
      if (settled) return
      if (stderr.length < 1) stderr.push(chunk.subarray(0, maxOutputBytes))
    }
    const onError = (error: unknown) => {
      if (signal.aborted) requestStop()
      else fail(error)
    }
    const onClose = (code: number | null) => {
      const stdoutText = decodeUtf8(stdout)
      const stderrText = decodeUtf8(stderr)
      finish({
        exitCode: code ?? 1,
        stdout: stdoutText,
        stderr: stderrText,
      })
    }
    signal.addEventListener('abort', onAbort, { once: true })
    child.stdout?.on('data', onStdoutData)
    child.stderr?.on('data', onStderrData)
    child.on('error', onError)
    child.on('close', onClose)

    const removeChildListeners = () => {
      child.stdout?.removeListener('data', onStdoutData)
      child.stderr?.removeListener('data', onStderrData)
      child.removeListener('error', onError)
      child.removeListener('close', onClose)
    }
    const previousCleanup = cleanup
    // Keep signal and child listener cleanup in the same settlement path.
    cleanup = () => {
      previousCleanup()
      removeChildListeners()
    }
  })
}

function decodeUtf8(chunks: readonly Buffer[]): string {
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))
  } catch {
    return ''
  }
}
