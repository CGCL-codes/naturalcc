import { lstat } from 'node:fs/promises'
import { relative } from 'node:path'

import type { ToolExecutionContext } from '../types.js'
import { resolveWorkspacePath } from '../workspace.js'
import { operationError, withFileDiagnostics } from '../diagnostics.js'
import type { RipgrepRunner } from '../backends/ripgrep/types.js'
import { buildSafeGlobArgs, RipgrepError } from '../backends/ripgrep/index.js'
import {
  assertOutputBudget,
  boundedJsonResult,
  DEFAULT_FILE_LIMIT,
  throwIfAborted,
} from '../toolUtils.js'
import { DEFAULT_MAX_DEPTH, parseListFilesInput } from './schema.js'

const MAX_RIPGREP_OUTPUT_BYTES = 4_000_000
export async function executeListFiles(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  return withFileDiagnostics(() => listWorkspaceFiles(input, context, runner))
}

async function listWorkspaceFiles(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  const options = parseListFilesInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const root = await resolveWorkspacePath(context.projectDir, options.path)
  const rootInfo = await lstat(root)
  if (!rootInfo.isDirectory() && !rootInfo.isFile()) {
    throw operationError(`Unsupported list path: ${options.path}`, 'not_regular_file', { field: 'path' })
  }
  throwIfAborted(context.signal)

  const rootRelative = relativePath(context.projectDir, root)
  const args = ['--no-config', '--files', '--null', '--color', 'never']
  if (options.includeIgnored) args.push('--no-ignore-vcs')
  if (options.includeHidden) args.push('--hidden')
  args.push(...buildSafeGlobArgs({ pattern: options.pattern, rootRelative }))
  if (rootInfo.isDirectory() && !options.recursive) args.push('--max-depth', '1')
  else if (rootInfo.isDirectory() && options.maxDepth < DEFAULT_MAX_DEPTH) args.push('--max-depth', String(options.maxDepth))
  args.push('--', rootRelative)

  if (options.mode === 'count') return executeCountMode(args, context, runner)

  const limit = options.limit ?? DEFAULT_FILE_LIMIT
  let observedRecords = 0

  const result = await runner.run({
    args,
    cwd: context.projectDir,
    signal: context.signal,
    timeoutMs: context.timeoutMs,
    maxOutputBytes: MAX_RIPGREP_OUTPUT_BYTES,
    stdoutRecordDelimiter: 0,
    onStdoutRecord: (record) => {
      if (record.length > 0) observedRecords += 1
      return observedRecords < limit + 1
    },
  })
  throwRipgrepFailure(result, 'list_files')
  // `--null` records are atomic: an incomplete final record after a byte
  // budget/early stop must never become a fabricated path.
  const paths = result.stdout.split('\0').slice(0, -1).filter(Boolean)
  const entries = paths
    .slice(0, limit)
    .map((path) => ({ path: normalizeRelative(path), type: 'file' as const }))
    .sort((a, b) => a.path.localeCompare(b.path))
  const truncated = result.truncated || result.stoppedEarly === true || paths.length > limit
  return {
    content: boundedJsonResult(
      { entries, truncated },
      budget,
      { entries: [], truncated: true },
    ),
  }
}

async function executeCountMode(
  args: readonly string[],
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  const budget = assertOutputBudget(context.maxOutputChars)
  let count = 0

  const result = await runner.run({
    args,
    cwd: context.projectDir,
    signal: context.signal,
    timeoutMs: context.timeoutMs,
    maxOutputBytes: MAX_RIPGREP_OUTPUT_BYTES,
    stdoutRecordDelimiter: 0,
    stdoutRecordRetention: 'discard',
    onStdoutRecord: (record) => {
      if (record.length === 0) return true
      if (!Number.isSafeInteger(count + 1)) {
        throw new RipgrepError('list_files count exceeded a safe integer', 'failed')
      }
      count += 1
      return true
    },
  })
  throwRipgrepFailure(result, 'list_files')
  if (result.truncated || result.stoppedEarly) {
    throw new RipgrepError('list_files count did not observe a complete result', 'failed')
  }
  return {
    content: boundedJsonResult(
      { mode: 'count' as const, count, complete: true as const },
      budget,
    ),
  }
}

function throwRipgrepFailure(
  result: { exitCode: number; stderr: string; timedOut: boolean; interrupted: boolean; truncated?: boolean; stoppedEarly?: boolean },
  toolName: string,
): void {
  if (result.interrupted) throw new RipgrepError(`${toolName} interrupted`, 'interrupted')
  if (result.timedOut) throw new RipgrepError(`${toolName} timed out`, 'timeout')
  if (!result.stoppedEarly && !result.truncated && result.exitCode !== 0 && result.exitCode !== 1) {
    throw new RipgrepError(
      `${toolName} ripgrep failed with exit code ${result.exitCode}: ${result.stderr.trim() || 'unknown error'}`,
      'failed',
    )
  }
}

function relativePath(projectDir: string, path: string): string {
  return normalizeRelative(relative(projectDir, path)) || '.'
}

function normalizeRelative(path: string): string {
  const normalized = path.split('\\').join('/') || '.'
  return normalized === '.' ? '.' : normalized.replace(/^\.\//, '')
}
