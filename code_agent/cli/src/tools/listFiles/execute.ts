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
// [maint-note:75d3fedff161]
// 维护导航：executeListFiles；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 runner：RipgrepRunner。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：withFileDiagnostics(() => listWorkspaceFiles(input, context, runner))。
// 直接调用：withFileDiagnostics。
// [/maint-note:75d3fedff161]
export async function executeListFiles(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  return withFileDiagnostics(() => listWorkspaceFiles(input, context, runner))
}

// [maint-note:9903c121b2aa]
// 维护导航：listWorkspaceFiles；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 runner：RipgrepRunner。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!rootInfo.isDirectory() && !rootInfo.isFile()。
// 分支条件：options.includeIgnored。
// 分支条件：options.includeHidden。
// 分支条件：rootInfo.isDirectory() && !options.recursive。
// 分支条件：rootInfo.isDirectory() && options.maxDepth < DEFAULT_MAX_DEPTH。
// 分支条件：options.mode === 'count'。
// 返回路径：executeCountMode(args, context, runner)。
// 返回路径：{ content: boundedJsonResult( { entries, truncated }, budget, { entries: [], truncated: true }, ), }。
// 显式抛出：operationError(`Unsupported list path: ${options.path}`, 'not_regular_file', { field: 'path' })。
// 直接调用：parseListFilesInput。
// 直接调用：assertOutputBudget。
// 直接调用：resolveWorkspacePath。
// 直接调用：lstat。
// 直接调用：rootInfo.isDirectory。
// 直接调用：rootInfo.isFile。
// 直接调用：operationError。
// 直接调用：throwIfAborted。
// 直接调用：relativePath。
// 直接调用：args.push。
// 直接调用：buildSafeGlobArgs。
// 直接调用：String。
// 直接调用：executeCountMode。
// 直接调用：runner.run。
// 直接调用：throwRipgrepFailure。
// 直接调用：result.stdout.split('\0').slice(0, -1).filter。
// 直接调用：result.stdout.split('\0').slice。
// 直接调用：result.stdout.split。
// 直接调用：paths .slice(0, limit) .map((path) => ({ path: normalizeRelative(path), type: 'file' as const })) .sort。
// 直接调用：paths .slice(0, limit) .map。
// 直接调用：paths .slice。
// 直接调用：boundedJsonResult。
// [/maint-note:9903c121b2aa]
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

// [maint-note:0e0f7665b0f9]
// 维护导航：executeCountMode；下列为本函数源码中的接口与路径索引。
// 输入 args：readonly string[]。
// 输入 context：ToolExecutionContext。
// 输入 runner：RipgrepRunner。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：result.truncated || result.stoppedEarly。
// 返回路径：{ content: boundedJsonResult( { mode: 'count' as const, count, complete: true as const }, budget, ), }。
// 显式抛出：new RipgrepError('list_files count did not observe a complete result', 'failed')。
// 直接调用：assertOutputBudget。
// 直接调用：runner.run。
// 直接调用：throwRipgrepFailure。
// 直接调用：boundedJsonResult。
// [/maint-note:0e0f7665b0f9]
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

// [maint-note:250c91df2b60]
// 维护导航：throwRipgrepFailure；下列为本函数源码中的接口与路径索引。
// 输入 result：{ exitCode: number; stderr: string; timedOut: boolean; interrupted: boolean; truncated?: boolean; stoppedEarly?: boolean }。
// 输入 toolName：string。
// 返回契约：void。
// 分支条件：result.interrupted。
// 分支条件：result.timedOut。
// 分支条件：!result.stoppedEarly && !result.truncated && result.exitCode !== 0 && result.exitCode !== 1。
// 显式抛出：new RipgrepError(`${toolName} interrupted`, 'interrupted')。
// 显式抛出：new RipgrepError(`${toolName} timed out`, 'timeout')。
// 显式抛出：new RipgrepError( `${toolName} ripgrep failed with exit code ${result.exitCode}: ${result.stderr.trim() || 'unknown error'}`, 'failed', )。
// 直接调用：result.stderr.trim。
// [/maint-note:250c91df2b60]
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

// [maint-note:2a8f218c5da4]
// 维护导航：relativePath；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 path：string。
// 返回契约：string。
// 返回路径：normalizeRelative(relative(projectDir, path)) || '.'。
// 直接调用：normalizeRelative。
// 直接调用：relative。
// [/maint-note:2a8f218c5da4]
function relativePath(projectDir: string, path: string): string {
  return normalizeRelative(relative(projectDir, path)) || '.'
}

// [maint-note:51a72d5abf53]
// 维护导航：normalizeRelative；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 返回契约：string。
// 返回路径：normalized === '.' ? '.' : normalized.replace(/^\.\//, '')。
// 直接调用：path.split('\\').join。
// 直接调用：path.split。
// 直接调用：normalized.replace。
// [/maint-note:51a72d5abf53]
function normalizeRelative(path: string): string {
  const normalized = path.split('\\').join('/') || '.'
  return normalized === '.' ? '.' : normalized.replace(/^\.\//, '')
}
