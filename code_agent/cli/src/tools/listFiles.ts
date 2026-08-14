import { lstat } from 'node:fs/promises'
import { relative } from 'node:path'

import type { Tool, ToolExecutionContext } from './types.js'
import { resolveWorkspacePath } from './workspace.js'
import type { RipgrepRunner } from './ripgrep/types.js'
import { buildSafeGlobArgs, createRipgrepRunner, RipgrepError } from './ripgrep/index.js'
import {
  assertKnownKeys,
  assertObject,
  assertOutputBudget,
  boundedJsonResult,
  MAX_FILE_LIMIT,
  optionalBoolean,
  optionalLimit,
  optionalPositiveInteger,
  optionalString,
  throwIfAborted,
} from './toolUtils.js'

const DEFAULT_MAX_DEPTH = 64
const MAX_RIPGREP_OUTPUT_BYTES = 4_000_000

interface ListFilesInput {
  path: string
  pattern?: string
  recursive: boolean
  maxDepth: number
  limit: number
  includeIgnored: boolean
  includeHidden: boolean
}

export function createListFilesTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    name: 'list_files',
    description: '通过系统 ripgrep 递归列出项目内文件；默认遵循 .gitignore 且不含隐藏文件。显式设置 recursive=false 时只列出目标目录的直接文件。',
    readOnly: true,
    requiresApproval: false,
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '相对项目根目录的目录或文件路径，默认 .' },
        pattern: { type: 'string', description: 'ripgrep glob，例如 **/*.ts 或 *.ts' },
        recursive: {
          type: 'boolean',
          default: true,
          description: '默认递归；设为 false 时只检查目标目录的直接文件，即使 pattern 含 ** 也不会进入子目录',
        },
        maxDepth: { type: 'integer', minimum: 1, maximum: DEFAULT_MAX_DEPTH },
        limit: { type: 'integer', minimum: 1, maximum: MAX_FILE_LIMIT },
        includeIgnored: { type: 'boolean', description: '包含 VCS ignored 文件' },
        includeHidden: { type: 'boolean', description: '包含隐藏文件' },
      },
      additionalProperties: false,
    },
    parse: parseListFilesInput,
    execute: (input, context) => executeListFiles(input, context, runner),
  }
}

export const listFilesTool: Tool = createListFilesTool()

export function parseListFilesInput(input: Record<string, unknown>): ListFilesInput {
  assertObject(input)
  assertKnownKeys(input, [
    'path', 'pattern', 'recursive', 'maxDepth', 'limit', 'includeIgnored', 'includeHidden',
  ])
  return {
    path: optionalString(input, 'path', '.'),
    pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''),
    recursive: optionalBoolean(input, 'recursive', true),
    maxDepth: optionalPositiveInteger(input, 'maxDepth', DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH),
    limit: optionalLimit(input),
    includeIgnored: optionalBoolean(input, 'includeIgnored', false),
    includeHidden: optionalBoolean(input, 'includeHidden', false),
  }
}

async function executeListFiles(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  const options = parseListFilesInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const root = await resolveWorkspacePath(context.projectDir, options.path)
  const rootInfo = await lstat(root)
  if (!rootInfo.isDirectory() && !rootInfo.isFile()) {
    throw new Error(`Unsupported list path: ${options.path}`)
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
      return observedRecords < options.limit + 1
    },
  })
  throwRipgrepFailure(result, 'list_files')
  // `--null` records are atomic: an incomplete final record after a byte
  // budget/early stop must never become a fabricated path.
  const paths = result.stdout.split('\0').slice(0, -1).filter(Boolean)
  const entries = paths
    .slice(0, options.limit)
    .map((path) => ({ path: normalizeRelative(path), type: 'file' as const }))
    .sort((a, b) => a.path.localeCompare(b.path))
  const truncated = result.truncated || result.stoppedEarly === true || paths.length > options.limit
  return {
    content: boundedJsonResult(
      { entries, truncated },
      budget,
      { entries: [], truncated: true },
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
