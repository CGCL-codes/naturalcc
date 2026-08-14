import { lstat } from 'node:fs/promises'
import { relative } from 'node:path'

import type { Tool, ToolExecutionContext } from './types.js'
import { ToolInputError } from './types.js'
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
  optionalString,
  requiredString,
  throwIfAborted,
} from './toolUtils.js'

const MAX_RIPGREP_OUTPUT_BYTES = 8_000_000

interface SearchTextInput {
  query: string
  path: string
  pattern?: string
  caseSensitive: boolean
  regex: boolean
  limit: number
  includeIgnored: boolean
  includeHidden: boolean
}

interface SearchMatch {
  path: string
  line: number
  column: number
  text: string
}

export function createSearchTextTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    name: 'search_text',
    description: '通过系统 ripgrep 搜索项目内字面字符串；默认遵循 .gitignore 且不含隐藏文件。',
    readOnly: true,
    requiresApproval: false,
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', minLength: 1, maxLength: 2000 },
        path: { type: 'string' },
        pattern: { type: 'string', description: '文件路径 glob，由 ripgrep 解释' },
        caseSensitive: { type: 'boolean' },
        limit: { type: 'integer', minimum: 1, maximum: MAX_FILE_LIMIT },
        includeIgnored: { type: 'boolean', description: '包含 VCS ignored 文件' },
        includeHidden: { type: 'boolean', description: '包含隐藏文件' },
      },
      required: ['query'],
      additionalProperties: false,
    },
    parse: parseSearchTextInput,
    execute: (input, context) => executeSearchText(input, context, runner),
  }
}

export const searchTextTool: Tool = createSearchTextTool()

export function parseSearchTextInput(input: Record<string, unknown>): SearchTextInput {
  assertObject(input)
  assertKnownKeys(input, [
    'query', 'path', 'pattern', 'caseSensitive', 'regex', 'limit', 'includeIgnored', 'includeHidden',
  ])
  const query = requiredString(input, 'query')
  const regex = optionalBoolean(input, 'regex', false)
  if (regex) throw new ToolInputError('regex search is disabled in the MVP; use a literal query')
  return {
    query,
    path: optionalString(input, 'path', '.'),
    pattern: input.pattern === undefined ? undefined : optionalString(input, 'pattern', ''),
    caseSensitive: optionalBoolean(input, 'caseSensitive', false),
    regex,
    limit: optionalLimit(input),
    includeIgnored: optionalBoolean(input, 'includeIgnored', false),
    includeHidden: optionalBoolean(input, 'includeHidden', false),
  }
}

async function executeSearchText(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  const options = parseSearchTextInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const root = await resolveWorkspacePath(context.projectDir, options.path)
  const rootInfo = await lstat(root)
  if (!rootInfo.isDirectory() && !rootInfo.isFile()) {
    throw new Error(`Unsupported search path: ${options.path}`)
  }
  throwIfAborted(context.signal)

  const rootRelative = relativePath(context.projectDir, root)
  const args = ['--no-config', '--json', '--fixed-strings', '--color', 'never']
  if (!options.caseSensitive) args.push('--ignore-case')
  if (options.includeIgnored) args.push('--no-ignore-vcs')
  if (options.includeHidden) args.push('--hidden')
  args.push(...buildSafeGlobArgs({ pattern: options.pattern, rootRelative }))
  args.push('--', options.query, rootRelative)

  const observedMatches: SearchMatch[] = []

  const result = await runner.run({
    args,
    cwd: context.projectDir,
    signal: context.signal,
    timeoutMs: context.timeoutMs,
    maxOutputBytes: MAX_RIPGREP_OUTPUT_BYTES,
    stdoutRecordDelimiter: 10,
    onStdoutRecord: (record) => {
      const match = parseRipgrepRecord(record, context.projectDir)
      if (match) observedMatches.push(match)
      return observedMatches.length < options.limit + 1
    },
  })
  if (result.interrupted) throw new RipgrepError('search_text interrupted', 'interrupted')
  if (result.timedOut) throw new RipgrepError('search_text timed out', 'timeout')
  if (!result.stoppedEarly && !result.truncated && result.exitCode !== 0 && result.exitCode !== 1) {
    throw new RipgrepError(
      `search_text ripgrep failed with exit code ${result.exitCode}: ${result.stderr.trim() || 'unknown error'}`,
      'failed',
    )
  }

  observedMatches.sort((a, b) => a.path.localeCompare(b.path) || a.line - b.line || a.column - b.column)
  const items = observedMatches.slice(0, options.limit)
  const truncated = result.truncated || result.stoppedEarly === true || observedMatches.length > options.limit
  return {
    content: boundedJsonResult(
      { matches: items, truncated },
      budget,
      { matches: [], truncated: true },
    ),
  }
}

function parseRipgrepRecord(
  record: Buffer,
  projectDir: string,
): SearchMatch | undefined {
  let event: unknown
  try {
    event = JSON.parse(record.toString('utf8'))
  } catch (error) {
    throw new RipgrepError(
      `search_text returned invalid ripgrep JSON: ${error instanceof Error ? error.message : String(error)}`,
      'failed',
    )
  }

  if (!isRecord(event) || typeof event.type !== 'string') {
    throw new RipgrepError('search_text returned an invalid ripgrep event envelope', 'failed')
  }
  if (event.type !== 'match') {
    if (event.type === 'begin' || event.type === 'end' || event.type === 'summary' || event.type === 'context') {
      return undefined
    }
    throw new RipgrepError(`search_text returned unsupported ripgrep event type: ${event.type}`, 'failed')
  }

  if (!isRecord(event.data)) {
    throw new RipgrepError('search_text returned a match without valid data', 'failed')
  }
  const data = event.data
  if (!isRecord(data.path) || typeof data.path.text !== 'string' || !data.path.text) {
    throw new RipgrepError('search_text returned a match without a valid path', 'failed')
  }
  if (!Number.isInteger(data.line_number) || data.line_number < 1) {
    throw new RipgrepError('search_text returned a match without a valid line number', 'failed')
  }
  if (!isRecord(data.lines) || typeof data.lines.text !== 'string') {
    throw new RipgrepError('search_text returned a match without valid lines', 'failed')
  }
  if (!Array.isArray(data.submatches) || data.submatches.length === 0) {
    throw new RipgrepError('search_text returned a match without valid submatches', 'failed')
  }
  const first = data.submatches[0]
  if (
    !isRecord(first) ||
    !isRecord(first.match) ||
    typeof first.match.text !== 'string' ||
    !Number.isInteger(first.start) ||
    first.start < 0
  ) {
    throw new RipgrepError('search_text returned a match without a valid submatch', 'failed')
  }
  return {
    path: normalizeRelative(data.path.text),
    line: data.line_number,
    column: first.start + 1,
    text: data.lines.text.replace(/\r?\n$/, ''),
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function relativePath(projectDir: string, path: string): string {
  return normalizeRelative(relative(projectDir, path)) || '.'
}

function normalizeRelative(path: string): string {
  const normalized = path.split('\\').join('/') || '.'
  return normalized === '.' ? '.' : normalized.replace(/^\.\//, '')
}
