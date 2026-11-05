import { lstat } from 'node:fs/promises'
import { relative } from 'node:path'

import type { ToolExecutionContext } from '../types.js'
import { resolveWorkspacePath } from '../workspace.js'
import { formatToolError, operationError, toolDiagnostic, withFileDiagnostics } from '../diagnostics.js'
import type { RipgrepRunner } from '../backends/ripgrep/types.js'
import { buildSafeGlobArgs, RipgrepError } from '../backends/ripgrep/index.js'
import {
  assertOutputBudget,
  boundedText,
  boundedJsonResult,
  throwIfAborted,
} from '../toolUtils.js'
import { parseSearchTextInput } from './schema.js'

const MAX_RIPGREP_OUTPUT_BYTES = 8_000_000
const REGEX_SIZE_LIMIT = '1M'
const DFA_SIZE_LIMIT = '1M'
const MAX_RIPGREP_DIAGNOSTIC_CHARS = 2_000

interface SearchMatch {
  path: string
  line: number
  column: number
  text: string
}

export async function executeSearchText(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  return withFileDiagnostics(() => searchWorkspaceText(input, context, runner))
}

async function searchWorkspaceText(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  const options = parseSearchTextInput(input)
  const budget = assertOutputBudget(context.maxOutputChars)
  const root = await resolveWorkspacePath(context.projectDir, options.path)
  const rootInfo = await lstat(root)
  if (!rootInfo.isDirectory() && !rootInfo.isFile()) {
    throw operationError(`Unsupported search path: ${options.path}`, 'not_regular_file', { field: 'path' })
  }
  throwIfAborted(context.signal)

  const rootRelative = relativePath(context.projectDir, root)
  const args = ['--no-config', '--json', '--color', 'never']
  if (options.mode === 'literal') {
    args.push('--fixed-strings')
  } else {
    args.push(
      '--engine', 'default',
      '--no-multiline',
      '--regex-size-limit', REGEX_SIZE_LIMIT,
      '--dfa-size-limit', DFA_SIZE_LIMIT,
    )
  }
  if (!options.caseSensitive) args.push('--ignore-case')
  if (options.includeIgnored) args.push('--no-ignore-vcs')
  if (options.includeHidden) args.push('--hidden')
  args.push(...buildSafeGlobArgs({ pattern: options.pattern, rootRelative }))
  args.push('-e', options.query, '--', rootRelative)

  const observedMatches: SearchMatch[] = []
  let observedRecords = 0

  const result = await runner.run({
    args,
    cwd: context.projectDir,
    signal: context.signal,
    timeoutMs: context.timeoutMs,
    maxOutputBytes: MAX_RIPGREP_OUTPUT_BYTES,
    stdoutRecordDelimiter: 10,
    onStdoutRecord: (record) => {
      observedRecords += 1
      const match = parseRipgrepRecord(record, context.projectDir)
      if (match) observedMatches.push(match)
      return observedMatches.length < options.limit + 1
    },
  })
  if (result.interrupted) throw new RipgrepError('search_text interrupted', 'interrupted')
  if (result.timedOut) throw new RipgrepError('search_text timed out', 'timeout')
  if (!result.stoppedEarly && !result.truncated && result.exitCode !== 0 && result.exitCode !== 1) {
    const diagnostic = formatRipgrepDiagnostic(result.stderr, MAX_RIPGREP_DIAGNOSTIC_CHARS)
    if (options.mode === 'regex' && isRegexInputDiagnostic({
      exitCode: result.exitCode,
      stderr: result.stderr,
      query: options.query,
      observedRecords,
    })) {
      // The existing source check above is the classification authority.
      // Keep the compiler's final cause, without echoing its query/body frame.
      const lines = result.stderr.trimEnd().split('\n')
      const cause = lines.find(line => line.startsWith('error: ')) ?? lines[0] ?? 'regex parse error'
      return {
        content: formatToolError('tool_validation_error', '', budget, toolDiagnostic('invalid_regex', {
          field: 'query', message: `Invalid regex query (regex parse error): ${cause}`,
        })),
        isError: true,
        errorCode: 'tool_validation_error',
      }
    }
    throw new RipgrepError(
      `search_text ripgrep failed with exit code ${result.exitCode}: ${diagnostic}`,
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

function isRegexInputDiagnostic(input: {
  exitCode: number
  stderr: string
  query: string
  observedRecords: number
}): boolean {
  if (input.exitCode !== 2 || input.observedRecords !== 0) return false
  const normalized = normalizeDiagnosticStructure(input.stderr)
  if (normalized === undefined) return false

  if (/^rg: compiled regex exceeds size limit of [1-9][0-9]*$/.test(normalized)) {
    return true
  }
  if (input.query.includes('\\n') && normalized === [
    'rg: the literal "\\n" is not allowed in a regex',
    '',
    'Consider enabling multiline mode with the --multiline flag (or -U for short).',
    'When multiline mode is enabled, new line characters can be matched.',
  ].join('\n')) {
    return true
  }

  const lines = normalized.split('\n')
  if (
    lines.length < 4 ||
    lines[0] !== 'rg: regex parse error:' ||
    lines[1] !== `    (?:${input.query})` ||
    !/^ {4}[ \t]*\^+(?:[ \t]+\^+)*[ \t]*$/.test(lines[2] ?? '') ||
    !isRegexCompilerErrorLine(lines[3] ?? '')
  ) {
    return false
  }
  if (lines.length === 4) return true
  return lines.length === 7 &&
    lines[4] === '' &&
    lines[5] === 'Consider enabling PCRE2 with the --pcre2 flag, which can handle backreferences' &&
    lines[6] === 'and look-around.'
}

function normalizeDiagnosticStructure(stderr: string): string | undefined {
  let normalized = stderr.replaceAll('\r\n', '\n')
  if (normalized.includes('\r')) return undefined
  if (normalized.endsWith('\n')) normalized = normalized.slice(0, -1)
  if (normalized.endsWith('\n')) return undefined
  return normalized
}

function isRegexCompilerErrorLine(line: string): boolean {
  if (!/^error: \S(?:.*\S)?$/.test(line)) return false
  return !/(?:permission denied|access is denied|\(os error [0-9]+\))/.test(line.toLowerCase())
}

function formatRipgrepDiagnostic(stderr: string, maxChars: number): string {
  const normalized = stderr
    .replaceAll('\r\n', '\n')
    .replaceAll('\r', '\n')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, '\uFFFD')
    .trim()
  if (!normalized) return 'unknown error'
  return boundedText(normalized, maxChars).text
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
