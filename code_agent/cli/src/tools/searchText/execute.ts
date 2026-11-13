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

// [maint-note:4ad7ddda9f98]
// 数据契约导航：SearchMatch；字段含义以消费端校验为准。
// 成员 path：string。
// 成员 line：number。
// 成员 column：number。
// 成员 text：string。
// [/maint-note:4ad7ddda9f98]
interface SearchMatch {
  path: string
  line: number
  column: number
  text: string
}

// [maint-note:08f9505a65a8]
// 维护导航：executeSearchText；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 runner：RipgrepRunner。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 返回路径：withFileDiagnostics(() => searchWorkspaceText(input, context, runner))。
// 直接调用：withFileDiagnostics。
// [/maint-note:08f9505a65a8]
export async function executeSearchText(
  input: Record<string, unknown>,
  context: ToolExecutionContext,
  runner: RipgrepRunner,
) {
  return withFileDiagnostics(() => searchWorkspaceText(input, context, runner))
}

// [maint-note:e464a47dec3b]
// 维护导航：searchWorkspaceText；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 输入 context：ToolExecutionContext。
// 输入 runner：RipgrepRunner。
// 异步边界：调用方需等待 Promise；拒绝路径由调用方或上层捕获。
// 分支条件：!rootInfo.isDirectory() && !rootInfo.isFile()。
// 分支条件：options.mode === 'literal'。
// 分支条件：!options.caseSensitive。
// 分支条件：options.includeIgnored。
// 分支条件：options.includeHidden。
// 分支条件：result.interrupted。
// 分支条件：result.timedOut。
// 分支条件：!result.stoppedEarly && !result.truncated && result.exitCode !== 0 && result.exitCode !== 1。
// 分支条件：options.mode === 'regex' && isRegexInputDiagnostic({ exitCode: result.exitCode, stderr: result.stderr, query: options.query, observedRecords, })。
// 返回路径：{ content: formatToolError('tool_validation_error', '', budget, toolDiagnostic('invalid_regex', { field: 'query', message: `Invalid regex query (regex parse error): ${cause}`, })), isError: true, errorCode: 'tool_validation_error', }。
// 返回路径：{ content: boundedJsonResult( { matches: items, truncated }, budget, { matches: [], truncated: true }, ), }。
// 显式抛出：operationError(`Unsupported search path: ${options.path}`, 'not_regular_file', { field: 'path' })。
// 显式抛出：new RipgrepError('search_text interrupted', 'interrupted')。
// 显式抛出：new RipgrepError('search_text timed out', 'timeout')。
// 显式抛出：new RipgrepError( `search_text ripgrep failed with exit code ${result.exitCode}: ${diagnostic}`, 'failed', )。
// 直接调用：parseSearchTextInput。
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
// 直接调用：runner.run。
// 直接调用：formatRipgrepDiagnostic。
// 直接调用：isRegexInputDiagnostic。
// 直接调用：result.stderr.trimEnd().split。
// 直接调用：result.stderr.trimEnd。
// 直接调用：lines.find。
// 直接调用：formatToolError。
// 直接调用：toolDiagnostic。
// 直接调用：observedMatches.sort。
// 直接调用：observedMatches.slice。
// 直接调用：boundedJsonResult。
// [/maint-note:e464a47dec3b]
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

// [maint-note:3ecd9415e82c]
// 维护导航：isRegexInputDiagnostic；下列为本函数源码中的接口与路径索引。
// 输入 input：{ exitCode: number stderr: string query: string observedRecords: number }。
// 返回契约：boolean。
// 分支条件：input.exitCode !== 2 || input.observedRecords !== 0。
// 分支条件：normalized === undefined。
// 分支条件：/^rg: compiled regex exceeds size limit of [1-9][0-9]*$/.test(normalized)。
// 分支条件：input.query.includes('\\n') && normalized === [ 'rg: the literal "\\n" is not allowed in a regex', '', 'Consider enabling multiline mode with the --multiline flag (or -U for short).', 'When multiline mode is enabled, new line characters can。
// 分支条件：lines.length < 4 || lines[0] !== 'rg: regex parse error:' || lines[1] !== ` (?:${input.query})` || !/^ {4}[ \t]*\^+(?:[ \t]+\^+)*[ \t]*$/.test(lines[2] ?? '') || !isRegexCompilerErrorLine(lines[3] ?? '')。
// 分支条件：lines.length === 4。
// 返回路径：false。
// 返回路径：true。
// 返回路径：lines.length === 7 && lines[4] === '' && lines[5] === 'Consider enabling PCRE2 with the --pcre2 flag, which can handle backreferences' && lines[6] === 'and look-around.'。
// 直接调用：normalizeDiagnosticStructure。
// 直接调用：/^rg: compiled regex exceeds size limit of [1-9][0-9]*$/.test。
// 直接调用：input.query.includes。
// 直接调用：[ 'rg: the literal "\\n" is not allowed in a regex', '', 'Consider enabling multiline mode with the --multiline flag (or -U for short).', 'When multiline mode is enabled, new line characters can be matched.', ].join。
// 直接调用：normalized.split。
// 直接调用：/^ {4}[ \t]*\^+(?:[ \t]+\^+)*[ \t]*$/.test。
// 直接调用：isRegexCompilerErrorLine。
// [/maint-note:3ecd9415e82c]
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

// [maint-note:649c964192f4]
// 维护导航：normalizeDiagnosticStructure；下列为本函数源码中的接口与路径索引。
// 输入 stderr：string。
// 返回契约：string | undefined。
// 分支条件：normalized.includes('\r')。
// 分支条件：normalized.endsWith('\n')。
// 返回路径：undefined。
// 返回路径：normalized。
// 写入目标：normalized。
// 直接调用：stderr.replaceAll。
// 直接调用：normalized.includes。
// 直接调用：normalized.endsWith。
// 直接调用：normalized.slice。
// [/maint-note:649c964192f4]
function normalizeDiagnosticStructure(stderr: string): string | undefined {
  let normalized = stderr.replaceAll('\r\n', '\n')
  if (normalized.includes('\r')) return undefined
  if (normalized.endsWith('\n')) normalized = normalized.slice(0, -1)
  if (normalized.endsWith('\n')) return undefined
  return normalized
}

// [maint-note:f00b82dfbbdf]
// 维护导航：isRegexCompilerErrorLine；下列为本函数源码中的接口与路径索引。
// 输入 line：string。
// 返回契约：boolean。
// 分支条件：!/^error: \S(?:.*\S)?$/.test(line)。
// 返回路径：false。
// 返回路径：!/(?:permission denied|access is denied|\(os error [0-9]+\))/.test(line.toLowerCase())。
// 直接调用：/^error: \S(?:.*\S)?$/.test。
// 直接调用：/(?:permission denied|access is denied|\(os error [0-9]+\))/.test。
// 直接调用：line.toLowerCase。
// [/maint-note:f00b82dfbbdf]
function isRegexCompilerErrorLine(line: string): boolean {
  if (!/^error: \S(?:.*\S)?$/.test(line)) return false
  return !/(?:permission denied|access is denied|\(os error [0-9]+\))/.test(line.toLowerCase())
}

// [maint-note:37c7a42ced2d]
// 维护导航：formatRipgrepDiagnostic；下列为本函数源码中的接口与路径索引。
// 输入 stderr：string。
// 输入 maxChars：number。
// 返回契约：string。
// 分支条件：!normalized。
// 返回路径：'unknown error'。
// 返回路径：boundedText(normalized, maxChars).text。
// 直接调用：stderr .replaceAll('\r\n', '\n') .replaceAll('\r', '\n') .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, '\uFFFD') .trim。
// 直接调用：stderr .replaceAll('\r\n', '\n') .replaceAll('\r', '\n') .replace。
// 直接调用：stderr .replaceAll('\r\n', '\n') .replaceAll。
// 直接调用：stderr .replaceAll。
// 直接调用：boundedText。
// [/maint-note:37c7a42ced2d]
function formatRipgrepDiagnostic(stderr: string, maxChars: number): string {
  const normalized = stderr
    .replaceAll('\r\n', '\n')
    .replaceAll('\r', '\n')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, '\uFFFD')
    .trim()
  if (!normalized) return 'unknown error'
  return boundedText(normalized, maxChars).text
}

// [maint-note:0e91dce65c37]
// 维护导航：parseRipgrepRecord；下列为本函数源码中的接口与路径索引。
// 输入 record：Buffer。
// 输入 projectDir：string。
// 返回契约：SearchMatch | undefined。
// 分支条件：!isRecord(event) || typeof event.type !== 'string'。
// 分支条件：event.type !== 'match'。
// 分支条件：event.type === 'begin' || event.type === 'end' || event.type === 'summary' || event.type === 'context'。
// 分支条件：!isRecord(event.data)。
// 分支条件：!isRecord(data.path) || typeof data.path.text !== 'string' || !data.path.text。
// 分支条件：!Number.isInteger(data.line_number) || data.line_number < 1。
// 分支条件：!isRecord(data.lines) || typeof data.lines.text !== 'string'。
// 分支条件：!Array.isArray(data.submatches) || data.submatches.length === 0。
// 分支条件：!isRecord(first) || !isRecord(first.match) || typeof first.match.text !== 'string' || !Number.isInteger(first.start) || first.start < 0。
// 返回路径：undefined。
// 返回路径：{ path: normalizeRelative(data.path.text), line: data.line_number, column: first.start + 1, text: data.lines.text.replace(/\r?\n$/, ''), }。
// 显式抛出：new RipgrepError( `search_text returned invalid ripgrep JSON: ${error instanceof Error ? error.message : String(error)}`, 'failed', )。
// 显式抛出：new RipgrepError('search_text returned an invalid ripgrep event envelope', 'failed')。
// 显式抛出：new RipgrepError(`search_text returned unsupported ripgrep event type: ${event.type}`, 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without valid data', 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without a valid path', 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without a valid line number', 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without valid lines', 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without valid submatches', 'failed')。
// 显式抛出：new RipgrepError('search_text returned a match without a valid submatch', 'failed')。
// 写入目标：event。
// 直接调用：JSON.parse。
// 直接调用：record.toString。
// 直接调用：String。
// 直接调用：isRecord。
// 直接调用：Number.isInteger。
// 直接调用：Array.isArray。
// 直接调用：normalizeRelative。
// 直接调用：data.lines.text.replace。
// [/maint-note:0e91dce65c37]
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

// [maint-note:83035f02a3d5]
// 维护导航：isRecord；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：value is Record<string, any>。
// 返回路径：Boolean(value && typeof value === 'object' && !Array.isArray(value))。
// 直接调用：Boolean。
// 直接调用：Array.isArray。
// [/maint-note:83035f02a3d5]
function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

// [maint-note:5ffe9677f7be]
// 维护导航：relativePath；下列为本函数源码中的接口与路径索引。
// 输入 projectDir：string。
// 输入 path：string。
// 返回契约：string。
// 返回路径：normalizeRelative(relative(projectDir, path)) || '.'。
// 直接调用：normalizeRelative。
// 直接调用：relative。
// [/maint-note:5ffe9677f7be]
function relativePath(projectDir: string, path: string): string {
  return normalizeRelative(relative(projectDir, path)) || '.'
}

// [maint-note:328e64f09674]
// 维护导航：normalizeRelative；下列为本函数源码中的接口与路径索引。
// 输入 path：string。
// 返回契约：string。
// 返回路径：normalized === '.' ? '.' : normalized.replace(/^\.\//, '')。
// 直接调用：path.split('\\').join。
// 直接调用：path.split。
// 直接调用：normalized.replace。
// [/maint-note:328e64f09674]
function normalizeRelative(path: string): string {
  const normalized = path.split('\\').join('/') || '.'
  return normalized === '.' ? '.' : normalized.replace(/^\.\//, '')
}
