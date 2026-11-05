import { ToolDiagnosticError, ToolInputError, type ToolDiagnostic } from './types.js'

const descriptions: Record<string, readonly [string, string]> = {
  invalid_call: ['Invalid tool call identity.', 'Send a tool name and a unique non-empty call ID.'],
  invalid_json: ['Tool arguments are not valid JSON.', 'Send one JSON object using the declared tool fields.'],
  invalid_argument_shape: ['Tool arguments must be an object.', 'Use the tool schema to provide a JSON object.'],
  tool_not_found: ['The requested tool is unavailable.', 'Use a tool from the current tool list.'],
  missing_required_field: ['A required field is missing.', 'Provide the required field in the tool arguments.'],
  invalid_type: ['The field has the wrong type.', 'Use the declared field type without coercion.'],
  invalid_value: ['The field value is invalid.', 'Use a non-empty value or one of the declared choices.'],
  out_of_range: ['The field is outside its allowed range.', 'Use a value within the declared bounds.'],
  unknown_field: ['An undeclared field was supplied.', 'Remove fields not declared in the tool schema.'],
  conflicting_fields: ['These fields cannot be combined.', 'Supply only one of the conflicting options.'],
  workspace_path_invalid: ['The workspace path is not permitted.', 'Use a relative path inside the authorized workspace.'],
  file_not_found: ['The workspace file was not found.', 'Use list_files or search_text to confirm the path.'],
  permission_denied: ['The file operation was denied.', 'Check workspace permissions; do not bypass the access policy.'],
  io_error: ['The workspace file operation failed.', 'Inspect the workspace state before retrying.'],
  not_regular_file: ['The target is not a regular file.', 'Use list_files to locate an ordinary text file.'],
  invalid_encoding: ['The file is not UTF-8 text.', 'Choose a supported text file.'],
  output_limit: ['The result exceeds its output budget.', 'Request a smaller page or narrower scope.'],
  page_out_of_range: ['The requested page is past EOF.', 'Re-read from a valid line in the current file.'],
  patch_format_error: ['The patch format is invalid.', 'Use *** Begin Patch and *** End Patch with Codex file directives.'],
  patch_hunk_error: ['The patch hunk is invalid.', 'Use @@ or @@ -old[,count] +new[,count] @@ and prefixed body lines.'],
  patch_context_mismatch: ['Patch context does not match the current file.', 'Re-read the file and build a minimal patch against its current content.'],
  patch_target_changed: ['The patch target changed after preflight.', 'Re-read the workspace before constructing a new patch.'],
  patch_restoration_incomplete: ['Workspace restoration was incomplete after the patch failed.', 'Inspect the workspace before continuing; some changes may remain.'],
  file_exists: ['The file already exists.', 'Read the existing file before choosing an update.'],
  invalid_regex: ['The regex query is invalid.', 'Use the supported single-line default regex engine, or mode:"literal".'],
  backend_unavailable: ['The tool backend is unavailable.', 'Check the local backend installation.'],
  backend_failed: ['The tool backend failed.', 'Inspect the workspace and backend before retrying.'],
  timeout: ['The tool timed out.', 'Inspect the workspace before retrying; effects may be unknown.'],
  interrupted: ['The tool was interrupted.', 'Inspect the workspace before continuing; effects may be unknown.'],
  stale_revision: ['The plan revision is stale.', 'Read the current plan and use its revision.'],
  invalid_step_ids: ['The step IDs do not match the current plan.', 'Use each current step ID exactly once.'],
  task_operation_rejected: ['The task operation was rejected.', 'Review the current plan and operation requirements.'],
  approval_denied: ['The tool was not authorized.', 'Respect the denial; do not bypass it using another tool.'],
  approval_provider_failed: ['The approval provider failed.', 'No execution was authorized by this request.'],
  unknown_error: ['The tool failed without a trusted diagnostic.', 'Inspect the workspace before retrying; effects may be unknown.'],
}

export function toolDiagnostic(reason: string, detail: Partial<Omit<ToolDiagnostic, 'reason'>> = {}): ToolDiagnostic {
  const defaults = Object.hasOwn(descriptions, reason) ? descriptions[reason]! : descriptions.unknown_error!
  return { reason, message: detail.message ?? defaults[0], hint: detail.hint ?? defaults[1],
    ...(detail.field === undefined ? {} : { field: detail.field }),
    ...(detail.expected === undefined ? {} : { expected: detail.expected }) }
}

export function inputError(message: string, reason: string, field?: string, expected?: string, hint?: string): ToolInputError {
  return new ToolInputError(message, toolDiagnostic(reason, { field, expected, hint }))
}

export function operationError(message: string, reason: string, detail: Partial<Omit<ToolDiagnostic, 'reason'>> = {}): ToolDiagnosticError {
  return new ToolDiagnosticError(message, toolDiagnostic(reason, detail))
}

export function errorDiagnostic(error: unknown): ToolDiagnostic | undefined {
  try { return error instanceof ToolDiagnosticError ? error.diagnostic : undefined } catch { return undefined }
}

/** Called only around owned filesystem operations, after existing workspace authorization. */
export async function withFileDiagnostics<T>(run: () => Promise<T>): Promise<T> {
  try { return await run() } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    if (error instanceof ToolDiagnosticError) throw error
    const code = error && typeof error === 'object' ? (error as NodeJS.ErrnoException).code : undefined
    const reason = code === 'ENOENT' ? 'file_not_found' : code === 'EACCES' || code === 'EPERM' ? 'permission_denied'
      : code === 'EISDIR' || code === 'ENOTDIR' ? 'not_regular_file' : code === 'EIO' ? 'io_error' : undefined
    if (!reason) throw error
    throw operationError(error instanceof Error ? error.message : 'File operation failed', reason)
  }
}

function safeText(value: string, limit = 1024): string {
  let text = ''
  for (const point of value) {
    if (text.length + point.length > limit) break
    const cp = point.codePointAt(0)!
    text += cp >= 0xd800 && cp <= 0xdfff ? '�' : point
  }
  const truncated = text.length < value.length
  const safe = redactSensitiveAssignments(text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/gu, '')
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069\u2028\u2029]/gu, ' ')
  )
    .replace(/\b(?:sk-|sk_|ghp_)[a-z0-9_-]+/giu, '[redacted]')
    .replace(/(^|[\s"'=(])(?:[a-z]:[\\/]|\/)[^\s"'<>]*/giu, '$1[path]')
  return truncated ? clipped(safe, limit - ' …omitted'.length) + ' …omitted' : safe
}

/** Scan only the capped diagnostic prefix; an unfinished quoted value ends at the cap. */
function redactSensitiveAssignments(text: string): string {
  const assignment = /(?<![\w-])(["']?)(?:api[_-]?key|authorization|password|secret|token)\1\s*[:=]\s*/giu
  let result = '', retainedUntil = 0
  for (let match = assignment.exec(text); match; match = assignment.exec(text)) {
    let end = assignment.lastIndex
    const bearer = /^Bearer\s+/iu.exec(text.slice(end))
    if (bearer) end += bearer[0].length
    const quote = text[end]
    if (quote === '"' || quote === "'") {
      end += 1
      while (end < text.length) {
        if (text[end] === '\\') {
          end = Math.min(text.length, end + 2)
        } else if (text[end++] === quote) {
          break
        }
      }
    } else {
      while (end < text.length && !/[\s,;}\]]/u.test(text[end]!)) end += 1
    }
    result += text.slice(retainedUntil, match.index) + '[redacted]'
    retainedUntil = end
    assignment.lastIndex = end
  }
  return result + text.slice(retainedUntil)
}

function clipped(text: string, max: number): string {
  let result = ''
  for (const point of text) {
    if (result.length + point.length > max) break
    result += point
  }
  return result
}

/** One final text serializer for framework, validation and thrown execution failures. */
export function formatToolError(code: string, message: string, maxOutputChars = 8000, diagnostic?: ToolDiagnostic): string {
  const budget = Number.isSafeInteger(maxOutputChars) && maxOutputChars >= 96 ? maxOutputChars : 96
  const safeCode = typeof code === 'string' && /^[a-z][a-z0-9_]{0,63}$/u.test(code) ? code : 'tool_execution_error'
  const prefix = `Tool error (${safeCode}): `
  try {
    const reason = diagnostic && Object.hasOwn(descriptions, diagnostic.reason) ? diagnostic.reason : undefined
    const body = safeText(diagnostic?.message ?? message)
    const base = prefix + (reason ? `[${reason}] ` : '') + body
    if (base.length > budget) return clipped(base, budget - ' …omitted'.length) + ' …omitted'
    let result = base
    // Optional details never displace the code, reason or human-readable cause.
    for (const [label, value] of [
      ['Field', diagnostic?.field], ['Expected', diagnostic?.expected], ['Hint', diagnostic?.hint],
    ]) {
      if (typeof value !== 'string') continue
      if (label === 'Field' && !/^(?:path|query|mode|scope|cursor|command|reason|patch|startLine|endLine|maxLines|includePagePlan|recursive|maxDepth|limit|pattern|caseSensitive|includeIgnored|includeHidden|regex|operation|revision|snapshotId|sourceOffset|sourceLimit|maxOutputChars|steps(?:\[\d+\](?:\.(?:id|content|status))?)?|stepIds(?:\[\d+\])?)$/u.test(value)) continue
      const part = ` ${label}: ${safeText(value, 512)}`
      if (result.length + part.length <= budget) result += part
    }
    return result
  } catch {
    return clipped(prefix + 'Error details unavailable.', budget)
  }
}

/** Native JSON failures keep their envelope. Shrink optional details before the cause. */
export function formatToolErrorJson(
  error: Record<string, unknown>,
  maxOutputChars = 8000,
  envelope: Record<string, unknown> = { ok: false },
): string {
  const budget = Number.isSafeInteger(maxOutputChars) && maxOutputChars >= 96 ? maxOutputChars : 96
  let fallbackCode = 'tool_execution_error'
  try {
    if (typeof error.code === 'string' && /^[a-z][a-z0-9_]{0,63}$/u.test(error.code)) fallbackCode = error.code
  } catch { /* A hostile diagnostic still gets a bounded fallback. */ }
  try {
    const fields = { ...error }
    for (const key of ['message', 'hint', 'expected', 'field']) {
      if (typeof fields[key] === 'string') fields[key] = safeText(fields[key] as string)
    }
    const serialize = () => JSON.stringify({ ...envelope, error: fields })
    if (serialize().length <= budget) return serialize()
    for (const key of ['hint', 'expected', 'field']) {
      delete fields[key]
      if (serialize().length <= budget) return serialize()
    }
    if (typeof fields.message === 'string') {
      const message = fields.message
      fields.message = '…'
      if (serialize().length <= budget) {
        const points = [...message]
        let low = 0, high = points.length
        while (low < high) {
          const mid = Math.ceil((low + high) / 2)
          fields.message = points.slice(0, mid).join('') + '…'
          if (serialize().length <= budget) low = mid
          else high = mid - 1
        }
        fields.message = points.slice(0, low).join('') + '…'
        return serialize()
      }
      delete fields.message
    }
    if (serialize().length <= budget) return serialize()
    delete fields.reason
    if (serialize().length <= budget) return serialize()
  } catch { /* No raw exception or partially serialized JSON escapes the fallback. */ }
  return JSON.stringify({ ...envelope, error: { code: fallbackCode } })
}

/** Older tools keep their canonical error schema; oversized errors get a safe bounded fallback. */
export function boundedErrorContent(content: string, code: string, budget = 8000): string {
  if (content.length <= budget) return content
  try {
    JSON.parse(content)
    const compact = JSON.stringify({ error: { code, reason: 'output_limit' } })
    if (compact.length <= budget) return compact
  } catch { /* Plain-text legacy errors use the same text formatter. */ }
  return formatToolError(code, '', budget, toolDiagnostic('output_limit'))
}
