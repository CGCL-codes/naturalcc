import type { TaskPlanOperation, TaskStepStatus } from '../../task/contracts.js'
import type { ToolSchema } from '../types.js'
import { inputError } from '../diagnostics.js'
import { assertKnownKeys, assertObject } from '../toolUtils.js'

export const MAX_TASK_STEPS = 24
export const MAX_TASK_STEP_TEXT_BYTES = 512
export const MAX_TASK_UPDATE_REASON_BYTES = 512
export const MAX_TASK_SNAPSHOT_QUERY_LIMIT = 8
export const MAX_TASK_SOURCE_PAGE_CHARS = 4_000
const MAX_TASK_ID_CHARS = 128

const stepProperties = {
  id: { type: 'string', minLength: 1, maxLength: MAX_TASK_ID_CHARS },
  content: { type: 'string', minLength: 1, maxLength: MAX_TASK_STEP_TEXT_BYTES },
  status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] },
}

export const updateTaskPlanSchema: ToolSchema = {
  name: 'update_task_plan',
  description: [
    'Manage an optional task plan for this run and read bounded historical run snapshots.',
    'Simple answers, local explanations, and one clear operation should normally proceed without a plan.',
    'Create a short plan only for multi-stage work or when complexity grows; update it after a stage completes, focus changes, or material replanning is needed.',
    'Once a plan exists it cannot be cleared. Before a normal final answer, use claim_run_complete with the current revision and every current step ID exactly once.',
    'A completion claim records model-declared completion only; it is not quality verification.',
    'list_snapshots and read_snapshot report sourceReadable/sourceChars for each historical run whose full original prompt still exists in the committed thread.',
    'When sourceReadable is true, call read_snapshot again with the same snapshotId plus sourceOffset and sourceLimit (at most 4000 characters per page) to page through that original text; sourceExcerpt.nextOffset continues the range. This is the way to recover details that fell out of the current request view.',
    'Fields per operation (validated strictly after parsing): create_plan sends steps[{content,status}] only (no step id — ids are assigned); update_plan sends revision, reason, and steps[{id?,content,status}]; claim_run_complete sends revision and stepIds covering every current step ID exactly once; list_snapshots sends only an optional limit; read_snapshot sends snapshotId plus optional sourceOffset/sourceLimit to page through the original prompt text.',
    'Do not send fields that do not belong to the chosen operation; such calls are rejected before any state changes.',
  ].join(' '),
  readOnly: true,
  requiresApproval: false,
  repeatKind: 'read',
  taskManagement: true,
  // the WIRE schema is a single root object with an explicit
  // `type: "object"` and one shared `operation` enum — the earlier root-level
  // oneOf (branches typed as objects but the root itself untyped) was
  // rejected by real OpenAI-compatible endpoints with a schema 400 on every
  // request. Per-operation required/forbidden fields stay enforced by
  // parseUpdateTaskPlanInput before any state effect; the wire shape is
  // deliberately broader for endpoint compatibility.
  parameters: {
    type: 'object',
    properties: {
      operation: {
        type: 'string',
        enum: ['create_plan', 'update_plan', 'claim_run_complete', 'list_snapshots', 'read_snapshot'],
      },
      steps: {
        type: 'array',
        minItems: 1,
        maxItems: MAX_TASK_STEPS,
        items: {
          type: 'object',
          properties: stepProperties,
          required: ['content', 'status'],
          additionalProperties: false,
        },
      },
      revision: { type: 'integer', minimum: 1 },
      reason: { type: 'string', minLength: 1, maxLength: MAX_TASK_UPDATE_REASON_BYTES },
      stepIds: {
        type: 'array',
        minItems: 1,
        maxItems: MAX_TASK_STEPS,
        items: { type: 'string', minLength: 1, maxLength: MAX_TASK_ID_CHARS },
      },
      limit: { type: 'integer', minimum: 1, maximum: MAX_TASK_SNAPSHOT_QUERY_LIMIT },
      snapshotId: { type: 'string', minLength: 1, maxLength: MAX_TASK_ID_CHARS },
      sourceOffset: { type: 'integer', minimum: 0 },
      sourceLimit: { type: 'integer', minimum: 1, maximum: MAX_TASK_SOURCE_PAGE_CHARS },
    },
    required: ['operation'],
    additionalProperties: false,
  },
}

// [maint-note:0b5557d26799]
// 维护导航：parseUpdateTaskPlanInput；下列为本函数源码中的接口与路径索引。
// 输入 input：Record<string, unknown>。
// 返回契约：TaskPlanOperation。
// 分支条件：typeof input.operation !== 'string'。
// 分支条件：switch input.operation。
// 分支条件：case 'create_plan'。
// 分支条件：case 'update_plan'。
// 分支条件：case 'claim_run_complete'。
// 分支条件：case 'list_snapshots'。
// 分支条件：case 'read_snapshot'。
// 返回路径：{ operation: input.operation, steps: parseSteps(input.steps, false) }。
// 返回路径：{ operation: input.operation, revision: positiveInteger(input.revision, 'revision'), reason: boundedString(input.reason, 'reason', MAX_TASK_UPDATE_REASON_BYTES), steps: parseSteps(input.steps, true), }。
// 返回路径：{ operation: input.operation, revision: positiveInteger(input.revision, 'revision'), stepIds: parseIds(input.stepIds), }。
// 返回路径：{ operation: input.operation, limit: input.limit === undefined ? undefined : boundedInteger(input.limit, 'limit', MAX_TASK_SNAPSHOT_QUERY_LIMIT), }。
// 返回路径：{ operation: input.operation, snapshotId: boundedString(input.snapshotId, 'snapshotId', MAX_TASK_ID_CHARS), ...(input.sourceOffset === undefined ? {} : { sourceOffset: nonNegativeInteger(input.sourceOffset, 'sourceOffset') }), ...(input.sou。
// 显式抛出：inputError('operation must be a supported string', input.operation === undefined ? 'missing_required_field' : 'invalid_type', 'operation', 'supported operation string')。
// 显式抛出：inputError(`Unsupported task plan operation: ${input.operation}`, 'invalid_value', 'operation', 'create_plan, update_plan, claim_run_complete, list_snapshots or read_snapshot')。
// 直接调用：assertObject。
// 直接调用：inputError。
// 直接调用：assertKnownKeys。
// 直接调用：parseSteps。
// 直接调用：positiveInteger。
// 直接调用：boundedString。
// 直接调用：parseIds。
// 直接调用：boundedInteger。
// 直接调用：nonNegativeInteger。
// [/maint-note:0b5557d26799]
export function parseUpdateTaskPlanInput(input: Record<string, unknown>): TaskPlanOperation {
  assertObject(input)
  if (typeof input.operation !== 'string') {
    throw inputError('operation must be a supported string', input.operation === undefined ? 'missing_required_field' : 'invalid_type', 'operation', 'supported operation string')
  }
  switch (input.operation) {
    case 'create_plan':
      assertKnownKeys(input, ['operation', 'steps'])
      return { operation: input.operation, steps: parseSteps(input.steps, false) }
    case 'update_plan':
      assertKnownKeys(input, ['operation', 'revision', 'reason', 'steps'])
      return {
        operation: input.operation,
        revision: positiveInteger(input.revision, 'revision'),
        reason: boundedString(input.reason, 'reason', MAX_TASK_UPDATE_REASON_BYTES),
        steps: parseSteps(input.steps, true),
      }
    case 'claim_run_complete':
      assertKnownKeys(input, ['operation', 'revision', 'stepIds'])
      return {
        operation: input.operation,
        revision: positiveInteger(input.revision, 'revision'),
        stepIds: parseIds(input.stepIds),
      }
    case 'list_snapshots':
      assertKnownKeys(input, ['operation', 'limit'])
      return {
        operation: input.operation,
        limit: input.limit === undefined
          ? undefined
          : boundedInteger(input.limit, 'limit', MAX_TASK_SNAPSHOT_QUERY_LIMIT),
      }
    case 'read_snapshot':
      assertKnownKeys(input, ['operation', 'snapshotId', 'sourceOffset', 'sourceLimit'])
      return {
        operation: input.operation,
        snapshotId: boundedString(input.snapshotId, 'snapshotId', MAX_TASK_ID_CHARS),
        ...(input.sourceOffset === undefined
          ? {}
          : { sourceOffset: nonNegativeInteger(input.sourceOffset, 'sourceOffset') }),
        ...(input.sourceLimit === undefined
          ? {}
          : { sourceLimit: boundedInteger(input.sourceLimit, 'sourceLimit', MAX_TASK_SOURCE_PAGE_CHARS) }),
      }
    default:
      throw inputError(`Unsupported task plan operation: ${input.operation}`, 'invalid_value', 'operation', 'create_plan, update_plan, claim_run_complete, list_snapshots or read_snapshot')
  }
}

// [maint-note:e7db6636ec57]
// 维护导航：parseSteps；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 allowId：boolean。
// 返回契约：Array<{ id?: string; content: string; status: TaskStepStatus }>。
// 分支条件：!Array.isArray(value) || value.length < 1 || value.length > MAX_TASK_STEPS。
// 分支条件：inProgress > 1。
// 返回路径：steps。
// 显式抛出：inputError(`steps must contain 1 to ${MAX_TASK_STEPS} items`, value === undefined ? 'missing_required_field' : !Array.isArray(value) ? 'invalid_type' : 'out_of_range', 'steps', `array with 1..${MAX_TASK_STEPS} items`)。
// 显式抛出：inputError('At most one task step may be in_progress', 'conflicting_fields', 'steps', 'at most one in_progress step')。
// 直接调用：Array.isArray。
// 直接调用：inputError。
// 直接调用：value.map。
// [/maint-note:e7db6636ec57]
function parseSteps(
  value: unknown,
  allowId: boolean,
): Array<{ id?: string; content: string; status: TaskStepStatus }> {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_TASK_STEPS) {
    throw inputError(`steps must contain 1 to ${MAX_TASK_STEPS} items`, value === undefined ? 'missing_required_field' : !Array.isArray(value) ? 'invalid_type' : 'out_of_range', 'steps', `array with 1..${MAX_TASK_STEPS} items`)
  }
  let inProgress = 0
  const steps = value.map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw inputError(`steps[${index}] must be an object`, 'invalid_type', `steps[${index}]`, 'object')
    }
    const step = item as Record<string, unknown>
    assertKnownKeys(step, allowId ? ['id', 'content', 'status'] : ['content', 'status'])
    const id = step.id === undefined
      ? undefined
      : boundedString(step.id, `steps[${index}].id`, MAX_TASK_ID_CHARS)
    const content = boundedString(step.content, `steps[${index}].content`, MAX_TASK_STEP_TEXT_BYTES)
    const status = stepStatus(step.status, index)
    if (status === 'in_progress') inProgress += 1
    return { ...(id === undefined ? {} : { id }), content, status }
  })
  if (inProgress > 1) throw inputError('At most one task step may be in_progress', 'conflicting_fields', 'steps', 'at most one in_progress step')
  return steps
}

// [maint-note:38d444a58789]
// 维护导航：parseIds；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 返回契约：string[]。
// 分支条件：!Array.isArray(value) || value.length < 1 || value.length > MAX_TASK_STEPS。
// 返回路径：value.map((item, index) => boundedString(item, `stepIds[${index}]`, MAX_TASK_ID_CHARS))。
// 显式抛出：inputError(`stepIds must contain 1 to ${MAX_TASK_STEPS} items`, value === undefined ? 'missing_required_field' : !Array.isArray(value) ? 'invalid_type' : 'out_of_range', 'stepIds', `array with 1..${MAX_TASK_STEPS} items`)。
// 直接调用：Array.isArray。
// 直接调用：inputError。
// 直接调用：value.map。
// [/maint-note:38d444a58789]
function parseIds(value: unknown): string[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_TASK_STEPS) {
    throw inputError(`stepIds must contain 1 to ${MAX_TASK_STEPS} items`, value === undefined ? 'missing_required_field' : !Array.isArray(value) ? 'invalid_type' : 'out_of_range', 'stepIds', `array with 1..${MAX_TASK_STEPS} items`)
  }
  return value.map((item, index) => boundedString(item, `stepIds[${index}]`, MAX_TASK_ID_CHARS))
}

// [maint-note:dcd728e4264d]
// 维护导航：stepStatus；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 index：number。
// 返回契约：TaskStepStatus。
// 分支条件：value === 'pending' || value === 'in_progress' || value === 'completed'。
// 返回路径：value。
// 显式抛出：inputError(`steps[${index}].status must be pending, in_progress, or completed`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', `steps[${index}].status`, 'pending, in_progress o。
// 直接调用：inputError。
// [/maint-note:dcd728e4264d]
function stepStatus(value: unknown, index: number): TaskStepStatus {
  if (value === 'pending' || value === 'in_progress' || value === 'completed') return value
  throw inputError(`steps[${index}].status must be pending, in_progress, or completed`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', `steps[${index}].status`, 'pending, in_progress or completed')
}

// [maint-note:4c2bdc6844d2]
// 维护导航：boundedString；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 label：string。
// 输入 maxBytes：number。
// 返回契约：string。
// 分支条件：typeof value !== 'string' || !value.trim()。
// 分支条件：value.includes('\0')。
// 分支条件：Buffer.byteLength(value, 'utf8') > maxBytes。
// 返回路径：value。
// 显式抛出：inputError(`${label} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', label, 'non-empty string')。
// 显式抛出：inputError(`${label} must not contain NUL`, 'invalid_value', label, 'text without NUL')。
// 显式抛出：inputError(`${label} exceeds ${maxBytes} UTF-8 bytes`, 'out_of_range', label, `at most ${maxBytes} UTF-8 bytes`)。
// 直接调用：value.trim。
// 直接调用：inputError。
// 直接调用：value.includes。
// 直接调用：Buffer.byteLength。
// [/maint-note:4c2bdc6844d2]
function boundedString(value: unknown, label: string, maxBytes: number): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw inputError(`${label} must be a non-empty string`, value === undefined ? 'missing_required_field' : typeof value !== 'string' ? 'invalid_type' : 'invalid_value', label, 'non-empty string')
  }
  if (value.includes('\0')) throw inputError(`${label} must not contain NUL`, 'invalid_value', label, 'text without NUL')
  if (Buffer.byteLength(value, 'utf8') > maxBytes) {
    throw inputError(`${label} exceeds ${maxBytes} UTF-8 bytes`, 'out_of_range', label, `at most ${maxBytes} UTF-8 bytes`)
  }
  return value
}

// [maint-note:74e8519bc938]
// 维护导航：positiveInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 label：string。
// 返回契约：number。
// 返回路径：boundedInteger(value, label, Number.MAX_SAFE_INTEGER)。
// 直接调用：boundedInteger。
// [/maint-note:74e8519bc938]
function positiveInteger(value: unknown, label: string): number {
  return boundedInteger(value, label, Number.MAX_SAFE_INTEGER)
}

// [maint-note:18a690c7366a]
// 维护导航：nonNegativeInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 label：string。
// 返回契约：number。
// 分支条件：typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0。
// 返回路径：value。
// 显式抛出：inputError(`${label} must be a non-negative integer`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', label, 'safe integer >= 0')。
// 直接调用：Number.isSafeInteger。
// 直接调用：inputError。
// [/maint-note:18a690c7366a]
function nonNegativeInteger(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw inputError(`${label} must be a non-negative integer`, typeof value !== 'number' ? 'invalid_type' : 'out_of_range', label, 'safe integer >= 0')
  }
  return value
}

// [maint-note:ec38aa014932]
// 维护导航：boundedInteger；下列为本函数源码中的接口与路径索引。
// 输入 value：unknown。
// 输入 label：string。
// 输入 maximum：number。
// 返回契约：number。
// 分支条件：typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > maximum。
// 返回路径：value。
// 显式抛出：inputError(`${label} must be an integer from 1 to ${maximum}`, value === undefined ? 'missing_required_field' : typeof value !== 'number' ? 'invalid_type' : 'out_of_range', label, `integer 1..${maximum}`)。
// 直接调用：Number.isSafeInteger。
// 直接调用：inputError。
// [/maint-note:ec38aa014932]
function boundedInteger(value: unknown, label: string, maximum: number): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > maximum) {
    throw inputError(`${label} must be an integer from 1 to ${maximum}`, value === undefined ? 'missing_required_field' : typeof value !== 'number' ? 'invalid_type' : 'out_of_range', label, `integer 1..${maximum}`)
  }
  return value
}
