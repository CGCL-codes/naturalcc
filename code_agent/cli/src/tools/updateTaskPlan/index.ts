import type { Tool } from '../types.js'
import { executeUpdateTaskPlan } from './execute.js'
import { parseUpdateTaskPlanInput, updateTaskPlanSchema } from './schema.js'

export const updateTaskPlanTool: Tool = {
  ...updateTaskPlanSchema,
  parse: parseUpdateTaskPlanInput,
  execute: executeUpdateTaskPlan,
}

export {
  MAX_TASK_SNAPSHOT_QUERY_LIMIT,
  MAX_TASK_STEPS,
  MAX_TASK_STEP_TEXT_BYTES,
  MAX_TASK_UPDATE_REASON_BYTES,
  parseUpdateTaskPlanInput,
} from './schema.js'
