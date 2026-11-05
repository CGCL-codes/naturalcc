import type { Tool } from '../types.js'
import { executeReadFile } from './execute.js'
import { parseReadFileInput, readFileSchema } from './schema.js'

export const readFileTool: Tool = {
  ...readFileSchema,
  parse: parseReadFileInput,
  execute: executeReadFile,
}

export {
  DEFAULT_READ_PAGE_LINES,
  MAX_READ_PAGE_LINES,
  MAX_SUGGESTED_CONTINUATION_RANGES,
  MIN_READ_FILE_OUTPUT_CHARS,
  parseReadFileInput,
} from './schema.js'
export type { ContinuationRange, PagePlan, ReadFileInput } from './schema.js'
export {
  getLastPlanMeasuredSegments,
  PLAN_HORIZON_LINES,
  serializeReadFilePage,
} from './execute.js'
