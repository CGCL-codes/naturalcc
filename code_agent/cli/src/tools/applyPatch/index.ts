import type { Tool } from '../types.js'
import { executeApplyPatch } from './execute.js'
import { parseApplyPatchInput } from './parser.js'
import { applyPatchSchema } from './schema.js'
import type { ApplyPatchDependencies } from './types.js'

export function createApplyPatchTool(
  dependencies: ApplyPatchDependencies = {},
): Tool {
  return {
    ...applyPatchSchema,
    parse: parseApplyPatchInput,
    execute: (input, context) => executeApplyPatch(input, context, dependencies),
  }
}

export const applyPatchTool: Tool = createApplyPatchTool()

export { parseApplyPatchInput, parseHunkHeader } from './parser.js'
export type {
  ApplyPatchDisplayFile,
  ApplyPatchDisplayPayload,
  ApplyPatchDisplayRecord,
  ParsedPatchHunkHeader,
} from './types.js'
