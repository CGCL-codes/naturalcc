import type { Tool } from '../types.js'
import { executeApplyPatch } from './execute.js'
import { parseApplyPatchInput } from './parser.js'
import { applyPatchSchema } from './schema.js'
import type { ApplyPatchDependencies } from './types.js'

// [maint-note:7b4cea65cadc]
// 维护导航：createApplyPatchTool；下列为本函数源码中的接口与路径索引。
// 输入 dependencies：ApplyPatchDependencies；缺省表达式 {}。
// 返回契约：Tool。
// 返回路径：{ ...applyPatchSchema, parse: parseApplyPatchInput, execute: (input, context) => executeApplyPatch(input, context, dependencies), }。
// [/maint-note:7b4cea65cadc]
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
