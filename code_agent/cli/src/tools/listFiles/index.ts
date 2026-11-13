import type { Tool } from '../types.js'
import { createRipgrepRunner, type RipgrepRunner } from '../backends/ripgrep/index.js'
import { executeListFiles } from './execute.js'
import { listFilesSchema, parseListFilesInput } from './schema.js'

// [maint-note:2f43d4cef2ef]
// 维护导航：createListFilesTool；下列为本函数源码中的接口与路径索引。
// 输入 runner：RipgrepRunner；缺省表达式 createRipgrepRunner()。
// 返回契约：Tool。
// 返回路径：{ ...listFilesSchema, parse: parseListFilesInput, execute: (input, context) => executeListFiles(input, context, runner), }。
// [/maint-note:2f43d4cef2ef]
export function createListFilesTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    ...listFilesSchema,
    parse: parseListFilesInput,
    execute: (input, context) => executeListFiles(input, context, runner),
  }
}

export const listFilesTool: Tool = createListFilesTool()

export { parseListFilesInput } from './schema.js'
