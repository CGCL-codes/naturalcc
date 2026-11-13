import type { Tool } from '../types.js'
import { createRipgrepRunner, type RipgrepRunner } from '../backends/ripgrep/index.js'
import { executeSearchText } from './execute.js'
import { parseSearchTextInput, searchTextSchema } from './schema.js'

// [maint-note:8d02faab3ca0]
// 维护导航：createSearchTextTool；下列为本函数源码中的接口与路径索引。
// 输入 runner：RipgrepRunner；缺省表达式 createRipgrepRunner()。
// 返回契约：Tool。
// 返回路径：{ ...searchTextSchema, parse: parseSearchTextInput, execute: (input, context) => executeSearchText(input, context, runner), }。
// [/maint-note:8d02faab3ca0]
export function createSearchTextTool(runner: RipgrepRunner = createRipgrepRunner()): Tool {
  return {
    ...searchTextSchema,
    parse: parseSearchTextInput,
    execute: (input, context) => executeSearchText(input, context, runner),
  }
}

export const searchTextTool: Tool = createSearchTextTool()

export {
  MAX_SEARCH_QUERY_UTF8_BYTES,
  parseSearchTextInput,
} from './schema.js'
export type { SearchTextInput } from './schema.js'
